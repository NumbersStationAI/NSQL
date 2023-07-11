"""Text2SQL dataset class."""

import copy
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any

import jsonlines
import sqlglot
from data_utils import (
    clean_str,
    convert_criteria2sql_instance,
    convert_css_nvbench_instance,
    convert_kummerfeld_instance,
    convert_sede_instance,
    convert_spider_instance,
    convert_sql_create_context_instance,
    convert_squall_instance,
    convert_wikisql_instance,
    escape_everything,
    read_tables_json,
    serialize_dict_to_str,
)
from datasets import load_dataset
from prompt_formatters import RajkumarFormatter
from pydantic import BaseModel
from rich.console import Console
from schema import ForeignKey, Table, TableColumn
from sqlglot import parse_one
from tqdm.auto import tqdm
from transformers import AutoTokenizer

console = Console(soft_wrap=True)


class Text2SQLData(BaseModel):
    """Text2SQL data class."""

    instruction: str
    output: str
    source: str


class Text2SQLDataset(ABC):
    """Text2SQL dataset class."""

    def __init__(
        self,
        name: str,
        train_data_file: str,
        val_data_file: str,
        test_data_file: str,
        schema_file: str,
        context_length: int,
        tokenizer_name: str,
        **kwargs: Any,
    ) -> None:
        """Initialize."""
        self.name = name
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.test_data_file = test_data_file
        self.schema_file = schema_file
        self.context_length = context_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.clean_question = True
        self.process_init_kwargs(**kwargs)

    def process_init_kwargs(self, **kwargs: Any) -> None:
        """Process init kwargs."""
        pass

    @abstractmethod
    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        raise NotImplementedError

    @abstractmethod
    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema."""
        raise NotImplementedError

    def _is_parseable(self, sql: str) -> bool:
        try:
            res: sqlglot.expressions.Expression | None = parse_one(sql, read="sqlite")
            return res is not None
        except Exception:
            return False

    def _format_example(
        self,
        ex: dict[str, Any],
        schema: dict[str, dict[str, Table]],
        prompt_formatter: RajkumarFormatter,
        gold_sql_key: str,
    ) -> tuple[str, str] | None:
        if not self._is_parseable(ex[gold_sql_key]):
            print("BAD:::", ex[gold_sql_key])
            return None

        db_id = ex.get("db_id", "database")
        db_schema = schema[db_id]
        tables_to_add = list(db_schema.keys())

        if self.clean_question:
            question = clean_str(ex["question"]).strip("'").strip('"')
        else:
            question = ex["question"].strip("'").strip('"')
        table_text = prompt_formatter.table_sep.join(
            prompt_formatter.format_all_tables(
                [db_schema[t] for t in tables_to_add], question
            )
        )

        input_str = prompt_formatter.format_prompt(question, table_text)
        output_str = prompt_formatter.format_gold_output(ex[gold_sql_key])
        return input_str, output_str

    def format_example(
        self,
        example: dict[str, Any],
        schema: dict[str, dict[str, Table]],
        prompt_formatter: RajkumarFormatter,
    ) -> dict[str, Any] | None:
        """Format example."""

        result = self._format_example(
            example,
            schema,
            prompt_formatter,
            "sql",
        )
        if not result:
            return None
        input_str, output_str = result
        input_str = input_str.strip() + "\n"
        output_str = output_str.strip()
        data_ex = dict(
            instruction=input_str,
            output=output_str,
            source=self.name,
        )
        return data_ex


class KummerfeldText2SQL(Text2SQLDataset):
    """Kummerfeld text2sql dataset from the text2sql-data repo."""

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        data_pathobj = Path(self.train_data_file)
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        for raw_ex in tqdm(json.load(data_pathobj.open()), desc="Loading data"):
            for ex in convert_kummerfeld_instance(raw_ex, schema=schema):
                if ex:
                    splits[ex["split"]].append(ex)
        return splits

    def mine_for_fks(
        self, data: list[list[str]], header: list[str]
    ) -> dict[str, list[tuple[str, str, list[str]]]]:
        """Mine for fks from schema."""
        # The Is Foreign Key column is not always correct so mine via exact match
        cur_tablename = None
        cur_database = None
        schema: dict = {}
        cur_table: dict[str, list] = {}
        for ex in data:
            if len(header) != len(ex):
                ex = ex[: len(header)]
            row = {h: r.strip() for h, r in zip(header, ex)}
            # Keep the type as only the first key
            # e.g. varchar(255) default null -> varchar
            table_name = row["Table Name"].lower()
            field_name = row["Field Name"].lower()
            database_name = row.get("Database name", "database").lower()
            if (
                table_name == "-"
                or table_name != cur_tablename
                or database_name != cur_database
            ):
                if len(cur_table) > 0:
                    if cur_database not in schema:
                        schema[cur_database] = {}
                    schema[cur_database][cur_tablename] = cur_table
                cur_table = {}
                cur_database = None
                cur_tablename = None
            if cur_tablename is None and table_name != "-":
                cur_tablename = table_name
                cur_table = {
                    "columns": [],
                    "pks": [],
                }
            if cur_database is None and database_name != "-":
                cur_database = database_name
            if cur_tablename is not None:
                assert cur_database is not None
                cur_table["columns"].append(field_name)
                if row["Is Primary Key"].strip().lower() in [
                    "yes",
                    "true",
                    "y",
                    "t",
                    "pri",
                ]:
                    cur_table["pks"].append(field_name)

        # Add last table
        assert cur_database is not None
        assert cur_tablename is not None
        schema[cur_database][cur_tablename] = cur_table

        # Find Fks by matching on field_name
        fks: dict[str, list[tuple[str, str, list[str]]]] = {}
        for database in schema:
            fks[database] = []
            for referenced_table in schema[database]:
                # Only want one key per column
                used_columns = set()
                for references_table in schema[database]:
                    if referenced_table == references_table:
                        continue
                    # Find all columns in referenced table that are in references table
                    matching_cols = [
                        c
                        for c in schema[database][referenced_table]["columns"]
                        if c in schema[database][references_table]["columns"]
                        and c not in used_columns
                    ]
                    matching_pk_cols = [
                        c
                        for c in matching_cols
                        if c in schema[database][references_table]["pks"]
                        and c.lower() != "id"
                    ]
                    used_columns.update(matching_cols)
                    if len(matching_pk_cols) > 0:
                        # Use the fk
                        fks[database].append(
                            (referenced_table, references_table, matching_pk_cols)
                        )
        return fks

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_pathobj = Path(self.schema_file)
        # Header is Table Name, Field Name, Is Primary Key, Is Foreign Key, Type
        data = [l.strip().split(",") for l in schema_pathobj.open().readlines()]
        header = [h.strip() for h in data[0]]
        data = data[1:]

        all_fks = self.mine_for_fks(data, header)
        schema: dict[str, dict[str, Table]] = {}
        cur_tablename = None
        cur_database = None
        cur_table: dict[str, list] = {}
        for ex in data:
            if len(header) != len(ex):
                ex = ex[: len(header)]
            row = {h: r.strip() for h, r in zip(header, ex)}
            # Keep the type as only the first key
            # e.g. varchar(255) default null -> varchar
            row_type = row["Type"].split("(")[0].lower()
            table_name = row["Table Name"].lower()
            field_name = row["Field Name"].lower()
            database_name = row.get("Database name", "database").lower()
            if (
                table_name == "-"
                or table_name != cur_tablename
                or database_name != cur_database
            ):
                if len(cur_table) > 0:
                    assert cur_database is not None
                    if cur_database not in schema:
                        schema[cur_database] = {}  # type: ignore
                    schema[cur_database][cur_tablename] = Table(  # type: ignore
                        name=cur_tablename,
                        columns=[
                            TableColumn(name=cn, dtype=ct)
                            for cn, ct in cur_table["columns"]
                        ],
                        pks=[
                            TableColumn(name=cn, dtype=ct)
                            for cn, ct in cur_table["pks"]
                        ],
                        fks=[
                            ForeignKey(
                                column=TableColumn(name=cn, dtype=ct),
                                references_name=rtn,
                                references_column=TableColumn(name=rn, dtype=rt),
                            )
                            for ((cn, ct), rtn, (rn, rt)) in cur_table["fks"]
                        ],
                        examples=[],
                    )
                cur_table = {}
                cur_database = None
                cur_tablename = None
            if cur_tablename is None and table_name != "-":
                cur_tablename = table_name
                cur_table = {
                    "columns": [],
                    "pks": [],
                    "fks": [],
                }
            if cur_database is None and database_name != "-":
                cur_database = database_name
            if cur_tablename is not None:
                assert cur_database is not None
                cur_table["columns"].append((field_name, row_type))
                if row["Is Primary Key"].strip().lower() in [
                    "yes",
                    "true",
                    "y",
                    "t",
                    "pri",
                ]:
                    cur_table["pks"].append((field_name, row_type))
                for fk_tuple in all_fks[cur_database]:
                    # referenced_table, references_table, matching_pk_cols
                    if fk_tuple[0] == cur_tablename and field_name in fk_tuple[2]:
                        cur_table["fks"].append(
                            (
                                (field_name, row_type),
                                fk_tuple[1],
                                (field_name, row_type),
                            )
                        )

        # Add last table
        assert cur_database is not None
        assert cur_tablename is not None
        schema[cur_database][cur_tablename] = Table(
            name=cur_tablename,
            columns=[TableColumn(name=cn, dtype=ct) for cn, ct in cur_table["columns"]],
            pks=[TableColumn(name=cn, dtype=ct) for cn, ct in cur_table["pks"]],
            fks=[
                ForeignKey(
                    column=TableColumn(name=cn, dtype=ct),
                    references_name=rtn,
                    references_column=TableColumn(name=rn, dtype=rt),
                )
                for ((cn, ct), rtn, (rn, rt)) in cur_table["fks"]
            ],
            examples=[],
        )
        return schema


class SedeText2SQL(Text2SQLDataset):
    """Sede text2sql dataset from the text2sql-data repo."""

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        for split in splits:
            if split == "dev":
                to_read_split = self.val_data_file
            elif split == "test":
                to_read_split = self.test_data_file
            else:
                to_read_split = self.train_data_file
            data_file = Path(to_read_split)
            for line in tqdm(
                data_file.open().readlines(), desc=f"Loading {split} data"
            ):
                raw_ex = json.loads(line)
                ex = convert_sede_instance(raw_ex, schema=schema)
                if ex:
                    splits[split].append(ex)
        return splits

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct = read_tables_json(self.schema_file)
        return schema_dct


class SpiderText2SQL(Text2SQLDataset):
    """Spider text2sql dataset adapted from Huggingface/Picard."""

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        all_data_for_demos: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "dev": [],
            "test": [],
        }
        for split in splits:
            if split in "dev":
                to_read_files = [Path(self.val_data_file)] if self.val_data_file else []
            elif split == "train":
                to_read_files = [Path(p) for p in self.train_data_file.split("@")]
            elif split == "test":
                to_read_files = (
                    [Path(self.test_data_file)] if self.test_data_file else []
                )
            else:
                to_read_files = []
            console.print(f"Loading {split} data", style="bold blue")
            for file in to_read_files:
                data_file = Path(file)
                try:
                    data = json.load(data_file.open())
                except json.decoder.JSONDecodeError:
                    data = [json.loads(line) for line in data_file.open()]
                convert_function = partial(
                    convert_spider_instance,
                    schema=schema,
                )
                for ex in tqdm(
                    map(
                        convert_function,
                        data,
                    ),
                    desc=f"Loading {split} data from {data_file.name}",
                    total=len(data),
                ):
                    if ex:
                        splits[split].append(ex)
                        all_data_for_demos[split].append(copy.deepcopy(ex))

        return splits

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct = read_tables_json(self.schema_file, lowercase=True)
        return schema_dct


class WikiSQL2SQL(Text2SQLDataset):
    """WikiSQL text2sql dataset from the text2sql-data repo."""

    example2table_name: dict[str, str] = {}

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        self.dataset = load_dataset("wikisql")
        schema_dct: dict[str, dict[str, Table]] = {}
        for split in sorted(self.dataset):
            for ex in tqdm(self.dataset[split], desc=f"Loading {split} data schema"):
                table = ex["table"]
                key = serialize_dict_to_str(ex)
                if key not in self.example2table_name:
                    self.example2table_name[
                        key
                    ] = f"table_{len(self.example2table_name)}"
                table_name = self.example2table_name[key]
                # Quote column names to handle spaces
                column_names = [
                    f'"{escape_everything(col)}"' for col in table["header"]
                ]
                if table_name in schema_dct:
                    continue
                columns = [
                    TableColumn(name=col, dtype=typ)
                    for col, typ in zip(column_names, table["types"])
                ]
                examples = [
                    {column_names[i]: row[i] for i in range(len(row))}
                    for row in table["rows"]
                ]
                if table_name not in schema_dct:
                    schema_dct[table_name] = {}
                    # WikiSQL uses table_name for both db and table name
                    schema_dct[table_name][table_name] = Table(
                        name=table_name, columns=columns, examples=examples
                    )
        return schema_dct

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        for split in splits:
            split_to_use = split
            if split == "dev":
                split_to_use = "validation"
            for line in tqdm(self.dataset[split_to_use], desc=f"Loading {split} data"):
                key = serialize_dict_to_str(line)
                line_to_use = line
                line_to_use["table"]["name"] = self.example2table_name[key]
                ex = convert_wikisql_instance(line_to_use, schema=schema)
                if ex:
                    splits[split].append(ex)
            console.print(
                f"Loaded {split} data: {len(splits[split])} over {len(self.dataset[split_to_use])}"
            )
        return splits

    def _get_table_schema(self, table: Table) -> list[str]:
        "Get table schema from Table."
        return sorted([_.name.lower() for _ in table.columns]) if table.columns else []


class MimicsqlText2SQL(Text2SQLDataset):
    """Mimicsql text2sql dataset from the TREQS repo."""

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        data_file_mapping = {
            "train": self.train_data_file,
            "dev": self.val_data_file,
            "test": self.test_data_file,
        }
        for split in splits:
            to_read_files = (
                [Path(p) for p in data_file_mapping[split].split("@")]
                if split in data_file_mapping
                else []
            )
            console.print(f"Loading {split} data", style="bold blue")

            for file in to_read_files:
                data_file = Path(file)
                try:
                    data = json.load(data_file.open())
                except json.decoder.JSONDecodeError:
                    data = [json.loads(line) for line in data_file.open()]
                # Convert the data to spider compatible
                for i in range(len(data)):
                    data[i]["db_id"] = "mimicsql"
                    data[i]["question"] = data[i]["question_refine"]
                    data[i]["query"] = data[i]["sql"]
                convert_function = partial(
                    convert_spider_instance,
                    schema=schema,
                )
                for ex in tqdm(
                    map(
                        convert_function,
                        data,
                    ),
                    desc=f"Loading {split} data from {data_file.name}",
                    total=len(data),
                ):
                    if ex:
                        splits[split].append(ex)

        return splits

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct = read_tables_json(self.schema_file)
        return schema_dct


class Criteria2SQL2SQL(Text2SQLDataset):
    """Criteria2SQL text2sql dataset from https://github.com/xiaojingyu92/Criteria2SQL."""

    def process_init_kwargs(self, **kwargs: Any) -> None:
        """Process kwargs."""
        self.train_schema_file = kwargs.pop("train_schema_file", "")
        self.val_schema_file = kwargs.pop("val_schema_file", "")
        self.test_schema_file = kwargs.pop("test_schema_file", "")

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct: dict[str, dict[str, Table]] = {}
        schema_file_mapping = {
            "train": self.train_schema_file,
            "dev": self.val_schema_file,
            "test": self.test_schema_file,
        }

        for split in sorted(schema_file_mapping):
            with jsonlines.open(schema_file_mapping[split], "r") as f:
                for table in tqdm(
                    [line for line in f], desc=f"Loading {split} data schema"
                ):
                    table_name = f"table_{split}_{table['id']}"
                    # Quote column names to handle spaces
                    column_names = [
                        f'"{escape_everything(col)}"' for col in table["header"]
                    ]
                    columns = [
                        TableColumn(name=col, dtype=typ)
                        for col, typ in zip(column_names, table["types"])
                    ]
                    examples = [
                        {column_names[i]: row[i] for i in range(len(row))}
                        for row in table["rows"]
                    ]
                    if table_name not in schema_dct:
                        schema_dct[table_name] = {}
                        # Criteria2SQL is similar to WikiSQL and it uses table_name for
                        # both db and table name
                        schema_dct[table_name][table_name] = Table(
                            name=table_name, columns=columns, examples=examples
                        )
        return schema_dct

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        data_file_mapping = {
            "train": self.train_data_file,
            "dev": self.val_data_file,
            "test": self.test_data_file,
        }

        for split in sorted(data_file_mapping):
            with jsonlines.open(data_file_mapping[split], "r") as f:
                all_samples = [line for line in f]
                for line in tqdm(all_samples, desc=f"Loading {split} data"):
                    line_to_use = line
                    line_to_use["db_id"] = f"table_{split}_{line['table_id']}"
                    ex = convert_criteria2sql_instance(line_to_use, schema=schema)
                    if ex:
                        splits[split].append(ex)
                console.print(
                    f"Loaded {split} data: {len(splits[split])} over {len(all_samples)}"
                )
        return splits


class SqlCreateContext2SQL(Text2SQLDataset):
    """sql-create-context text2sql dataset from huggingface."""

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct: dict[str, dict[str, Table]] = {}
        self.dataset = load_dataset("b-mc2/sql-create-context")
        for db_id, ex in tqdm(enumerate(self.dataset["train"])):
            for table_context in ex["context"].split(";"):
                table_context = table_context.strip()
                assert table_context.startswith("CREATE TABLE ")
                table_context = table_context[len("CREATE TABLE ") :].strip()
                table_name = table_context[: table_context.find("(")].strip()
                col_context = table_context[len(table_name) :].strip()[1:-1]
                cols = [col.strip().split(" ") for col in col_context.split(",")]
                columns = [TableColumn(name=col, dtype=typ) for col, typ in cols]

                if db_id not in schema_dct:
                    schema_dct[db_id] = {}
                if table_name not in schema_dct[db_id]:
                    schema_dct[db_id][table_name] = Table(
                        name=table_name, columns=columns
                    )
        return schema_dct

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}

        for split in sorted(self.dataset):
            for db_id, ex in tqdm(
                enumerate(self.dataset[split]), desc=f"Loading {split} data"
            ):
                line_to_use = ex
                line_to_use["db_id"] = db_id
                ex = convert_sql_create_context_instance(line_to_use, schema=schema)
                if ex:
                    splits[split].append(ex)
            console.print(
                f"Loaded {split} data: {len(splits[split])} over {len(self.dataset[split])}"
            )
        return splits


class Squall2SQL(Text2SQLDataset):
    """Squall text2sql dataset from huggingface."""

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct: dict[str, dict[str, Table]] = {}
        self.data = json.load(open(self.train_data_file, "r"))
        for i, ex in enumerate(self.data):
            table_name = f"table_{ex['tbl']}"
            cols = [["id", "number"]] + [
                [
                    f'"{escape_everything(col[0])}"',
                    col[3] if col[3] == "number" else "text",
                ]
                for col in ex["columns"]
            ]
            # Skip the table with duplicate column names
            if len(set([col[0] for col in cols])) != len(cols):
                continue
            # Skip the table with empty column name
            if '""' in [col[0] for col in cols]:
                continue
            columns = [TableColumn(name=col, dtype=typ) for col, typ in cols]
            if table_name not in schema_dct:
                schema_dct[table_name] = {}
            if table_name not in schema_dct[table_name]:
                schema_dct[table_name][table_name] = Table(
                    name=table_name, columns=columns
                )
        return schema_dct

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        split = "train"
        for i, ex in enumerate(self.data):
            line_to_use = ex
            line_to_use["db_id"] = f"table_{ex['tbl']}"
            if line_to_use["db_id"] not in schema:
                continue
            ex = convert_squall_instance(line_to_use, schema=schema)
            if ex:
                splits[split].append(ex)
        console.print(
            f"Loaded {split} data: {len(splits[split])} over {len(self.data)}"
        )
        return splits


class CSS2SQL(Text2SQLDataset):
    """CSS2SQL text2sql dataset from huggingface."""

    def process_init_kwargs(self, **kwargs: Any) -> None:
        """Process kwargs."""
        self.clean_question = False

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct = read_tables_json(self.schema_file)
        return schema_dct

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        ds = load_dataset("zhanghanchong/css")
        data_split_mapping = {
            "train": self.train_data_file,
            "dev": self.val_data_file,
            "test": self.test_data_file,
        }
        for split in splits:
            split_cnt = 0
            if data_split_mapping[split] is None:
                continue
            ex_cache: set = set([])
            to_read_splits = (
                [p for p in data_split_mapping[split].split("@")]
                if split in data_split_mapping
                else []
            )
            console.print(f"Loading {split} data", style="bold blue")
            for spt in to_read_splits:
                split_cnt += len(ds[spt])
                for line in ds[spt]:
                    line["question"] = line["question"].strip()
                    ex = convert_css_nvbench_instance(line, schema=schema)
                    if ex:
                        key = f"{ex['db_id']}###{ex['question']}"
                        if key in ex_cache:
                            continue
                        ex_cache.add(key)
                        splits[split].append(ex)
            console.print(f"Loaded {split} data: {len(splits[split])} over {split_cnt}")
        return splits


class NVBENCH2SQL(Text2SQLDataset):
    """NVBENCH2SQL text2sql dataset from https://github.com/TsinghuaDatabaseGroup/nvBench."""

    def load_schema(self) -> dict[str, dict[str, Table]]:
        """Load schema for each table in the database."""
        schema_dct: dict[str, dict[str, Table]] = {}

        db_files = [
            file
            for file in glob(self.schema_file + "**", recursive=True)
            if file.endswith(".sqlite")
        ]

        for db_file in db_files:
            db_id = os.path.basename(os.path.dirname(db_file))
            if db_id not in schema_dct:
                schema_dct[db_id] = {}

            # Connect to the SQLite database
            conn = sqlite3.connect(db_file)

            # Create a cursor object to execute SQL queries
            cursor = conn.cursor()

            # Get the list of tables in the database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            # Iterate over the tables and retrieve their schemas
            for table in tables:
                table_name = table[0]
                # Execute a PRAGMA query to get the schema of the table
                cursor.execute(f"PRAGMA table_info({table_name});")
                schema = cursor.fetchall()

                # Get the schema details
                columns = [
                    TableColumn(
                        name=column[1] if " " not in column[1] else f'"{column[1]}"',
                        dtype=column[2],
                    )
                    for column in schema
                ]

                if table_name not in schema_dct[db_id]:
                    schema_dct[db_id][table_name] = Table(
                        name=table_name, columns=columns
                    )

            # Close the cursor and the database connection
            cursor.close()
            conn.close()

        return schema_dct

    def load_data(
        self, schema: dict[str, dict[str, Table]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load data."""
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        data_file_mapping = {
            "train": self.train_data_file,
            "dev": self.val_data_file,
            "test": self.test_data_file,
        }
        for split in splits:
            split_cnt = 0
            if data_file_mapping[split] is None:
                continue
            ex_cache: set = set([])
            to_read_files = (
                [Path(p) for p in data_file_mapping[split].split("@")]
                if split in data_file_mapping
                else []
            )
            console.print(f"Loading {split} data", style="bold blue")
            for data_file in to_read_files:
                data = json.load(open(data_file, "r"))
                for k, v in data.items():
                    for nl_query in v["nl_queries"]:
                        split_cnt += 1
                        if len(nl_query.strip()) == 0:
                            continue
                        line = {
                            "db_id": v["db_id"],
                            "query": v["vis_query"]["data_part"]["sql_part"].strip(),
                            "question": nl_query.strip(),
                        }
                        ex = convert_css_nvbench_instance(line, schema=schema)
                        if ex:
                            key = f"{ex['db_id']}###{ex['question']}"
                            if key in ex_cache:
                                continue
                            ex_cache.add(key)
                            splits[split].append(ex)
            console.print(f"Loaded {split} data: {len(splits[split])} over {split_cnt}")
        return splits
