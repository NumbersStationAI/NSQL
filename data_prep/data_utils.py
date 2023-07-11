"""Training data prep utils."""
import json
import re
from collections import defaultdict
from typing import Any

import sqlglot
from kummerfeld_utils import preprocess_for_jsql
from schema import ForeignKey, Table, TableColumn

_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "OP"]


def escape_everything(string: str) -> str:
    """Escape everything.

    Args:
        string: string to escape

    Returns:
        Escaped string.
    """
    return json.dumps(string)[1:-1]


def serialize_dict_to_str(d: dict) -> str:
    """
    Serialize a dict into a str.

    Args:
        d: dict to serialize.

    Returns:
        serialized dict.
    """
    return json.dumps(d, sort_keys=True)


def read_tables_json(
    schema_file: str,
    lowercase: bool = False,
) -> dict[str, dict[str, Table]]:
    """Read tables json."""
    data = json.load(open(schema_file))
    db_to_tables = {}
    for db in data:
        db_name = db["db_id"]
        table_names = db["table_names_original"]
        db["column_names_original"] = [
            [x[0], x[1]] for x in db["column_names_original"]
        ]
        db["column_types"] = db["column_types"]
        if lowercase:
            table_names = [tn.lower() for tn in table_names]
        pks = db["primary_keys"]
        fks = db["foreign_keys"]
        tables = defaultdict(list)
        tables_pks = defaultdict(list)
        tables_fks = defaultdict(list)
        for idx, ((ti, col_name), col_type) in enumerate(
            zip(db["column_names_original"], db["column_types"])
        ):
            if ti == -1:
                continue
            if lowercase:
                col_name = col_name.lower()
                col_type = col_type.lower()
            if idx in pks:
                tables_pks[table_names[ti]].append(
                    TableColumn(name=col_name, dtype=col_type)
                )
            for fk in fks:
                if idx == fk[0]:
                    other_column = db["column_names_original"][fk[1]]
                    other_column_type = db["column_types"][fk[1]]
                    other_table = table_names[other_column[0]]
                    tables_fks[table_names[ti]].append(
                        ForeignKey(
                            column=TableColumn(name=col_name, dtype=col_type),
                            references_name=other_table,
                            references_column=TableColumn(
                                name=other_column[1], dtype=other_column_type
                            ),
                        )
                    )
            tables[table_names[ti]].append(TableColumn(name=col_name, dtype=col_type))
        db_to_tables[db_name] = {
            table_name: Table(
                name=table_name,
                columns=tables[table_name],
                pks=tables_pks[table_name],
                fks=tables_fks[table_name],
            )
            for table_name in tables
        }
    return db_to_tables


def clean_str(target: str) -> str:
    """Clean string for question."""
    if not target:
        return target

    target = re.sub(r"[^\x00-\x7f]", r" ", target)
    line = re.sub(r"''", r" ", target)
    line = re.sub(r"``", r" ", line)
    line = re.sub(r"\"", r"'", line)
    line = re.sub(r"[\t ]+", " ", line)
    return line.strip()


def case_sql(query: str) -> str:
    """Case sql query."""
    try:
        cased_sql = sqlglot.parse_one(query).sql()  # type: ignore
        # SQLGlot makes NOT <col> IN. We want <col> NOT IN for Spider
        cased_sql = re.sub(r"NOT\s+([^\s]+)\s+IN", r"\1 NOT IN", cased_sql)
        # Replace <> with !=
        cased_sql = cased_sql.replace("<>", "!=")
        return cased_sql
    except Exception:
        print("Cannot CASE this SQL")
        return query


def crude_remove_aliases(sql: str) -> str:
    """Cruder way of cleaning up aliases."""
    alias2cleanalias = {}
    new_sql = re.sub(r"[\t\s\n]+", " ", sql)
    for word in sql.split():
        if "." in word:
            alias = word.split(".")[0]
            if "alias" in alias:
                clean_alias = alias.split("alias")[0] + "_" + alias.split("alias")[1]
                alias2cleanalias[alias] = clean_alias
    for alias, clean_alias in alias2cleanalias.items():
        new_sql = new_sql.replace(alias, clean_alias)
    return new_sql


def remove_aliases(sql: str) -> str:
    """Remove aliases from SQL."""
    new_sql = re.sub(r"[\t\s\n]+", " ", sql)
    # Handle from
    alias2table = {}
    table2alias: dict[str, list[str]] = {}
    # Get substring from FROM to WHERE or to GROUP BY or to end
    inside_from = re.search(
        r"FROM (.*?) (WHERE|GROUP BY|ORDER BY|LIMIT|;)", new_sql, re.DOTALL
    )
    if not inside_from:
        inside_from = re.search(r"FROM (.*?)$", new_sql, re.DOTALL)
    if not inside_from:
        print("BAD FROM", sql)
    for from_clause in re.split(
        r",| INNER JOIN| OUTER JOIN| LEFT JOIN| RIGHT JOIN| JOIN| EXCEPT",
        inside_from.group(1),  # type: ignore
    ):
        # If JOIN table ON XXX, remove the ON XXX
        if " ON " in from_clause:
            from_clause = from_clause.split(" ON ")[0]
        if " AS " in from_clause:
            table = from_clause.split(" AS ")[0].strip()
            alias = from_clause.split(" AS ")[1].strip()
            alias2table[alias] = table
            # If we have two of the same tables in the from clause
            # must keep and handle aliases differently
            if table in table2alias:
                # If only one already in, start creating new aliases
                if len(table2alias[table]) == 1:
                    old_alias = table2alias[table][0]
                    table2alias[table] = [f"{table}_{len(table2alias[table])-1}"]
                    alias2table[old_alias] = table2alias[table][-1]
                table2alias[table].append(f"{table}_{len(table2alias[table])}")
                alias2table[alias] = table2alias[table][-1]
            else:
                table2alias[table] = [alias]
    # Now replace AS alias in from clauses where we can
    for from_clause in re.split(
        r",| INNER JOIN| OUTER JOIN| LEFT JOIN| RIGHT JOIN| JOIN| EXCEPT",
        inside_from.group(1),  # type: ignore
    ):
        if " ON " in from_clause:
            from_clause = from_clause.split(" ON ")[0]
        if " AS " in from_clause:
            table = from_clause.split(" AS ")[0].strip()
            alias = from_clause.split(" AS ")[1].strip()
            if len(table2alias[table]) == 1:
                new_sql = new_sql.replace(from_clause, " " + table)

    # Replace old aliases with new ones (or og table name)
    for al, table in alias2table.items():
        new_sql = new_sql.replace(al, table)

    # Replace table references as not needed with one table
    if len(alias2table) == 1:
        table = list(alias2table.values())[0]
        new_sql = new_sql.replace(table + ".", "")

    new_sql = re.sub(r"[\t\s\n]+", " ", new_sql)
    return new_sql


def get_table_alias_to_ref_map(sql: str) -> dict[str, set[str]]:
    """Get all aliases and the reference tables they point to.

    Key of None will be all unaliased tables.

    This accounts for both table AS T1 clauses and subexpressions.
    """
    try:
        parsed: sqlglot.expressions.Expression = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return defaultdict(set)
    # Get all table aliases - including CTEs
    mapping = defaultdict(set)
    all_table_aliases = list(parsed.find_all(sqlglot.exp.TableAlias))
    for tbl_alias in all_table_aliases:
        sql_parent = tbl_alias.parent
        if sql_parent:
            tbls = [
                table.name
                for table in sql_parent.find_all(sqlglot.exp.Table)
                if table.name != tbl_alias.name
            ]
            if tbls:
                mapping[tbl_alias.name].update(tbls)
    # Add any table without alias
    for table in parsed.find_all(sqlglot.exp.Table):
        if not table.alias or table.alias == table.name:
            mapping[None].add(table.name)
    return mapping


def format_to_match_schema(
    sql: str,
    schema: dict[str, Table],
) -> str:
    """Format the tables and columns in the query to match the schema."""
    table_alias_to_ref = get_table_alias_to_ref_map(sql)
    all_tables = set().union(*table_alias_to_ref.values())
    all_tables_lower = {t.lower() for t in all_tables}
    tablename2colset = {
        tbl.name: set([c.name for c in tbl.columns])
        for tbl in schema.values()
        if tbl.name.lower() in all_tables_lower
    }

    def transformer(node: sqlglot.Expression) -> sqlglot.Expression:
        if isinstance(node, sqlglot.exp.Column):
            for tbl in tablename2colset:
                for col in tablename2colset[tbl]:
                    # Due to table aliases, we don't want to make this a joint
                    # condition on the column and alias
                    if node.table and node.table.lower() == tbl.lower():
                        node.args["table"] = tbl
                    if node.name.lower() == col.lower():
                        node.args["this"] = col
                        break
        elif isinstance(node, sqlglot.exp.Table):
            for tbl in tablename2colset:
                if node.name.lower() == tbl.lower():
                    node.args["this"] = tbl
                    break
        return node

    parsed: sqlglot.expressions.Expression = sqlglot.parse_one(sql, read="sqlite")
    transformed_parsed = parsed.transform(transformer)
    return transformed_parsed.sql()


def convert_kummerfeld_instance(
    data: dict[str, Any],
    schema: dict[str, dict[str, Table]] = {},
    keep_vars: bool = False,
    keep_sql_vars: bool = False,
) -> list[dict[str, Any]]:
    """Convert a single instance of the data into a list of examples.

    Used for the text2sql-data repo from jkkummerfeld.
    """
    var_sql = None
    var_sql = data["sql"][0]  #
    parsed_results: list[dict[str, Any]] = []
    for sentence in data["sentences"]:
        text = sentence["text"]
        sql = preprocess_for_jsql(
            var_sql
        )  # Needed to do variable replacement correctly
        if not sql:
            raise ValueError(f"No SQL for sentence {sentence}")
        sql = str(sql)
        cleaned_sql = remove_aliases(sql)
        cleaned_sql = case_sql(cleaned_sql)
        crude_cleaned_sql = crude_remove_aliases(sql)
        crude_cleaned_sql = case_sql(crude_cleaned_sql)
        # Variable replacement
        if not keep_vars:
            for name in sentence["variables"]:
                value = sentence["variables"][name]
                if len(value) == 0:
                    for variable in data["variables"]:
                        if variable["name"] == name:
                            value = variable["example"]
                text = value.join(text.split(name))
                if not keep_sql_vars:
                    cleaned_sql = value.join(cleaned_sql.split(name))
                    crude_cleaned_sql = value.join(crude_cleaned_sql.split(name))
                    sql = value.join(sql.split(name))  # type: ignore

        # Query split is either train/dev/test or 0-9 for cross validation
        # We use test/0 for test, dev/1 for dev and the rest for train
        if data["query-split"] == "N/A":
            # Flip a coin to decide if it's train or test or valid
            output_file = sentence["question-split"]
        else:
            if data["query-split"] == "test" or data["query-split"] == "0":
                output_file = "test"
            elif data["query-split"] == "dev" or data["query-split"] == "1":
                output_file = "dev"
            else:
                output_file = "train"

        db_id = sentence.get("database", "database").lower()
        try:
            cleaned_sql = format_to_match_schema(cleaned_sql, schema[db_id])
        except Exception:
            print("ERROR")
            continue
        parsed_results.append(
            {
                "question": text,
                "sql": cleaned_sql,
                "split": output_file,
                "db_id": db_id,
            }
        )

    return parsed_results


def convert_sede_instance(
    data: dict[str, str],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example.

    Used for the sede dataset.
    """
    # clean title and description
    cleaned_title = clean_str(data["Title"])
    cleaned_description = clean_str(data["Description"])

    # clean SQL query
    cleaned_sql = None
    # cleaned_sql_with_values = None
    if data["QueryBody"]:
        target_with_values = str(preprocess_for_jsql(data["QueryBody"]))
        target_with_values = case_sql(target_with_values)
        if target_with_values:
            target_tokens = target_with_values.strip(";").split()
            target = case_sql(" ".join(target_tokens))
            cleaned_sql = target
        # Handle With statement by removing WITH part before SELECT
        # Example:
        #    WITH no activity in last 6 months SELECT * FROM TABLE
        #  -->
        #   SELECT * FROM TABLE
        index_s = cleaned_sql.lower().find("select")
        index_w = cleaned_sql.lower().find("with")
        if index_w < index_s and index_w == 0:
            prefix = re.sub(r"\s\s+", " ", cleaned_sql[index_w:index_s][4:].strip())
            # Ignore the valid CTE: With a AS(...)
            # Don't want to skip With a AS ...
            # since no () means it won't use the defined var in the SQL
            if not (
                prefix.lower().endswith(" as (") or prefix.lower().endswith(" as(")
            ):
                print("ORI:", cleaned_sql)
                print("NEW:", case_sql(cleaned_sql[index_s:]))
                cleaned_sql = case_sql(cleaned_sql[index_s:])

    # Try to convert from TSQL to SQLite
    try:
        cleaned_sql = sqlglot.transpile(cleaned_sql, read="tsql", write="sqlite")[0]
    except Exception:
        pass

    if cleaned_title and cleaned_sql:
        try:
            cleaned_sql = format_to_match_schema(cleaned_sql, schema["stackexchange"])
            cleaned_sql = case_sql(cleaned_sql)
        except Exception:
            print("ERROR:::", cleaned_sql)
            cleaned_sql = None
        if cleaned_sql:
            preprocessed_annotated_sql = {
                "question": (
                    cleaned_title.strip() + ". " + (cleaned_description or "").strip()
                ).strip(),
                "db_id": "stackexchange",
                "sql": cleaned_sql,
            }
        else:
            preprocessed_annotated_sql = {}
    else:
        preprocessed_annotated_sql = {}

    return preprocessed_annotated_sql


def convert_spider_instance(
    data: dict[str, str],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example.

    Used for the spider dataset.
    """
    query = data["query"]
    question = data["question"]
    db_id = data["db_id"]
    target = case_sql(query)
    target = format_to_match_schema(target, schema[db_id])
    sql = {
        "question": question,
        "db_id": db_id,
        "sql": target,
    }
    # Check if example is impossible to answer
    if not data.get("is_impossible", False):
        return sql
    return {}


def convert_wikisql_instance(
    data: dict[str, Any],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example.

    Used for the wikisql dataset.
    """

    def _convert_to_human_readable(
        table_name: str,
        sel: int,
        agg: int,
        columns: list[str],
        conditions: list[tuple[int, int, str]],
    ) -> str:
        """Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10"""  # noqa: E501
        strip_quotes = lambda x: x.strip('"').strip("'").replace("'", "''")
        quoted_columns = [f'"{escape_everything(c)}"' for c in columns]
        if _AGG_OPS[agg] == "":
            rep = f"SELECT {quoted_columns[sel] if quoted_columns is not None else f'col{sel}'} FROM {table_name}"  # noqa: E501
        else:
            rep = f"SELECT {_AGG_OPS[agg]}({quoted_columns[sel] if quoted_columns is not None else f'col{sel}'}) FROM {table_name}"  # noqa: E501

        if conditions:
            rep += " WHERE " + " AND ".join(
                [
                    f"{quoted_columns[i]} {_COND_OPS[o]} '{strip_quotes(v)}'"
                    for i, o, v in conditions
                ]
            )
        return " ".join(rep.split())

    conds = data["sql"]["conds"]
    iov_list = list(
        zip(conds["column_index"], conds["operator_index"], conds["condition"])
    )
    query = _convert_to_human_readable(
        data["table"]["name"],
        data["sql"]["sel"],
        data["sql"]["agg"],
        data["table"]["header"],
        iov_list,
    )
    question = data["question"]
    db_id = data["table"]["name"]
    target = case_sql(query)

    try:
        target = format_to_match_schema(target, schema[db_id])
    except Exception as e:
        print("ERROR:::")
        print(target)
        print(e)
        return {}
    sql = {
        "question": question,
        "db_id": db_id,
        "sql": target,
    }
    return sql


def convert_criteria2sql_instance(
    data: dict[str, Any],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example.

    Modified from the criteria2sql dataset.
    """
    # We want to use the 'real' table name and all columns in the query
    assert data["query"].startswith("select id from records")
    query = data["query"]
    query = query.replace("select id from records", f"select * from {data['db_id']}")
    question = data["question"]
    db_id = data["db_id"]
    target = case_sql(query)

    try:
        target = format_to_match_schema(target, schema[db_id])
    except Exception as e:
        print("ERROR:::")
        print(target)
        print(e)
        return {}

    sql = {
        "question": question,
        "db_id": db_id,
        "sql": target,
    }
    return sql


def convert_sql_create_context_instance(
    data: dict[str, Any],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example."""
    query = data["answer"]
    question = data["question"]
    db_id = data["db_id"]
    target = case_sql(query)

    try:
        target = format_to_match_schema(target, schema[db_id])
    except Exception as e:
        print("ERROR:::")
        print(target)
        print(e)
        return {}

    sql = {
        "question": question,
        "db_id": db_id,
        "sql": target,
    }
    return sql


def convert_squall_instance(
    data: dict[str, Any],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example."""
    db_id = data["db_id"]
    question = " ".join(data["nl"])
    sql_toks = []
    for tok in data["sql"]:
        if tok[0] in ["Literal.Number", "Literal.String"]:
            sql_toks.append(tok[1])
        elif tok[0] == "Keyword":
            sql_toks.append(tok[1] if tok[1] != "w" else db_id)
        else:
            if "_" in tok[1]:
                idx = int(tok[1][1 : tok[1].find("_")])
            else:
                idx = int(tok[1][1])
            sql_toks.append(schema[db_id][db_id].columns[idx].name)
    query = " ".join(sql_toks)
    # Fix not null error
    query = query.replace("not null", "is not null")
    target = case_sql(query)

    try:
        target = format_to_match_schema(target, schema[db_id])
    except Exception as e:
        print("ERROR:::")
        print(target)
        print(e)
        return {}

    sql = {
        "question": question,
        "db_id": db_id,
        "sql": target,
    }
    return sql


def convert_css_nvbench_instance(
    data: dict[str, Any],
    schema: dict[str, dict[str, Table]] = {},
) -> dict[str, Any]:
    """Convert a single instance of the data into an example."""
    db_id = data["db_id"]
    question = data["question"]
    query = data["query"]
    target = case_sql(query)
    try:
        target = format_to_match_schema(target, schema[db_id])
    except Exception as e:
        print("ERROR:::")
        print(target)
        print(e)
        return {}

    sql = {
        "question": question,
        "db_id": db_id,
        "sql": target,
    }
    return sql
