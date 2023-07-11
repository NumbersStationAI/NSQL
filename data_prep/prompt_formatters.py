"""Rajkumar prompt formatter."""

from abc import ABC
from random import shuffle

from schema import Table


class RajkumarFormatter(ABC):
    """RajkumarFormatter class.

    From https://arxiv.org/pdf/2204.00498.pdf.
    """

    table_sep: str = "\n\n"
    shuffle_table_order: bool = True
    _cache: dict[tuple[str, str, str], list[str]] = {}

    @classmethod
    def format_table(cls, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table.name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table.name}"
        return create_tbl

    @classmethod
    def format_all_tables(cls, tables: list[Table], instruction: str) -> list[str]:
        """Get all tables format."""
        table_texts = [cls.format_table(table) for table in tables]
        key = ("tables", instruction, str(tables))
        if key not in cls._cache:
            shuffle(table_texts)
            cls._cache[key] = table_texts
        else:
            table_texts = cls._cache[key]
        return table_texts

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
    ) -> str:
        """Get prompt format."""
        return f"""{table_text}\n\n\n-- Using valid SQLite, answer the following questions for the tables provided above.\n\n-- {instruction}\n"""  # noqa: E501

    @classmethod
    def format_model_output(cls, output_sql: str, prompt: str) -> str:
        """Format model output."""
        return output_sql

    @classmethod
    def format_gold_output(cls, output_sql: str) -> str:
        """Format gold output for demonstration."""
        return output_sql
