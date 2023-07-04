from pydantic import BaseModel


class TableColumn(BaseModel):
    """Table column."""

    name: str
    dtype: str | None


class ForeignKey(BaseModel):
    """Foreign key."""

    # Referenced column
    column: TableColumn
    # References table name
    references_name: str
    # References column
    references_column: TableColumn


class Table(BaseModel):
    """Table."""

    name: str
    columns: list[TableColumn] | None
    pks: list[TableColumn] | None
    # FK from this table to another column in another table
    fks: list[ForeignKey] | None


class RajkumarFormatter:
    """RajkumarFormatter class.

    From https://arxiv.org/pdf/2204.00498.pdf.
    """

    table_sep: str = "\n\n"

    def __init__(self, tables: list[Table]) -> None:
        self.tables = tables
        self.table_str = self.format_tables(tables)

    def format_table(self, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        table_name = table.name
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table.pks:
            table_fmt.append(
                f"    primary key ({', '.join(pk.name for pk in table.pks)})"
            )
        for fk in table.fks or []:
            table_fmt.append(
                f"    foreign key ({fk.column.name}) references {fk.references_name}({fk.references_column.name})"  # noqa: E501
            )
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table_name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table_name}"
        return create_tbl

    def format_tables(self, tables: list[Table]) -> str:
        """Get tables format."""
        return self.table_sep.join(self.format_table(table) for table in tables)

    def format_prompt(
        self,
        instruction: str,
    ) -> str:
        """Get prompt format."""
        sql_prefix = "SELECT"
        return f"""{self.table_str}\n\n\n-- Using valid SQLite, answer the following questions for the tables provided above.\n\n-- {instruction}\n{sql_prefix}"""  # noqa: E501

    def format_model_output(self, output_sql: str) -> str:
        """Format model output.

        Our prompt ends with SELECT so we need to add it back.
        """
        if not output_sql.lower().startswith("select"):
            output_sql = "SELECT " + output_sql.strip()
        return output_sql
