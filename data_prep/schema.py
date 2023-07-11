"""Text2SQL schemas."""

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

    name: str | None
    columns: list[TableColumn] | None
    pks: list[TableColumn] | None
    # FK from this table to another column in another table
    fks: list[ForeignKey] | None
    examples: list[dict] | None
    # Is the table a source or intermediate reference table
    is_reference_table: bool = False
