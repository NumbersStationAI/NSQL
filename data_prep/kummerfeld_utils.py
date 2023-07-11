"""SQL processing utils.

Adapted from https://github.com/hirupert/sede
"""
import re

ALIAS_PATTERN = re.compile(r"\[([^\]]+)]", re.MULTILINE | re.IGNORECASE)
TAGS_PATTERN = re.compile(
    r"([^'%])(##[a-z0-9_?:]+##)([^'%]?)", re.MULTILINE | re.IGNORECASE
)
TOP_TAGS_PATTERN = re.compile(
    r"(top|percentile_cont)([ ]+)?[\(]?[ ]?(##[a-z0-9_]+(:[a-z]+)?(\?([0-9.]+))?##)[ ]?[\)]?",
    re.IGNORECASE,
)
SQL_TOKENS = {
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
    "join",
    "on",
    "as",
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
    "none",
    "max",
    "min",
    "count",
    "sum",
    "avg",
    "or",
    "and",
}


def _remove_comment_at_beginning(cleaned_query: str) -> str:
    """Remove comments at the beginning of the line."""
    return re.sub(r"^([- ]+|(result))+", "", cleaned_query, re.MULTILINE)


def remove_comments(sql: str) -> str:
    """Remove comments from sql.""" ""
    # remove comments at the beginning of line
    sql = _remove_comment_at_beginning(sql)

    # remove comments at the end of lines
    sql = re.sub(r"--(.+)?\n", "", sql)

    # remove comments at the end of lines
    sql = re.sub(r"\n;\n", " ", sql)

    sql = re.sub(" +", " ", sql)

    return sql.strip()


def remove_comments_after_removing_new_lines(sql: str) -> str:
    """Remove comments and newlines from sql."""
    # remove comments at the end of the query
    sql = re.sub(r"--(.?)+$", "", sql, re.MULTILINE)

    # remove comments like /* a comment */
    sql = re.sub(r"/\*[^*/]+\*/", "", sql, re.MULTILINE)

    sql = re.sub(" +", " ", sql)

    return sql.strip()


def _surrounded_by_apostrophes(sql: str, start_index: int, end_index: int) -> bool:
    """Check if the string is surrounded by apostrophes."""
    max_steps = 10

    starts_with_apostrophe = False
    step_count = 0
    while start_index >= 0 and step_count < max_steps:
        if sql[start_index] == "'":
            starts_with_apostrophe = True
            break
        if sql[start_index] == " ":
            starts_with_apostrophe = False
            break
        start_index -= 1
        step_count += 1

    end_with_apostrophe = False
    step_count = 0
    while end_index < len(sql) and step_count < max_steps:
        if sql[end_index] == "'":
            end_with_apostrophe = True
            break
        if sql[end_index] == " ":
            end_with_apostrophe = False
            break
        end_index += 1
        step_count += 1

    return starts_with_apostrophe and end_with_apostrophe


# pylint: disable=too-many-branches
def preprocess_for_jsql(sql: str) -> str | None:
    """Preprocess sql for jsql."""
    # replace all alias like "as [User Id]" to "as 'user_id'"
    match = re.search(ALIAS_PATTERN, sql)
    while match is not None:
        group_one = match.group(1)
        if not _surrounded_by_apostrophes(sql, match.start(), match.end()):
            new_alias = f"'{group_one.lower()}'"
        else:
            new_alias = group_one.lower()

        if " " in new_alias:
            new_alias = new_alias.replace(" ", "_")
        sql = sql.replace(match.group(0), new_alias)
        match = re.search(ALIAS_PATTERN, sql)

    # replace all parameters like "TOP ##topn:int?200##" to "TOP 200"
    match = re.search(TOP_TAGS_PATTERN, sql)
    while match is not None:
        group_zero = match.group(0)
        default_number = match.group(6)

        if default_number is not None:
            new_alias = f"{match.group(1)} ({default_number})"
        else:
            new_alias = f"{match.group(1)} (100)"

        sql = sql.replace(group_zero, new_alias)
        match = re.search(TOP_TAGS_PATTERN, sql)

    # replace all parameters like ##tagName:Java## to '##tagName:Java##'
    new_sql = ""
    match = re.search(TAGS_PATTERN, sql)
    while match is not None:
        group_two = match.group(2)

        if not _surrounded_by_apostrophes(sql, match.start(), match.end()):
            new_alias = f"{match.group(1)}'{group_two}'{match.group(3)}"
            new_sql = new_sql + sql[0 : match.start()] + new_alias
        else:
            new_sql = new_sql + sql[0 : match.start()] + match.group(0)

        sql = sql[match.end() :]
        match = re.search(TAGS_PATTERN, sql)
    if sql:
        new_sql = new_sql + sql
    sql = new_sql

    # convert FORMAT function to CONVERT function to support JSQL
    sql = re.sub(r" format\(", " convert(", sql, flags=re.IGNORECASE)

    # remove comments from SQL
    sql = remove_comments(sql)

    # replace N'%Kitchener%' with '%Kitchener%'
    sql = re.sub(r" N'", " '", sql, re.IGNORECASE)

    # remove declares with a new line
    sql = re.sub(
        r"(DECLARE|declare) [^\n]+\n",
        " ",
        sql,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    # remove new lines
    sql = re.sub(r"[\n\t\r]+", " ", sql)

    sql = remove_comments_after_removing_new_lines(sql)

    # remove declares
    sql = re.sub(
        r"(DECLARE|declare) [^;]+;", " ", sql, re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    sql = re.sub(r"(DECLARE|declare) (?:.(?!(SELECT|select)))", "SELECT", sql)

    if "))))))))))))))))))))" in sql or "((((((((((((((((((((" in sql:
        return None

    if "cast(avg(cast(avg(cast(avg(cast(avg(cast(avg(cast(avg(cast(avg(" in sql:
        return None

    sql = re.sub(r"[^\x00-\x7f]", r" ", sql)
    sql = re.sub(r"``", r"'", sql)
    sql = re.sub(r"\"", r"'", sql)
    sql = re.sub(r" +", " ", sql).strip()

    if not sql:
        return None

    if sql[-1] == ";":
        sql = sql[0:-1]

    if ";" in sql:
        sql = sql.split(";")[-1]

    return sql
