import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, Format, StatementState

from src.config import get_databricks_host, get_databricks_token

WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "7fa6c5266d794420")

_client: WorkspaceClient | None = None


def get_workspace_client() -> WorkspaceClient:
    """Return an authenticated Databricks WorkspaceClient (cached)."""
    global _client
    if _client is None:
        _client = WorkspaceClient(
            host=get_databricks_host(),
            token=get_databricks_token(),
        )
    return _client


def run_sql(sql: str, *, catalog: str = "samples", schema: str = "default") -> list[dict]:
    """Execute SQL via the Starter Warehouse and return rows as dicts."""
    client = get_workspace_client()
    response = client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=WAREHOUSE_ID,
        catalog=catalog,
        schema=schema,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout="50s",
    )

    # Poll if the warehouse is still starting up or query is pending
    while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        response = client.statement_execution.get_statement(response.statement_id)

    if response.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Query failed ({response.status.state}): {response.status.error}")

    columns = [col.name for col in response.manifest.schema.columns]
    rows = []
    if response.result and response.result.data_array:
        for row_data in response.result.data_array:
            rows.append(dict(zip(columns, row_data, strict=False)))
    return rows


def print_table(rows: list[dict], max_rows: int = 20, max_col_width: int = 40):
    """Pretty-print query results as a formatted table."""
    if not rows:
        print("  (no results)")
        return

    columns = list(rows[0].keys())
    display_rows = rows[:max_rows]

    col_widths = {}
    for col in columns:
        values = [str(r.get(col, "")) for r in display_rows]
        col_widths[col] = min(max(len(col), max(len(v) for v in values)), max_col_width)

    header = " | ".join(col.ljust(col_widths[col])[: col_widths[col]] for col in columns)
    separator = "-+-".join("-" * col_widths[col] for col in columns)
    print(f"  {header}")
    print(f"  {separator}")

    for row in display_rows:
        line = " | ".join(str(row.get(col, "")).ljust(col_widths[col])[: col_widths[col]] for col in columns)
        print(f"  {line}")

    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows)")
