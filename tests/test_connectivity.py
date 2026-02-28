"""Integration tests for Databricks workspace connectivity."""

from databricks.sdk import WorkspaceClient


class TestDatabricksAuth:
    """Verify authentication and basic SDK operations."""

    def test_workspace_client_authenticates(self, workspace_client: WorkspaceClient):
        """SDK client can authenticate and reach the workspace."""
        me = workspace_client.current_user.me()
        assert me.user_name is not None
        assert "@" in me.user_name

    def test_host_is_reachable(self, databricks_host: str):
        """DATABRICKS_HOST is set and looks like a valid URL."""
        assert databricks_host.startswith("https://")
        assert ".cloud.databricks.com" in databricks_host or ".azuredatabricks.net" in databricks_host


class TestSQLWarehouse:
    """Verify SQL warehouse connectivity."""

    def test_simple_query(self, workspace_client: WorkspaceClient):
        """Can execute a trivial SQL statement via the statement execution API."""
        from src.connect import run_sql

        rows = run_sql("SELECT 1 AS ok")
        assert len(rows) == 1
        assert rows[0]["ok"] == "1"

    def test_list_catalogs(self, workspace_client: WorkspaceClient):
        """Can list catalogs via SQL."""
        from src.connect import run_sql

        rows = run_sql("SHOW CATALOGS")
        catalog_names = [r["catalog"] for r in rows]
        assert "workspace" in catalog_names

    def test_unity_catalog_schema_exists(self, workspace_client: WorkspaceClient):
        """The ml_models schema exists in the workspace catalog."""
        from src.connect import run_sql

        rows = run_sql("SHOW SCHEMAS IN workspace")
        schema_names = [r["databaseName"] for r in rows]
        assert "ml_models" in schema_names
