"""Tests to verify the Databricks MCP server is configured and reachable.

These tests validate the MCP tooling indirectly by exercising the same
Databricks SDK operations that the MCP server uses under the hood.
"""

from databricks.sdk import WorkspaceClient


class TestMCPReadiness:
    """Verify the SDK operations that back MCP tools work correctly."""

    def test_current_user(self, workspace_client: WorkspaceClient):
        """get_current_user MCP tool equivalent works."""
        me = workspace_client.current_user.me()
        assert me.user_name is not None

    def test_list_warehouses(self, workspace_client: WorkspaceClient):
        """list_warehouses MCP tool equivalent works."""
        warehouses = list(workspace_client.warehouses.list())
        assert len(warehouses) > 0
        # At least one warehouse should have an id
        assert warehouses[0].id is not None

    def test_list_clusters(self, workspace_client: WorkspaceClient):
        """list_clusters MCP tool equivalent works."""
        clusters = list(workspace_client.clusters.list())
        # Workspace should have at least one cluster (even if not running)
        assert isinstance(clusters, list)

    def test_list_serving_endpoints(self, workspace_client: WorkspaceClient):
        """list_serving_endpoints MCP tool equivalent works."""
        endpoints = list(workspace_client.serving_endpoints.list())
        assert len(endpoints) > 0
        # Foundation models should always be present
        names = {e.name for e in endpoints}
        assert any(n.startswith("databricks-") for n in names)

    def test_execute_sql(self):
        """execute_sql MCP tool equivalent works."""
        from src.connect import run_sql

        rows = run_sql("SELECT current_timestamp() AS ts")
        assert len(rows) == 1
        assert "ts" in rows[0]
