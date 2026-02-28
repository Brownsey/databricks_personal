"""Shared fixtures for integration tests."""

import pytest
from databricks.sdk import WorkspaceClient

from src.config import get_databricks_host, get_databricks_token
from src.connect import get_workspace_client


@pytest.fixture(scope="session")
def workspace_client() -> WorkspaceClient:
    """Return an authenticated WorkspaceClient for the test session."""
    return get_workspace_client()


@pytest.fixture(scope="session")
def databricks_host() -> str:
    return get_databricks_host()


@pytest.fixture(scope="session")
def databricks_token() -> str:
    return get_databricks_token()
