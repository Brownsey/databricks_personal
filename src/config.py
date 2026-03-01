import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def is_on_databricks() -> bool:
    """Return True when running inside a Databricks cluster or serverless."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_databricks_host() -> str:
    host = os.environ.get("DATABRICKS_HOST")
    if not host:
        raise OSError("DATABRICKS_HOST is not set in .env")
    return host.rstrip("/")


def get_databricks_token() -> str:
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        raise OSError("DATABRICKS_TOKEN is not set in .env")
    return token


def configure_mlflow() -> None:
    """Ensure MLflow environment variables are set for Databricks-managed tracking + UC registry."""
    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
    os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
    if not is_on_databricks():
        get_databricks_host()
        get_databricks_token()
