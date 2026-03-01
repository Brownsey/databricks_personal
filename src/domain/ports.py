"""Port interfaces for model registration pipeline.

Following hexagonal architecture: ports define contracts that
adaptor implementations must satisfy.
"""

from typing import Any, Protocol

from src.domain.errors import ModelLoadError, RegistryError, ServingError, TrackingError
from src.domain.result import Result


class ModelLoaderPort(Protocol):
    """Port for loading a pre-trained model from an external source."""

    def load_model(self, model_id: str, task: str) -> Result[Any, ModelLoadError]:
        """Download and return a model ready for logging."""
        ...


class ExperimentTrackerPort(Protocol):
    """Port for logging a model and metadata to an experiment tracking system."""

    def log_model(
        self,
        model: Any,
        task: str,
        model_id: str,
        registered_model_name: str,
    ) -> Result[str, TrackingError]:
        """Log model artifacts and register. Returns the run ID on success."""
        ...


class ModelRegistryPort(Protocol):
    """Port for creating schemas and verifying model registration in a catalog."""

    def ensure_schema(self, catalog: str, schema: str) -> Result[None, RegistryError]:
        """Create the target schema if it does not exist."""
        ...

    def get_model_info(self, full_name: str) -> Result[dict, RegistryError]:
        """Retrieve metadata for a registered model."""
        ...

    def list_model_versions(self, full_name: str) -> Result[list[dict], RegistryError]:
        """List all versions of a registered model."""
        ...


class ModelServingPort(Protocol):
    """Port for deploying and querying model serving endpoints."""

    def deploy_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        *,
        redeploy: bool = False,
    ) -> Result[dict, ServingError]:
        """Deploy a serving endpoint for the given model version.

        If redeploy is True, an existing endpoint is deleted and recreated.
        If redeploy is False and the endpoint already exists, returns an error.
        """
        ...

    def wait_for_ready(self, endpoint_name: str) -> Result[None, ServingError]:
        """Block until the endpoint reaches READY state."""
        ...

    def configure_ai_gateway(self, endpoint_name: str) -> Result[dict, ServingError]:
        """Apply AI Gateway configuration (usage tracking, inference tables, rate limits)."""
        ...
