"""Unity Catalog adaptor — implements ModelRegistryPort."""

import logging

from databricks.sdk import WorkspaceClient

from src.domain.errors import RegistryError
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)


class UnityCatalogAdaptor:
    """Manages schemas and verifies model registration via the Databricks SDK."""

    def __init__(self, client: WorkspaceClient) -> None:
        self._client = client

    def ensure_schema(self, catalog: str, schema: str) -> Result[None, RegistryError]:
        """Create the target schema if it does not already exist."""
        try:
            existing = [s.name for s in self._client.schemas.list(catalog_name=catalog)]
            if schema in existing:
                logger.info("Schema '%s.%s' already exists", catalog, schema)
                return Ok(None)

            logger.info("Creating schema '%s.%s'...", catalog, schema)
            self._client.schemas.create(name=schema, catalog_name=catalog)
            logger.info("Schema '%s.%s' created", catalog, schema)
            return Ok(None)
        except Exception as e:
            return Err(RegistryError(reason=f"Failed to ensure schema {catalog}.{schema}: {e}"))

    def get_model_info(self, full_name: str) -> Result[dict, RegistryError]:
        """Retrieve metadata for a registered model by its three-level name."""
        try:
            model = self._client.registered_models.get(full_name=full_name)
            return Ok(
                {
                    "full_name": model.full_name,
                    "created_at": str(model.created_at),
                    "updated_at": str(model.updated_at),
                    "owner": model.owner,
                }
            )
        except Exception as e:
            return Err(RegistryError(reason=f"Failed to get model '{full_name}': {e}"))

    def list_model_versions(self, full_name: str) -> Result[list[dict], RegistryError]:
        """List all versions of a registered model."""
        try:
            versions = list(self._client.model_versions.list(full_name=full_name))
            return Ok(
                [
                    {
                        "version": v.version,
                        "status": str(v.status),
                        "run_id": v.run_id,
                        "created_at": str(v.created_at),
                    }
                    for v in versions
                ]
            )
        except Exception as e:
            return Err(RegistryError(reason=f"Failed to list versions for '{full_name}': {e}"))
