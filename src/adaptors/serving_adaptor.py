"""Model serving adaptor — implements ModelServingPort."""

import logging
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

from src.domain.errors import ServingError
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30
MAX_WAIT = 1200  # 20 minutes


class ServingAdaptor:
    """Deploys and manages model serving endpoints via the Databricks SDK."""

    def __init__(self, client: WorkspaceClient) -> None:
        self._client = client

    def deploy_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
    ) -> Result[dict, ServingError]:
        """Create or update a serving endpoint for the given UC model version."""
        try:
            entity = ServedEntityInput(
                entity_name=model_name,
                entity_version=model_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
            config = EndpointCoreConfigInput(name=endpoint_name, served_entities=[entity])

            # Check if endpoint already exists
            try:
                self._client.serving_endpoints.get(name=endpoint_name)
                logger.info("Endpoint '%s' exists, updating config...", endpoint_name)
                self._client.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_entities=[entity],
                )
                return Ok({"name": endpoint_name, "action": "updated"})
            except Exception:
                pass  # Does not exist, create it

            logger.info("Creating serving endpoint '%s'...", endpoint_name)
            self._client.serving_endpoints.create(
                name=endpoint_name,
                config=config,
            )
            return Ok({"name": endpoint_name, "action": "created"})
        except Exception as e:
            return Err(ServingError(reason=f"Failed to deploy endpoint '{endpoint_name}': {e}"))

    def wait_for_ready(self, endpoint_name: str) -> Result[None, ServingError]:
        """Poll until the endpoint is READY or timeout."""
        try:
            elapsed = 0
            while elapsed < MAX_WAIT:
                ep = self._client.serving_endpoints.get(name=endpoint_name)
                state = ep.state.ready if ep.state else None
                logger.info(
                    "Endpoint '%s' state: %s (waited %ds)",
                    endpoint_name,
                    state,
                    elapsed,
                )
                if state and state.value == "READY":
                    return Ok(None)
                time.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

            return Err(ServingError(reason=f"Endpoint '{endpoint_name}' not ready after {MAX_WAIT}s"))
        except Exception as e:
            return Err(ServingError(reason=f"Error waiting for endpoint '{endpoint_name}': {e}"))
