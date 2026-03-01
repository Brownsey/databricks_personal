"""Model serving adaptor — implements ModelServingPort."""

import logging
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayRateLimit,
    AiGatewayRateLimitKey,
    AiGatewayRateLimitRenewalPeriod,
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)

from src.domain.errors import ServingError
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 30


class ServingAdaptor:
    """Deploys and manages model serving endpoints via the Databricks SDK."""

    def __init__(
        self,
        client: WorkspaceClient,
        *,
        workload_size: str = "Small",
        scale_to_zero: bool = True,
        max_wait: int = 1200,
        region: str = "us-east-2",
        enable_usage_tracking: bool = False,
        enable_inference_tables: bool = False,
        inference_table_catalog: str | None = None,
        inference_table_schema: str | None = None,
        inference_table_prefix: str | None = None,
        rate_limit_calls: int | None = None,
    ) -> None:
        self._client = client
        self._workload_size = workload_size
        self._scale_to_zero = scale_to_zero
        self._max_wait = max_wait
        self._region = region
        self._enable_usage_tracking = enable_usage_tracking
        self._enable_inference_tables = enable_inference_tables
        self._inference_table_catalog = inference_table_catalog
        self._inference_table_schema = inference_table_schema
        self._inference_table_prefix = inference_table_prefix
        self._rate_limit_calls = rate_limit_calls

    def _build_ai_gateway_config(self) -> AiGatewayConfig | None:
        """Build AI Gateway config from constructor parameters, or None if nothing enabled."""
        usage = None
        inference = None
        rate_limits = None

        if self._enable_usage_tracking:
            usage = AiGatewayUsageTrackingConfig(enabled=True)

        if self._enable_inference_tables:
            inference = AiGatewayInferenceTableConfig(
                enabled=True,
                catalog_name=self._inference_table_catalog,
                schema_name=self._inference_table_schema,
                table_name_prefix=self._inference_table_prefix,
            )

        if self._rate_limit_calls is not None:
            # Endpoint-level rate limit (calls per minute)
            rate_limits = [
                AiGatewayRateLimit(
                    key=AiGatewayRateLimitKey.ENDPOINT,
                    calls=self._rate_limit_calls,
                    renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
                ),
                # Future: per-user rate limits
                # AiGatewayRateLimit(
                #     key=AiGatewayRateLimitKey.USER,
                #     principal="user@example.com",
                #     calls=100,
                #     renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
                # ),
                # Future: per-group rate limits
                # AiGatewayRateLimit(
                #     key=AiGatewayRateLimitKey.USER_GROUP,
                #     principal="data_scientists",
                #     calls=500,
                #     renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
                # ),
                # Future: per-service-principal rate limits
                # AiGatewayRateLimit(
                #     key=AiGatewayRateLimitKey.SERVICE_PRINCIPAL,
                #     principal="<application-id>",
                #     calls=200,
                #     renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
                # ),
            ]

        if not any([usage, inference, rate_limits]):
            return None

        return AiGatewayConfig(
            usage_tracking_config=usage,
            inference_table_config=inference,
            rate_limits=rate_limits,
        )

    def configure_ai_gateway(self, endpoint_name: str) -> Result[dict, ServingError]:
        """Apply AI Gateway configuration (usage tracking, inference tables, rate limits)."""
        gw = self._build_ai_gateway_config()
        if gw is None:
            return Ok({"ai_gateway": "no features enabled"})

        try:
            logger.info("Configuring AI Gateway on endpoint '%s'...", endpoint_name)
            self._client.serving_endpoints.put_ai_gateway(
                name=endpoint_name,
                usage_tracking_config=gw.usage_tracking_config,
                inference_table_config=gw.inference_table_config,
                rate_limits=gw.rate_limits,
            )

            features = []
            if gw.usage_tracking_config:
                features.append("usage_tracking")
            if gw.inference_table_config:
                features.append("inference_tables")
            if gw.rate_limits:
                features.append(f"rate_limit({self._rate_limit_calls} calls/min)")
            logger.info("AI Gateway configured: %s", ", ".join(features))

            return Ok({"ai_gateway": features})
        except Exception as e:
            return Err(ServingError(reason=f"Failed to configure AI Gateway on '{endpoint_name}': {e}"))

    def deploy_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        *,
        redeploy: bool = False,
    ) -> Result[dict, ServingError]:
        """Deploy a serving endpoint for the given UC model version.

        If redeploy is True, an existing endpoint is deleted and recreated.
        If redeploy is False and the endpoint already exists, returns an error.
        """
        try:
            entity = ServedEntityInput(
                entity_name=model_name,
                entity_version=model_version,
                workload_size=self._workload_size,
                scale_to_zero_enabled=self._scale_to_zero,
            )
            config = EndpointCoreConfigInput(name=endpoint_name, served_entities=[entity])

            # Check if endpoint already exists
            exists = True
            try:
                self._client.serving_endpoints.get(name=endpoint_name)
            except NotFound:
                exists = False

            if exists and not redeploy:
                return Err(
                    ServingError(
                        reason=f"Endpoint '{endpoint_name}' already exists. "
                        "Use --redeploy to delete and recreate it."
                    )
                )

            if exists:
                logger.info("Deleting existing endpoint '%s' for redeploy...", endpoint_name)
                self._client.serving_endpoints.delete(name=endpoint_name)

            logger.info("Creating serving endpoint '%s'...", endpoint_name)
            self._client.serving_endpoints.create(
                name=endpoint_name,
                config=config,
            )
            logger.info("Endpoint region: %s", self._region)
            action = "redeployed" if exists else "created"
            return Ok({"name": endpoint_name, "action": action})
        except Exception as e:
            return Err(ServingError(reason=f"Failed to deploy endpoint '{endpoint_name}': {e}"))

    def wait_for_ready(self, endpoint_name: str) -> Result[None, ServingError]:
        """Poll until the endpoint is READY or timeout."""
        try:
            elapsed = 0
            while elapsed < self._max_wait:
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
                time.sleep(_POLL_INTERVAL)
                elapsed += _POLL_INTERVAL

            return Err(ServingError(reason=f"Endpoint '{endpoint_name}' not ready after {self._max_wait}s"))
        except Exception as e:
            return Err(ServingError(reason=f"Error waiting for endpoint '{endpoint_name}': {e}"))
