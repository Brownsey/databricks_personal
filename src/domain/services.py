"""Domain services — pipeline orchestration without infrastructure details."""

import logging

from src.domain.errors import ModelPipelineError
from src.domain.models import ModelRegistrationResult
from src.domain.ports import (
    ExperimentTrackerPort,
    ModelLoaderPort,
    ModelRegistryPort,
    ModelServingPort,
)
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)


def register_model_pipeline(
    loader: ModelLoaderPort,
    tracker: ExperimentTrackerPort,
    registry: ModelRegistryPort,
    catalog: str,
    schema: str,
    model_name: str,
    model_id: str,
    task: str,
    *,
    serving: ModelServingPort | None = None,
    deploy: bool = False,
    redeploy: bool = False,
    register: bool = True,
    endpoint_name: str | None = None,
    artifact_path: str = "sentiment_model",
) -> Result[ModelRegistrationResult, ModelPipelineError]:
    """Run the full model registration pipeline.

    1. Ensure target schema exists in Unity Catalog (if register=True)
    2. Load model from external source
    3. Log model to experiment tracker and register in catalog
    4. Verify registration (if register=True)
    5. (Optional) Deploy to a serving endpoint

    Args:
        loader: Port for loading the pre-trained model
        tracker: Port for logging and registering the model
        registry: Port for schema management and verification
        catalog: Unity Catalog catalog name
        schema: Schema name within the catalog
        model_name: Name to register the model under
        model_id: External model identifier (e.g. HuggingFace model ID)
        task: Model task type (e.g. "sentiment-analysis")
        serving: Port for deploying model serving endpoints (required if deploy=True)
        deploy: Whether to deploy the model to a serving endpoint (default: False)
        register: Whether to register the model in Unity Catalog (default: True)
        endpoint_name: Override for serving endpoint name (default: derived from model_name)
        artifact_path: MLflow artifact path for the logged model (default: "sentiment_model")

    Returns:
        Result containing ModelRegistrationResult or error
    """
    registered_model_name = f"{catalog}.{schema}.{model_name}" if register else None

    # Step 1: Ensure target schema exists (only when registering in UC)
    if register:
        logger.info("Ensuring schema %s.%s exists", catalog, schema)
        schema_result = registry.ensure_schema(catalog, schema)
        if isinstance(schema_result, Err):
            return schema_result  # type: ignore[return-value]

    # Step 2: Load model from external source
    logger.info("Loading model: %s", model_id)
    model_result = loader.load_model(model_id, task)
    if isinstance(model_result, Err):
        return model_result  # type: ignore[return-value]

    model = model_result.value

    # Step 3: Log model (and register in Unity Catalog if register=True)
    log_target = registered_model_name or "(MLflow only, no UC registration)"
    logger.info("Logging model to tracker and registering as %s", log_target)
    run_result = tracker.log_model(
        model=model,
        task=task,
        model_id=model_id,
        registered_model_name=registered_model_name,
    )
    if isinstance(run_result, Err):
        return run_result  # type: ignore[return-value]

    run_id = run_result.value

    # If not registering in UC, return early with minimal result
    if not register:
        return Ok(
            ModelRegistrationResult(
                registered_model_name=f"{catalog}.{schema}.{model_name}",
                run_id=run_id,
                model_uri=f"runs:/{run_id}/{artifact_path}",
                model_info={},
                versions=[],
                endpoint_name=None,
            )
        )

    # Step 4: Verify registration
    logger.info("Verifying registration for %s", registered_model_name)
    info_result = registry.get_model_info(registered_model_name)
    if isinstance(info_result, Err):
        return info_result  # type: ignore[return-value]

    versions_result = registry.list_model_versions(registered_model_name)
    if isinstance(versions_result, Err):
        return versions_result  # type: ignore[return-value]

    # Step 5: Optionally deploy to a serving endpoint
    resolved_endpoint: str | None = None
    if deploy and serving is not None:
        latest_version = str(max(int(v["version"]) for v in versions_result.value))
        resolved_endpoint = endpoint_name or model_name.replace("_", "-")
        logger.info(
            "Deploying %s v%s to endpoint '%s'",
            registered_model_name,
            latest_version,
            resolved_endpoint,
        )
        deploy_result = serving.deploy_endpoint(
            endpoint_name=resolved_endpoint,
            model_name=registered_model_name,
            model_version=latest_version,
            redeploy=redeploy,
        )
        if isinstance(deploy_result, Err):
            return deploy_result  # type: ignore[return-value]

        ready_result = serving.wait_for_ready(resolved_endpoint)
        if isinstance(ready_result, Err):
            return ready_result  # type: ignore[return-value]

        logger.info("Endpoint '%s' is READY", resolved_endpoint)

        # Step 6: Configure AI Gateway (usage tracking, inference tables, rate limits)
        gw_result = serving.configure_ai_gateway(resolved_endpoint)
        if isinstance(gw_result, Err):
            return gw_result  # type: ignore[return-value]

    return Ok(
        ModelRegistrationResult(
            registered_model_name=registered_model_name,
            run_id=run_id,
            model_uri=f"runs:/{run_id}/{artifact_path}",
            model_info=info_result.value,
            versions=versions_result.value,
            endpoint_name=resolved_endpoint,
        )
    )
