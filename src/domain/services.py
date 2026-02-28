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
) -> Result[ModelRegistrationResult, ModelPipelineError]:
    """Run the full model registration pipeline.

    1. Ensure target schema exists in Unity Catalog
    2. Load model from external source
    3. Log model to experiment tracker and register in catalog
    4. Verify registration
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

    Returns:
        Result containing ModelRegistrationResult or error
    """
    registered_model_name = f"{catalog}.{schema}.{model_name}"

    # Step 1: Ensure target schema exists
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

    # Step 3: Log model and register in Unity Catalog
    logger.info("Logging model to tracker and registering as %s", registered_model_name)
    run_result = tracker.log_model(
        model=model,
        task=task,
        model_id=model_id,
        registered_model_name=registered_model_name,
    )
    if isinstance(run_result, Err):
        return run_result  # type: ignore[return-value]

    run_id = run_result.value

    # Step 4: Verify registration
    logger.info("Verifying registration for %s", registered_model_name)
    info_result = registry.get_model_info(registered_model_name)
    if isinstance(info_result, Err):
        return info_result  # type: ignore[return-value]

    versions_result = registry.list_model_versions(registered_model_name)
    if isinstance(versions_result, Err):
        return versions_result  # type: ignore[return-value]

    # Step 5: Optionally deploy to a serving endpoint
    endpoint_name: str | None = None
    if deploy and serving is not None:
        latest_version = str(max(int(v["version"]) for v in versions_result.value))
        endpoint_name = model_name.replace("_", "-")
        logger.info(
            "Deploying %s v%s to endpoint '%s'",
            registered_model_name,
            latest_version,
            endpoint_name,
        )
        deploy_result = serving.deploy_endpoint(
            endpoint_name=endpoint_name,
            model_name=registered_model_name,
            model_version=latest_version,
        )
        if isinstance(deploy_result, Err):
            return deploy_result  # type: ignore[return-value]

        ready_result = serving.wait_for_ready(endpoint_name)
        if isinstance(ready_result, Err):
            return ready_result  # type: ignore[return-value]

        logger.info("Endpoint '%s' is READY", endpoint_name)

    return Ok(
        ModelRegistrationResult(
            registered_model_name=registered_model_name,
            run_id=run_id,
            model_uri=f"runs:/{run_id}/sentiment_model",
            model_info=info_result.value,
            versions=versions_result.value,
            endpoint_name=endpoint_name,
        )
    )
