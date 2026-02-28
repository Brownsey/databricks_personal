"""Composition root: wires adaptors to domain services and registers a HuggingFace model.

Usage:
    uv run python register_model.py
    uv run python register_model.py --deploy   # also deploy to serving endpoint
"""

import argparse
import logging
import sys

from src.adaptors.huggingface_adaptor import HuggingFaceAdaptor
from src.adaptors.mlflow_adaptor import MLflowAdaptor
from src.adaptors.serving_adaptor import ServingAdaptor
from src.adaptors.unity_catalog_adaptor import UnityCatalogAdaptor
from src.config import configure_mlflow
from src.connect import get_workspace_client
from src.domain.result import Err
from src.domain.services import register_model_pipeline

# ── Constants ──────────────────────────────────────────────────────────
CATALOG = "workspace"
SCHEMA = "ml_models"
MODEL_NAME = "distilbert_sentiment"
HF_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
HF_TASK = "sentiment-analysis"


def main():
    parser = argparse.ArgumentParser(description="Register a HuggingFace model in Databricks UC")
    parser.add_argument(
        "--deploy",
        action="store_true",
        default=False,
        help="Deploy the model to a serving endpoint after registration",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("  HuggingFace Model -> Databricks Unity Catalog Registration")
    print("=" * 70)

    # Configure MLflow environment
    configure_mlflow()

    # Resolve current user for the experiment path
    client = get_workspace_client()
    current_user = client.current_user.me().user_name
    experiment_name = f"/Users/{current_user}/sentiment-model-registration"

    # Wire adaptors
    loader = HuggingFaceAdaptor()
    tracker = MLflowAdaptor(experiment_name=experiment_name)
    registry = UnityCatalogAdaptor(client=client)
    serving = ServingAdaptor(client=client) if args.deploy else None

    # Run pipeline via domain service
    result = register_model_pipeline(
        loader=loader,
        tracker=tracker,
        registry=registry,
        catalog=CATALOG,
        schema=SCHEMA,
        model_name=MODEL_NAME,
        model_id=HF_MODEL_ID,
        task=HF_TASK,
        serving=serving,
        deploy=args.deploy,
    )

    if isinstance(result, Err):
        logging.error("Pipeline failed: %s", result.error)
        sys.exit(1)

    # Print summary
    reg = result.value
    print(f"\n{'=' * 70}")
    print(f"  Model registered: {reg.registered_model_name}")
    print(f"  MLflow run ID:    {reg.run_id}")
    print(f"  Model URI:        {reg.model_uri}")
    print(f"  Owner:            {reg.model_info.get('owner', '?')}")
    print(f"  Created:          {reg.model_info.get('created_at', '?')}")
    if reg.endpoint_name:
        print(f"  Endpoint:         {reg.endpoint_name}")
    print(f"\n  Versions ({len(reg.versions)}):")
    for v in reg.versions:
        print(f"    v{v['version']}: status={v['status']}, run_id={v['run_id']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
