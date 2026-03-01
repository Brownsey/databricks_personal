"""Composition root: wires adaptors to domain services and registers a HuggingFace model.

Usage:
    uv run python register_model.py
    uv run python register_model.py --deploy
    uv run python register_model.py --model-id bert-base-uncased --task fill-mask
    uv run python register_model.py --catalog my_catalog --schema my_schema --region us-west-2
    uv run python register_model.py --no-register   # log to MLflow only, skip UC registration
"""

import argparse
import logging
import os
import sys

from src.adaptors.huggingface_adaptor import HuggingFaceAdaptor
from src.adaptors.mlflow_adaptor import MLflowAdaptor
from src.adaptors.serving_adaptor import ServingAdaptor
from src.adaptors.unity_catalog_adaptor import UnityCatalogAdaptor
from src.config import configure_mlflow
from src.connect import get_workspace_client
from src.domain.result import Err
from src.domain.services import register_model_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register a HuggingFace model in Databricks Unity Catalog",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model Source ──────────────────────────────────────────────────
    source = parser.add_argument_group("Model Source")
    source.add_argument(
        "--model-id",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HuggingFace model ID to download",
    )
    source.add_argument(
        "--task",
        default="sentiment-analysis",
        help="HuggingFace pipeline task (e.g. sentiment-analysis, fill-mask, text-generation)",
    )

    # ── Unity Catalog ─────────────────────────────────────────────────
    uc = parser.add_argument_group("Unity Catalog")
    uc.add_argument(
        "--catalog",
        default="workspace",
        help="Unity Catalog catalog name",
    )
    uc.add_argument(
        "--schema",
        default="ml_models",
        help="Schema name within the catalog",
    )
    uc.add_argument(
        "--model-name",
        default="distilbert_sentiment",
        help="Name to register the model under in UC",
    )
    uc.add_argument(
        "--no-register",
        action="store_true",
        default=False,
        help="Skip Unity Catalog registration (log to MLflow only)",
    )

    # ── Deployment ────────────────────────────────────────────────────
    deploy_group = parser.add_argument_group("Deployment")
    deploy_group.add_argument(
        "--deploy",
        action="store_true",
        default=False,
        help="Deploy the model to a serving endpoint after registration",
    )
    deploy_group.add_argument(
        "--endpoint-name",
        default=None,
        help="Serving endpoint name (default: derived from --model-name)",
    )
    deploy_group.add_argument(
        "--workload-size",
        default="Small",
        choices=["Small", "Medium", "Large"],
        help="Serving endpoint workload size",
    )
    deploy_group.add_argument(
        "--no-scale-to-zero",
        action="store_true",
        default=False,
        help="Disable scale-to-zero on the serving endpoint",
    )
    deploy_group.add_argument(
        "--deploy-timeout",
        type=int,
        default=1200,
        help="Max seconds to wait for endpoint readiness",
    )
    deploy_group.add_argument(
        "--region",
        default="us-east-2",
        help="Target region for deployment (avoids cross-region data transfer)",
    )

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow_group = parser.add_argument_group("MLflow")
    mlflow_group.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment path (default: /Users/<current_user>/<model-name>-registration)",
    )
    mlflow_group.add_argument(
        "--artifact-path",
        default="sentiment_model",
        help="MLflow artifact path for the logged model",
    )

    # ── AI Gateway ──────────────────────────────────────────────────
    gw = parser.add_argument_group("AI Gateway")
    gw.add_argument(
        "--enable-usage-tracking",
        action="store_true",
        default=False,
        help="Enable usage tracking via system tables",
    )
    gw.add_argument(
        "--enable-inference-tables",
        action="store_true",
        default=False,
        help="Enable inference table logging for request/response payloads",
    )
    gw.add_argument(
        "--inference-table-catalog",
        default=None,
        help="Catalog for inference tables (defaults to --catalog if not set)",
    )
    gw.add_argument(
        "--inference-table-schema",
        default=None,
        help="Schema for inference tables (defaults to --schema if not set)",
    )
    gw.add_argument(
        "--inference-table-prefix",
        default=None,
        help="Table name prefix for inference tables",
    )
    gw.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        metavar="CALLS",
        help="Endpoint-level rate limit (calls per minute). Off by default",
    )

    # ── Infrastructure ────────────────────────────────────────────────
    infra = parser.add_argument_group("Infrastructure")
    infra.add_argument(
        "--warehouse-id",
        default=None,
        help="SQL warehouse ID (default: DATABRICKS_WAREHOUSE_ID env var or built-in fallback)",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Override warehouse ID if provided via CLI
    if args.warehouse_id:
        os.environ["DATABRICKS_WAREHOUSE_ID"] = args.warehouse_id

    print("=" * 70)
    print("  HuggingFace Model -> Databricks Unity Catalog Registration")
    print("=" * 70)
    print(f"  Model ID:      {args.model_id}")
    print(f"  Task:          {args.task}")
    print(f"  Catalog:       {args.catalog}")
    print(f"  Schema:        {args.schema}")
    print(f"  Model name:    {args.model_name}")
    print(f"  Region:        {args.region}")
    print(f"  Register in UC: {not args.no_register}")
    print(f"  Deploy:        {args.deploy}")
    if args.deploy:
        ep = args.endpoint_name or args.model_name.replace("_", "-")
        print(f"  Endpoint:      {ep}")
        print(f"  Workload size: {args.workload_size}")
        print(f"  Scale-to-zero: {not args.no_scale_to_zero}")
        print(f"  Deploy timeout: {args.deploy_timeout}s")
        print(f"  Usage tracking: {args.enable_usage_tracking}")
        print(f"  Inference tables: {args.enable_inference_tables}")
        if args.rate_limit is not None:
            print(f"  Rate limit:    {args.rate_limit} calls/min (endpoint)")
    print("=" * 70)

    # Configure MLflow environment
    configure_mlflow()

    # Resolve current user for the experiment path
    client = get_workspace_client()
    current_user = client.current_user.me().user_name

    experiment_name = args.experiment_name or f"/Users/{current_user}/{args.model_name}-registration"

    # Wire adaptors
    loader = HuggingFaceAdaptor()
    tracker = MLflowAdaptor(
        experiment_name=experiment_name,
        artifact_path=args.artifact_path,
        region=args.region,
    )
    registry = UnityCatalogAdaptor(client=client)
    serving = (
        ServingAdaptor(
            client=client,
            workload_size=args.workload_size,
            scale_to_zero=not args.no_scale_to_zero,
            max_wait=args.deploy_timeout,
            region=args.region,
            enable_usage_tracking=args.enable_usage_tracking,
            enable_inference_tables=args.enable_inference_tables,
            inference_table_catalog=args.inference_table_catalog or args.catalog,
            inference_table_schema=args.inference_table_schema or args.schema,
            inference_table_prefix=args.inference_table_prefix,
            rate_limit_calls=args.rate_limit,
        )
        if args.deploy
        else None
    )

    # Run pipeline via domain service
    result = register_model_pipeline(
        loader=loader,
        tracker=tracker,
        registry=registry,
        catalog=args.catalog,
        schema=args.schema,
        model_name=args.model_name,
        model_id=args.model_id,
        task=args.task,
        serving=serving,
        deploy=args.deploy,
        register=not args.no_register,
        endpoint_name=args.endpoint_name,
        artifact_path=args.artifact_path,
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
