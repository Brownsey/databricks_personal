"""Test all deployed pyfunc model serving endpoints.

Discovers endpoints backed by Unity Catalog models in the configured catalog/schema,
sends a test query to each, and prints a summary table.

Usage:
    uv run task test-deployment
    uv run python test_deployments.py
    uv run python test_deployments.py --catalog workspace --schema ml_models
"""

import argparse
import io
import sys
import time

# Reconfigure stdout/stderr to UTF-8 for Windows compatibility.
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.config import configure_mlflow
from src.connect import get_workspace_client

# Test prompts per task type. The key is matched against MLflow tags or model config.
_TEST_INPUTS = {
    "sentiment-analysis": [{"text": "I love this product!"}],
    "text-classification": [{"text": "I love this product!"}],
    "text-generation": [{"text": "Hi"}],
    "ner": [{"text": "John Smith works at Google in London."}],
    "fill-mask": [{"text": "The capital of France is [MASK]."}],
    "question-answering": [{"text": "What is Python? Python is a programming language."}],
    "zero-shot-classification": [{"text": "This is a sports article about football."}],
    "feature-extraction": [{"text": "Hello world."}],
}
_DEFAULT_INPUT = [{"text": "This is a test."}]


def _detect_task(endpoint) -> str:
    """Best-effort detection of the HuggingFace task from endpoint tags."""
    tags = {}
    if endpoint.tags:
        for tag in endpoint.tags:
            tags[tag.key] = tag.value

    if "hf_task" in tags:
        return tags["hf_task"]

    # Infer from the model name as a fallback
    name = endpoint.name.lower()
    if "sentiment" in name:
        return "sentiment-analysis"
    if "ner" in name:
        return "ner"
    if "qwen" in name or "gpt" in name or "gen" in name:
        return "text-generation"

    return "unknown"


def _query_endpoint(client, name, task):
    """Send a test query and return (success, latency_ms, preview)."""
    records = _TEST_INPUTS.get(task, _DEFAULT_INPUT)
    start = time.perf_counter()
    try:
        response = client.serving_endpoints.query(
            name=name,
            dataframe_records=records,
        )
        latency = (time.perf_counter() - start) * 1000
        predictions = response.predictions
        if predictions and len(predictions) > 0:
            preview = str(predictions[0])
            if len(preview) > 80:
                preview = preview[:77] + "..."
            return True, latency, preview
        return True, latency, "(empty response)"
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        msg = str(e)
        if len(msg) > 80:
            msg = msg[:77] + "..."
        return False, latency, msg


def test_deployments(catalog: str = "workspace", schema: str = "ml_models"):
    """Discover and test all pyfunc endpoints in the given catalog/schema."""
    configure_mlflow()
    client = get_workspace_client()

    print("=" * 90)
    print("  Deployment Test — Discovering serving endpoints")
    print("=" * 90)

    # List all endpoints and filter to ones backed by our catalog.schema
    all_endpoints = list(client.serving_endpoints.list())
    prefix = f"{catalog}.{schema}."

    our_endpoints = []
    for ep in all_endpoints:
        if not ep.config or not ep.config.served_entities:
            continue
        for entity in ep.config.served_entities:
            if entity.entity_name and entity.entity_name.startswith(prefix):
                our_endpoints.append((ep, entity))
                break

    if not our_endpoints:
        print(f"\n  No endpoints found serving models from {catalog}.{schema}")
        print(f"  Total endpoints in workspace: {len(all_endpoints)}")
        return 0

    print(f"\n  Found {len(our_endpoints)} endpoint(s) serving from {catalog}.{schema}\n")

    # Collect results
    results = []
    for ep, entity in our_endpoints:
        state = "UNKNOWN"
        if ep.state and ep.state.ready:
            state = ep.state.ready.value

        model_name = entity.entity_name.removeprefix(prefix)
        version = entity.entity_version or "?"
        task = _detect_task(ep)

        print(f"  Testing: {ep.name} ({model_name} v{version}, task={task}, state={state})")

        if state != "READY":
            results.append({
                "endpoint": ep.name,
                "model": model_name,
                "version": version,
                "task": task,
                "state": state,
                "status": "SKIPPED",
                "latency_ms": "-",
                "response": f"Endpoint not ready ({state})",
            })
            continue

        success, latency, preview = _query_endpoint(client, ep.name, task)
        results.append({
            "endpoint": ep.name,
            "model": model_name,
            "version": version,
            "task": task,
            "state": state,
            "status": "PASS" if success else "FAIL",
            "latency_ms": f"{latency:.0f}",
            "response": preview,
        })

    # Print summary table
    print(f"\n{'=' * 90}")
    print("  DEPLOYMENT TEST SUMMARY")
    print(f"{'=' * 90}\n")

    # Column widths
    cols = [
        ("Endpoint", "endpoint", 24),
        ("Model", "model", 22),
        ("Ver", "version", 4),
        ("Task", "task", 22),
        ("State", "state", 9),
        ("Test", "status", 7),
        ("Latency", "latency_ms", 9),
    ]

    header = " | ".join(c[0].ljust(c[2]) for c in cols)
    separator = "-+-".join("-" * c[2] for c in cols)
    print(f"  {header}")
    print(f"  {separator}")

    for r in results:
        row = " | ".join(str(r[c[1]]).ljust(c[2])[:c[2]] for c in cols)
        print(f"  {row}")

    # Print response previews
    print(f"\n  {'─' * 60}")
    print("  Response Previews:\n")
    for r in results:
        marker = "+" if r["status"] == "PASS" else ("~" if r["status"] == "SKIPPED" else "x")
        print(f"  [{marker}] {r['endpoint']}: {r['response']}")

    # Summary line
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"{'=' * 90}")

    return 1 if failed > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Test deployed model serving endpoints")
    parser.add_argument("--catalog", default="workspace", help="Unity Catalog catalog name")
    parser.add_argument("--schema", default="ml_models", help="Schema name")
    args = parser.parse_args()
    sys.exit(test_deployments(catalog=args.catalog, schema=args.schema))


if __name__ == "__main__":
    main()
