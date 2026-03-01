# HuggingFace to Databricks Model Pipeline

A CLI tool that downloads pre-trained models from HuggingFace, registers them in Databricks Unity Catalog, and optionally deploys them to serving endpoints with AI Gateway controls -- all from a single command.

## Quick Start

```bash
# Install dependencies
uv sync

# Set up your .env file
echo "DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com" > .env
echo "DATABRICKS_TOKEN=dapi..." >> .env
```

### Example Commands

**Register the default sentiment model in Unity Catalog:**

```bash
uv run python register_model.py
```

**Register and deploy to a serving endpoint:**

```bash
uv run python register_model.py --deploy
```

**Use a different HuggingFace model (e.g. text generation):**

```bash
uv run python register_model.py \
  --model-id gpt2 \
  --task text-generation \
  --model-name gpt2_generator
```

**Register a named entity recognition model:**

```bash
uv run python register_model.py \
  --model-id dslim/bert-base-NER \
  --task ner \
  --model-name bert_ner \
  --catalog workspace \
  --schema nlp_models
```

**Deploy a zero-shot classifier to a specific region with rate limiting:**

```bash
uv run python register_model.py \
  --model-id facebook/bart-large-mnli \
  --task zero-shot-classification \
  --model-name bart_zero_shot \
  --deploy \
  --region us-west-2 \
  --workload-size Medium \
  --rate-limit 500 \
  --enable-usage-tracking
```

**Deploy with full AI Gateway features (inference tables + usage tracking):**

```bash
uv run python register_model.py \
  --model-id distilbert-base-uncased-finetuned-sst-2-english \
  --task sentiment-analysis \
  --model-name distilbert_sentiment \
  --deploy \
  --enable-usage-tracking \
  --enable-inference-tables \
  --inference-table-prefix sentiment_logs \
  --rate-limit 1000
```

**Log a model to MLflow only (skip Unity Catalog registration):**

```bash
uv run python register_model.py \
  --model-id bert-base-uncased \
  --task fill-mask \
  --no-register
```

**Override the endpoint name and disable scale-to-zero for production:**

```bash
uv run python register_model.py \
  --model-id distilbert-base-uncased-finetuned-sst-2-english \
  --task sentiment-analysis \
  --deploy \
  --endpoint-name prod-sentiment-v2 \
  --no-scale-to-zero \
  --workload-size Large
```

Run `uv run python register_model.py --help` for the full argument reference.

---

## Architecture

The project follows **hexagonal architecture** (ports and adaptors), which cleanly separates business logic from infrastructure concerns.

```
register_model.py          CLI entry point + composition root
    |
    v
src/domain/
    services.py            Pipeline orchestration (pure business logic)
    ports.py               Protocol interfaces (contracts)
    models.py              Domain data models (frozen dataclasses)
    result.py              Rust-style Result[T, E] type
    errors.py              Typed error hierarchy
    |
    v
src/adaptors/
    huggingface_adaptor    Downloads models from HuggingFace Hub
    mlflow_adaptor         Logs models to Databricks-managed MLflow
    unity_catalog_adaptor  Manages UC schemas and model registration
    serving_adaptor        Deploys endpoints + configures AI Gateway
    |
    v
src/
    config.py              Loads .env credentials
    connect.py             Databricks WorkspaceClient singleton
```

### How it works

The pipeline runs through six steps, each backed by a port/adaptor pair:

| Step | Description | Port | Adaptor |
|------|-------------|------|---------|
| 1 | Ensure the target UC schema exists | `ModelRegistryPort` | `UnityCatalogAdaptor` |
| 2 | Download the model from HuggingFace | `ModelLoaderPort` | `HuggingFaceAdaptor` |
| 3 | Log the model to MLflow and register in UC | `ExperimentTrackerPort` | `MLflowAdaptor` |
| 4 | Verify registration (metadata + versions) | `ModelRegistryPort` | `UnityCatalogAdaptor` |
| 5 | Deploy to a serving endpoint (optional) | `ModelServingPort` | `ServingAdaptor` |
| 6 | Configure AI Gateway (optional) | `ModelServingPort` | `ServingAdaptor` |

Steps 1 and 4 are skipped when `--no-register` is passed. Steps 5-6 only run when `--deploy` is passed.

### Error handling

The pipeline uses a **Rust-style `Result` type** instead of exceptions. Every operation returns `Ok(value)` or `Err(error)`, and errors propagate up the call chain with typed context:

- `ModelLoadError` -- HuggingFace download failed
- `TrackingError` -- MLflow logging/registration failed
- `RegistryError` -- Unity Catalog operation failed
- `ServingError` -- Endpoint deployment or AI Gateway configuration failed

### Benefits of this architecture

- **Testable** -- Domain logic has zero infrastructure imports. Mock any port to unit test the pipeline.
- **Swappable** -- Replace `HuggingFaceAdaptor` with a local file loader, or swap `MLflowAdaptor` for W&B, without touching business logic.
- **Explicit errors** -- No hidden exceptions. Every failure path is typed and visible in the function signature.
- **Fully configurable** -- Every parameter that was previously hardcoded is now a CLI argument with sensible defaults.

---

## CLI Reference

Arguments are organized into logical groups:

### Model Source

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model ID |
| `--task` | `sentiment-analysis` | HuggingFace pipeline task |

### Unity Catalog

| Flag | Default | Description |
|------|---------|-------------|
| `--catalog` | `workspace` | UC catalog name |
| `--schema` | `ml_models` | Schema within the catalog |
| `--model-name` | `distilbert_sentiment` | Name to register under |
| `--no-register` | off | Skip UC registration (MLflow-only) |

### Deployment

| Flag | Default | Description |
|------|---------|-------------|
| `--deploy` | off | Deploy to a serving endpoint |
| `--endpoint-name` | derived from `--model-name` | Override endpoint name |
| `--workload-size` | `Small` | `Small` / `Medium` / `Large` |
| `--no-scale-to-zero` | off | Keep endpoint always warm |
| `--deploy-timeout` | `1200` | Max seconds to wait for readiness |
| `--region` | `us-east-2` | Target region (avoid cross-region transfer) |

### AI Gateway

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-usage-tracking` | off | Track usage via system tables |
| `--enable-inference-tables` | off | Log request/response payloads |
| `--inference-table-catalog` | same as `--catalog` | Catalog for inference tables |
| `--inference-table-schema` | same as `--schema` | Schema for inference tables |
| `--inference-table-prefix` | none | Table name prefix |
| `--rate-limit` | off | Endpoint-level calls per minute |

### MLflow

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment-name` | `/Users/<you>/<model-name>-registration` | MLflow experiment path |
| `--artifact-path` | `sentiment_model` | Artifact path in the run |

### Infrastructure

| Flag | Default | Description |
|------|---------|-------------|
| `--warehouse-id` | `DATABRICKS_WAREHOUSE_ID` env var | SQL warehouse ID |

---

## Project Structure

```
.
├── register_model.py              # CLI entry point
├── tasks.py                       # Task runner (lint, test, format)
├── pyproject.toml                 # Dependencies and tool config
├── .env                           # Databricks credentials (not committed)
├── src/
│   ├── config.py                  # .env loader + MLflow env setup
│   ├── connect.py                 # WorkspaceClient + SQL utilities
│   ├── domain/
│   │   ├── services.py            # Pipeline orchestration
│   │   ├── ports.py               # Protocol interfaces
│   │   ├── models.py              # Domain data models
│   │   ├── result.py              # Ok / Err result type
│   │   └── errors.py              # Typed error classes
│   └── adaptors/
│       ├── huggingface_adaptor.py  # HuggingFace model loading
│       ├── mlflow_adaptor.py       # MLflow logging + UC registration
│       ├── unity_catalog_adaptor.py# Schema management + model queries
│       └── serving_adaptor.py      # Endpoint deployment + AI Gateway
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── test_connectivity.py       # Databricks auth + SQL smoke tests
    ├── test_model_serving.py      # Endpoint integration tests
    └── test_mcp.py                # MCP server readiness tests
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Lint
uv run python tasks.py lint

# Format
uv run python tasks.py format

# Run tests
uv run python tasks.py test

# Full pipeline (install + lint + test)
uv run python tasks.py all
```

## Requirements

- Python >= 3.10
- A Databricks workspace with Unity Catalog enabled
- A `.env` file with `DATABRICKS_HOST` and `DATABRICKS_TOKEN`
