# HuggingFace to Databricks Model Pipeline

A CLI tool that downloads pre-trained models from HuggingFace, registers them in Databricks Unity Catalog via MLflow, and optionally deploys them to serving endpoints with AI Gateway controls -- all from a single command.

Works from **any platform** (Windows, macOS, Linux). The model wrapper is serialised via CloudPickle with an inline class definition, so the serving container never needs the project's source code on its path.

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

**Register and deploy to a serving endpoint (fails if endpoint already exists):**

```bash
uv run python register_model.py --deploy
```

**Redeploy to an existing endpoint (deletes and recreates it):**

```bash
uv run python register_model.py --deploy --redeploy
```

**Deploy Qwen2.5-0.5B for text generation:**

```bash
uv run python register_model.py \
  --model-id Qwen/Qwen2.5-0.5B \
  --task text-generation \
  --model-name qwen25_05b \
  --artifact-path qwen_model \
  --deploy
```

**Use a different HuggingFace model (e.g. GPT-2 text generation):**

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

## Tested Models

| Model | Task | Size | Status | Notes |
|-------|------|------|--------|-------|
| `distilbert-base-uncased-finetuned-sst-2-english` | `sentiment-analysis` | ~260 MB | Registered + Deployed | Default model. Fast inference on CPU small compute. |
| `Qwen/Qwen2.5-0.5B` | `text-generation` | ~1 GB | Registered + Deployed | Endpoint takes ~9 min to reach READY. Slow CPU inference. See [limitations](#limitations). |

## Supported Tasks

The pipeline supports any HuggingFace `pipeline()` task. The MLflow signature and predict wrapper adapt automatically based on the task type:

| Task | Output Format | Recommended Models |
|------|---------------|--------------------|
| `sentiment-analysis` / `text-classification` | label + score | DistilBERT, RoBERTa |
| `text-generation` | generated text (`max_new_tokens` configurable) | GPT-2, Qwen2.5-0.5B |
| `ner` | token-level entities | `dslim/bert-base-NER` |
| `fill-mask` | token predictions | `bert-base-uncased` |
| `question-answering` | extractive answer + score | `deepset/roberta-base-squad2` |
| `zero-shot-classification` | label rankings | `facebook/bart-large-mnli` |
| `feature-extraction` | embedding vectors | `sentence-transformers/all-MiniLM-L6-v2` |

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

**Deployment behaviour:** By default, `--deploy` fails fast if the endpoint already exists. Pass `--redeploy` to delete the existing endpoint and recreate it from scratch.

### Model serialisation (CloudPickle inline class)

The MLflow adaptor defines the pyfunc wrapper class **inline** inside `log_model()`. This is a deliberate design choice:

- **CloudPickle serialises local/nested classes by value** (full bytecode), not by module reference. The serving container receives the complete class definition in the pickle payload and doesn't need the project's `src/` package on its `sys.path`.
- **Cross-platform safety.** Because CloudPickle doesn't store file paths, models logged from Windows work identically on the Linux serving containers. No path translation or monkey-patching required.
- **Self-contained artifacts.** Each logged model version carries everything it needs. The HuggingFace pipeline is downloaded at serving time inside `load_context()`, keeping the artifact small.

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
| `--redeploy` | off | Delete and recreate the endpoint if it already exists |
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
├── register_model.py              # CLI entry point + composition root
├── test_deployments.py            # Test all deployed serving endpoints
├── tasks.py                       # Task runner (lint, test, format, deploy tests)
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
│       ├── mlflow_adaptor.py       # MLflow logging + UC registration (inline pyfunc wrapper)
│       ├── unity_catalog_adaptor.py# Schema management + model queries
│       └── serving_adaptor.py      # Endpoint deployment + AI Gateway
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── test_connectivity.py       # Databricks auth + SQL smoke tests
    ├── test_model_serving.py      # Endpoint integration tests
    └── test_mcp.py                # MCP server readiness tests
```

## Development

All tasks are run via `tasks.py`:

```bash
uv run python tasks.py help       # Show all available tasks
```

### Setup

| Task | Command | Description |
|------|---------|-------------|
| `install` | `uv run python tasks.py install` | Install all dependencies with UV |

### Code Quality

| Task | Command | Description |
|------|---------|-------------|
| `ruff` | `uv run python tasks.py ruff` | Run ruff linter |
| `ruff-fix` | `uv run python tasks.py ruff-fix` | Run ruff with auto-fix |
| `format` | `uv run python tasks.py format` | Format code with ruff |
| `format-check` | `uv run python tasks.py format-check` | Check code formatting |
| `ty` | `uv run python tasks.py ty` | Run ty type checker (Astral) |
| `lint` | `uv run python tasks.py lint` | Run all linters (ruff + format check) |

### Testing

| Task | Command | Description |
|------|---------|-------------|
| `test` | `uv run python tasks.py test` | Run unit/integration tests with pytest |
| `test-deployment` | `uv run python tasks.py test-deployment` | Test all deployed serving endpoints |
| `coverage` | `uv run python tasks.py coverage` | Run tests with coverage report |

### Pipeline

| Task | Command | Description |
|------|---------|-------------|
| `all` | `uv run python tasks.py all` | Full pipeline: install, lint, test |

### Testing Deployed Endpoints

`test-deployment` discovers all serving endpoints backed by models in your catalog/schema, sends a test query to each READY endpoint, and prints a summary:

```bash
uv run python tasks.py test-deployment
```

You can also run it directly with custom catalog/schema:

```bash
uv run python test_deployments.py --catalog workspace --schema ml_models
```

Example output:

```
  Endpoint                 | Model                  | Ver  | Task                   | State     | Test    | Latency
  -------------------------+------------------------+------+------------------------+-----------+---------+----------
  distilbert-sentiment     | distilbert_sentiment   | 24   | sentiment-analysis     | READY     | PASS    | 856
  qwen25-05b               | qwen25_05b             | 2    | text-generation        | READY     | PASS    | 12780

  Response Previews:
  [+] distilbert-sentiment: {'label': 'POSITIVE', 'score': 0.9998855590820312}
  [+] qwen25-05b: {'generated_text': '! I'm 16 and I'm just trying to find a good math tutor...

  Total: 2 | Passed: 2 | Failed: 0 | Skipped: 0
```

## Requirements

- Python >= 3.10
- A Databricks workspace with Unity Catalog enabled
- A `.env` file with `DATABRICKS_HOST` and `DATABRICKS_TOKEN`

---

## Limitations

### Databricks Free Tier / Serverless Small Compute

- **~2 CPU cores, limited RAM.** Adequate for encoder models (BERT, DistilBERT, RoBERTa) but marginal for generative models.
- **DAB deployment fails** on free-tier workspaces due to S3 path-style URL issues (`PermanentRedirect`). Running the CLI locally works fine. Expected to work on paid workspaces.

### CPU-Only Inference

- The pipeline uses **CPU-only PyTorch** to keep container images small (~200 MB vs 4+ GB with GPU).
- **Encoder models** (DistilBERT, RoBERTa) return results in **milliseconds** on CPU — ideal for this setup.
- **Generative models** (Qwen2.5-0.5B, GPT-2) are **very slow on CPU** — expect several seconds per response. Not practical for production use without GPU compute.
- Models above ~1B parameters will likely OOM or timeout on Small compute.

### Model Size and Cold Start

- Model weights are **downloaded from HuggingFace Hub at serving time** (each container startup). Large models increase cold-start time significantly.
- No local caching between container restarts on serverless compute.
- Practical maximum for a smooth experience: **~1 GB** of model weights. Qwen2.5-0.5B (~1 GB) is at the upper limit and took ~9 minutes for the endpoint to reach READY.
- Models above ~2 GB (Qwen2.5-1.5B, LLaMA 3B, BART-large) will likely timeout during container startup.

### Text Generation Specifically

- Output is capped at `max_new_tokens=100` by default to prevent CPU timeouts. Override via the `params` argument when querying the endpoint.
- `return_full_text=False` — responses contain only generated text, not the input prompt.
- A 0.5B parameter model produces limited quality output. Expect short, sometimes incoherent text continuations. This is a deployment demonstration, not a production LLM.

### General

- **Input format**: All models expect a single-column text DataFrame. Multi-input tasks (e.g., question-answering with separate question + context fields) require the caller to format input accordingly.
- **No streaming**: Endpoints return the full response synchronously. SSE/streaming is not implemented.
- **Single model per endpoint**: Each deployment creates one endpoint serving one model version.
- **No quantisation**: Models run at full precision (fp32). INT8/INT4 quantisation could improve performance for generative models but is not currently supported.

---

## Next Steps: Databricks Asset Bundle

The natural evolution is to run this pipeline as a **Databricks Asset Bundle (DAB) job** on remote compute, removing the need for a local machine entirely. A parameterised DAB job would accept `--python-params` at run time, making it easy to register any HuggingFace model from CI/CD or a scheduled trigger:

```bash
databricks bundle run register_model -t prod \
  --python-params '["--model-id", "gpt2", "--task", "text-generation", "--model-name", "gpt2_gen", "--deploy", "--redeploy"]'
```

**Known issue on Databricks Community / free-tier workspaces:** Serverless compute on these workspaces uses path-style S3 URLs (`s3.amazonaws.com/bucket`) when copying model artifacts to Unity Catalog storage. The S3 bucket rejects these with a `PermanentRedirect`, requiring virtual-hosted-style URLs (`bucket.s3.amazonaws.com`) instead. This causes model versions to get stuck in `PENDING_REGISTRATION` indefinitely. The issue does not affect the local CLI approach (which uploads artifacts directly) and is expected to work correctly on paid Databricks workspaces with standard compute or dedicated S3 endpoints.
