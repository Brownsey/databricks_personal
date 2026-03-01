"""MLflow adaptor — implements ExperimentTrackerPort."""

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.pyfunc import PythonModel

from src.domain.errors import TrackingError
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)


class MLflowAdaptor:
    """Logs models to Databricks-managed MLflow and registers via Unity Catalog."""

    def __init__(
        self,
        experiment_name: str,
        *,
        artifact_path: str = "sentiment_model",
        region: str = "us-east-2",
    ) -> None:
        self._experiment_name = experiment_name
        self._artifact_path = artifact_path
        self._region = region

    def log_model(
        self,
        model: Any,
        task: str,
        model_id: str,
        registered_model_name: str,
    ) -> Result[str, TrackingError]:
        """Log a pyfunc model to MLflow and register in Unity Catalog."""
        try:
            mlflow.set_experiment(self._experiment_name)
            logger.info("MLflow experiment: %s", self._experiment_name)

            with mlflow.start_run(run_name=f"register-{registered_model_name}") as run:
                run_id = run.info.run_id
                logger.info("MLflow run started: %s", run_id)

                mlflow.set_tag("hf_model_id", model_id)
                mlflow.set_tag("hf_task", task)
                mlflow.set_tag("region", self._region)
                mlflow.log_param("model_source", "huggingface")
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("task", task)

                logger.info("Logging model and registering as '%s'...", registered_model_name)

                # CPU-only torch keeps container small (~200 MB vs 4+ GB GPU)
                conda_env = {
                    "channels": ["defaults"],
                    "dependencies": [
                        "python=3.10",
                        "pip",
                        {
                            "pip": [
                                "--index-url https://download.pytorch.org/whl/cpu",
                                "--extra-index-url https://pypi.org/simple",
                                "transformers",
                                "torch",
                                "pandas",
                                "mlflow",
                            ]
                        },
                    ],
                    "name": "mlflow-env",
                }

                signature = mlflow.models.infer_signature(
                    pd.DataFrame(["text input"], columns=["text"]),
                    pd.DataFrame([{"label": "POSITIVE", "score": 0.99}]),
                )

                # Define wrapper inline so CloudPickle serialises it by value
                # (not by module reference). This avoids the serving container
                # needing our project's src/ package on its sys.path.
                class _HFPyfuncWrapper(PythonModel):
                    def load_context(self, context) -> None:
                        from transformers import pipeline as hf_pipeline

                        cfg = context.model_config
                        self._pipe = hf_pipeline(cfg["task"], model=cfg["model_id"])

                    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
                        texts = model_input.iloc[:, 0].tolist()
                        results = self._pipe(texts)
                        return pd.DataFrame(results)

                model_info = mlflow.pyfunc.log_model(
                    artifact_path=self._artifact_path,
                    python_model=_HFPyfuncWrapper(),
                    model_config={"model_id": model_id, "task": task},
                    registered_model_name=registered_model_name,
                    signature=signature,
                    conda_env=conda_env,
                )
                logger.info("Model logged. URI: %s", model_info.model_uri)

            return Ok(run_id)
        except Exception as e:
            return Err(TrackingError(reason=str(e)))
