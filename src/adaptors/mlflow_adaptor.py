"""MLflow adaptor — implements ExperimentTrackerPort."""

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.domain.errors import TrackingError
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)


class _SentimentPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a HuggingFace pipeline as an MLflow pyfunc model."""

    def __init__(self, model_id: str, task: str) -> None:
        self.model_id = model_id
        self.task = task

    def load_context(self, context):
        from transformers import pipeline

        self._pipe = pipeline(self.task, model=self.model_id)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        texts = model_input.iloc[:, 0].tolist()
        results = self._pipe(texts)
        return pd.DataFrame(results)


class MLflowAdaptor:
    """Logs models to Databricks-managed MLflow and registers via Unity Catalog."""

    def __init__(self, experiment_name: str) -> None:
        self._experiment_name = experiment_name

    def log_model(
        self,
        model: Any,
        task: str,
        model_id: str,
        registered_model_name: str,
    ) -> Result[str, TrackingError]:
        """Log a pyfunc-wrapped model to MLflow and register in Unity Catalog."""
        try:
            mlflow.set_experiment(self._experiment_name)
            logger.info("MLflow experiment: %s", self._experiment_name)

            with mlflow.start_run(run_name=f"register-{registered_model_name}") as run:
                run_id = run.info.run_id
                logger.info("MLflow run started: %s", run_id)

                mlflow.set_tag("hf_model_id", model_id)
                mlflow.set_tag("hf_task", task)
                mlflow.log_param("model_source", "huggingface")
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("task", task)

                logger.info("Logging model and registering as '%s'...", registered_model_name)
                input_example = pd.DataFrame(["This movie was fantastic!"], columns=["text"])
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="sentiment_model",
                    python_model=_SentimentPyfuncWrapper(model_id=model_id, task=task),
                    registered_model_name=registered_model_name,
                    input_example=input_example,
                    pip_requirements=["transformers", "torch", "pandas"],
                )
                logger.info("Model logged. URI: %s", model_info.model_uri)

            return Ok(run_id)
        except Exception as e:
            return Err(TrackingError(reason=str(e)))
