"""Standalone pyfunc model definition for MLflow serving.

``SentimentPyfuncWrapper`` is instantiated and passed to
``mlflow.pyfunc.log_model(python_model=...)`` in the MLflow adaptor.
The wrapper has no state at pickle time — the HuggingFace pipeline is
downloaded at serving time inside ``load_context()``.
"""

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel


class SentimentPyfuncWrapper(PythonModel):
    """Wraps a HuggingFace pipeline as an MLflow pyfunc model."""

    def load_context(self, context) -> None:
        from transformers import pipeline

        cfg = context.model_config
        self._pipe = pipeline(cfg["task"], model=cfg["model_id"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        texts = model_input.iloc[:, 0].tolist()
        results = self._pipe(texts)
        return pd.DataFrame(results)


# set_model is only available in MLflow 3.x (code-based logging).
# Guard for compatibility with older runtimes that use CloudPickle.
if hasattr(mlflow.models, "set_model"):
    mlflow.models.set_model(SentimentPyfuncWrapper())
