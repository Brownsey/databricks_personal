"""Standalone pyfunc model definition for MLflow serving.

This file is passed as a *path* to ``mlflow.pyfunc.log_model(python_model=...)``,
which stores the **source code** instead of cloudpickle bytecode.  This avoids
Python-version incompatibilities (e.g. 3.14 → 3.10 code-object mismatch).

At serving time MLflow imports this module, finds the model registered via
``set_model()``, and uses it to handle prediction requests.
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


mlflow.models.set_model(SentimentPyfuncWrapper())
