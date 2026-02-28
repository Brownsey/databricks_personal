"""HuggingFace adaptor — implements ModelLoaderPort."""

import logging
from typing import Any

from src.domain.errors import ModelLoadError
from src.domain.result import Err, Ok, Result

logger = logging.getLogger(__name__)


class HuggingFaceAdaptor:
    """Loads pre-trained models from HuggingFace Hub via the transformers library."""

    def load_model(self, model_id: str, task: str) -> Result[Any, ModelLoadError]:
        """Download a HuggingFace pipeline for the given model and task."""
        try:
            from transformers import pipeline

            logger.info("Downloading HuggingFace model: %s (task=%s)", model_id, task)
            pipe = pipeline(task, model=model_id)

            # Sanity check
            test_result = pipe("This is a test sentence.")
            logger.info("Sanity check passed: %s", test_result)

            return Ok(pipe)
        except Exception as e:
            return Err(ModelLoadError(model_id=model_id, reason=str(e)))
