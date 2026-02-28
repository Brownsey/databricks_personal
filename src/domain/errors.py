"""Typed error hierarchy for model registration pipeline."""

from dataclasses import dataclass


class ModelPipelineError(Exception):
    """Base exception for all model pipeline errors."""

    pass


@dataclass(frozen=True)
class ModelLoadError(ModelPipelineError):
    """Failed to load a model from an external source."""

    model_id: str
    reason: str

    def __str__(self) -> str:
        return f"Failed to load model '{self.model_id}': {self.reason}"


@dataclass(frozen=True)
class TrackingError(ModelPipelineError):
    """Failed to log a model to the experiment tracker."""

    reason: str

    def __str__(self) -> str:
        return f"Experiment tracking error: {self.reason}"


@dataclass(frozen=True)
class RegistryError(ModelPipelineError):
    """Failed to interact with the model registry."""

    reason: str

    def __str__(self) -> str:
        return f"Model registry error: {self.reason}"


@dataclass(frozen=True)
class ServingError(ModelPipelineError):
    """Failed to deploy or query a model serving endpoint."""

    reason: str

    def __str__(self) -> str:
        return f"Model serving error: {self.reason}"
