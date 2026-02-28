"""Domain layer — core business logic, ports, and types."""

from src.domain.errors import (
    ModelLoadError,
    ModelPipelineError,
    RegistryError,
    TrackingError,
)
from src.domain.models import ModelRegistrationResult
from src.domain.ports import (
    ExperimentTrackerPort,
    ModelLoaderPort,
    ModelRegistryPort,
)
from src.domain.result import Err, Ok, Result
from src.domain.services import register_model_pipeline

__all__ = [
    # Result types
    "Ok",
    "Err",
    "Result",
    # Errors
    "ModelPipelineError",
    "ModelLoadError",
    "TrackingError",
    "RegistryError",
    # Models
    "ModelRegistrationResult",
    # Ports
    "ModelLoaderPort",
    "ExperimentTrackerPort",
    "ModelRegistryPort",
    # Services
    "register_model_pipeline",
]
