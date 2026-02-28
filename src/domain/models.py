"""Domain models for model registration pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRegistrationResult:
    """Result of a successful model registration."""

    registered_model_name: str
    run_id: str
    model_uri: str
    model_info: dict
    versions: list[dict]
    endpoint_name: str | None = None
