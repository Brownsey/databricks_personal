"""Rust-style Result type for explicit error handling without exceptions."""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True)


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful result containing a value."""

    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    """Error result containing an error."""

    error: E


Result = Ok[T] | Err[E]
