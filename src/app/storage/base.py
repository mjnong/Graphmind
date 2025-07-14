# app/storage/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

class BaseStore(ABC):
    """
    Minimal contract all storage back-ends must fulfil.
    Every method is synchronous on purpose so you can call
    it from Celery tasks as well as FastAPI routes without
    juggling event-loops.
    """

    @abstractmethod
    def put(self, file: BinaryIO, object_key: str) -> None:
        """Save *file* (opened in binary mode) under *object_key*."""
        ...

    @abstractmethod
    def get(self, object_key: str, dest: Path) -> Path:
        """Download the object to *dest* (a local path) and return that path."""
        ...

    @abstractmethod
    def delete(self, object_key: str) -> None:
        """Remove the object."""
        ...

    @abstractmethod
    def url(self, object_key: str, expires: int = 3600) -> str:
        """Return a publicly accessible (or pre-signed) URL."""
        ...