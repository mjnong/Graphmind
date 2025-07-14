# app/storage/local.py
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import urljoin

from .base import BaseStore

class LocalStore(BaseStore):
    """
    Stores files under <base_path>/<object_key>
    where *object_key* can include slashes (e.g. uploads/2025/07/uuid.pdf).

    Because everything is already local, `get()` just copies/links
    the file into *dest* so workers can treat both back-ends the same way.
    """

    def __init__(self, base_path: str = "/var/_uploads", base_url: str = "/files/"):
        self.root = Path(base_path).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/") + "/"

    # ---------- helpers ---------- #
    def _full(self, key: str) -> Path:
        return self.root.joinpath(key).resolve()

    # ---------- API ---------- #
    def put(self, file, object_key: str) -> None:
        dst = self._full(object_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as out:
            shutil.copyfileobj(file, out)

    def get(self, object_key: str, dest: Path) -> Path:
        src = self._full(object_key)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return dest

    def delete(self, object_key: str) -> None:
        try:
            self._full(object_key).unlink()
        except FileNotFoundError:
            pass

    def url(self, object_key: str, expires: int = 3600) -> str:
        # No signing—just return the static URL.
        return urljoin(self.base_url, object_key)