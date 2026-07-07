from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from amsrr.schemas.common import canonical_json


def stable_hash(data: Any) -> str:
    return sha256(canonical_json(data).encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

