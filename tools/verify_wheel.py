"""Verify wheel metadata, typed marker, entry point, and every RECORD digest."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
from pathlib import Path
from zipfile import ZipFile


def verify(path: Path) -> int:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Expected exactly one wheel metadata directory")
        root = metadata_paths[0].rsplit("/", 1)[0]
        metadata = archive.read(metadata_paths[0]).decode()
        for expected in ("Name: rag-evaluation-toolkit", "Version: 0.1.0", "Requires-Python: >=3.11"):
            if expected not in metadata:
                raise ValueError(f"Missing metadata: {expected}")
        if "rag_eval_toolkit/py.typed" not in names:
            raise ValueError("Missing py.typed marker")
        entry_points = archive.read(root + "/entry_points.txt").decode()
        if "rag-eval = rag_eval_toolkit.cli.app:main" not in entry_points:
            raise ValueError("Missing CLI entry point")
        rows = list(csv.reader(io.StringIO(archive.read(root + "/RECORD").decode())))
        if {row[0] for row in rows} != names:
            raise ValueError("Wheel file list does not match RECORD")
        checked = 0
        for name, recorded_hash, size in rows:
            if name == root + "/RECORD":
                continue
            payload = archive.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            if recorded_hash != "sha256=" + digest or int(size) != len(payload):
                raise ValueError(f"RECORD integrity mismatch: {name}")
            checked += 1
    return checked


if __name__ == "__main__":
    wheels = [Path(sys.argv[1])] if len(sys.argv) > 1 else sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit("Provide exactly one wheel, or build one wheel in dist/")
    count = verify(wheels[0])
    print(f"Verified {wheels[0].name}: {count} file digests, metadata, types, and CLI entry point")
