"""Build using the installed PEP 517 backend, without downloading build dependencies.

Normally use `python -m build`. In an offline environment with setuptools and
wheel already installed, `python tools/build_local.py` calls the same backend.
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    from setuptools import build_meta

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = root / "dist"
    output.mkdir(exist_ok=True)
    print(build_meta.build_wheel(str(output)))
    print(build_meta.build_sdist(str(output)))


if __name__ == "__main__":
    main()
