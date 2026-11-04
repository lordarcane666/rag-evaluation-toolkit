"""Local output operations with explicit overwrite and same-file protection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rag_eval_toolkit.core.errors import ReportError


def write_output(
    path: str | Path, content: str, *, overwrite: bool = False,
    protected: tuple[Path, ...] = (),
) -> None:
    target = Path(path)
    resolved = target.resolve()
    if any(resolved == source.resolve() for source in protected):
        raise ReportError("Output must not replace an input dataset, configuration, or report")
    if target.is_symlink():
        raise ReportError("Refusing to write through an output symlink")
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            # Exclusive creation prevents a check-then-write overwrite race.
            with target.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=target.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, target)
    except FileExistsError:
        raise ReportError("Output exists; choose another path or pass --overwrite") from None
    except OSError:
        raise ReportError("Unable to write output file") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
