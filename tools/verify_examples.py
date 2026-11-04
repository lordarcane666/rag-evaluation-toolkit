"""Execute offline README/API/CLI examples; never run the opt-in live-judge example."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    python_examples = re.findall(r"```python\n(.*?)\n```", readme, flags=re.DOTALL)
    for example in python_examples:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(example, "README.md", "exec"), {})
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(root / "examples/custom_metric.py"), run_name="__main__")
    checked = 0

    def run(*args: str, expected: int = 0) -> str:
        nonlocal checked
        process = subprocess.run(
            [sys.executable, "-m", "rag_eval_toolkit", *args],
            capture_output=True, encoding="utf-8", timeout=60, check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if process.returncode != expected:
            raise RuntimeError(f"Example command failed: {args}; exit={process.returncode}")
        checked += 1
        return process.stdout

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        run("version")
        run("--help")
        run("metrics")
        run("validate", "examples/demo.jsonl")
        run("evaluate", "examples/demo.jsonl", "--metrics", "exact_match,token_f1,context_precision,context_recall")
        for format in ("json", "markdown", "csv"):
            output = directory / f"result.{format}"
            run("evaluate", "examples/demo.jsonl", "--format", format, "--output", str(output))
            if not output.read_text(encoding="utf-8"):
                raise ValueError("Empty report output")
        run("benchmark", "examples/benchmark.json", "--output", str(directory / "benchmark.json"))
        for label, filename in (("baseline", "demo"), ("candidate", "candidate")):
            run("evaluate", f"examples/{filename}.jsonl", "--metrics", "token_f1,context_recall",
                "--format", "json", "--output", str(directory / f"{label}.json"))
        comparison = run("compare", str(directory / "baseline.json"), str(directory / "candidate.json"), "--threshold", "0.01")
        if json.loads(comparison)["has_regression"]:
            raise ValueError("Unexpected regression in synthetic improvement example")
        run("compare", str(directory / "candidate.json"), str(directory / "baseline.json"), expected=1)
        fixed = ("evaluate", "examples/demo.jsonl", "--format", "json", "--timestamp", "2026-01-01T00:00:00Z")
        if run(*fixed) != run(*fixed):
            raise ValueError("Deterministic JSON is not reproducible")
    print(f"Verified {checked} CLI executions, {len(python_examples)} README Python example(s), and custom metric example")


if __name__ == "__main__":
    main()
