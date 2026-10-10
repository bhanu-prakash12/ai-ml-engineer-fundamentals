#!/usr/bin/env python
"""Execute one or more notebooks in place, saving populated outputs.

Usage:
    python scripts/run_notebooks.py path/to/one.ipynb [more.ipynb ...]
    python scripts/run_notebooks.py --all day01_python_data_math   # run every .ipynb under a dir
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_notebook(path: Path, timeout: int = 600) -> tuple[bool, float, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    start = time.time()
    try:
        client.execute()
        status, note = True, "ok"
    except CellExecutionError as exc:
        status, note = False, str(exc).splitlines()[0][:200]
    elapsed = time.time() - start
    nbformat.write(nb, path)
    return status, elapsed, note


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="notebook paths")
    parser.add_argument("--all", metavar="DIR", help="run every .ipynb found under DIR")
    args = parser.parse_args()

    targets: list[Path] = [Path(p) for p in args.paths]
    if args.all:
        targets.extend(sorted((REPO_ROOT / args.all).rglob("*.ipynb")))
    if not targets:
        parser.error("provide notebook path(s) or --all DIR")

    failed = []
    for nb_path in targets:
        ok, elapsed, note = run_notebook(nb_path)
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {nb_path} ({elapsed:.1f}s) {note}")
        if not ok:
            failed.append(nb_path)

    if failed:
        print(f"\n{len(failed)} notebook(s) failed: {[str(p) for p in failed]}")
        return 1
    print(f"\nAll {len(targets)} notebook(s) executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
