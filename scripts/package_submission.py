#!/usr/bin/env python3
"""Package available lab evidence and sources; omit weights, secrets and caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "submission/Lab22-NguyenTranBaoTam-2A202602408.zip"


def included_files() -> list[Path]:
    files = list(REPO.glob("*.md")) + list(REPO.glob("requirements*.txt"))
    files += [REPO / name for name in (".gitignore", "LICENSE", "Makefile", "pyproject.toml", "setup-colab.sh", "setup-laptop.sh")]
    folders = {
        "lab22": {".py"}, "scripts": {".py"}, "notebooks": {".py", ".ipynb"},
        "colab": {".ipynb"}, "kaggle": {".ipynb", ".md"}, "docs": {".md"},
        "solutions": {".md"}, "data/pref": {".parquet", ".json"},
        "data/eval": {".json", ".jsonl"}, "adapters/dpo": {".json"},
        "adapters/sft-mini": {".json"}, "submission": {".md", ".json", ".txt", ".png", ".jpg", ".jpeg"},
    }
    blocked = {"__pycache__", ".ipynb_checkpoints", "runs", ".pytest_cache"}
    for folder, suffixes in folders.items():
        for path in (REPO / folder).rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and not blocked.intersection(path.relative_to(REPO).parts):
                files.append(path)
    return sorted({p for p in files if p.is_file()})


def main() -> int:
    files = included_files()
    recovery = json.loads((REPO / "submission/notebook-recovery.json").read_text(encoding="utf-8"))
    missing = [name for name in recovery["missing_from_notebook_outputs"] if not (REPO / name).is_file()]
    if not (REPO / "models/sft-merged/config.json").is_file():
        missing.append("models/sft-merged/config.json")
    manifest = {
        "student": "Nguyễn Trần Bảo Tâm", "student_id": "2A202602408", "lab": 22,
        "package_kind": "Available local evidence and reproducible sources",
        "kaggle_core_verification": "Passed, recorded in executed notebook output",
        "local_artifacts_complete": not missing,
        "missing_artifacts": missing,
        "weights_included": False, "secrets_included": False,
        "sha256": {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    TARGET.parent.mkdir(exist_ok=True)
    temporary = TARGET.with_suffix(".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(REPO).as_posix())
        archive.writestr("PACKAGE-MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for name, checksum in manifest["sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise RuntimeError(f"Checksum mismatch: {name}")
    temporary.replace(TARGET)
    print(f"Created {TARGET.relative_to(REPO)} ({len(files)} files, {TARGET.stat().st_size:,} bytes)")
    print(f"Missing original artifacts: {len(missing)}; see PACKAGE-MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
