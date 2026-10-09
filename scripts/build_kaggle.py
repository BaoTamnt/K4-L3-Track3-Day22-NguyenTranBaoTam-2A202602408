#!/usr/bin/env python3
"""Generate the standalone Kaggle T4 core notebook from the lab sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_colab import REPO, WORKDIR, code, md, render

TARGET = REPO / "kaggle" / "Lab22_DPO_T4_Kaggle.ipynb"
KAGGLE_WORK = "/kaggle/working/lab22"


def render_kaggle() -> dict:
    notebook = render("T4", core_only=True)
    for cell in notebook["cells"]:
        source = "".join(cell["source"]).replace(WORKDIR, KAGGLE_WORK)
        source = source.replace("from google.colab import files\n", "")
        source = source.replace("files.download(str(archive))", "print(f'Download from Kaggle Output: {archive}')")
        source = source.replace("from google.colab import userdata;", "Kaggle Secrets (optional):")
        source = source.replace("Colab secrets", "Kaggle Secrets")
        if cell["cell_type"] == "code" and source.startswith("import os\n"):
            source = source.replace("import os\n", 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n', 1)
        if cell["cell_type"] == "code" and source.startswith("!pip install"):
            # NB0-NB4 do not need GGUF or benchmark dependencies.
            from build_colab import requirements
            specs = [s for s in requirements() if not s.startswith(("llama-cpp-python", "lm-eval"))]
            source = "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', " + ", ".join(repr(s) for s in specs) + "], check=True)"
        cell["source"] = source.splitlines(keepends=True)
    notebook["cells"][0] = md(
        "# Lab 22 — Kaggle T4 · Nguyễn Trần Bảo Tâm\n\n"
        "Import this notebook into Kaggle. Select GPU T4 ×2 and enable Internet, then Run All. "
        "Only GPU 0 is used; the two GPUs do not pool VRAM. Core NB0–NB4 only. "
        "The final cells generate REFLECTION.md from actual outputs, verify the submission, "
        "and save `/kaggle/working/lab22/lab22-results.zip`. "
        "Download the executed notebook too. If a stage fails, the ZIP from previous stages is retained."
    )
    # Keep every source and the verifier available inside the execution environment.
    extras = [code("(WORK / 'notebooks').mkdir(exist_ok=True)\n(WORK / 'scripts').mkdir(exist_ok=True)")]
    for source in sorted((REPO / "notebooks").glob("*.py")):
        extras.append(code(f"%%writefile {KAGGLE_WORK}/notebooks/{source.name}\n" + source.read_text(encoding="utf-8")))
    for name in ("verify.py", "generate_reflection.py"):
        extras.append(code(f"%%writefile {KAGGLE_WORK}/scripts/{name}\n" + (REPO / "scripts" / name).read_text(encoding="utf-8")))
    insert = next(i for i, c in enumerate(notebook["cells"]) if "def export_evidence(stage):" in "".join(c["source"]))
    notebook["cells"][insert:insert] = extras
    # Report and verification run before the final export; a failed gate still leaves the earlier ZIP.
    notebook["cells"][-1:-1] = [
        md("## Phản tư từ số liệu thật và kiểm tra bài\n\nBản phản tư được tạo từ JSON/log thực nghiệm, không dùng số liệu mẫu. Đọc lại các ví dụ và nhận xét trước khi nộp. Nếu kiểm tra lỗi, đọc lỗi và sửa; không coi ZIP là bằng chứng bài đã hoàn tất."),
        code("subprocess.run([sys.executable, str(WORK / 'scripts/generate_reflection.py')], check=True)\n"
             "subprocess.run([sys.executable, str(WORK / 'scripts/verify.py')], check=True)\n"
             "(WORK / 'submission/verification.txt').write_text('Core checks passed in the Kaggle training environment.\\n', encoding='utf-8')"),
    ]
    # Include the report, verification status and SFT log in evidence exports.
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "def export_evidence(stage):" in source:
            source = source.replace("    config = WORK", "    for name in ('submission/REFLECTION.md', 'submission/verification.txt'):\n        item = WORK / name\n        if item.is_file():\n            evidence.append(item)\n    config = WORK")
            cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"].pop("colab", None)
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebook = render_kaggle()
    if args.check:
        if not TARGET.exists() or json.loads(TARGET.read_text(encoding="utf-8")) != notebook:
            print("Kaggle notebook is stale: run python scripts/build_kaggle.py")
            return 1
    else:
        TARGET.parent.mkdir(exist_ok=True)
        TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {TARGET} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
