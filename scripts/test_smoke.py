"""CPU-only structural checks: sources parse, trainer APIs match TRL 1.13 /
transformers 5, and the Colab bundles are in sync with the sources.

Run:  pytest -q scripts/   (or `make test`).
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "00_dpo_loss_from_scratch", "01_sft_mini", "02_preference_data", "03_dpo_train",
    "03b_dpo_variants", "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
    "07_grpo_bonus",
]
SOURCES = [REPO / "notebooks" / f"{nb}.py" for nb in NOTEBOOKS] + sorted((REPO / "scripts").glob("*.py")) + sorted(
    (REPO / "lab22").glob("*.py")
)


def test_sources_exist_and_parse():
    for p in SOURCES:
        assert p.exists(), f"missing {p}"
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


def test_no_removed_trainer_arguments():
    # tokenizer= (TRL >= 0.13), warmup_ratio (transformers 5), max_prompt_length (TRL 1.x DPOConfig).
    banned = re.compile(r"\btokenizer\s*=\s*tokenizer\b|\bwarmup_ratio\s*=|\bmax_prompt_length\s*=")
    offenders = [str(p.relative_to(REPO)) for p in SOURCES if banned.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"removed trainer arguments in {offenders}"


def test_no_hardcoded_judge_model():
    for p in SOURCES:
        if p.name == "test_smoke.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert "gpt-4o-mini" not in text and "claude-haiku" not in text, f"hard-coded judge id in {p}"


def test_colab_bundles_are_valid_and_current():
    from build_colab import render

    for tier, path in (("T4", "Lab22_DPO_T4.ipynb"), ("BIGGPU", "Lab22_DPO_BigGPU.ipynb"), ("T4", "Lab22_DPO_T4_Core.ipynb")):
        on_disk = json.loads((REPO / "colab" / path).read_text(encoding="utf-8"))
        assert on_disk == render(tier, core_only=path.endswith("_Core.ipynb")), f"colab/{path} is stale: run `make colab`"


def test_core_evidence_backup(tmp_path, monkeypatch):
    """Backups retain evidence and hashes, omit weights, and refresh safely."""
    import hashlib
    import sys
    import types
    from zipfile import ZipFile
    from build_colab import render

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace()
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    names = ["data/eval/judge_summary.json", "adapters/dpo/adapter_config.json",
             "models/sft-merged/config.json", "submission/screenshots/example.png"]
    for name in names + ["adapters/dpo/adapter_model.safetensors"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"test evidence")
    nb = render("T4", core_only=True)
    sources = ["".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "def export_evidence(stage):" in "".join(c["source"])]
    assert len(sources) == 1
    scope = {"WORK": tmp_path}
    exec(compile(sources[0], "evidence-cell", "exec"), scope)
    scope["export_evidence"]("NB3")
    (tmp_path / names[0]).write_bytes(b"updated")
    scope["export_evidence"]("NB3")
    with ZipFile(scope["archive"]) as archive:
        assert set(archive.namelist()) == set(names + ["submission/run_manifest.json"])
        manifest = json.loads(archive.read("submission/run_manifest.json"))
        assert manifest["completed_stages"] == ["NB3"]
        assert manifest["weights_included"] is False
        for name in names:
            assert manifest["sha256"][name] == hashlib.sha256(archive.read(name)).hexdigest()
        assert archive.read(names[0]) == b"updated"


def test_kaggle_bundle_is_portable_and_current():
    from build_kaggle import TARGET, render_kaggle

    notebook = render_kaggle()
    assert json.loads(TARGET.read_text(encoding="utf-8")) == notebook
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    assert not any("from google.colab" in s or "files.download(" in s or "/content/lab22" in s for s in sources)
    assert any('os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in s for s in sources)
    assert sum("def export_evidence(stage):" in s for s in sources) == 1
    assert sum("export_evidence('0" in s for s in sources) == 6
    assert any("scripts/verify.py')], check=True" in s for s in sources)
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            if source.startswith("%%writefile"):
                source = source.split("\n", 1)[1]
            ast.parse(source)


def test_reflection_refuses_missing_results(tmp_path):
    import pytest
    from generate_reflection import generate

    with pytest.raises(FileNotFoundError):
        generate(tmp_path)
    assert not (tmp_path / "submission/REFLECTION.md").exists()


def test_reflection_uses_matching_results_and_keeps_uncertainty(tmp_path):
    import hashlib
    import pandas as pd
    import pytest
    from generate_reflection import generate
    from lab22.judge import summarize

    def save(name, payload):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    save("adapters/sft-mini/sft_metrics.json", {
        "dataset": "test-sft", "samples": 1000, "epochs": 1,
        "training_loss": 1.5, "history": [{"loss": 2.0}, {"loss": 1.0}],
    })
    save("adapters/dpo/dpo_metrics.json", {
        "compute_tier": "T4", "base_model": "test-model", "beta": 0.1, "lr": 5e-6, "epochs": 1,
        "end_reward_gap": 2.0, "eval_chosen_reward": -1.0, "eval_rejected_reward": -3.0,
        "eval_reward_gap": 2.0, "eval_reward_accuracy": 0.75,
        "diagnosis": "LIKELIHOOD DISPLACEMENT", "training_seconds": 123, "peak_allocated_gb": 10.0,
        "history": [{"step": i, "rewards/chosen": -i, "rewards/rejected": -3 * i} for i in (1, 2)] +
                   [{"step": i, "eval_rewards/chosen": -i / 2, "eval_rewards/rejected": -3 * i / 2} for i in (1, 2)],
    })
    save("data/pref/stats.json", {"dataset": "test-pref", "language": "Vietnamese", "chosen_median": 20,
                                 "rejected_median": 10, "chosen_longer_frac": 0.75})
    for split in ("train", "eval"):
        rows = [{"prompt": [{"role": "user", "content": f"{split} question {i}"}],
                 "chosen": [{"role": "assistant", "content": "chosen"}],
                 "rejected": [{"role": "assistant", "content": "rejected"}]} for i in range(3)]
        pd.DataFrame(rows).to_parquet(tmp_path / f"data/pref/{split}.parquet")
    rows = [{"id": f"e{i}", "category": "heldout", "prompt": "held question", "sft": "sft answer",
             "dpo": "dpo answer", "winner": "tie"} for i in range(50)]
    rows += [{"id": f"{prefix}{i}", "category": category, "prompt": "fixed question", "sft": "sft answer",
              "dpo": "dpo answer", "winner": "tie"}
             for prefix, category in (("h", "helpfulness"), ("s", "safety")) for i in range(1, 5)]
    output = tmp_path / "data/eval/side_by_side.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    fingerprint = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {"judge": "rm-panel:test", "sanity_accuracy": 1.0, "outputs_sha256": fingerprint,
               **{category: summarize([r for r in rows if r["category"] == category])
                  for category in ("heldout", "helpfulness", "safety")}}
    save("data/eval/judge_summary.json", summary)
    save("data/eval/judge_results_rm.json", {"outputs_sha256": fingerprint, "records": rows})
    report = generate(tmp_path).read_text(encoding="utf-8")
    assert "LIKELIHOOD DISPLACEMENT" in report
    assert "chưa đủ bằng chứng DPO tốt hơn SFT" in report
    assert "50.00%" in report and "-1.0000 / -3.0000" in report
    assert "### Ví dụ helpfulness: h3" in report and "### Ví dụ safety: s2" in report
    summary["outputs_sha256"] = "stale"
    save("data/eval/judge_summary.json", summary)
    with pytest.raises(ValueError, match="does not match"):
        generate(tmp_path)
