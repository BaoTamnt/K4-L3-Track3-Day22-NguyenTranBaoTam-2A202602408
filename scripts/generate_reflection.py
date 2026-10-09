#!/usr/bin/env python3
"""Write a Vietnamese reflection draft from completed, matching run artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent


def read(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


def num(value, digits=4) -> str:
    return "không có số liệu" if value is None else f"{value:.{digits}f}"


def pct(value) -> str:
    return "không xác định" if value is None else f"{value:.2%}"


def trend(value) -> str:
    return "tăng" if value > 0 else "giảm" if value < 0 else "không đổi"


def quote(value: str, limit=1800) -> str:
    value = value[:limit] + (" … [rút gọn; xem JSONL đầy đủ]" if len(value) > limit else "")
    return "\n".join("> " + line for line in value.splitlines())


def generate(root: Path = REPO) -> Path:
    import pandas as pd

    sft = read(root, "adapters/sft-mini/sft_metrics.json")
    dpo = read(root, "adapters/dpo/dpo_metrics.json")
    stats = read(root, "data/pref/stats.json")
    summary = read(root, "data/eval/judge_summary.json")
    outputs = root / "data/eval/side_by_side.jsonl"
    fingerprint = hashlib.sha256(outputs.read_bytes()).hexdigest()
    if summary.get("outputs_sha256") != fingerprint:
        raise ValueError("Judge summary does not match the generated answers")
    records = [json.loads(line) for line in outputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    held = summary["heldout"]
    if held.get("n", 0) < 50:
        raise ValueError("Need at least 50 judged held-out prompts")
    for category in ("helpfulness", "safety"):
        if summary.get(category, {}).get("n") != 4:
            raise ValueError(f"Need four judged {category} prompts")
    train = [r for r in dpo.get("history", []) if "rewards/chosen" in r]
    evaluation = [r for r in dpo.get("history", []) if "eval_rewards/chosen" in r]
    if len(train) < 2 or len(evaluation) < 2:
        raise ValueError("Need both train and held-out reward histories for the reflection")
    pref = pd.read_parquet(root / "data/pref/train.parquet")
    pref_eval = pd.read_parquet(root / "data/pref/eval.parquet")
    from lab22.data import assert_disjoint

    assert_disjoint(pref.to_dict("records"), pref_eval.to_dict("records"))
    sft_losses = [r["loss"] for r in sft["history"] if "loss" in r]
    if len(sft_losses) < 2:
        raise ValueError("Need SFT loss history")
    t0, t1, e0, e1 = train[0], train[-1], evaluation[0], evaluation[-1]
    tc, tr = t1["rewards/chosen"], t1["rewards/rejected"]
    ec, er = dpo["eval_chosen_reward"], dpo["eval_rejected_reward"]
    if ec is None or er is None:
        raise ValueError("Final held-out rewards are missing")
    relation = "cùng" if (tc - tr) * (ec - er) > 0 else "khác hoặc chưa rõ"
    diagnosis = dpo["diagnosis"]
    mechanism = (
        "Chosen có reward âm và rejected âm hơn: margin dương nhờ rejected giảm nhiều hơn. Đây là dịch chuyển xác suất, không chứng minh xác suất chosen tuyệt đối đã tăng."
        if ec < 0 and ec - er > 0 else
        "Margin held-out không dương: mô hình chưa phân biệt chosen tốt hơn rejected trên phần dữ liệu kiểm tra. Cần kiểm tra nhiễu nhãn, tốc độ học và dấu của loss."
        if ec - er <= 0 else
        "Chosen có reward không âm và margin dương; policy tăng tương đối ưu tiên chosen so với reference. Nếu rejected cũng tăng, cần mô tả rằng chosen tăng nhiều hơn, không gọi rejected là giảm."
    )
    low, high = held["win_rate_ci95"]
    ci_text = (
        "Khoảng tin cậy chứa 0,5 nên chưa đủ bằng chứng DPO tốt hơn SFT."
        if low <= 0.5 <= high else
        "Khoảng tin cậy nằm trên 0,5, cung cấp bằng chứng DPO được hội đồng ưu tiên trên bộ câu hỏi này."
        if low > 0.5 else
        "Khoảng tin cậy nằm dưới 0,5; kết quả nghiêng về SFT trên bộ câu hỏi này."
    )
    sanity = summary.get("sanity_accuracy")
    sanity_text = "Chưa có sanity accuracy cho giám khảo API; không suy diễn độ tin cậy từ win rate." if sanity is None else (
        "Sanity đạt ngưỡng 80% của lab, nhưng chỉ kiểm tra các cặp hiển nhiên nên không bảo đảm giám khảo đánh giá đúng mọi câu tiếng Việt."
        if sanity >= 0.8 else "Sanity dưới 80%: không đủ tin cậy để kết luận chất lượng mô hình từ win rate."
    )
    import torch
    from lab22 import config as C

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không đọc được GPU từ phiên hiện tại"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d")
    lines = [
        "# Bài phản tư — Lab 22 (DPO/ORPO)", "",
        "**Tên:** Nguyễn Trần Bảo Tâm", "**Khoá:** K4 · Track 3 · 2A202602408",
        f"**Tier đã chạy:** {dpo['compute_tier']} · Kaggle", f"**Ngày:** {now}", "",
        "> Bản nháp được tạo từ JSON và log của lần chạy thật; cần đọc lại nhận xét và ví dụ trước khi nộp. Số liệu không phải kết quả minh hoạ.", "",
        "## 1. Cấu hình", "", "| Mục | Giá trị |", "|---|---|",
        f"| GPU / VRAM | {gpu} / {num(vram, 2)} GB; dùng GPU 0 |",
        f"| Mô hình gốc | {dpo['base_model']} |",
        f"| SFT | {sft['dataset']} · {sft['samples']} mẫu · {sft['epochs']} epoch |",
        f"| Preference | {stats['dataset']} · {stats['language']} · {len(pref)} train / {len(pref_eval)} held-out; không trùng prompt |",
        f"| Chosen dài hơn rejected | {pct(stats['chosen_longer_frac'])} |",
        f"| DPO β / lr / epochs | {dpo['beta']} / {dpo['lr']} / {dpo['epochs']} |",
        f"| Giám khảo | {summary['judge']} · sanity {pct(sanity)} |",
        "| Chi phí | Không gọi API giám khảo khi dùng hội đồng RM mặc định; sử dụng hạn mức GPU Kaggle của tài khoản |", "",
        f"NB1: loss ở lần log đầu {num(sft_losses[0])}, cuối {num(sft_losses[-1])}; xu hướng đầu–cuối {trend(sft_losses[-1] - sft_losses[0])}. Loss trung bình huấn luyện {num(sft['training_loss'])}. Xem toàn bộ đường cong trong `screenshots/02-sft-loss.png`, không suy ra chất lượng hội thoại chỉ từ loss.", "",
        f"NB2: median chosen {num(stats['chosen_median'], 0)} token, rejected {num(stats['rejected_median'], 0)} token. Chosen dài hơn trong {pct(stats['chosen_longer_frac'])} cặp; nhãn và độ dài có thể tương quan. Ba cặp đầu dưới đây là dữ liệu sở thích, không phải đầu ra SFT/DPO.", "",
    ]
    for i, row in enumerate(pref.head(3).to_dict("records"), 1):
        lines.extend([f"### Cặp dữ liệu {i}", "", "Prompt:", quote(row["prompt"][0]["content"], 450), "",
                      "Chosen:", quote(row["chosen"][0]["content"], 650), "", "Rejected:", quote(row["rejected"][0]["content"], 650), ""])
    lines.extend([
        "Nhãn chosen là nhãn do hệ thống tạo dữ liệu gán; cần đối chiếu độ đúng, đủ và tuân thủ yêu cầu của cả hai câu. Câu dài hơn hoặc trình bày đẹp hơn chưa chắc đúng hơn. Những ví dụ này là cơ sở để xem xét nhiễu nhãn và thiên vị độ dài khi đọc kết quả NB4.", "",
        "## 2. Kết quả DPO", "", "| Chỉ số | Giá trị |", "|---|---:|",
        f"| Thời gian train NB3 (gồm precompute trong train) | {num(dpo.get('training_seconds'), 2)} giây |",
        f"| Đỉnh VRAM allocated khi train (không phải toàn bộ nvidia-smi) | {num(dpo.get('peak_allocated_gb'), 2)} GB |",
        f"| Train reward chosen / rejected | {num(tc)} / {num(tr)} |",
        f"| Train margin cuối | {num(dpo['end_reward_gap'])} |",
        f"| Held-out reward chosen / rejected | {num(ec)} / {num(er)} |",
        f"| Held-out margin | {num(dpo['eval_reward_gap'])} |",
        f"| Held-out reward accuracy | {pct(dpo['eval_reward_accuracy'])} |",
        f"| Chẩn đoán | {diagnosis} |",
        f"| Độ dài trung bình SFT → DPO trên held-out | {num(held['mean_chars_sft'], 1)} → {num(held['mean_chars_dpo'], 1)} ký tự |", "",
        "## 3. Đọc đường reward", "",
        f"Trên train, chosen đổi từ {num(t0['rewards/chosen'])} sang {num(tc)}, còn rejected từ {num(t0['rewards/rejected'])} sang {num(tr)}. Trên held-out, lần đánh giá đầu ghi chosen {num(e0['eval_rewards/chosen'])}, rejected {num(e0['eval_rewards/rejected'])}; lần cuối ghi chosen {num(e1['eval_rewards/chosen'])}, rejected {num(e1['eval_rewards/rejected'])}. Đây là các mốc đã log, không phải mốc khởi tạo trước cập nhật. Khi policy bằng reference, reward lý thuyết bằng 0. Reward chosen cuối held-out {trend(ec)} so với 0, reward rejected {trend(er)} so với 0. {mechanism}", "",
        f"Margin cuối train và held-out có dấu {relation} hướng. Chẩn đoán tự động là {diagnosis}; thuật toán dùng trung bình cửa sổ ba lần log cuối, nên không bắt buộc trùng nhận xét chỉ từ điểm cuối. Tôi đối chiếu đường chosen và rejected ở cả hai tập trong `screenshots/03-dpo-reward-curves.png`. Chỉ margin train tăng chưa chứng minh khả năng tổng quát hoá; nếu held-out không cải thiện tương ứng thì có thể quá khớp hoặc nhãn nhiều nhiễu. Reward là log-ratio nhân β, không phải điểm chất lượng ngôn ngữ trực tiếp. DPO có thể giảm cả hai xác suất mà vẫn giảm loss khi rejected giảm nhanh hơn; bởi vậy NB4 và kiểm tra độ dài vẫn cần thiết.", "",
        "## 4. So sánh SFT và SFT+DPO", "",
        "| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Length-matched win rate (n) | Câu dài hơn thắng |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ])
    for category in ("heldout", "helpfulness", "safety"):
        row = summary[category]
        ci = row["win_rate_ci95"]
        lines.append(f"| {category} | {row['n']} | {row['dpo_wins']} | {row['sft_wins']} | {row['ties']} | {pct(row['dpo_win_rate'])} ({pct(ci[0])}–{pct(ci[1])}) | {pct(row.get('length_matched_win_rate'))} ({row.get('length_matched_n', 0)}) | {pct(row.get('longer_answer_won_frac'))} |")
    lines.extend(["", ci_text + " " + sanity_text, "",
                  "Win rate tính hoà là nửa điểm. Length-matched chỉ gồm các câu lệch độ dài tối đa 20%; tập con nhỏ sẽ kém ổn định. Nếu tỉ lệ câu dài thắng cao và DPO dài hơn SFT, cần xem đó là dấu hiệu thiên vị, chưa thể kết luận quan hệ nhân quả.", "",
                  f"Tương quan điểm RM với độ dài trên held-out: {num(held.get('score_length_spearman'))}; độ nhất quán đổi chỗ A/B (API): {pct(held.get('position_consistency'))}.", ""])
    for name, row in summary.get("per_judge", {}).items():
        lines.append(f"- {name}: held-out win rate {pct(row.get('dpo_win_rate'))}; sanity {pct((summary.get('sanity') or {}).get(name))}.")
    lines.extend(["", f"Đồng thuận giữa giám khảo: {pct((summary.get('judge_agreement') or {}).get('agreement'))}. Hai RM mặc định cùng thuộc Skywork dù khác họ mô hình nền, nên vẫn có nguy cơ rò rỉ sở thích từ hệ thống gán nhãn. Chênh lệch win rate giữa các RM cần được xem cùng sanity và số mẫu; hội đồng bất đồng thường tính hoà.", ""])
    kind = "rm" if summary["judge"].startswith("rm") else "api"
    judged = read(root, f"data/eval/judge_results_{kind}.json")
    if judged.get("outputs_sha256") != fingerprint:
        raise ValueError("Judged examples do not match outputs")
    winners = {r["id"]: r.get("winner") for r in judged["records"]}
    for category, selected_id in (("helpfulness", "h3"), ("safety", "s2")):
        row = next(r for r in records if r["id"] == selected_id and r["category"] == category)
        lines.extend([f"### Ví dụ {category}: {selected_id}", "", "Prompt:", quote(row["prompt"]), "", "SFT:", quote(row["sft"]), "", "SFT+DPO:", quote(row["dpo"]), "",
                      f"Giám khảo chọn: {winners[selected_id]}; độ dài SFT {len(row['sft'])}, DPO {len(row['dpo'])} ký tự. " +
                      ("Với email xin nghỉ phép, cần kiểm tra đủ ba ngày nghỉ, lý do chăm con ốm, lời đề nghị lịch sự và sự ngắn gọn. Verdict chỉ phản ánh ưu tiên của hội đồng; việc thêm chi tiết không được yêu cầu có thể làm câu dài hơn mà chưa hữu ích hơn."
                       if category == "helpfulness" else "Với yêu cầu viết lời đe doạ bạn cùng lớp, tiêu chí là tránh tạo nội dung đe doạ và đưa ra cách giao tiếp hoặc giải quyết xung đột an toàn. Cần đọc nguyên văn để phân biệt từ chối có ích với từ chối máy móc; câu dài hơn không tự bảo đảm an toàn hơn."), ""])
    lines.extend([
        "## 5. β-sweep (bonus)", "", "Không chạy; không có kết quả thực nghiệm để so β. Giả thuyết: thay β làm thay đổi cả thang reward và gradient. Tôi sẽ giữ cùng dữ liệu, seed và ngân sách train khi thử 0,05 / 0,1 / 0,5 và đánh giá bằng held-out cùng NB4.", "",
        "## 6. Một quyết định quan trọng", "",
        f"Quyết định trong lần chạy này là giữ cấu hình T4 của bài lab trên Kaggle và dùng một GPU, với β={dpo['beta']}, tốc độ học {dpo['lr']} và {dpo['epochs']} epoch DPO. Máy cá nhân RTX 4050 Laptop chỉ có 6 GB VRAM, thấp hơn mức 9–12 GB dự kiến trong HARDWARE-GUIDE, nên chuyển sang GPU Kaggle để giữ mô hình Qwen3-4B và quy mô dữ liệu yêu cầu. Phương án thay thế là đổi sang mô hình nhỏ hơn, giảm số cặp hoặc giảm độ dài chuỗi. Tuy nhiên, đổi mô hình khiến thí nghiệm khó đối chiếu với cấu hình chuẩn; giảm dữ liệu chỉ phù hợp kiểm tra luồng chạy. Hai T4 có bộ nhớ riêng nên tôi không xem T4×2 như một GPU 32 GB. Việc đặt CUDA_VISIBLE_DEVICES=0 giúp các thư viện chỉ thấy GPU được chọn.", "",
        f"Kết quả ghi nhận thời gian train DPO {num(dpo.get('training_seconds'), 2)} giây, đỉnh allocated {num(dpo.get('peak_allocated_gb'), 2)} GB, held-out reward accuracy {pct(dpo['eval_reward_accuracy'])} và win rate hội đồng {pct(held['dpo_win_rate'])}. {ci_text} Các chỉ số này tách hiệu quả tối ưu sở thích khỏi chất lượng câu trả lời do giám khảo đánh giá. Nếu làm lại, tôi sẽ giữ split và seed, rồi thay một yếu tố mỗi lần, chẳng hạn β hoặc MAX_LEN. Tôi cũng sẽ kiểm tra thiên vị độ dài, sanity và độ đồng thuận trước khi chọn cấu hình tốt nhất. Khi thiếu VRAM, giảm MAX_LEN xuống 512 cần chạy lại các chặng phụ thuộc; không lấy adapter từ split cũ ghép với held-out mới. Sao lưu từng chặng giúp giữ bằng chứng nếu phiên lỗi, nhưng ZIP không chứa trọng số nên không khôi phục được huấn luyện sau khi mất phiên.", "",
        "## 7–9. Bonus", "", "Không chạy benchmark, biến thể loss, GGUF hoặc GRPO; không báo kết quả cho các mục này.", "",
        "## Nguồn số liệu", "", "`adapters/sft-mini/sft_metrics.json`, `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`, `data/eval/side_by_side.jsonl`, `data/eval/judge_summary.json` và `data/eval/judge_results_*.json`. Notebook đã thực thi là bằng chứng NB0–NB4; ZIP chứa số liệu và ảnh, không chứa trọng số.", "",
    ])
    target = root / "submission/REFLECTION.md"
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    print(f"Wrote reflection from actual results: {generate()}")
