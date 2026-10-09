# Gói bằng chứng Lab 22

File ZIP: `Lab22-NguyenTranBaoTam-2A202602408.zip` trong thư mục này.

Gói gồm mã nguồn và hướng dẫn lab, notebook Kaggle có output NB0–NB4, bản notebook đã rà soát, phản tư, bốn biểu đồ, dữ liệu preference hiện có, số liệu DPO và tổng hợp judge khôi phục từ output thật. Không chứa trọng số mô hình, môi trường ảo hay khoá API.

Notebook gốc: `kaggle/evidence/notebooke3ecbcfb2f-completed-original.ipynb`. Output xác nhận verifier đã qua **trên Kaggle**. Bản rà soát `kaggle/Lab22_DPO_T4_Kaggle_Executed.ipynb` sửa ký tự thừa ở cell xuất ZIP cuối cùng; output và execution count của cell đã sửa được xoá để không trình bày như một cell đã chạy lại.

Kết quả chính: held-out reward accuracy 69%; win rate DPO 49%, CI 95% 41–57%. Chưa đủ bằng chứng DPO tốt hơn SFT. Giám khảo Qwen không qua sanity và đã bị loại khỏi kết quả chính.

## Giới hạn của gói hiện tại

ZIP này được đóng gói tại máy từ bằng chứng đã có, không phải ZIP tải nguyên vẹn từ Kaggle. Các file sau chưa khôi phục được từ output notebook:

- `data/eval/side_by_side.jsonl`: toàn bộ 58 cặp câu trả lời.
- `data/eval/judge_results_rm.json`: verdict và score từng câu.
- `adapters/dpo/adapter_config.json`, `adapters/dpo/split.json`.
- `adapters/sft-mini/adapter_config.json`, `adapters/sft-mini/sft_metrics.json`.
- `models/sft-merged/config.json` và manifest gốc của Kaggle.

Không dựng lại các file này từ đoạn trích hoặc điền số liệu giả. Verifier cục bộ hiện chưa qua vì thiếu artifact. Để đủ bằng chứng theo README/rubric, cần tải ZIP kết quả thật từ Kaggle rồi bổ sung. `PACKAGE-MANIFEST.json` bên trong ZIP liệt kê SHA-256 và trạng thái này.

## Tạo lại ZIP

```text
python scripts/package_submission.py
```

Nộp bằng link repo: ZIP và các file bằng chứng cần được commit và push lên repo trước khi gửi link. Chỉ tạo ZIP trong thư mục làm việc chưa làm file xuất hiện trên GitHub.
