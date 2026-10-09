# Lab 22 trên Kaggle T4

Notebook: `Lab22_DPO_T4_Kaggle.ipynb`. Được sinh từ mã nguồn bằng:

```text
python scripts/build_kaggle.py
```

1. Mở notebook Kaggle của bạn, chọn **File → Import Notebook**, tải file trên lên.
2. Trong **Settings**, chọn **GPU T4×2**, bật **Internet**. Notebook dùng GPU 0.
3. Chọn **Run All**. Bản này chạy NB0–NB4, không chạy bonus. Ước tính trong README là 1,5–2 giờ trên T4; thời gian Kaggle thực tế có thể khác.
4. Khi chạy xong, phần cuối tạo phản tư từ JSON/log thật và chạy verifier. Chỉ khi thấy `Core checks passed` mới xem kiểm tra trong môi trường huấn luyện là thành công.
5. Dùng **Save Version → Save & Run All** nếu muốn có phiên bản thực thi và output được lưu. Sau khi phiên bản hoàn tất, tải notebook có output và `lab22/lab22-results.zip` trong **Output**. Việc Save & Run All sẽ chạy lại từ đầu; đừng bắt đầu thêm một phiên khi phiên trước còn chạy.
6. Giải nén ZIP vào repo, lưu notebook đã chạy trong `kaggle/`, đọc lại phản tư và các ví dụ trước khi nộp.

ZIP được cập nhật sau từng NB hoàn tất, chứa ảnh, dữ liệu, JSON adapter, cấu hình mô hình SFT, phản tư (khi NB4 hoàn tất) và manifest SHA-256. Không chứa trọng số mô hình hay khoá API. ZIP của một lần chạy dang dở không được coi là bài hoàn tất.

Nếu hết bộ nhớ, thêm `os.environ["MAX_LEN"] = "512"` vào cell cấu hình đầu tiên rồi chạy lại từ đầu. Không giảm `JUDGE_PROMPTS` dưới 50 cho bài chính thức. Nếu mất phiên GPU, ZIP không đủ để khôi phục mô hình; cần chạy lại NB1.

Verifier gốc so đường dẫn reference tại nơi huấn luyện. Do đó notebook chạy verifier ngay trên Kaggle và lưu `submission/verification.txt`; chạy lại verifier gốc trên Windows sau khi chỉ tải ZIP bằng chứng có thể báo khác reference hoặc thiếu trọng số. Không sửa `adapter_config.json` để giả đường dẫn Windows.

Trạng thái hiện tại: đã chuẩn bị notebook và kiểm tra cấu trúc tại máy; chưa có lần chạy GPU Kaggle hoàn tất. Không có số liệu SFT/DPO/judge thực nghiệm mới.
