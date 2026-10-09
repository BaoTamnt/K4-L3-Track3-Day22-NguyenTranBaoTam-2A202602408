# Chạy lab trên Colab T4

1. Mở https://colab.research.google.com, chọn Upload notebook và tải `colab/Lab22_DPO_T4_Core.ipynb`.
2. Chọn Runtime → Change runtime type → T4 GPU.
3. Chọn Runtime → Run all. Bản Core chỉ chạy NB0, NB1, NB2, NB3, NB4; không chạy bonus. Dự kiến 1,5–2 giờ theo README.
4. Cell cuối tải `lab22-results.zip`. Tải thêm notebook qua File → Download → Download .ipynb để giữ output.
5. Giải nén ZIP vào thư mục lab, lưu notebook có output trong `colab/`, rồi dùng số liệu thật để hoàn thiện `submission/REFLECTION.md`.
6. Chạy `python -X utf8 scripts/verify.py`. Lưu ý verifier hiện kiểm tra đường dẫn reference trên máy đang chạy và file model cục bộ; artifact từ Colab có reference `/content/lab22/models/sft-merged`. Không sửa đường dẫn trong adapter để giả vờ đã huấn luyện trên Windows. Kết quả Colab cần được đối chiếu trong môi trường đã chạy.

## Sao lưu và xử lý lỗi

Notebook Core cập nhật `/content/lab22/lab22-results.zip` sau mỗi NB hoàn tất. Nếu NB3 hoặc NB4 lỗi, tạo cell mới và chạy `files.download(str(archive))` để tải bằng chứng các chặng trước. ZIP có ảnh, dữ liệu, JSON của adapter, cấu hình SFT đã gộp và `submission/run_manifest.json` ghi cấu hình, GPU, chặng hoàn tất cùng SHA-256 của từng file. ZIP không chứa trọng số nên không dùng để khôi phục huấn luyện sau khi Colab xoá phiên.

Nếu gặp OOM, thêm `os.environ["MAX_LEN"] = "512"` vào cell cài đặt đầu tiên và chạy lại từ đầu. Không giảm số câu held-out chấm xuống dưới 50 khi làm bài chính thức. Nếu Colab báo không có GPU, kiểm tra lại loại runtime và hạn mức GPU của tài khoản.

## Trạng thái hiện tại

NB0 đã chạy thành công trên CPU, output lưu tại `notebooks/00_dpo_loss_from_scratch.ipynb`. Đã cài hàm `my_dpo_loss` và giải thích likelihood displacement trong NB0. NB2 đã có 800 cặp train, 100 cặp held-out, `stats.json` và ảnh `02b-pref-length.png`; chosen dài hơn trong 65,875% cặp. Máy hiện có RTX 4050 Laptop 6 GB; đã chọn Colab T4 cho phần huấn luyện theo HARDWARE-GUIDE. Notebook Core có sao lưu từng chặng và tải kết quả. Chưa có kết quả SFT/DPO/judge; chưa sẵn sàng nộp bài.

Để tiếp tục hoàn thiện phản tư, cần kết quả ZIP và notebook đã chạy trên Colab. Không sử dụng số liệu ví dụ làm kết quả thực nghiệm.
