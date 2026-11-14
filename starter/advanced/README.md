# Bonus: báo cáo Ngân hàng Thế giới

## Nguồn và lập chỉ mục

Nguồn là *World Development Report 2024: The Middle-Income Trap*, World Bank (2024), DOI `10.1596/978-1-4648-2078-6`. Bản PDF được truy cập qua [trang báo cáo của World Bank](https://www.worldbank.org/en/publication/wdr2024) và link tải chính thức `https://bit.ly/WDR2024_FullReport_ENG`. Trang bản quyền trong PDF ghi giấy phép CC BY 3.0 IGO. Giữ phần attribution trong `data/manifest.json` khi trích dẫn.

Tạo corpus bằng PyMuPDF:

```bash
python3 starter/advanced/index_wdr.py
```

Nếu đã tải PDF, có thể truyền đường dẫn cụ thể:

```bash
python3 starter/advanced/index_wdr.py --pdf /path/to/wdr2024.pdf
```

Lệnh tạo `data/chunks.json` và `data/manifest.json`. Manifest lưu SHA-256, số trang, cách trích xuất, giới hạn và danh sách trang. Corpus hiện có 339 chunk cho 276 trang: 263 trang có text trích xuất và 23 chú giải trực quan. Chú giải được lập thành chunk riêng để giảm nhiễu khi retrieval tìm giá trị trong bảng, biểu đồ và sơ đồ. Vị trí chú giải giữ cả số trang PDF, số trang in và vùng hình.

PyMuPDF trích text native; không chạy OCR. Đã render kiểm tra 13 trang không có text trích xuất; đó là các trang trắng và không được index. Raster pages được kiểm kê trong manifest. Các hình có thông tin ở trang PDF 41, 49, 92, 115, 173, 182 và 186 có chú giải đã đối chiếu trực quan; trang 6 là trang bản quyền có text native và dấu trang trí. `visual_notes.json` cũng ghi các bảng/biểu đồ/sơ đồ vector được chọn để bổ sung evidence trực quan. Những hình không có chú giải riêng vẫn phụ thuộc vào phần text có thể trích xuất; không dùng nội dung chưa kiểm chứng làm gold.

## Benchmark

`questions.json` tách split trước khi chạy agent:

- Dev: 18 câu, gồm 15 có đáp án và 3 thiếu evidence.
- Test: 30 câu, gồm 24 có đáp án và 6 thiếu evidence. Nhóm chính có 6 câu bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop và 4 text.

Kiểm tra toàn bộ gold anchor và số lượng tối thiểu bằng `starter/baseline.py` và `starter/benchmark.py`. Kiểm tra anchor chuẩn hóa ngắt dòng, khoảng trắng không ngắt dòng và soft hyphen do PDF để trích dẫn sạch vẫn đối chiếu được với text gốc. Chú giải trực quan được kiểm tra với PDF gốc; không đưa đáp án hoặc gold vào lời gọi agent.

## Chạy và kết quả hiện có

TF-IDF baseline trên dev với context budget 9.000 ký tự:

- Hit@1: 12/15
- Evidence Hit@3: 15/15
- Evidence trong context: 15/15
- Chi phí dịch vụ retrieval: $0

Trace và summary nằm trong [`results/tfidf-dev-9k/`](results/tfidf-dev-9k/). Có thể tái lập bằng:

```bash
python3 starter/baseline.py \
  --chunks starter/advanced/data/chunks.json \
  --questions starter/advanced/questions.json \
  --split dev --budget 9000 \
  --output starter/advanced/results/tfidf-dev-9k
```

Chạy Gemini agent từ thư mục `starter` để `python-dotenv` nạp `starter/.env`:

```bash
cd starter
python3 run_agent.py \
  --chunks advanced/data/chunks.json \
  --questions advanced/questions.json \
  --split dev --methods lexical \
  --agent-model gemini-2.5-flash-lite \
  --top-k 3 --candidate-k 20 --max-searches 4 --budget 9000 \
  --request-interval 6.2 --agent-cost-per-case-usd 0 \
  --output advanced/results/agent-dev
```

`--request-interval` giúp tránh vượt giới hạn requests-per-minute; `--agent-cost-per-case-usd 0` ghi chi phí $0 cho tài khoản Free Tier vào từng trace và summary. Không đưa `.env` hoặc khóa API vào bài nộp.

**Bonus chưa hoàn tất.** Agent run trên dev chưa đủ mẫu: model `gemini-3.5-flash-lite` gặp lỗi 503 do quá tải; `gemini-2.5-flash-lite` chạy được một số câu nhưng Free Tier trả lỗi 429 do giới hạn 20 requests/ngày. Các thư mục `results/lexical-dev*` lưu những lần chạy dở và không phải kết quả benchmark hoàn chỉnh. Chưa chạy/chấm toàn bộ test, chưa có tỷ lệ đạt cuối cùng và chưa cập nhật checklist bonus. Không kết luận đủ 90% hoặc đủ điều kiện nhận điểm cho tới khi có trace đầy đủ, chấm thủ công từng câu test và báo cáo n/N theo nhóm.
