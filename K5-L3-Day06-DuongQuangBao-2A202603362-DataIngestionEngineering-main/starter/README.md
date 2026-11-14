# Starter · Gemini + PostgreSQL/pgvector

Starter hiện thực một baseline cho lab: corpus công khai, pipeline ingest/chunk/embed/vector search, ba cấu hình A/B/C, benchmark dev/test và report. Kết quả grounding, metadata correctness và coherence vẫn cần người chấm; chương trình không tự bịa nhãn hay câu trả lời LLM.

## Corpus và quyền sử dụng

Mẫu dùng *Alice's Adventures in Wonderland*, Project Gutenberg eBook #11. Tác phẩm gốc thuộc public domain ở Hoa Kỳ; nếu ở ngoài Hoa Kỳ, tự xác minh luật địa phương trước khi tái phân phối. Nguồn, giấy phép, thời điểm truy cập và quy tắc chia thành source unit nằm ở `../data/manifest.json`; corpus tại `../data/alice-wonderland.txt`; bộ câu hỏi/evidence tại `../benchmark/questions.json`.

Corpus gồm các đơn vị nội dung theo chương, mỗi đơn vị tối thiểu 1.500 ký tự đã chuẩn hóa. Nếu file chưa có, runner tải từ URL trong manifest. Có thể tải trước:

```bash
cd starter
python -m scripts.fetch_corpus
```

## Setup

Cần Python **3.11–3.14**, Docker Compose và mạng để tải corpus/dependencies lần đầu.

```bash
cd starter
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env  # chỉ khi chưa có starter/.env; không ghi đè file đang có key
```

Trong `.env`, điền `GEMINI_API_KEY`. Cấu hình mẫu dùng `EMBEDDING_MODEL=gemini-embedding-2` và `EMBEDDING_DIMENSIONS=768`. Runner mặc định giới hạn tốc độ ở 20.000 token/phút ước tính để giảm lỗi quota; điều chỉnh `GEMINI_EMBEDDING_TPM_LIMIT` theo quota thực tế của project nếu cần. Không in hoặc commit API key. Model embedding khác hoặc số chiều khác phải dùng nhất quán cho document/query và bảng pgvector.

Khởi động và kiểm tra database:

```bash
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

## Chạy lab

Chuẩn bị corpus, benchmark, chunk statistics và report mà **không gọi Gemini hoặc database**:

```bash
python -m src.run --prepare-only
```

Chạy đủ ingest/retrieval (gọi Gemini API và dùng database local):

```bash
python -m src.run
```

Tùy chọn: tạo **gợi ý** coherence/grounding bằng Gemini (dùng quota/API riêng; không điền điểm cuối vào audit):

```bash
python -m src.judge_suggestions --model gemini-2.5-flash
```

Gợi ý được lưu riêng ở `../results/coherence_ai_suggestions.csv` và `../results/grounding_ai_suggestions.csv`; hãy kiểm tra từng gợi ý với nguồn/trace, rồi xác nhận hoặc sửa nhãn audit chính thức.

Pipeline dùng cùng corpus/model/câu hỏi cho cả ba chiến lược:

- **A:** fixed-size 1.500 ký tự, overlap 100.
- **B:** chunk theo đoạn, tối đa 1.500 ký tự.
- **C:** cùng chunk/embedding với B, chỉ thêm filter `section_id` nếu câu hỏi nêu chương.

Cấu hình chọn trước khi xem test: tối đa Evidence Hit@3 trên dev, tie-break bằng Hit@1; test chỉ để báo cáo. Câu hỏi có `filter` được áp dụng chỉ cho C; top-3 của A/B được đo xem có tự thỏa filter hay không.

## Output

`../results/` chứa benchmark đã xác minh theo source unit, inventory ranh giới/độ dài, chunk stats, Hit@1/Evidence Hit@3, top-3 trace (`top3_trace.jsonl`), run metadata và audit. `metadata_precheck.csv` đối chiếu metadata với source unit; `grounding_precheck.csv` gom exact-evidence hits theo từng câu/strategy nhưng không tự gán điểm rubric. Runner sinh `manual_review.md` để hướng dẫn xác nhận nhãn. Báo cáo được tạo từ `../REPORT_TEMPLATE.md` tại `../report/individual/2A202603362_DuongQuangBao.md`. `coherence_audit.csv` ghi 10 source-unit đầu cho cả ba chiến lược; `metadata_grounding_audit.csv` chuẩn bị top-3 của 5 câu test để audit metadata. Nếu dùng Gemini judge, giữ gợi ý tách riêng, ghi rõ phương pháp trong báo cáo và tự xem xét từng nhãn theo nguồn/trace trước khi gọi đó là nhãn thủ công. Precheck không thay đánh giá usefulness; không suy ra nhãn chỉ từ cosine score.

Gemini có thể tính phí hoặc giới hạn quota. Runner ghi số request và token nếu API trả về; chi phí chính xác phải kiểm tra tại Google AI Studio. `--prepare-only` không dùng key và không gọi dịch vụ.
