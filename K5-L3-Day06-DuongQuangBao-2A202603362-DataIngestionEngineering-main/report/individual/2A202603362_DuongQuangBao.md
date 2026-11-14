# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline
- Chủ đề, nguồn/quyền, số trang và quy tắc tính trang: Alice's Adventures in Wonderland — Lewis Carroll; nguồn https://www.gutenberg.org/cache/epub/11/pg11.txt; quyền: Project Gutenberg eBook #11; public domain in the United States. The original work is public domain in many jurisdictions, but users outside the United States must verify local law. Attribution: Project Gutenberg, https://www.gutenberg.org/ebooks/11.; 50 đơn vị nội dung từ ít nhất 1500 ký tự/đơn vị (142,884 ký tự đã chuẩn hóa). Page là logical source unit (không phải trang in): The plain-text source has no physical page numbers; page is a one-based ordinal for each paragraph-aligned logical source unit in book order. Quy tắc chia: Remove Gutenberg wrapper and chapter headings, normalize NFC/whitespace, group paragraphs toward 3000 characters, split oversized paragraphs at word boundaries, merge a short final unit with its predecessor when possible, and skip/log any remaining unit under 1500 characters. Unit offsets are relative to normalized chapter text.
- Chuẩn hóa, lỗi và giới hạn: Project Gutenberg header/footer được bỏ; NFC/whitespace chuẩn hóa; 50 đơn vị hợp lệ, lỗi/rỗng được log.
- Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí: Gemini Embeddings API; `gemini-embedding-2`, 768 chiều. PostgreSQL 16.16 (Debian 16.16-1.pgdg12+1) + pgvector 0.8.7; cosine distance, exact search. Từ `starter/`: `python -m src.run --prepare-only` để chuẩn bị offline; `python -m src.run` để chạy đầy đủ. Số request embedding: 273; token theo metadata API: không trả về. Chi phí cần xác minh trong Google AI Studio. Query embedding chung: 10.22s.
- Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn: Mỗi chunk có `document_id`, `source`, `page`, `section_id`, `section`, `language`; filter `section_id` lấy từ tiêu đề chương trong nguồn.

## Cấu hình và benchmark
- A / B / C (tham số cụ thể, khác biệt duy nhất B–C): A fixed 1500 chars / overlap 100; B paragraph-aware, tối đa 1500 chars; C giống B và chỉ thêm filter theo section khi câu hỏi nêu chương. Thời gian A/B gồm embedding riêng; C là thời gian tăng thêm vì tái dùng embedding của B. Thời gian embedding câu hỏi dùng chung không được cộng vào từng strategy. Chunk counts: A=126, B=127, C=127.
- ≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence: 20 câu, 15 dev / 5 test; 5 câu filter, 2 thuộc test. Evidence được kiểm tra khớp nguyên văn và trang trước khi chạy.
- Tiêu chí chọn chiến lược đã chốt trước khi xem test: Chọn trên dev theo Evidence Hit@3, tie-break bằng Hit@1; grounding/coherence thủ công là bước xác nhận. Test chỉ báo cáo sau khi khóa lựa chọn.

## Kết quả
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian (A/B gồm embedding; C incremental) | Chi phí |
|---|---|---|---|---|---|---|
| A | dev | 10/15 (66.7%) | 15/15 (100.0%) | 126 | 141.74s | Check Gemini billing |
| A | test | 4/5 (80.0%) | 4/5 (80.0%) | 126 | 141.74s | Check Gemini billing |
| B | dev | 13/15 (86.7%) | 15/15 (100.0%) | 127 | 168.13s | Check Gemini billing |
| B | test | 4/5 (80.0%) | 4/5 (80.0%) | 127 | 168.13s | Check Gemini billing |
| C | dev | 13/15 (86.7%) | 15/15 (100.0%) | 127 | 0.40s (incremental) | Check Gemini billing |
| C | test | 4/5 (80.0%) | 4/5 (80.0%) | 127 | 0.40s (incremental) | Check Gemini billing |

| Strategy | Coherence đã ghi /2 (n; chờ xác nhận) | Metadata đủ/đúng, hữu dụng (n/N; chờ xác nhận) | Filter đúng trên test (câu; top-3 hits) | Grounding đã ghi 0/1/2 (n; chờ xác nhận) |
|---|---|---|---|---|
| A | 0.00/2 (n=10) | 15/15 | 0/2; 3/6 | 0/0/5; trung bình 2.00/2 |
| B | 2.00/2 (n=10) | 15/15 | 1/2; 5/6 | 0/0/5; trung bình 2.00/2 |
| C | 2.00/2 (n=10) | 15/15 | 2/2; 6/6 | 0/1/4; trung bình 1.80/2 |

Đường dẫn trace và audit: Trace `results/top3_trace.jsonl`; coherence `results/coherence_audit.csv`; metadata `results/metadata_grounding_audit.csv`; grounding `results/grounding_precheck.csv`. Precheck và inventory: `results/metadata_precheck.csv`, `results/grounding_precheck.csv`, `results/source_unit_inventory.csv`. Mẫu coherence là 10 source-unit đầu theo thứ tự; cấu hình/giả thuyết ở `artifacts/experiment_log.csv`.

**Phương pháp và trạng thái xác nhận nhãn:** Gemini 2.5 Flash đã được dùng để tạo gợi ý coherence/grounding riêng theo rubric; mã và prompt nằm trong `starter/src/judge_suggestions.py`. Trong lần rà soát hiện tại, cả điểm lẫn rationale trong nhãn cuối trùng gợi ý AI ở 30/30 dòng coherence và 15/15 dòng grounding. Đây là tín hiệu cần công khai, không phải bằng chứng rằng tác giả đã/không đã xem xét từng dòng. Các nhãn này chưa được xác nhận độc lập trong hồ sơ. File `metadata_grounding_audit.csv` hiện ghi 45/45 hit đủ/đúng và hữu dụng, kèm rationale (15/15 mỗi cấu hình); cần đối chiếu với trace/inventory và xác nhận đây là đánh giá của người chấm trước khi gọi là nhãn thủ công. Các kết quả audit trên là nhãn đang được lưu, không phải xác nhận độc lập. Người chấm là tác giả; cần nêu nguy cơ thiên lệch tự đánh giá.

## Phân tích và kết luận
- So sánh A–B và B–C: Dev Evidence Hit@3 A/B/C: 15/15, 15/15, 15/15; A–B đo chunking, B–C đo filter. Test chỉ báo cáo, không dùng để chọn cấu hình.
- Tuân thủ filter trên hai câu test có filter (q16, q20): A có 0/2 câu mà cả top-3 cùng đúng chương (3/6 hit đúng chương); B có 1/2 câu (5/6 hit); C có 2/2 câu (6/6 hit). A/B không áp dụng filter; đây là phép đo top-3 có tình cờ thuộc đúng chương không, còn C áp dụng filter trước truy xuất.
- Chiến lược chọn, đánh đổi, điều kiện áp dụng: Trên dev, B và C hòa nhau ở Evidence Hit@3 (15/15) và Hit@1 (13/15), nên tiêu chí đã đăng ký không phân định được hai cấu hình. B là baseline đơn giản hơn; C phù hợp khi câu hỏi có filter theo chapter. Trên test cả ba đạt 4/5 Evidence Hit@3.
- Ví dụ lỗi và giới hạn: q18 (gold `alice-wonderland`, logical page 40) là exact-evidence Hit@3 miss ở cả A/B/C; top-3 có nội dung liên quan nhưng không khớp nguyên văn gold quote. Mẫu coherence của A có 10/10 chunk bị cắt giữa ý (ví dụ chunk `5b7b9437c941bf049ffd5bb2`, page 1), trong khi B/C đạt 2/2 ở 10 mẫu. Test chỉ có 5 câu và corpus là một tác phẩm tiếng Anh nên không đại diện rộng.
- Chi phí, xác nhận nhãn và nộp bài: `run_metadata.json` hiện ghi `cost_usd: 0` và `cost_note: "Free tier"`; cần đảm bảo đã đối chiếu billing thực tế, bao gồm cả các lần gọi Gemini judge. Nhãn audit chờ tác giả xác nhận như mô tả ở trên. Repo bài nộp đã được điền trong `STUDENT.md`; commit chốt vẫn còn trống.
- Nếu làm lại hoặc mở rộng, cải tiến gì? Tăng số tác phẩm và câu test độc lập; kiểm tra thêm câu có filter; lưu bằng chứng xác nhận billing và rà soát nhãn audit độc lập trước khi kết luận.
