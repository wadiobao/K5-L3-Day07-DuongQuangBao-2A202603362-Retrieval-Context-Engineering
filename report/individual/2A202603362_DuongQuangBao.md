# Báo cáo cá nhân — Day 7

## Dữ liệu và bộ đánh giá

- **Nguồn:** *Alice's Adventures in Wonderland*, Project Gutenberg eBook #11, [văn bản thuần](https://www.gutenberg.org/cache/epub/11/pg11.txt). Tác phẩm gốc thuộc phạm vi công cộng tại Hoa Kỳ; người dùng ở nơi khác cần tự xác minh luật địa phương trước khi tái phân phối.
- **Các đoạn dữ liệu (chunk):** dùng chỉ mục Day 6, strategy C; 127 đoạn từ một tài liệu. Trường `page` là số thứ tự của đơn vị nguồn logic trong ngữ liệu, không phải số trang in.
- **Kho vector và embedding:** PostgreSQL 16.16 + pgvector 0.8.7; khoảng cách cosine, tìm kiếm chính xác. Gemini Embedding 2, 768 chiều. Cơ sở dữ liệu cục bộ dùng cổng 55433; không đưa thông tin xác thực vào báo cáo. Day 6 ghi nhận 273 lượt gọi embedding; người dùng xác nhận dùng gói miễn phí, chi phí $0.00.
- **Tác nhân (agent):** `gemini-3.5-flash-lite`, gọi hàm qua Gemini API; truy hồi dense từ pgvector, truy hồi từ vựng TF-IDF và truy hồi kết hợp bằng phương pháp hợp nhất thứ hạng nghịch đảo (RRF).
- **Bộ đánh giá:** 24 câu: dev có 15 câu có đáp án và 2 câu thiếu bằng chứng; test có 5 câu có đáp án và 2 câu thiếu bằng chứng. Nhãn chuẩn (gold) gồm nguồn, trang logic và đoạn bằng chứng. Các câu thiếu bằng chứng hỏi tên chính xác của trường, địa chỉ nhà, ngày cụ thể và tên đầy đủ. Toàn bộ câu hỏi test: [questions.json](../../starter/data/day7/questions.json); các đoạn ngữ liệu: [chunks.json](../../starter/data/day7/chunks.json).

## Truy hồi và thử nghiệm

Hit@1, Evidence Hit@3 và Context Evidence Rate chỉ tính các câu có đáp án. Ngân sách ngữ cảnh được đo bằng ký tự; thời gian truy hồi là trung vị mỗi truy vấn trong bản ghi trace. TF-IDF chạy cục bộ nên không phát sinh phí dịch vụ.

| Cấu hình | Phương pháp, top-k/bộ lọc/xếp hạng lại, ngân sách | Giả thuyết hoặc thay đổi | Tập | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Dense | Day 6 strategy C, top-3, ngân sách 6000 | Mốc đối chứng truy hồi vector | dev | 13/15 | 15/15 | 15/15 | trung vị 423 ms/truy vấn; embedding $0.00 (gói miễn phí) |
| Lexical | TF-IDF, top-3, ngân sách 6000 | So sánh phương pháp truy hồi từ vựng | dev | 12/15 | 15/15 | 15/15 | trung vị 0.39 ms/truy vấn; chạy cục bộ |
| Hybrid RRF | Dense + TF-IDF; 20 ứng viên, top-3, ngân sách 6000 | Thử kết hợp thứ hạng | dev | 14/15 | 15/15 | 15/15 | trung vị 12.47 ms/truy vấn; embedding $0.00 (gói miễn phí) |
| Lexical, ngân sách 800 | TF-IDF, top-3, ngân sách 800 | Mốc đối chứng với ngữ cảnh ngắn | dev | 12/15 | 15/15 | 2/15 | trung vị 0.67 ms/truy vấn; chạy cục bộ |
| Lexical, ngân sách 6000 | TF-IDF, top-3, ngân sách 6000 | Tăng ngân sách ngữ cảnh để giữ đủ bằng chứng | dev | 12/15 | 15/15 | 15/15 | trung vị 0.54 ms/truy vấn; chạy cục bộ |
| Hybrid top-5 | RRF với 20 ứng viên, trả về 5; so sánh thứ hạng tại ngưỡng 3 | Kiểm tra việc tăng số kết quả trong ngữ cảnh | dev | 14/15 | 15/15 | 15/15 | trung vị 445 ms/truy vấn; embedding $0.00 (gói miễn phí), không tốt hơn top-3 |
| Agent dense | Day 6 strategy C, top-3, ngân sách 6000 | Tác nhân truy hồi dense đối chứng trên test | test | 4/5 | 4/5 | 4/5 | thời gian xử lý của tác nhân trung vị 4.06 giây/truy vấn; chi phí $0.00 (gói miễn phí, theo xác nhận người dùng) |
| Agent hybrid RRF | 20 ứng viên, top-3, ngân sách 6000 | Cấu hình được chọn trên dev | test | 3/5 | 4/5 | 4/5 | truy hồi trung vị 469 ms/truy vấn; thời gian xử lý của tác nhân trung vị 3.37 giây/truy vấn; chi phí $0.00 (gói miễn phí, theo xác nhận người dùng) |

Hai thay đổi có lợi trên dev: hybrid RRF tăng Hit@1 từ 13/15 (dense) lên 14/15; tăng ngân sách ngữ cảnh từ 800 lên 6000 làm Context Evidence Rate của TF-IDF tăng từ 2/15 lên 15/15 mà không đổi thứ hạng truy hồi. Top-5 không cải thiện các chỉ số so với top-3 nên giữ top-3. Cấu hình được chọn dựa trên dev, không dựa vào test. Trên tập test nhỏ này, hybrid không tăng Hit@1 so với dense; cả hai đạt Evidence Hit@3 4/5. Context Evidence Rate 4/5 là kết quả đối chiếu chặt theo trang/đoạn bằng chứng gold. Riêng q18, ngữ cảnh không có đúng đoạn gold ở trang 40 nhưng có đoạn thay thế ở trang 39 đủ hỗ trợ các bước của điệu nhảy; điểm ngữ cảnh chấm thủ công là 2.

## Tác nhân (agent), bản ghi trace và nhãn đánh giá

Đã đọc và chấm toàn bộ 7 câu test cho mốc đối chứng dense và hybrid. Việc chấm gồm điểm câu trả lời có căn cứ /2, mức độ hỗ trợ của từng trích dẫn và tính đúng đắn của việc từ chối trả lời. Các nhãn và lý do:

- Dense test: [nhãn đánh giá](../../starter/results/agent-dense-test/dense-test-reviews.json), [chỉ số](../../starter/results/agent-dense-test/dense-test-metrics.json), [trace đã gộp](../../starter/results/agent-dense-test/dense-test-traces.json).
- Hybrid test: [nhãn đánh giá](../../starter/results/agent-individual/hybrid-test-reviews.json), [chỉ số](../../starter/results/agent-individual/hybrid-test-metrics.json), [trace](../../starter/results/agent-individual/hybrid-test-traces.json).
- Hybrid dev: [nhãn đánh giá](../../starter/results/agent-individual/hybrid-dev-reviews.json), [chỉ số](../../starter/results/agent-individual/hybrid-dev-metrics.json).

| Mã câu | Dense test | Hybrid test | Nhận xét chấm thủ công |
|---|---|---|---|
| q16 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Nhím làm bóng; hồng hạc làm gậy đánh croquet. |
| q17 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Reeling và Writhing là hai môn học được nêu đầu tiên. |
| q18 | Điểm 2; cả hai trích dẫn đều hỗ trợ; ngữ cảnh 2 | Điểm 2; cả hai trích dẫn đều hỗ trợ; ngữ cảnh 2 | Các đoạn được trích ở trang 39 độc lập hỗ trợ việc ném tôm hùm ra biển, bơi theo, nhào lộn rồi quay lại bờ. |
| q19 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Một chiếc bàn cùng đĩa bánh tart lớn được đặt ở giữa phòng xử án. |
| q20 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Điểm 2; trích dẫn hỗ trợ; ngữ cảnh 2 | Alice làm đổ hộp bồi thẩm bằng gấu váy. |
| q23 | Từ chối đúng | Từ chối đúng | Ngữ liệu không nêu ngày dương lịch chính xác. |
| q24 | Từ chối đúng | Từ chối đúng | Ngữ liệu không nêu tên riêng đầy đủ của Thỏ Trắng. |

Cả hai lượt chạy test đều gọi công cụ truy hồi cho 7/7 câu và không có lỗi từ nhà cung cấp. Mỗi lượt dùng 21 lần gọi Gemini để sinh nội dung và 14 lần gọi tìm kiếm; số lượt tìm kiếm mỗi câu là 1 (4 câu), 2 (1 câu) và 4 (2 câu). Cấu hình dense dùng 24.540 token sinh nội dung theo siêu dữ liệu; hybrid dùng 25.300. Cả hai đạt 5/5 câu có đáp án ở mức điểm 2, có 6/6 trích dẫn được hỗ trợ, có trích dẫn ở 5/5 câu trả lời, từ chối đúng 2/2 câu thiếu bằng chứng và không từ chối nhầm câu có đáp án nào (0/5). Thời gian xử lý trung vị của tác nhân là 4.06 giây/truy vấn với dense và 3.37 giây/truy vấn với hybrid. Chi phí API của tác nhân trong các lượt chạy dev/test được ghi là $0.00 theo xác nhận của người dùng rằng các lệnh gọi dùng gói miễn phí.

Chỉ số hybrid dev: sử dụng công cụ 17/17; câu trả lời có căn cứ gồm 14 câu điểm 2 và một câu điểm 0 (trung bình 1.87/2); trích dẫn được hỗ trợ 15/16; từ chối đúng 1/2; từ chối nhầm 1/15. Lượt chạy tác nhân dùng truy hồi dense trên dev bị giới hạn hạn mức: thành công 7/17, có 10 lỗi `429 RESOURCE_EXHAUSTED`; do đó không dùng làm kết quả so sánh đầy đủ về chất lượng câu trả lời.

### Hai lỗi dev và phân tích

1. **q15 — từ chối nhầm:** tác nhân dừng với trạng thái `abstain`. Kết quả tìm kiếm có đoạn ở trang 27 với câu “he poured a little hot tea upon its nose”, nhưng đoạn này không nằm trong ngữ cảnh cuối; tác nhân không trích dẫn bằng chứng. Điểm câu trả lời là 0, điểm ngữ cảnh là 0. Đây là lỗi giữ/chọn ngữ cảnh sau truy hồi, không phải do thiếu bằng chứng trong ngữ liệu.
2. **q21 — trạng thái và trích dẫn không nhất quán:** câu hỏi yêu cầu tên chính xác của trường. Nội dung trả lời đúng rằng tên trường không được nêu và Alice chỉ nói từng học ở một trường học ban ngày (`day-school`), nhưng trace ghi trạng thái `answered`; trích dẫn không chứng minh kết luận rằng toàn bộ ngữ liệu không có tên cụ thể. Theo chỉ số, đây là lượt từ chối không đúng (1/2 dev), và bộ điều khiển nên trả trạng thái `abstain` khi không có tên chính xác.

Các lỗi quota 429 trong lượt dense dev cho thấy kết quả agent phụ thuộc giới hạn tốc độ; không tính lỗi là câu trả lời đúng/sai và không chuyển sang gói trả phí để né hạn mức. Tập test chỉ có 5 câu có đáp án nên các chỉ số còn nhiều bất định; ngữ liệu chỉ gồm một tác phẩm tiếng Anh. Độ trễ truy hồi không bao gồm độ trễ agent.

## Lệnh chạy và tái lập

Xem hướng dẫn thiết lập và chạy trong [README của starter](../../starter/README.md). Cần có cơ sở dữ liệu Day 6 đã lập chỉ mục theo strategy C hoặc khởi tạo lại theo hướng dẫn Day 6; cấu hình cục bộ/cổng kết nối được nêu trong `.env.example`. Không đưa `.env` hoặc khóa API vào kho mã nguồn.

```bash
python3 -m pip install -r starter/requirements.txt
python3 starter/export_day6.py --output starter/data/day7/chunks.json --strategy C
python3 starter/run_agent.py --chunks starter/data/day7/chunks.json --questions starter/data/day7/questions.json --split test --methods dense,hybrid --strategy C --top-k 3 --candidate-k 20 --budget 6000 --agent-model gemini-3.5-flash-lite --output starter/results/agent-test
```

## Thử thách nâng cao

Không thực hiện; chỉ lập chỉ mục ngữ liệu của một tác phẩm.
