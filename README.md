# Kho tri thức AI — demo local

Ứng dụng hỏi đáp trên tài liệu riêng theo mô hình RAG. Giao diện chính là chatbot: tạo kho, thêm tài liệu và hỏi đáp có nguồn. Có thể đổi tên/xóa kho, tài liệu và hội thoại; xem chunk, nguồn và pipeline đã tạo câu trả lời.

## Chạy trên Windows

Yêu cầu Python 3.13 hoặc 3.12 và PowerShell. Script tự chọn Python 3.13 trước, sau đó thử Python 3.12.

1. Tạo API key trong Google AI Studio.
2. Tạo hoặc mở file `.env` trong thư mục dự án, thêm dòng `GEMINI_API_KEY=...`. Hoặc để trống và nhập key trong giao diện mỗi lần chạy.
3. Mở PowerShell tại thư mục dự án và chạy:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
   ```

   Cách gọi này chỉ đặt Execution Policy `Bypass` cho tiến trình PowerShell vừa mở để chạy script, không thay đổi policy cố định của máy. Script tạo `.venv`, cài dependencies (bao gồm Docling + RapidOCR cho OCR/fallback) rồi mở Streamlit. Lần cài đầu có thể lâu vì Docling tải nhiều thư viện; lần OCR đầu có thể tải model weights. OCR dự phòng được cấu hình cho tiếng Việt. Khi Streamlit in URL local, mở URL đó trên trình duyệt.

Dữ liệu ứng dụng được lưu dưới `data/`: file gốc ở `uploads/`, vector Chroma ở `chroma/`, lịch sử và metadata ở `app.sqlite3`. Thư mục này đã được thêm vào `.gitignore`.

## Công nghệ

- Python + Streamlit cho demo local.
- LangChain `RecursiveCharacterTextSplitter` và LangChain Chroma integration.
- Gemini API `gemini-3.5-flash` cho sinh câu trả lời; `gemini-embedding-001` riêng cho vector văn bản.
- Chroma lưu vector theo kho; SQLite lưu kho, file, chunk, hội thoại và tin nhắn.
- Parser nhanh cho PDF có text, DOCX, XLSX, PPTX, TXT/Markdown, CSV/TSV và HTML. Nếu định dạng chưa có parser nhanh, PDF scan/ảnh hoặc trích xuất rỗng, ứng dụng tự chuyển sang Docling/OCR và nhận diện cấu trúc bảng.
- Uploader cũng nhận các định dạng phổ biến khác như ảnh, DOC/XLS/PPT cũ, RTF, ODT/ODS/ODP, EPUB, AsciiDoc, LaTeX, XHTML, MHTML và email; định dạng không có parser nhanh sẽ thử Docling.

## Quota Gemini và lỗi `429 RESOURCE_EXHAUSTED`

Một file có thể tạo ra hàng chục/hàng trăm chunk. Gửi một request embedding cho mỗi chunk sẽ chạm RPM nhanh, kể cả khi người dùng chỉ upload một file. Adapter LangChain trong app gom tối đa `GEMINI_EMBEDDING_BATCH_SIZE` chunk vào mỗi request (mặc định 32), giới hạn nhịp ở phía app và retry 429/503 với exponential backoff, jitter, số lần thử hữu hạn. Chunk đã parse được lưu vào SQLite trước khi gọi embedding; sau lỗi quota có thể bấm **Thử lập chỉ mục lại** để không phải parse/OCR lần nữa.

Thiết lập trong `.env`:

```dotenv
GEMINI_EMBEDDING_RPM_LIMIT=10
GEMINI_GENERATION_RPM_LIMIT=10
GEMINI_EMBEDDING_BATCH_SIZE=32
GEMINI_MAX_RETRIES=5
```

Hai giới hạn RPM trong sidebar là giới hạn cục bộ của app; hãy đặt chúng thấp hơn hoặc bằng quota model đang hiển thị trong Google AI Studio. Nếu nhiều chương trình cùng dùng một Google Cloud/AI Studio project, requests của chúng chia sẻ hạn mức project. Chỉ thay API key thường không tăng quota. Khi API báo quota/ngân sách theo ngày hoặc spend cap, đợi quota reset hoặc kiểm tra billing/tier; retry ngắn không thể khắc phục.

Nếu thường xuyên nhập file lớn nhưng không cần lập chỉ mục ngay, Gemini Batch API có thể phù hợp cho xử lý nền, đổi lại thời gian hoàn tất không bảo đảm tức thời. Bản demo này dùng embedding batch đồng bộ để người dùng có kết quả ngay và không tạo batch job treo.

Tài liệu Google: [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [Troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting), [Batch API](https://ai.google.dev/gemini-api/docs/batch-api).

## Đọc bảng, OCR và phạm vi file

Parser nhanh đọc bảng PDF/DOCX/XLSX/PPTX thành Markdown có hàng/cột. Mỗi đoạn bảng được giữ tiêu đề cột khi chia thành chunk để câu trả lời không bị tách khỏi cấu trúc. Với PDF scan, ảnh, file legacy hoặc parser nhanh không lấy được nội dung, Docling + RapidOCR fallback xử lý layout/OCR bằng tiếng Việt và bật nhận diện bảng; số trang được ghi vào metadata khi parser cung cấp provenance.

Ứng dụng nhận các định dạng trong `rag_demo/config.py`. Không có parser nào bảo đảm hiểu mọi định dạng, sơ đồ hoặc bảng lồng nhau hoàn hảo; file lỗi, có mật khẩu, biểu đồ chỉ tồn tại dưới dạng ảnh hoặc OCR mờ có thể cần kiểm tra kết quả trong mục thu gọn **Tài liệu trong kho**. Một số định dạng Office cũ (`.doc`, `.xls`, `.ppt`) cần cài LibreOffice để Docling chuyển đổi. Định dạng chuyên biệt hoặc dữ liệu nhị phân tùy chỉnh có thể không đọc được dù uploader chấp nhận một số loại file mở rộng. Nếu file không được đọc, trạng thái lỗi giữ lại để xem nguyên nhân.

## Trình bày demo

Mở [DEMO_GUIDE.md](DEMO_GUIDE.md) để dùng kịch bản demo 5 phút, sơ đồ kiến trúc, danh sách vấn đề đã xử lý và checklist chuẩn bị khi trình bày.

## Hỏi đáp trong phạm vi toàn kho

- **Tự động / Hỏi đáp chi tiết** dùng vector search lấy top-k chunk từ toàn bộ tài liệu trong kho đang chọn.
- **Tổng hợp toàn kho** đọc riêng từng file đã lập chỉ mục, chia file lớn thành các phần rồi dùng map-reduce. Với yêu cầu tóm tắt toàn bộ, câu trả lời có thêm ghi chú theo từng file để dễ kiểm tra phạm vi. Chế độ **Tự động** cũng chuyển câu hỏi rộng sang luồng này dù giao diện đang chọn Nâng cao. Câu đếm số file và liệt kê tên file lấy trực tiếp từ danh mục SQLite, không gọi Gemini. Kho lớn vẫn cần nhiều lượt gọi Gemini và có thể mất thời gian/quota hơn.

Đổi model embedding hoặc số chiều vector sau khi đã nhập file cần tạo lại vector cho kho. Không trộn embedding từ các model khác nhau trong cùng collection.

## Pipeline nâng cao và so sánh

Trong **Kết nối AI & tùy chọn nâng cao**, có thể chọn **Cơ bản**, **Nâng cao** hoặc **So sánh cơ bản và nâng cao** cho cùng một câu hỏi. Chế độ so sánh lưu cả hai câu trả lời, nguồn và số liệu vào lịch sử để trình bày.

Pipeline nâng cao có thể bật/tắt Parent-Child context, Hybrid Search (Chroma dense + BM25), FlashRank reranker, Query Rewrite, Multi-query (tối đa hai query phụ), HyDE, nén ngữ cảnh trích xuất, answer grading, citation verification và LangGraph. Các query Rewrite/Multi-query/HyDE dùng chung một lần lập kế hoạch Gemini; các truy vấn embedding được gửi theo batch. Reranker chạy local sau lần tải model đầu.

Khi bật Query Rewrite, bước sinh câu trả lời vẫn nhận tối đa một lượt hội thoại gần nhất để hiểu câu hỏi nối tiếp. Lịch sử chỉ dùng để giải quyết chủ thể/đại từ; dữ kiện và trích dẫn phải dựa trên các nguồn được truy xuất ở lượt hiện tại.

Để truy vấn phụ không lấn át câu hỏi gốc, RRF dùng trọng số cao nhất cho Dense của câu gốc; Rewrite/Multi-query có trọng số thấp hơn và HyDE chỉ tham gia Dense với trọng số nhỏ (không dùng BM25). Candidate pool luôn giữ các Dense hit đầu của câu gốc; context dành tối đa hai nhóm nguồn riêng cho các hit đó và không nén chúng. Pipeline hiển thị top sau RRF, top sau reranker và nguồn thực sự gửi Gemini để kiểm tra trường hợp truy xuất bỏ sót.

Chi phí được khống chế bằng cách giữ pipeline cơ bản làm lựa chọn riêng, giới hạn query phụ và số ứng viên rerank, đặt ngân sách token cho đoạn tài liệu, giới hạn output planner/audit, gom chấm điểm và xác minh vào một lượt audit, và không tự gọi thêm lượt sửa câu trả lời. Mặc định planner/audit tắt để giảm độ trễ; nút **Tiết kiệm API** giữ cấu hình này, còn **Bật mọi kỹ thuật** bật toàn bộ để trình diễn. Có thể đổi từng công tắc riêng sau khi chọn preset. So sánh chủ động chạy cả pipeline cơ bản và nâng cao nên chi phí lượt đó gần bằng tổng hai lượt. Số request và token được lưu cùng câu trả lời; token lấy từ usage metadata Gemini nếu có, nếu không sẽ ước tính từ độ dài ký tự.

Mỗi lượt nâng cao cần một embedding batch và một generation call để trả lời; Rewrite/Multi-query/HyDE chia sẻ tối đa một generation call lập kế hoạch; grading/citation chia sẻ tối đa một generation call audit. Reranker, BM25, nén ngữ cảnh và LangGraph chạy cục bộ. Mở **So sánh cơ bản và nâng cao** để xem cả hai câu trả lời, file/trang/chunk, số nguồn trùng, API calls, token và thời gian. Mở **Nguồn đã dùng** để xem child chunk và, nếu parent/compression làm thay đổi văn bản, đúng ngữ cảnh đã gửi cho Gemini.

Parent chunks của tài liệu mới được tạo lúc nhập file. Khi mở dữ liệu cũ, ứng dụng tự tạo nhóm parent từ tối đa ba child chunks liên tiếp để dùng ngay; đây là parent dẫn xuất từ chunk cũ, còn nhập lại file sẽ tạo phân cấp mới từ nội dung gốc. Ngân sách ngữ cảnh nâng cao giới hạn lượng nội dung gửi cho câu trả lời.

## Phòng thử nghiệm pipeline từng bước

Chọn **🧪 Phòng thử nghiệm từng bước** ở đầu giao diện để mở một lượt chạy độc lập với chat thường. Tải một file lên, chọn **Chunk thường** hoặc **Parent-Child**, xem trước nội dung chunk (có thể bật đối chiếu cả hai), rồi xác nhận để tạo embedding vào collection Chroma tạm trong bộ nhớ. Từ bước planner trở đi, chọn lựa chọn nào là ứng dụng sẽ chạy đúng bước đó và hiện kết quả ngay; nút tiếp tục chỉ chốt kết quả đang xem. Planner có đủ 8 tổ hợp: từng kỹ thuật Rewrite, Multi-query, HyDE, ba tổ hợp đôi và cả ba kỹ thuật. Preview cho thấy query rewrite, query phụ hoặc câu trả lời HyDE trước khi retrieval chạy. Vector/Hybrid, reranker, nén context và audit cũng hiện đầu ra ngay sau khi chọn. Có thể mở lại kết quả từng bước để trình bày.

Phòng thử nghiệm dừng thủ công giữa các bước để người xem kiểm tra kết quả; phần chat thường điều phối pipeline tự động bằng LangGraph.

Trong lab, child luôn được ghi nhãn parent tương ứng; kết quả dùng số chunk/trang dễ đọc thay cho ID nội bộ. Bước context mở hai bản không nén/có nén cạnh nhau và chỉ gửi bản đã chọn. Bước sinh câu trả lời hiển thị câu trả lời, nguồn và gợi ý ngay tại màn hình trước khi tiếp tục. Mở **Giải thích thuật ngữ, chỉ số, thông số và công thức** để xem định nghĩa Dense/BM25/RRF, công thức tìm kiếm/nén, quota và các thông số cấu hình. Bộ đếm lab không hiển thị thứ tự request.

Các lựa chọn planner mới có thể cần một generation call cho mỗi tổ hợp mới; tổ hợp đã chạy được cache trong lượt thử. Vector và Hybrid dùng chung cache embedding cho cùng query, nên chuyển qua lại chỉ chạy lại retrieval/BM25 cục bộ. Reranker, dựng context và nén context chạy cục bộ; Answer Grading/Citation Verification dùng chung tối đa một generation call cho tổ hợp audit đã chọn.

Phòng thử nghiệm không thêm file vào kho thường trong lúc xem trước và chạy pipeline; index tạm biến mất khi phiên ứng dụng kết thúc. Sau khi hoàn tất, có thể bấm **Lưu file với cách chia đã chọn vào kho thật**. Bước lưu này tạo embedding lần nữa cho collection Chroma bền vững của kho. Số API calls và token của các bước thử được hiển thị trong phiên.

## Giới hạn truy xuất cần trình bày

Tìm kiếm cơ bản lấy top-k chunk gần nhất theo cosine distance nhưng hiện chưa có ngưỡng từ chối đã hiệu chỉnh. Vì vậy, câu hỏi ngoài phạm vi vẫn có thể đưa các chunk gần nhất vào Gemini. Distance là chỉ số xếp hạng, không phải xác suất câu trả lời đúng; không nên dùng một ngưỡng cố định như 0.70 cho mọi kho nếu chưa đánh giá bằng câu hỏi có đáp án và câu hỏi ngoài phạm vi. Pipeline nâng cao cũng không tự sửa câu trả lời khi điểm audit thấp.

Nén ngữ cảnh là phép chọn đoạn/dòng dựa trên từ khóa sau khi bỏ một số từ dừng tiếng Việt phổ biến. Khi không tìm được từ khóa nội dung trùng, ứng dụng giữ phần đầu và cuối thay vì chỉ lấy đầu đoạn; với bảng, tiêu đề được giữ và thứ tự các hàng được chọn vẫn theo tài liệu gốc. Đây vẫn là phép nén có mất mát, nên cần mở nội dung nguồn khi trả lời về số liệu/bảng quan trọng.
