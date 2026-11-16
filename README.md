# Kho tri thức AI — demo local

Ứng dụng hỏi đáp trên tài liệu riêng theo mô hình RAG. Giao diện chính là chatbot: tạo kho, thêm tài liệu và hỏi đáp có nguồn. Có thể đổi tên/xóa kho, tài liệu và hội thoại; xem chunk, nguồn và pipeline đã tạo câu trả lời. Tên kho được kiểm tra không trùng sau khi chuẩn hóa Unicode, chữ hoa/thường và khoảng trắng; nếu dữ liệu cũ đã có kho trùng tên, giao diện thêm mã ngắn để phân biệt.

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

## Kịch bản hội thoại và giọng nói tiếng Việt

Trong **💬 Chat**, hội thoại mới chào người dùng bằng menu gồm hai mẫu có sẵn: thẩm định Hồ sơ Đấu thầu (Chương III), phân tích Báo giá Website, cùng lựa chọn hỏi đáp tự do. Hai tác vụ phân tích yêu cầu người dùng chọn tài liệu sẵn có trong kho hoặc đính kèm file ngay trong ô chat. Tìm kiếm bị giới hạn vào các file đã chọn/tải lên; câu trả lời có trích dẫn theo trang và có thể hỏi tiếp trên cùng phạm vi tài liệu. Nút **⚙️ Thiết lập kịch bản** cho phép tạo/chỉnh sửa persona, kho tri thức, các bước hỏi, trường bắt buộc và khung sản phẩm đầu ra; mẫu tự tạo sẽ xuất hiện thêm trong menu Chat và thu thập câu trả lời/file tuần tự ngay trong timeline.

Ô chat hỗ trợ nhập văn bản, ghi âm và đính kèm file. Sau khi ghi âm, ứng dụng nhận dạng tiếng Việt và đưa transcript vào ô chat dưới dạng bản nháp; người dùng có thể sửa rồi chủ động nhấn Gửi để chạy câu hỏi. Nút **🔊 Tạo giọng đọc** xuất hiện dưới cả câu hỏi và câu trả lời; sau khi tạo WAV bằng Gemini TTS, người dùng bấm Play trên trình phát để nghe. Nhận dạng và tổng hợp giọng nói dùng quota Gemini generation; lần tạo giọng đọc đầu tiên cần quyền truy cập model TTS.

Trong project này, **LangChain** hỗ trợ một số thành phần xử lý tài liệu và truy xuất (text splitter, Chroma integration); lời gọi Gemini dùng `google-genai`. **LangGraph** điều phối các bước của pipeline RAG. Kịch bản là workflow xác định theo cấu hình trong SQLite, không phải agent tự quyết định bước hoặc công cụ; một phiên lưu snapshot mẫu, câu trả lời, trạng thái và phạm vi tài liệu để có thể tiếp tục sau rerun. Kết quả cuối chạy qua RAG trong phạm vi đã chọn, giữ nguồn trích dẫn và mở tiếp hội thoại trên cùng tập tài liệu.

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

Pipeline nâng cao có thể bật/tắt Parent-Child context, Hybrid Search (Chroma dense + BM25), FlashRank reranker, Query Rewrite, Multi-query (tối đa hai query phụ), HyDE, Query Decomposition (tối đa ba câu con), nén ngữ cảnh trích xuất, answer grading, citation verification, Corrective Retrieval, Reflection (sửa tối đa một lần), Function Calling Router và LangGraph. Các query Rewrite/Multi-query/HyDE/Decomposition dùng chung một lần lập kế hoạch Gemini; các truy vấn embedding được gửi theo batch. Reranker chạy local sau lần tải model đầu.

Khi bật Query Rewrite, bước sinh câu trả lời vẫn nhận tối đa một lượt hội thoại gần nhất để hiểu câu hỏi nối tiếp. Lịch sử chỉ dùng để giải quyết chủ thể/đại từ; dữ kiện và trích dẫn phải dựa trên các nguồn được truy xuất ở lượt hiện tại.

Để truy vấn phụ không lấn át câu hỏi gốc, RRF dùng trọng số cao nhất cho Dense của câu gốc; Rewrite/Multi-query có trọng số thấp hơn và HyDE chỉ tham gia Dense với trọng số nhỏ (không dùng BM25). Candidate pool luôn giữ các Dense hit đầu của câu gốc; context dành tối đa hai nhóm nguồn riêng cho các hit đó và không nén chúng. Pipeline hiển thị top sau RRF, top sau reranker và nguồn thực sự gửi Gemini để kiểm tra trường hợp truy xuất bỏ sót.

Chi phí được khống chế bằng cách giữ pipeline cơ bản làm lựa chọn riêng, giới hạn query phụ, số ứng viên rerank và số lượt correction/reflection, đặt ngân sách token cho context, gom grading và citation vào một lượt audit. **Corrective Retrieval** có thể tìm lại tối đa một lần rồi abstain; **Reflection** có thể viết lại một lần từ cùng context, audit lần nữa và abstain nếu chưa đạt. Cả hai đều dừng hữu hạn. **Function Calling Router**, decomposition và reflection mặc định tắt; dùng checkbox riêng hoặc preset **Agentic demo** để bật, còn **Bật mọi kỹ thuật** bật toàn bộ cho demo quota cao. So sánh chạy cả hai pipeline nên tốn tổng chi phí của hai lượt. Số request/token được lưu cùng câu trả lời; token lấy từ Gemini usage metadata nếu có, nếu không ước tính từ độ dài ký tự.

Mỗi lượt nâng cao cần một embedding batch và một generation call để trả lời; planner dùng chung một generation call; audit đầu dùng chung cho grading/citation/Reflection. Function Calling thêm tối đa một call; Reflection thêm tối đa hai call (viết lại + audit lại); Corrective Retrieval có thể thêm tối đa hai call và một embedding batch nếu phải tìm lại. Reranker, BM25, nén ngữ cảnh và LangGraph chạy cục bộ. Mở **So sánh cơ bản và nâng cao** để xem cả hai câu trả lời, nguồn, API calls, token và thời gian. Mở **Nguồn đã dùng** để xem child chunk và ngữ cảnh thực sự đã gửi cho Gemini.

Parent chunks của tài liệu mới được tạo lúc nhập file. Khi mở dữ liệu cũ, ứng dụng tự tạo nhóm parent từ tối đa ba child chunks liên tiếp để dùng ngay; đây là parent dẫn xuất từ chunk cũ, còn nhập lại file sẽ tạo phân cấp mới từ nội dung gốc. Ngân sách ngữ cảnh nâng cao giới hạn lượng nội dung gửi cho câu trả lời.

## Phòng thử nghiệm pipeline từng bước

Chọn **🧪 Phòng thử nghiệm từng bước** ở đầu giao diện để mở một lượt chạy độc lập với chat thường. Tải một file lên, chọn **Chunk thường** hoặc **Parent-Child**, xem trước nội dung chunk (có thể bật đối chiếu cả hai), rồi xác nhận để tạo embedding vào collection Chroma tạm trong bộ nhớ. Từ bước planner trở đi, chọn lựa chọn nào là ứng dụng sẽ chạy đúng bước đó và hiện kết quả ngay; nút tiếp tục chỉ chốt kết quả đang xem. Planner có đủ 8 tổ hợp: từng kỹ thuật Rewrite, Multi-query, HyDE, ba tổ hợp đôi và cả ba kỹ thuật. Preview cho thấy query rewrite, query phụ hoặc câu trả lời HyDE trước khi retrieval chạy. Vector/Hybrid, reranker, nén context và audit cũng hiện đầu ra ngay sau khi chọn. Có thể mở lại kết quả từng bước để trình bày.

Phòng thử nghiệm dừng thủ công giữa các bước để người xem kiểm tra kết quả; phần chat thường điều phối pipeline tự động bằng LangGraph.

Trong lab, child luôn được ghi nhãn parent tương ứng; kết quả dùng số chunk/trang dễ đọc thay cho ID nội bộ. Bước context mở hai bản không nén/có nén cạnh nhau và chỉ gửi bản đã chọn. Bước sinh câu trả lời hiển thị câu trả lời, nguồn và gợi ý ngay tại màn hình trước khi tiếp tục. Mở **Giải thích thuật ngữ, chỉ số, thông số và công thức** để xem định nghĩa Dense/BM25/RRF, công thức tìm kiếm/nén, quota và các thông số cấu hình. Bộ đếm lab không hiển thị thứ tự request.

Các lựa chọn planner mới có thể cần một generation call cho mỗi tổ hợp mới; tổ hợp đã chạy được cache trong lượt thử. Vector và Hybrid dùng chung cache embedding cho cùng query, nên chuyển qua lại chỉ chạy lại retrieval/BM25 cục bộ. Reranker, dựng context và nén context chạy cục bộ; Answer Grading/Citation Verification dùng chung tối đa một generation call cho tổ hợp audit đã chọn.

Phòng thử nghiệm không thêm file vào kho thường trong lúc xem trước và chạy pipeline; index tạm biến mất khi phiên ứng dụng kết thúc. Sau khi hoàn tất, có thể bấm **Lưu file với cách chia đã chọn vào kho thật**. Bước lưu này tạo embedding lần nữa cho collection Chroma bền vững của kho. Số API calls và token của các bước thử được hiển thị trong phiên.

## Giới hạn truy xuất cần trình bày

Tìm kiếm cơ bản lấy top-k chunk gần nhất theo cosine distance nhưng hiện chưa có ngưỡng từ chối đã hiệu chỉnh. Vì vậy, câu hỏi ngoài phạm vi vẫn có thể đưa các chunk gần nhất vào Gemini. Với câu hỏi trỏ rõ mục/điều như “trình bày mục 3.2”, cả hai pipeline có thêm exact heading match và lấy kèm tiêu đề bảng/trang tiếp theo khi nhận diện được để giữ ngữ cảnh cột và giá trị. Nếu tài liệu lặp lại cùng số mục ở nhiều bảng/mẫu, hãy xem nguồn; câu trả lời vẫn phụ thuộc chất lượng parser và model. Distance là chỉ số xếp hạng, không phải xác suất câu trả lời đúng; không nên dùng một ngưỡng cố định như 0.70 cho mọi kho nếu chưa đánh giá bằng câu hỏi có đáp án và câu hỏi ngoài phạm vi. Pipeline nâng cao cũng không tự sửa câu trả lời khi điểm audit thấp.

Nén ngữ cảnh là phép chọn đoạn/dòng dựa trên từ khóa sau khi bỏ một số từ dừng tiếng Việt phổ biến. Khi không tìm được từ khóa nội dung trùng, ứng dụng giữ phần đầu và cuối thay vì chỉ lấy đầu đoạn; với bảng, tiêu đề được giữ và thứ tự các hàng được chọn vẫn theo tài liệu gốc. Đây vẫn là phép nén có mất mát, nên cần mở nội dung nguồn khi trả lời về số liệu/bảng quan trọng.

## RAG Evaluation, Observability và Agentic Design Patterns

Chọn **📊 RAG Evaluation** để chạy cùng một JSONL dataset qua RAG cơ bản, nâng cao hoặc cả hai. Bộ demo 12 case bao gồm tra cứu dữ kiện/số liệu, hỏi thiếu bằng chứng và câu cần hai file; có thể chọn case cụ thể để kiểm soát quota, bật LLM Judge tùy chọn, xem nguồn/trang/chunk và tải kết quả CSV/JSON. Mặc định chọn case đại diện từ nhóm tra cứu, abstention và nhiều nguồn. JSON kèm manifest model, hash corpus/dataset, IDs case và cấu hình chạy để hỗ trợ tái lập.

Các phép đo gồm expected-source hit/recall/precision/MRR, citation ID hợp lệ và có citation, độ chính xác từ chối theo nhãn `answerable`, latency và request/token. Citation check yêu cầu citation ở case trả lời được và câu từ chối ở case không trả lời được; đây vẫn là phép kiểm tra định dạng/policy, chưa xác minh nội dung citation có hỗ trợ từng claim hay không. Đây là metric MVP cho demo: metric nguồn phụ thuộc nhãn file; từ chối dùng heuristic; LLM Judge có thể sai và tốn thêm một generation call mỗi đáp án. So sánh có ý nghĩa nhất khi dùng cùng kho, dataset, top-k và cấu hình. Không dùng điểm tổng hợp làm bằng chứng thống kê nếu số case nhỏ.

Prompt trả lời áp dụng contract chung theo ý định câu hỏi: tra cứu, trình bày/giải thích, tóm tắt, so sánh, liệt kê hoặc tính toán; yêu cầu giữ nguyên số liệu/điều kiện và chỉ dùng citation có trong nguồn. Sau mỗi câu trả lời, UI kiểm tra citation ID có tồn tại và câu trả lời có citation hoặc từ chối rõ ràng. Đây là kiểm tra hình thức, không chứng minh entailment; bật Citation Verification/LLM Judge hoặc đối chiếu nguồn khi cần kiểm tra ngữ nghĩa.

Phoenix tracing là tùy chọn và mặc định tắt. Cài dependency bổ sung rồi bật env để trace Gemini provider calls cùng span retrieval/context/generation/audit/retry. Khi chạy script local, Phoenix lưu dữ liệu ở `data/phoenix`. Gemini vẫn nhận request qua API Google.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-observability.txt
# Terminal 1: mở Phoenix local
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-phoenix.ps1
# Trong .env: đặt PHOENIX_TRACING_ENABLED=true
# Terminal 2: chạy chatbot bằng .\run.ps1
```

Các design patterns có trong demo:

- **Deterministic workflow:** LangGraph điều phối các node planner (tùy chọn), retrieval, context, answer và audit.
- **Bounded corrective retrieval:** context grader đánh giá độ đủ, đề xuất một truy vấn sửa, retry tối đa một lần rồi abstain nếu vẫn yếu. Đây là evaluator-optimizer có giới hạn chi phí, lấy cảm hứng từ corrective RAG; không tuyên bố là triển khai đầy đủ paper CRAG hay Self-RAG.
- **Routing theo rule:** câu hỏi danh mục kho được trả từ SQLite, câu tổng hợp rộng chuyển sang map-reduce toàn kho, câu thông thường dùng RAG top-k. Đây là router xác định, không phải agent tự chọn tool.
- **Function Calling router (opt-in):** Gemini chọn một trong bốn route được allowlist; Python vẫn thực thi nhánh và giới hạn kho hiện hành. Khi tắt, rule router cũ tiếp tục chạy. Nếu tool-router lỗi, app fallback về rule router và ghi lại lỗi.
- **Query decomposition (opt-in):** planner có thể tạo tối đa 3 câu hỏi con khác nhau, gom embedding trong một batch, giữ hit cho từng câu con rồi hợp nhất nguồn trước khi trả lời câu gốc. Đây là decomposition song song có giới hạn; chưa tự tạo hop phụ thuộc theo kết quả trung gian.
- **Reflection loop (opt-in):** audit kiểm tra answer/citation; nếu phát hiện vấn đề, model viết lại đúng một lần từ cùng context rồi audit lại. Nếu audit lỗi hoặc câu trả lời vẫn không đạt, app chặn bản nháp và báo cần kiểm tra/bổ sung tài liệu.
- **Observability spans:** provider instrumentation ghi lời gọi Gemini; span nghiệp vụ theo dõi retrieval, context, generation, audit và evaluation case.
- **Dataset-driven evaluation:** câu hỏi, đáp án tham chiếu và nguồn kỳ vọng được lưu JSONL; baseline và advanced chạy trên cùng case.

Màn hình **🧭 Agentic Patterns** cho chọn pattern và so sánh trên cùng câu hỏi. Hai nhánh giữ nguyên các retrieval options hiện bật ở thiết lập, bao gồm Corrective Retrieval; baseline tắt riêng Function Calling router, decomposition và reflection, còn nhánh thử nghiệm bật đúng lựa chọn của bạn. Màn hình vẽ pipeline dự kiến, rồi hiển thị route thực tế, câu con, audit, nguồn cuối, retry, latency và API/token. Route đếm/liệt kê dùng SQLite; route toàn kho dùng map-reduce nên các bước RAG top-k không chạy. Một lượt so sánh không chứng minh cải thiện tổng quát; dùng **📊 RAG Evaluation** để chạy bộ nhiều case.

Xem [DEMO_GUIDE.md](DEMO_GUIDE.md) để có kịch bản trình bày và [docs/RAG_EVALUATION_OBSERVABILITY.md](docs/RAG_EVALUATION_OBSERVABILITY.md) để biết cách diễn giải metric, chi phí và giới hạn.
