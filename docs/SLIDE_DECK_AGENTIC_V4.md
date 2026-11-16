# BỘ TÀI LIỆU THUYẾT TRÌNH BÁO CÁO THỰC TẬP (V4 - AGENTIC UPGRADE)

File trình chiếu PowerPoint thực tế đã được tạo và lưu tại:
👉 `Bao_Cao_Thuc_Tap_RAG_Chatbot_2026-11-16_v4_Agentic.pptx` (Tổng cộng: 17 slides).

Dưới đây là kịch bản và nội dung chi tiết của 3 Slide trọng tâm vừa được chèn vào bộ slide:

---

## 1. SLIDE 04: LANGCHAIN LÀ GÌ & HỆ SINH THÁI TRONG DỰ ÁN

* **Tiêu đề**: LangChain là gì & Hệ sinh thái trong dự án
* **Phụ đề**: Framework kết nối LLM với nguồn dữ liệu ngoài và chiến lược áp dụng có chọn lọc

### Nội dung 3 Khối Card:
1. **Khối 1: LangChain là gì? (Harrison Chase - Cuối 2022)**
   - Framework mã nguồn mở tiêu chuẩn cho các ứng dụng LLM.
   - Cung cấp các abstraction kết nối LLM với Vector DB, SQL, tài liệu và công cụ ngoài.
   - Giải quyết bài toán: LLM thiếu dữ liệu riêng và ngữ cảnh thời gian thực.
2. **Khối 2: Hệ sinh thái LangChain**
   - **LangChain Core**: Document Loaders, TextSplitters, Vectorstore Adapters.
   - **LangChain Chains (LCEL)**: Kết nối chuỗi xử lý 1 chiều (DAG tuần tự).
   - **LangGraph**: Đồ thị trạng thái chuyên biệt cho Agentic AI, hỗ trợ chu trình (Cycles) và lưu vết (Checkpointers).
3. **Khối 3: Chiến lược của Dự án (Tránh bẫy Framework Bloat)**
   - Chỉ dùng: TextSplitter (chunking) + Chroma Adapter (vector store).
   - Dùng LangGraph (StateGraph) để quản lý 2 vòng lặp tự sửa sai.
   - KHÔNG dùng Full LangChain Chains: Gọi trực tiếp SDK google-genai để kiểm soát Quota RPM và gắn OpenTelemetry Spans.

> 🎙️ **Speaker Note:**
> *"Thưa Hội đồng, LangChain ra đời cuối năm 2022 nhằm giải quyết bài toán kết nối mô hình ngôn ngữ với dữ liệu bên ngoài. Trong dự án này, em áp dụng chiến lược 'Component-based': Chỉ tận dụng những thành phần mạnh nhất là TextSplitter và LangGraph StateGraph để điều phối quy trình có chu trình tự sửa sai. Toàn bộ phần gọi mô hình Gemini, em gọi trực tiếp qua SDK gốc của Google để giữ toàn quyền kiểm soát quota RPM và gắn OpenTelemetry Spans truy vết chi tiết."*

---

## 2. SLIDE 09: AGENTIC DESIGN PATTERNS: KHÁI NIỆM & PHÂN LOẠI

* **Tiêu đề**: Agentic Design Patterns: Khái niệm & Phân loại
* **Phụ đề**: Nguồn gốc học thuật từ TS. Andrew Ng (DeepLearning.AI) và nghiên cứu của Anthropic

### Nội dung 4 Mẫu hình:
1. **REFLECTION (Tự đánh giá & Sửa sai)**:
   - *Định nghĩa*: Model tự kiểm tra đầu ra của chính mình và sửa lại.
   - *Cơ chế*: Generator sinh bản nháp $\rightarrow$ Critic đánh giá $\rightarrow$ Generator sửa.
   - *Trong dự án*: Closed-loop Reflection (Node `reflect_answer` tự sửa lỗi trích dẫn và ảo giác).
2. **TOOL USE (Sử dụng công cụ ngoài)**:
   - *Định nghĩa*: LLM tự quyết định gọi API, Database hoặc Search engine.
   - *Cơ chế*: Model phát ra Function Call có cấu trúc (JSON parameters).
   - *Trong dự án*: Function Calling Router tự chọn giữa SQLite, Map-Reduce và RAG top-k.
3. **PLANNING (Lập kế hoạch & Phân rã)**:
   - *Định nghĩa*: Chia nhỏ mục tiêu phức tạp thành các bước thực thi độc lập.
   - *Cơ chế*: Query Decomposition cho các câu hỏi so sánh đa tài liệu.
   - *Trong dự án*: Planner tách tối đa 3 câu con, cào song song bằng 1 batch embedding.
4. **MULTI-AGENT (Hợp tác đa tác tử)**:
   - *Định nghĩa*: Nhiều tác tử phân vai (chuyên gia, giám sát, phản biện).
   - *Đánh đổi*: Rất tốn token, độ trễ cao và khó kiểm soát lỗi.
   - *Trong dự án*: **CHƯA ÁP DỤNG** để tránh overengineering; ưu tiên Bounded Workflow.

> 🎙️ **Speaker Note:**
> *"Về mặt lý thuyết, khái niệm Agentic Design Patterns được TS. Andrew Ng hệ thống hóa vào đầu năm 2024 gồm 4 mẫu hình: Reflection, Tool Use, Planning và Multi-Agent. Dự án của em đã hiện thực hóa 3 mẫu hình đầu tiên dưới dạng Bounded Workflow theo chuẩn nghiên cứu của Anthropic: Có sự tự chủ của mô hình nhưng luôn bị chặn trần số bước lặp (Retry <= 1) để không gây bùng nổ chi phí và độ trễ."*

---

## 3. SLIDE 11: KIẾN TRÚC AGENTIC VỪA CẬP NHẬT: DUAL-LOOP SELF-CORRECTION

* **Tiêu đề**: Kiến trúc Agentic Vừa Cập Nhật: Dual-Loop Self-Correction
* **Phụ đề**: Mô hình 2 vòng lặp tự sửa sai khép kín kết hợp Function Calling Router & Query Decomposition

### Nội dung 4 Khối:
1. **VÒNG LẶP 1: Bounded Corrective Retrieval (CRAG)** *(Kiểm soát Đầu vào ở Tầng Tìm kiếm)*:
   - Evaluator (Context Grader) chấm độ đủ của context trước khi trả lời.
   - Nếu thiếu bằng chứng: Tự sửa truy vấn (Retry Query) và tìm lại tối đa 1 lần.
   - Nếu sau retry vẫn thiếu: Chủ động từ chối (Abstain), không đẩy rác vào LLM.
2. **VÒNG LẶP 2: Closed-loop Reflection (MỚI CẬP NHẬT)** *(Kiểm soát Đầu ra ở Tầng Sinh văn bản)*:
   - Critic (Audit Node) chấm điểm và bóc tách `unsupported_claims` & lỗi citation.
   - Nếu phát hiện ảo giác: Kích hoạt node `reflect_answer` sửa lại 1 lần từ feedback.
   - Re-Audit lần 2: Bắt buộc thẩm định lại bản sửa; nếu vẫn sai thì chặn bản nháp và Abstain.
3. **PATTERN MỚI: Function Calling Router** *(Tool Selection thay thế hoàn toàn Regex)*:
   - Model tự chọn 1 trong 4 routes: `document_count`, `document_list`, `all_documents`, `focused_rag`.
   - Tự động chuyển câu hỏi đếm/liệt kê file sang SQLite (0 token RAG, trả kết quả <100ms).
   - Cơ chế an toàn: Tự động fallback về Regex nếu gọi API gặp sự cố.
4. **PATTERN MỚI: Query Decomposition** *(Phân rã câu hỏi cho bài toán Multi-hop)*:
   - Nhận diện câu hỏi so sánh đa tài liệu để chia thành tối đa 3 câu con tự chứa.
   - Bảo tồn thực thể (Subject Anchoring) và gộp trong 1 batch embedding duy nhất.
   - Tổng hợp hit của từng câu con trước khi rerank, giải quyết triệt để bài toán thiếu nguồn.

> 🎙️ **Speaker Note:**
> *"Điểm đột phá nhất trong kiến trúc của em là mô hình Hai vòng lặp tự sửa sai khép kín (Dual-Loop Self-Correction): Vòng 1 kiểm soát tầng tìm kiếm dữ liệu (Corrective Retrieval), còn Vòng 2 kiểm soát tầng sinh văn bản (Reflection & Re-Audit). Cùng với Function Calling Router ở cổng vào và Query Decomposition ở tầng lập kế hoạch, hệ thống đã giải quyết trọn vẹn bài toán chống ảo giác từ đầu vào đến đầu ra."*
