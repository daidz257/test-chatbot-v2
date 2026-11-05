# RAG Evaluation, Observability và Agentic Workflow

## Mục tiêu demo

Trình bày thay đổi chất lượng bằng dữ liệu thay vì chỉ bật nhiều kỹ thuật rồi so một câu trả lời. Bộ evaluation giữ cố định câu hỏi, kho, nhãn đáp án/nguồn và có thể so sánh RAG cơ bản, cấu hình nâng cao và multi-agent.

## Chạy benchmark

1. Chọn kho có chứa các tài liệu được liệt kê trong `evals/demo_dataset.jsonl`, hoặc upload JSONL riêng phù hợp với kho đó.
2. Mở khu vực **RAG Evaluation**.
3. Chọn một hoặc nhiều pipeline: RAG cơ bản, nâng cao, Multi-agent RAG; để số case thấp lúc đầu nhằm kiểm soát quota. Khi bật Multi-agent, mỗi lượt giới hạn tối đa 3 case.
4. Xem ước tính request trước khi chạy. Planner/audit/corrective retrieval của advanced và số specialist/Critic của multi-agent được tính vào ước tính tối đa.
5. Chạy evaluation; xem bảng tổng hợp, rồi mở case để đối chiếu đáp án tham chiếu, nguồn kỳ vọng, file/trang/chunk đã truy xuất và câu trả lời.
6. Tải JSON/CSV làm artifact cho buổi review. JSON giữ metrics/diagnostics theo case; CSV thuận tiện cho bảng tính.

Bộ demo được viết cho hai tài liệu mẫu trong kho hiện tại: CV và báo giá website. Nếu file không có trong kho đang chọn, giao diện cảnh báo và metric retrieval không còn so sánh công bằng. Hãy sửa/upload một dataset khớp tài liệu của kho trước khi diễn giải số liệu.

## Dataset JSONL

Mỗi dòng là một case:

```json
{"id":"price-1","question":"Giá gói X là bao nhiêu?","reference_answer":"Giá là ...","expected_sources":["quote.pdf"],"answerable":true,"category":"numeric lookup"}
```

- `id`: duy nhất trong file.
- `question`: câu hỏi sẽ gửi lần lượt vào từng pipeline, không có lịch sử hội thoại trước.
- `reference_answer`: đáp án tham chiếu để người xem/LLM Judge đối chiếu; cần có với case trả lời được.
- `expected_sources`: tên file được gán thủ công là nguồn cần thiết. Case không trả lời được vẫn có thể kỳ vọng truy xuất đúng file nhưng câu trả lời phải từ chối suy đoán.
- `answerable`: nhãn của người tạo dataset, không do model tự sinh.
- `category`: phân nhóm tùy chọn, ví dụ `numeric lookup`, `table`, `multi-document`, `abstention`.
- `expected_agents`: tùy chọn; danh sách specialist ID được đánh giá là phù hợp, ví dụ `["table_analyst"]`. Đây là nhãn thủ công và chỉ được dùng tính routing metrics cho Multi-agent RAG.

## Metric trong giao diện

| Metric | Cách tính | Giới hạn |
| --- | --- | --- |
| Expected source hit | Có ít nhất một file kỳ vọng trong các file được truy xuất; so tên file sau Unicode NFKC, chuẩn khoảng trắng/underscore và không phân biệt hoa/thường | Không đánh giá thứ tự hoặc chất lượng đoạn bên trong file |
| Expected source recall | Số file kỳ vọng đã tìm thấy / số file kỳ vọng | Cần gán source labels chính xác |
| Retrieved document precision | File truy xuất khớp nhãn kỳ vọng / file duy nhất truy xuất được | Một file liên quan nhưng không gán nhãn vẫn bị tính là không đúng |
| Citation ID check | Mọi citation được phát ra đều thuộc nguồn đã trả về; nhận dạng cả dạng gộp như `[S1, S2]` | Không chứng minh câu được citation hỗ trợ về ngữ nghĩa |
| Citation check pass | Có citation hợp lệ ở câu trả lời được, hoặc câu từ chối hợp lệ ở case không trả lời được | Kiểm tra sự hiện diện/mã citation; chưa kiểm chứng claim-to-span đầy đủ |
| Correct abstention | Cụm từ từ chối khớp với `answerable=false`, hoặc không từ chối khi case có đáp án | Heuristic dò cụm từ tiếng Việt/Anh; cần xem thủ công |
| Latency/API/tokens | Lấy từ diagnostics `RequestMeter`, gồm token metadata hoặc ước tính | Request retry cũng được tính; token có thể ước tính theo ký tự khi Gemini không trả usage metadata |
| LLM Judge | Gemini chấm correctness, groundedness và answer relevance 1–5 với reference/context | Tốn thêm một generation call mỗi output; model judge có thiên lệch và không phải ground truth |
| Agent routing precision/recall/F1 | So tập `expected_agents` với ID specialist do Supervisor chọn | Chỉ có khi dataset gán expected_agents; lựa chọn agent không khẳng định câu trả lời đúng |

Benchmark nhỏ có ích cho regression/demo, không đủ để khẳng định cải thiện có ý nghĩa thống kê. Để so sánh công bằng, dùng cùng kho, dataset, top-k, model và phiên bản parser. Với bảng hoặc câu hỏi nhiều đoạn, thêm nhiều câu hỏi và spot-check context/source thủ công. Cân nhắc chạy lại một số case để nhận ra biến thiên của LLM.

Các phép đo hiện tại là custom MVP, chưa gọi thư viện Ragas. `expected source recall/precision` trong app là khớp tên file theo nhãn dataset; không đồng nghĩa hoàn toàn với metric Context Precision/Recall trong Ragas. LLM Judge có các chiều groundedness và answer relevance gần với ý tưởng RAG Triad của TruLens, nhưng app không triển khai toàn bộ RAG Triad hay các evaluator của TruLens.

## Tracing bằng Phoenix local

Phoenix là optional. Mặc định app không khởi tạo exporter, không mở cổng mới và không ghi trace. Cài dependency riêng:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-observability.txt
```

Trong `.env`, đặt:

```dotenv
PHOENIX_TRACING_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=rag-demo-intern
```

Mở terminal thứ nhất và chạy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-phoenix.ps1
```

Sau đó chạy Streamlit từ terminal thứ hai bằng `.un.ps1`. Phoenix UI mặc định ở `http://localhost:6006`; app xuất OTLP traces đến endpoint trong `.env`. Script đặt `PHOENIX_WORKING_DIR` thành `data/phoenix` để dữ liệu Phoenix của demo nằm cạnh dữ liệu app.

Instrument cả Google GenAI provider và LangChain orchestration, đồng thời tạo span cấp nghiệp vụ cho query planning, retrieval (kèm attempt), context grading/build, answer generation, audit, từng evaluation case và các node agent (`agent.supervisor.route`, `agent.specialist`, `agent.synthesis`, `agent.critic_citation_check`). Điều này phân biệt thời gian/lỗi provider với node retrieval/context/agent. Một lượt chat cơ bản có thể chỉ hiện các span provider vì các node cơ bản chưa phải toàn bộ custom spans.

Trace có thể chứa prompt, câu hỏi, nội dung chunk và câu trả lời. Chỉ bật trong buổi demo local khi phù hợp; việc lưu trace local không làm cho request Gemini trở thành xử lý local. Việc gọi model vẫn đi qua API Google.

## Agentic design patterns đang trình bày

### 1. Deterministic graph / workflow

LangGraph làm bộ điều phối luồng xác định: planner tùy chọn → retrieval → context build/compression → answer → audit tùy chọn. Các nhánh là cấu hình rõ, có trạng thái và meter; đây là workflow được điều phối bằng graph, không phải agent tự do gọi tool không giới hạn.

### 2. Bounded corrective retrieval

Khi bật công tắc **Tự đánh giá nguồn và tìm lại tối đa 1 lần**:

1. Sau retrieval, app dựng context cuối cùng, gồm dense-anchor được giữ lại và nén nếu bật; Gemini judge đánh giá đúng context sẽ được gửi ở bước trả lời.
2. Nếu chưa đủ và judge đưa ra truy vấn sửa, app embed/truy xuất lại đúng một lần.
3. Nếu lượt thứ hai vẫn bị đánh giá thiếu, pipeline abstain trước answer generation; các nguồn truy xuất cuối vẫn được giữ lại để xem.
4. Nếu judge lỗi, app ghi lỗi vào diagnostics rồi fail-open để không biến judge thành điểm lỗi bắt buộc của chatbot.

Chi phí tối đa theo cấu hình: hai context-grade generation calls và thêm một embedding batch; khi không cần retry có thể dừng sau lần grade đầu. Cờ tắt mặc định, trừ preset **Bật mọi kỹ thuật**. Đây là corrective retrieval lấy cảm hứng từ evaluator-optimizer/CRAG, không phải bản triển khai CRAG đầy đủ: không có retrieval evaluator độc lập đã hiệu chuẩn, web search hay nguồn sửa ngoài kho. Cũng không phải Self-RAG, vốn là hướng huấn luyện model bằng reflection tokens.

### 3. Vì sao đây là workflow agentic có giới hạn

LangGraph ở đây quản lý trạng thái, thứ tự node và nhánh retry/abstain. Model chỉ được dùng ở các điểm cụ thể (plan, context grade, answer, audit); nó không tự tạo tool mới hoặc lặp theo ý muốn. Đây là lựa chọn phù hợp demo vì chi phí, request cap và điều kiện kết thúc đọc được trên sơ đồ và trace. Khi cần mở rộng, có thể thêm node query decomposition, alternate retriever hoặc human review, rồi đo từng thay đổi bằng cùng eval dataset.

### 4. Multi-agent supervisor / worker

Khu vực **🧭 Kịch bản sản phẩm** và lựa chọn **Multi-agent** trong chat dùng graph riêng:

```text
Retrieve một lần → Supervisor route → Specialists → Synthesis → Critic → citation ID check
```

Supervisor chỉ chọn trong agent allowlist; tối đa 1–3 specialists. Mỗi specialist có role cụ thể: bằng chứng, bảng, đối chiếu tài liệu, yêu cầu hoặc rủi ro/thiếu dữ liệu. Các worker chạy tuần tự theo RPM gate, nhận cùng context và không tự tạo thêm retrieval. Synthesis gộp findings thành câu trả lời/gợi ý; Critic kiểm tra với nguồn gốc và có thể sửa câu trả lời một lần. Không có agent loop vô hạn hay tự do gọi tool.

Hai specialist và Critic bật tương đương tối đa 5 generation calls cộng 1 embedding batch. Evaluation giới hạn 3 case nếu chọn pipeline này và cho phép giảm worker/tắt Critic. Lịch sử scenario lưu snapshot cấu hình; evaluation run lưu hash dataset, flags, metrics và kết quả trong SQLite. Tab overview trong app giải thích LangChain/LangGraph, role của từng agent và chi phí.

`expected_agents` là optional JSONL label do người tạo dataset khai báo. Khi có nhãn, UI tính precision/recall/F1 của route; metric này chỉ đánh giá chọn specialist, không thay thế source recall, citation, correctness hoặc spot-check. Kết quả multi-agent có thể tốn token hơn RAG nâng cao; cần nhìn cùng retrieval/source/answer quality, latency và usage.

## Tài liệu tham khảo

- [Ragas — metrics overview](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [Ragas — experiments](https://docs.ragas.io/en/latest/concepts/experimentation/)
- [TruLens — RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- [Phoenix — self-hosting](https://arize.com/docs/phoenix/self-hosting/deploying-phoenix)
- [Phoenix — tracing integrations](https://arize.com/docs/ax/integrations)
- [OpenInference — supported instrumentation packages](https://github.com/Arize-ai/openinference)
- [Anthropic — building effective agents](https://www.anthropic.com/research/building-effective-agents)
- [Corrective RAG paper](https://arxiv.org/abs/2401.15884)
- [Self-RAG paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)

## Kịch bản trình bày nhanh

1. Chạy RAG Evaluation với vài case và cả hai pipeline.
2. Tìm một câu advanced cải thiện source recall, sau đó chỉ ra chi phí/latency tương ứng; đừng chọn riêng kết quả đẹp nhất mà bỏ qua case xấu.
3. Bật LLM Judge cho 1–2 case và giải thích đó là model đánh giá model, nên cần kiểm tra thủ công.
4. Bật Corrective Retrieval, chọn case bị thiếu nguồn và mở diagnostics để xem số lần retrieval, query sửa, quyết định đủ/thiếu và việc abstain.
5. Bật Phoenix local, hỏi lại một câu, mở trace để cho thấy thời gian theo node/provider và payload/citations.
6. Nêu rõ trade-off: observability giúp tìm nút lỗi; evaluation đo tập case đã gán nhãn; corrective loop tăng khả năng sửa retrieval nhưng tiêu thụ quota và judge có thể nhận định sai.
