import pptx
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    src_file = "Bao_Cao_Thuc_Tap_RAG_Chatbot_2026-11-13_v3.pptx"
    out_file = "Bao_Cao_Thuc_Tap_RAG_Chatbot_2026-11-16_v4_Agentic.pptx"
    prs = pptx.Presentation(src_file)

    blank_layout = prs.slide_layouts[5]

    # Helper to add standard header & slide number
    def add_header(slide, title_text, subtitle_text, slide_num):
        # Title
        tb_title = slide.shapes.add_textbox(Emu(685800), Emu(438150), Emu(10668000), Emu(628650))
        tf = tb_title.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(25)
        p.font.bold = True
        p.font.color.rgb = RGBColor(0x10, 0x2A, 0x43)

        # Accent line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(685800), Emu(1200150), Emu(704850), Emu(47625))
        line.fill.solid()
        line.fill.fore_color.rgb = RGBColor(0x0F, 0x6B, 0x78)
        line.line.fill.background()

        # Subtitle
        tb_sub = slide.shapes.add_textbox(Emu(685800), Emu(1409700), Emu(10668000), Emu(342900))
        tf_sub = tb_sub.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.name = "Arial"
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = RGBColor(0x52, 0x60, 0x6D)

        # Slide Number
        tb_num = slide.shapes.add_textbox(Emu(11010900), Emu(6362700), Emu(495300), Emu(228600))
        tf_num = tb_num.text_frame
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{slide_num:02d}"
        p_num.font.name = "Arial"
        p_num.font.size = Pt(10)
        p_num.font.color.rgb = RGBColor(0x82, 0x9A, 0xB1)

    # Helper to add a card
    def add_card(slide, left, top, width, height, title, subtitle="", bullets=None, bg_rgb=RGBColor(0xF0, 0xF4, 0xF8), border_rgb=RGBColor(0xDC, 0xE2, 0xEC), title_rgb=RGBColor(0x10, 0x2A, 0x43)):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(left), Emu(top), Emu(width), Emu(height))
        card.fill.solid()
        card.fill.fore_color.rgb = bg_rgb
        card.line.color.rgb = border_rgb
        card.line.width = Pt(1)

        tb = slide.shapes.add_textbox(Emu(left + 180000), Emu(top + 160000), Emu(width - 360000), Emu(height - 320000))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = title_rgb

        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(10.5)
            p_sub.font.bold = True
            p_sub.font.color.rgb = RGBColor(0x0F, 0x6B, 0x78)
            p_sub.space_before = Pt(4)

        if bullets:
            for b in bullets:
                p_b = tf.add_paragraph()
                p_b.text = f"• {b}"
                p_b.font.name = "Arial"
                p_b.font.size = Pt(9.5)
                p_b.font.color.rgb = RGBColor(0x33, 0x4E, 0x68)
                p_b.space_before = Pt(3)

    # Slide A: LangChain Overview
    slide_lc = prs.slides.add_slide(blank_layout)
    add_header(slide_lc, "LangChain là gì & Hệ sinh thái trong dự án", 
               "Framework kết nối LLM với nguồn dữ liệu ngoài và chiến lược áp dụng có chọn lọc", 15)
    
    add_card(slide_lc, 685800, 2000000, 3400000, 3900000, "1. LangChain là gì?", "Ra đời cuối năm 2022 (Harrison Chase)", [
        "Framework mã nguồn mở tiêu chuẩn cho các ứng dụng LLM.",
        "Cung cấp các abstraction kết nối LLM với Vector DB, SQL, tài liệu và công cụ ngoài.",
        "Giải quyết bài toán: LLM thiếu dữ liệu riêng và ngữ cảnh thời gian thực."
    ])
    add_card(slide_lc, 4350000, 2000000, 3400000, 3900000, "2. Hệ sinh thái LangChain", "3 trụ cột công nghệ chính", [
        "LangChain Core: Document Loaders, TextSplitters, Vectorstore Adapters.",
        "LangChain Chains (LCEL): Kết nối chuỗi xử lý 1 chiều (DAG tuần tự).",
        "LangGraph: Đồ thị trạng thái chuyên biệt cho Agentic AI, hỗ trợ chu trình (Cycles) và lưu vết (Checkpointers)."
    ])
    add_card(slide_lc, 8010000, 2000000, 3480000, 3900000, "3. Chiến lược của Dự án", "Tránh bẫy Framework Bloat", [
        "Chỉ dùng: TextSplitter (chunking) + Chroma Adapter (vector).",
        "Dùng LangGraph (StateGraph) để quản lý 2 vòng lặp tự sửa sai.",
        "KHÔNG dùng Full LangChain Chains: Gọi trực tiếp SDK google-genai để kiểm soát Quota RPM và gắn OpenTelemetry Spans."
    ], bg_rgb=RGBColor(0xEE, 0xF8, 0xF6), border_rgb=RGBColor(0x99, 0xD4, 0xCE), title_rgb=RGBColor(0x0F, 0x6B, 0x78))

    # Slide B: Agentic Design Patterns (Andrew Ng / Anthropic)
    slide_ap = prs.slides.add_slide(blank_layout)
    add_header(slide_ap, "Agentic Design Patterns: Khái niệm & Phân loại", 
               "Nguồn gốc học thuật từ TS. Andrew Ng (DeepLearning.AI) và nghiên cứu của Anthropic", 16)
    
    add_card(slide_ap, 685800, 1950000, 2550000, 4000000, "1. REFLECTION", "Tự đánh giá & Sửa sai", [
        "Định nghĩa: Model tự kiểm tra đầu ra của chính mình và sửa lại.",
        "Cơ chế: Generator sinh bản nháp -> Critic đánh giá -> Generator sửa.",
        "Trong dự án: Closed-loop Reflection (Node reflect_answer tự sửa lỗi trích dẫn/ảo giác)."
    ], bg_rgb=RGBColor(0xEE, 0xF8, 0xF6), title_rgb=RGBColor(0x0F, 0x6B, 0x78))

    add_card(slide_ap, 3450000, 1950000, 2550000, 4000000, "2. TOOL USE", "Sử dụng công cụ ngoài", [
        "Định nghĩa: LLM tự quyết định gọi API, Database hoặc Search engine.",
        "Cơ chế: Model phát ra Function Call có cấu trúc (JSON parameters).",
        "Trong dự án: Function Calling Router tự chọn giữa SQLite, Map-Reduce và RAG top-k."
    ], bg_rgb=RGBColor(0xEE, 0xF8, 0xF6), title_rgb=RGBColor(0x0F, 0x6B, 0x78))

    add_card(slide_ap, 6210000, 1950000, 2550000, 4000000, "3. PLANNING", "Lập kế hoạch & Phân rã", [
        "Định nghĩa: Chia nhỏ mục tiêu phức tạp thành các bước thực thi độc lập.",
        "Cơ chế: Query Decomposition cho các câu hỏi so sánh đa tài liệu.",
        "Trong dự án: Planner tách tối đa 3 câu con, cào song song bằng 1 batch embedding."
    ], bg_rgb=RGBColor(0xEE, 0xF8, 0xF6), title_rgb=RGBColor(0x0F, 0x6B, 0x78))

    add_card(slide_ap, 8970000, 1950000, 2520000, 4000000, "4. MULTI-AGENT", "Hợp tác đa tác tử", [
        "Định nghĩa: Nhiều tác tử phân vai (chuyên gia, giám sát, phản biện).",
        "Đánh đổi: Rất tốn token, độ trễ cao và khó kiểm soát lỗi.",
        "Trong dự án: CHƯA ÁP DỤNG để tránh overengineering; ưu tiên Bounded Workflow."
    ], bg_rgb=RGBColor(0xF8, 0xF9, 0xFA), title_rgb=RGBColor(0x62, 0x7D, 0x98))

    # Slide C: Chi tiết 3 Pattern mới vừa cập nhật & Dual Loop
    slide_new = prs.slides.add_slide(blank_layout)
    add_header(slide_new, "Kiến trúc Agentic Vừa Cập Nhật: Dual-Loop Self-Correction", 
               "Mô hình 2 vòng lặp tự sửa sai khép kín kết hợp Function Calling Router & Query Decomposition", 17)
    
    add_card(slide_new, 685800, 1950000, 5200000, 2050000, "VÒNG LẶP 1: Bounded Corrective Retrieval (CRAG)", "Kiểm soát Đầu vào ở Tầng Tìm kiếm", [
        "Evaluator (Context Grader) chấm độ đủ của context trước khi trả lời.",
        "Nếu thiếu bằng chứng: Tự sửa truy vấn (Retry Query) và tìm lại tối đa 1 lần.",
        "Nếu sau retry vẫn thiếu: Chủ động từ chối (Abstain), không đẩy rác vào LLM."
    ], bg_rgb=RGBColor(0xFB, 0xF5, 0xEA), border_rgb=RGBColor(0xF0, 0xDC, 0xB4), title_rgb=RGBColor(0xB4, 0x53, 0x09))

    add_card(slide_new, 6300000, 1950000, 5200000, 2050000, "VÒNG LẶP 2: Closed-loop Reflection (MỚI CẬP NHẬT)", "Kiểm soát Đầu ra ở Tầng Sinh văn bản", [
        "Critic (Audit Node) chấm điểm và bóc tách unsupported_claims & lỗi citation.",
        "Nếu phát hiện ảo giác: Kích hoạt node reflect_answer sửa lại 1 lần từ feedback.",
        "Re-Audit lần 2: Bắt buộc thẩm định lại bản sửa; nếu vẫn sai thì chặn bản nháp và Abstain."
    ], bg_rgb=RGBColor(0xEE, 0xF8, 0xF6), border_rgb=RGBColor(0x99, 0xD4, 0xCE), title_rgb=RGBColor(0x0F, 0x6B, 0x78))

    add_card(slide_new, 685800, 4200000, 5200000, 1900000, "PATTERN MỚI: Function Calling Router", "Tool Selection thay thế hoàn toàn Regex", [
        "Model tự chọn 1 trong 4 routes: document_count, document_list, all_documents, focused_rag.",
        "Tự động chuyển câu hỏi đếm/liệt kê file sang SQLite (0 token RAG, trả kết quả <100ms).",
        "Cơ chế an toàn: Tự động fallback về Regex nếu gọi API gặp sự cố."
    ])

    add_card(slide_new, 6300000, 4200000, 5200000, 1900000, "PATTERN MỚI: Query Decomposition", "Phân rã câu hỏi cho bài toán Multi-hop", [
        "Nhận diện câu hỏi so sánh đa tài liệu để chia thành tối đa 3 câu con tự chứa.",
        "Bảo tồn thực thể (Subject Anchoring) và gộp trong 1 batch embedding duy nhất.",
        "Tổng hợp hit của từng câu con trước khi rerank, giải quyết triệt để bài toán thiếu nguồn."
    ])

    # Reorder slides so that Slide LC is at index 3, Slide AP is at index 8, Slide New is at index 10
    sldIdLst = prs.slides._sldIdLst
    s_new_id = sldIdLst[-1]
    s_ap_id = sldIdLst[-2]
    s_lc_id = sldIdLst[-3]

    sldIdLst.remove(s_lc_id)
    sldIdLst.insert(3, s_lc_id)

    sldIdLst.remove(s_ap_id)
    sldIdLst.insert(8, s_ap_id)

    sldIdLst.remove(s_new_id)
    sldIdLst.insert(10, s_new_id)

    # Re-number all slide numbers in the footer
    for index, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.has_text_frame and shape.left > Emu(10500000) and shape.top > Emu(6000000):
                shape.text_frame.text = f"{index:02d}"

    prs.save(out_file)
    print(f"Successfully generated {out_file} with {len(prs.slides)} slides!")

if __name__ == "__main__":
    create_deck()
