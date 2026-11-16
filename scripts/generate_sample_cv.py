from __future__ import annotations

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def build_sample_cv(output_path: str = "CV_Backend_AI_Engineer_Le_Hoang_Nam.pdf") -> str:
    """Generate a realistic, professional, 1-page Vietnamese technical resume."""
    # Register Arial fonts for full Vietnamese Unicode support
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    name_style = ParagraphStyle(
        "NameStyle",
        fontName="Arial-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
    )

    title_style = ParagraphStyle(
        "TitleStyle",
        fontName="Arial-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2563EB"),
    )

    contact_style = ParagraphStyle(
        "ContactStyle",
        fontName="Arial",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4B5563"),
    )

    h1_style = ParagraphStyle(
        "H1Style",
        fontName="Arial-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4,
    )

    sub_style = ParagraphStyle(
        "SubStyle",
        fontName="Arial-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#111827"),
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        fontName="Arial",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#374151"),
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        fontName="Arial",
        fontSize=8.5,
        leading=12,
        leftIndent=12,
        firstLineIndent=-8,
        textColor=colors.HexColor("#374151"),
    )

    story = []

    # Header Table
    left_header = [
        Paragraph("LÊ HOÀNG NAM", name_style),
        Spacer(1, 3),
        Paragraph("SENIOR BACKEND & AI ENGINEER", title_style),
        Spacer(1, 4),
        Paragraph(
            "Kỹ sư phần mềm 3+ năm kinh nghiệm phát triển hệ thống backend chịu tải cao, "
            "Microservices và giải pháp AI/RAG ứng dụng.",
            body_style,
        ),
    ]

    right_header = [
        Paragraph("<b>Email:</b> lehoangnam.dev@gmail.com", contact_style),
        Paragraph("<b>Điện thoại:</b> 0987.654.321", contact_style),
        Paragraph("<b>Địa chỉ:</b> Cầu Giấy, Hà Nội", contact_style),
        Paragraph("<b>GitHub:</b> github.com/namlh-tech", contact_style),
        Paragraph("<b>LinkedIn:</b> linkedin.com/in/namlh-backend", contact_style),
    ]

    header_table = Table([[left_header, right_header]], colWidths=[330, 190])
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#1E3A8A"),
            spaceAfter=8,
            spaceBefore=2,
        )
    )

    # 1. KỸ NĂNG CHUYÊN MÔN
    story.append(Paragraph("KỸ NĂNG CHUYÊN MÔN (TECHNICAL SKILLS)", h1_style))
    skills_data = [
        [
            Paragraph("<b>Ngôn ngữ lập trình:</b>", body_style),
            Paragraph("Python (Chuyên sâu), Go, SQL, Bash script, JavaScript/TypeScript cơ bản", body_style),
        ],
        [
            Paragraph("<b>Framework & Web:</b>", body_style),
            Paragraph("FastAPI, Django REST Framework, Flask, Celery, gRPC", body_style),
        ],
        [
            Paragraph("<b>AI / GenAI & RAG:</b>", body_style),
            Paragraph("LangChain, LangGraph, LlamaIndex, Gemini API, OpenAI API, HuggingFace", body_style),
        ],
        [
            Paragraph("<b>Cơ sở dữ liệu & Cache:</b>", body_style),
            Paragraph("PostgreSQL, MySQL, Redis, ChromaDB, Qdrant, Milvus, MongoDB", body_style),
        ],
        [
            Paragraph("<b>DevOps & Cloud:</b>", body_style),
            Paragraph("Docker, Docker Compose, Kubernetes cơ bản, AWS (EC2, S3, RDS), GitLab CI/CD", body_style),
        ],
        [
            Paragraph("<b>Kiến trúc & Phương pháp:</b>", body_style),
            Paragraph("Microservices, Event-driven (Kafka, RabbitMQ), Clean Architecture, RESTful API", body_style),
        ],
    ]
    skills_table = Table(skills_data, colWidths=[140, 380])
    skills_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(skills_table)
    story.append(Spacer(1, 10))

    # 2. KINH NGHIỆM LÀM VIỆC
    story.append(Paragraph("KINH NGHIỆM LÀM VIỆC (WORK EXPERIENCE)", h1_style))

    # Job 1
    j1_header = Table(
        [
            [
                Paragraph("<b>Công ty Cổ phần Giải pháp Công nghệ NextWave</b>", sub_style),
                Paragraph(
                    "<i>06/2022 – Hiện tại | Hà Nội</i>",
                    ParagraphStyle("r", fontName="Arial", fontSize=8.5, alignment=2, textColor=colors.HexColor("#6B7280")),
                ),
            ]
        ],
        colWidths=[340, 180],
    )
    j1_header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(j1_header)
    story.append(
        Paragraph("<i>Vị trí: Senior Backend & AI Engineer</i>", ParagraphStyle("p", fontName="Arial-Italic", fontSize=8.5, textColor=colors.HexColor("#2563EB")))
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "• Thiết kế và triển khai hệ thống <b>Enterprise RAG Chatbot</b> cho 5.000+ nhân viên nội bộ, kết hợp Vector Search (ChromaDB) và BM25 reranking giúp tăng độ chính xác truy xuất tài liệu từ 72% lên 89%.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Xây dựng cụm Microservices với FastAPI, Redis Streams và PostgreSQL; chịu tải thực tế 1.500 requests/giây với độ trễ P95 < 65ms.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Tích hợp hệ thống Observability (OpenTelemetry & Prometheus) theo dõi sát sao token usage, latency và tỷ lệ hallucination của LLM.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Xây dựng pipeline CI/CD trên GitLab CI và triển khai container hóa qua Docker, giảm 60% thời gian triển khai ứng dụng.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 6))

    # Job 2
    j2_header = Table(
        [
            [
                Paragraph("<b>Công ty TNHH Phần mềm TechSol Vietnam</b>", sub_style),
                Paragraph(
                    "<i>08/2021 – 05/2022 | Hà Nội</i>",
                    ParagraphStyle("r2", fontName="Arial", fontSize=8.5, alignment=2, textColor=colors.HexColor("#6B7280")),
                ),
            ]
        ],
        colWidths=[340, 180],
    )
    j2_header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(j2_header)
    story.append(
        Paragraph("<i>Vị trí: Junior Python Backend Developer</i>", ParagraphStyle("p2", fontName="Arial-Italic", fontSize=8.5, textColor=colors.HexColor("#2563EB")))
    )
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "• Tham gia xây dựng RESTful API cho nền tảng CRM và phân hệ quản lý kho bán lẻ sử dụng Django REST Framework.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Tối ưu hóa các câu truy vấn cơ sở dữ liệu PostgreSQL, đánh index hợp lý giúp cải thiện 40% thời gian truy xuất báo cáo doanh thu.",
            bullet_style,
        )
    )
    story.append(
        Paragraph("• Viết unit test và integration test với pytest, duy trì code coverage trên 85%.", bullet_style)
    )
    story.append(Spacer(1, 10))

    # 3. DỰ ÁN TIÊU BIỂU
    story.append(Paragraph("DỰ ÁN TIÊU BIỂU (SELECTED PROJECTS)", h1_style))
    story.append(Paragraph("<b>1. Agentic RAG Platform (LangGraph & Multi-agent Orchestration)</b>", sub_style))
    story.append(
        Paragraph(
            "<i>Công nghệ: Python, FastAPI, LangGraph, ChromaDB, Gemini 2.5 Flash, Arize Phoenix, Docker</i>",
            ParagraphStyle("tech", fontName="Arial-Italic", fontSize=8, textColor=colors.HexColor("#4B5563")),
        )
    )
    story.append(
        Paragraph(
            "• Phát triển luồng Agentic RAG đa bước với khả năng: Router phân luồng câu hỏi, Query Decomposition chia nhỏ truy vấn, CRAG tự đánh giá tài liệu, và Closed-loop Reflection tự sửa câu trả lời trước khi phản hồi người dùng.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Tích hợp Scenario Engine cho phép thẩm định hồ sơ đấu thầu và phân tích báo giá tự động theo khung mẫu có cấu trúc.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 5))

    story.append(Paragraph("<b>2. Hệ thống Xử lý Giao dịch Bán lẻ Thời gian thực (Flash Sale Engine)</b>", sub_style))
    story.append(
        Paragraph(
            "<i>Công nghệ: Go, Python, Redis Streams, PostgreSQL, Docker Compose</i>",
            ParagraphStyle("tech2", fontName="Arial-Italic", fontSize=8, textColor=colors.HexColor("#4B5563")),
        )
    )
    story.append(
        Paragraph(
            "• Thiết kế cơ chế Distributed Locking bằng Redis để xử lý đồng thời 3.000 đơn hàng/giây trong các khung giờ flash sale, đảm bảo tính nhất quán tuyệt đối của tồn kho.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 10))

    # 4. HỌC VẤN & CHỨNG CHỈ
    story.append(Paragraph("HỌC VẤN & CHỨNG CHỈ (EDUCATION & CERTIFICATIONS)", h1_style))
    edu_data = [
        [
            Paragraph("<b>Đại học Bách Khoa Hà Nội (HUST)</b>", body_style),
            Paragraph("Cử nhân Kỹ thuật Công nghệ Thông tin | GPA: 3.42 / 4.0 (Loại Giỏi) | 2018 – 2022", body_style),
        ],
        [
            Paragraph("<b>Chứng chỉ AWS:</b>", body_style),
            Paragraph("AWS Certified Solutions Architect – Associate (SAA-C03)", body_style),
        ],
        [
            Paragraph("<b>Chứng chỉ Tiếng Anh:</b>", body_style),
            Paragraph("TOEIC 850 (Nghe: 440, Đọc: 410) – Giao tiếp và làm việc thành thạo với đối tác quốc tế", body_style),
        ],
    ]
    edu_table = Table(edu_data, colWidths=[150, 370])
    edu_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(edu_table)

    doc.build(story)
    return str(Path(output_path).resolve())


if __name__ == "__main__":
    path = build_sample_cv()
    print(f"Generated sample CV at: {path}")
