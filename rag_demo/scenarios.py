from __future__ import annotations

from typing import Any

from .multi_agent import AGENT_REGISTRY


SCENARIO_FIELD_TYPES = {
    "text": "Câu trả lời ngắn",
    "textarea": "Câu trả lời dài",
    "choice": "Chọn một phương án",
    "documents": "Chọn tài liệu trong kho",
}


def demo_cv_scenario() -> dict[str, Any]:
    return {
        "name": "Phân tích CV và đối chiếu yêu cầu tuyển dụng",
        "description": (
            "Thu thập CV và tiêu chí tuyển dụng, đối chiếu dữ kiện có trích dẫn, nêu điểm phù hợp, "
            "điểm còn thiếu và câu hỏi cần xác minh."
        ),
        "steps": [
            {
                "id": "documents",
                "label": "Chọn tài liệu đầu vào",
                "help": "Chọn CV và mô tả công việc đã thêm vào kho tài liệu.",
                "type": "documents",
                "required": True,
                "options": [],
            },
            {
                "id": "position",
                "label": "Vị trí cần tuyển",
                "help": "Ví dụ: Backend Intern. Có thể nêu mức độ kinh nghiệm mong muốn.",
                "type": "text",
                "required": True,
                "options": [],
            },
            {
                "id": "requirements",
                "label": "Tiêu chí cần đối chiếu",
                "help": "Liệt kê kỹ năng, học vấn, kinh nghiệm hoặc yêu cầu bắt buộc.",
                "type": "textarea",
                "required": True,
                "options": [],
            },
            {
                "id": "report_style",
                "label": "Dạng báo cáo mong muốn",
                "help": "Chọn cách trình bày để người tuyển dụng dễ đánh giá.",
                "type": "choice",
                "required": True,
                "options": ["Bảng điểm phù hợp", "Tóm tắt theo tiêu chí", "Danh sách câu hỏi phỏng vấn"],
            },
        ],
        "agent_ids": ["evidence_analyst", "requirements_analyst", "risk_analyst", "table_analyst"],
        "is_active": True,
    }


def validate_scenario(
    name: str, description: str, steps: list[dict[str, Any]], agent_ids: list[str]
) -> list[dict[str, Any]]:
    if not name.strip():
        raise ValueError("Hãy đặt tên cho kịch bản.")
    if not steps:
        raise ValueError("Kịch bản cần ít nhất một bước.")
    if len(steps) > 10:
        raise ValueError("Một kịch bản hỗ trợ tối đa 10 bước.")
    cleaned_steps: list[dict[str, Any]] = []
    seen_documents_step = False
    for index, raw in enumerate(steps, start=1):
        label = str(raw.get("label", "")).strip()
        kind = str(raw.get("type", "text"))
        options = [str(value).strip() for value in raw.get("options", []) if str(value).strip()]
        if not label:
            raise ValueError(f"Bước {index}: cần có tên bước/câu hỏi.")
        if kind not in SCENARIO_FIELD_TYPES:
            raise ValueError(f"Bước {index}: loại trường không hợp lệ.")
        if kind == "choice" and len(options) < 2:
            raise ValueError(f"Bước {index}: lựa chọn cần ít nhất 2 phương án.")
        if kind == "documents":
            if seen_documents_step:
                raise ValueError("Kịch bản chỉ hỗ trợ một bước chọn tài liệu để giới hạn rõ phạm vi truy xuất.")
            seen_documents_step = True
        cleaned_steps.append(
            {
                "id": f"step-{index}",
                "label": label[:120],
                "help": str(raw.get("help", "")).strip()[:700],
                "type": kind,
                "required": bool(raw.get("required", True)),
                "options": options[:20] if kind == "choice" else [],
            }
        )
    valid_agents = list(dict.fromkeys(agent_id for agent_id in agent_ids if agent_id in AGENT_REGISTRY))
    if not valid_agents:
        raise ValueError("Hãy chọn ít nhất một chuyên viên AI.")
    return cleaned_steps
