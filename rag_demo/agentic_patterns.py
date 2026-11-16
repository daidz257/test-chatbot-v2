from __future__ import annotations

from typing import Any

from google.genai import types

from .config import GENERATION_MODEL, GENERATION_RPM_LIMIT
from .quota import RequestMeter, call_with_quota_handling, make_client


ROUTE_OPTIONS = {
    "document_count": "Đếm file trong kho",
    "document_list": "Liệt kê file trong kho",
    "all_documents": "Tổng hợp / tìm trên toàn kho",
    "focused_rag": "Hỏi đáp RAG theo top-k",
}


def route_with_function_calling(
    api_key: str,
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    generation_rpm: int = GENERATION_RPM_LIMIT,
    progress=None,
) -> dict[str, Any]:
    """Ask Gemini to choose one bounded, read-only route for the current query."""
    meter = RequestMeter()
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "route": types.Schema(
                type=types.Type.STRING,
                enum=list(ROUTE_OPTIONS),
                description="Chọn đúng một nhánh xử lý có sẵn.",
            ),
            "reason": types.Schema(type=types.Type.STRING),
        },
        required=["route", "reason"],
    )
    declaration = types.FunctionDeclaration(
        name="select_chat_route",
        description=(
            "Chọn một nhánh đọc dữ liệu chỉ-đọc cho câu hỏi: đếm file, liệt kê file, "
            "tổng hợp/tìm trên toàn kho, hoặc hỏi đáp RAG top-k. Không trả lời câu hỏi."
        ),
        parameters=schema,
    )
    recent = "\n".join(
        f"{'Người dùng' if item.get('role') == 'user' else 'Trợ lý'}: "
        f"{str(item.get('content', ''))[:500]}"
        for item in (history or [])[-4:]
    )
    prompt = f"""Bạn là bộ chọn tuyến xử lý cho chatbot tài liệu tiếng Việt.
Chỉ chọn một route trong schema dựa trên yêu cầu hiện tại. Không trả lời nội dung tài liệu.
- document_count: hỏi số lượng file trong kho.
- document_list: hỏi tên/danh sách file trong kho.
- all_documents: yêu cầu tổng hợp toàn kho hoặc tìm tài liệu nào trong kho chứa một chủ đề.
- focused_rag: các câu hỏi còn lại, cần truy xuất một số đoạn liên quan.
Nếu không chắc, chọn focused_rag. Câu hỏi lịch sử chỉ giúp hiểu đại từ; không thay đổi ý định mới nhất.

LỊCH SỬ GẦN ĐÂY:
{recent or '(trống)'}

YÊU CẦU HIỆN TẠI:
{question}
"""
    if progress:
        progress("Function Calling đang chọn một tuyến xử lý trong danh sách cho phép…")
    client = make_client(api_key)
    response = call_with_quota_handling(
        lambda: client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[declaration])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["select_chat_route"],
                    )
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                max_output_tokens=180,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        ),
        purpose="generation",
        model=GENERATION_MODEL,
        rpm=generation_rpm,
        label="Function Calling chọn tuyến chatbot",
        progress=progress,
        meter=meter,
        prompt=prompt,
    )
    function_call = next(
        (
            part.function_call
            for candidate in (response.candidates or [])
            for part in (candidate.content.parts if candidate.content else [])
            if getattr(part, "function_call", None) is not None
        ),
        None,
    )
    if function_call is None or function_call.name != "select_chat_route":
        raise ValueError("Gemini không trả về function call chọn tuyến hợp lệ.")
    args = function_call.args or {}
    route = str(args.get("route", ""))
    if route not in ROUTE_OPTIONS:
        raise ValueError("Function Calling trả về tuyến nằm ngoài danh sách cho phép.")
    return {
        "route": route,
        "label": ROUTE_OPTIONS[route],
        "reason": str(args.get("reason", ""))[:300],
        "function_name": function_call.name,
        "meter": meter.as_dict(),
    }

