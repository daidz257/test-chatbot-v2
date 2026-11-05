from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from .config import GENERATION_MODEL, GENERATION_RPM_LIMIT
from .observability import trace_span
from .quota import RequestMeter, call_with_quota_handling


def transcribe_audio(
    api_key: str,
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    *,
    rpm_limit: int = GENERATION_RPM_LIMIT,
    meter: RequestMeter | None = None,
    progress=None,
) -> dict[str, Any]:
    if not audio_bytes:
        raise ValueError("Không có dữ liệu âm thanh để nhận dạng.")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    prompt = (
        "Chuyển lời nói trong âm thanh thành văn bản tiếng Việt. "
        "Giữ nguyên tên riêng, thuật ngữ, số liệu và ý định đặt câu hỏi. "
        "Không trả lời câu hỏi, không thêm lời dẫn; chỉ trả phần lời nói đã nhận dạng."
    )
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type or "audio/wav")
    with trace_span("rag.voice.transcription", {"gemini.model": GENERATION_MODEL}):
        response = call_with_quota_handling(
            lambda: client.models.generate_content(
                model=GENERATION_MODEL,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            ),
            purpose="generation",
            model=GENERATION_MODEL,
            rpm=rpm_limit,
            label="Nhận dạng giọng nói",
            progress=progress,
            meter=meter,
            prompt=prompt,
        )
    transcript = str(getattr(response, "text", "") or "").strip()
    if not transcript:
        raise ValueError("Không nhận được văn bản từ âm thanh. Hãy thử ghi âm rõ hơn.")
    return {"text": transcript, "model": GENERATION_MODEL}
