from __future__ import annotations

import random
import re
import threading
import time
import json
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_RPM_LIMIT,
    GENERATION_MODEL,
    GENERATION_RPM_LIMIT,
    MAX_RETRIES,
)

ProgressCallback = Callable[[str], None] | None
CHARS_PER_ESTIMATED_TOKEN = 3


class GeminiRequestError(RuntimeError):
    """A Gemini API error with actionable guidance for the local demo user."""


@dataclass
class RequestMeter:
    """Per-answer accounting for API requests and provider-reported/estimated tokens."""

    api_calls: int = 0
    embedding_calls: int = 0
    generation_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_token_calls: int = 0
    labels: list[str] = field(default_factory=list)

    @staticmethod
    def _prompt_chars(prompt: str | list[str] | None) -> int:
        if prompt is None:
            return 0
        if isinstance(prompt, str):
            return len(prompt)
        return sum(len(item) for item in prompt)

    def record_attempt(self, purpose: str, label: str) -> None:
        self.api_calls += 1
        if purpose == "embedding":
            self.embedding_calls += 1
        else:
            self.generation_calls += 1
        self.labels.append(label)

    def record_success(self, purpose: str, prompt: str | list[str] | None, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        input_count = None
        output_count = None
        if usage is not None:
            for name in ("prompt_token_count", "input_token_count", "total_input_tokens"):
                value = getattr(usage, name, None)
                if value is not None:
                    input_count = int(value)
                    break
            for name in ("candidates_token_count", "output_token_count", "total_output_tokens"):
                value = getattr(usage, name, None)
                if value is not None:
                    output_count = int(value)
                    break
        if input_count is None:
            input_count = (
                self._prompt_chars(prompt) + CHARS_PER_ESTIMATED_TOKEN - 1
            ) // CHARS_PER_ESTIMATED_TOKEN
            self.estimated_token_calls += 1
        self.input_tokens += input_count
        if purpose != "embedding":
            if output_count is None:
                response_text = getattr(response, "text", "") or ""
                output_count = (
                    len(response_text) + CHARS_PER_ESTIMATED_TOKEN - 1
                ) // CHARS_PER_ESTIMATED_TOKEN
                self.estimated_token_calls += 1
            self.output_tokens += output_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "embedding_calls": self.embedding_calls,
            "generation_calls": self.generation_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_token_calls": self.estimated_token_calls,
            "token_note": "Ước tính bảo thủ theo ký tự/3 khi request không có usage metadata từ Gemini.",
            "labels": self.labels,
        }


class RPMGate:
    """A process-local request pacer to avoid bursts against a per-minute limit."""

    def __init__(self, rpm: int):
        self.interval = 63.0 / max(1, rpm)
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self, progress: ProgressCallback = None) -> None:
        with self.lock:
            now = time.monotonic()
            wait_for = max(0.0, self.next_request_at - now)
            self.next_request_at = max(now, self.next_request_at) + self.interval
        if wait_for > 0:
            if progress:
                progress(f"Đang tự giãn nhịp gọi Gemini ({wait_for:.1f} giây) để tránh vượt RPM…")
            time.sleep(wait_for)


_gates: dict[tuple[str, str, int], RPMGate] = {}
_gates_lock = threading.Lock()


def _gate(purpose: str, model: str, rpm: int) -> RPMGate:
    key = (purpose, model, rpm)
    with _gates_lock:
        if key not in _gates:
            _gates[key] = RPMGate(rpm)
        return _gates[key]


def make_client(api_key: str) -> genai.Client:
    # Disable the SDK's hidden retry loop so the UI can show one bounded,
    # observable retry policy rather than stacking two retry policies.
    retry = types.HttpRetryOptions(attempts=1)
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(retry_options=retry),
    )


def _status_code(error: Exception) -> int | None:
    for value in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _message(error: Exception) -> str:
    return str(error) or error.__class__.__name__


def _is_permanent_quota(message: str) -> bool:
    lower = message.lower()
    # Gemini may use the generic phrase "quota exceeded" for a transient
    # requests-per-minute limit. Treat rate-window errors as retryable even if
    # the same message also includes that phrase.
    rate_window_terms = (
        "per minute",
        "per-minute",
        "requests/minute",
        "request limit per minute",
        "peak requests",
        "tokens per minute",
        "per second",
        "rate limit",
        "rate_limit_exceeded",
        "spend-based rate limit",
        "spend rate limit",
        "rolling 10-minute",
    )
    if any(token in lower for token in rate_window_terms):
        return False
    return any(
        token in lower
        for token in (
            "per day",
            "per-day",
            "daily quota",
            "requests per day",
            "prepay",
            "billing account",
            "billing limit",
            "insufficient prepay",
        )
    )


def _retry_after(error: Exception, message: str) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw:
            try:
                return min(60.0, max(0.0, float(raw)))
            except (TypeError, ValueError):
                pass
    patterns = (
        r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s(?:ec(?:ond)?s?)?",
        r"retryDelay[\"']?\s*[:=]\s*[\"']([0-9]+(?:\.[0-9]+)?)s",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return min(60.0, max(0.0, float(match.group(1))))
    return None


def call_with_quota_handling(
    operation: Callable[[], Any],
    *,
    purpose: str,
    model: str,
    rpm: int,
    label: str,
    progress: ProgressCallback = None,
    meter: RequestMeter | None = None,
    prompt: str | list[str] | None = None,
) -> Any:
    gate = _gate(purpose, model, rpm)
    last_error: Exception | None = None
    # MAX_RETRIES means retries after the initial request (the conventional
    # interpretation of the environment variable).
    for attempt in range(MAX_RETRIES + 1):
        gate.wait(progress)
        if meter:
            meter.record_attempt(purpose, label)
        try:
            response = operation()
            if meter:
                meter.record_success(purpose, prompt, response)
            return response
        except Exception as error:  # SDK error types vary by API endpoint/version.
            last_error = error
            message = _message(error)
            status = _status_code(error)
            lowered = message.lower()
            transient = status in {408, 429, 500, 502, 503, 504} or any(
                hint in lowered
                for hint in ("resource_exhausted", "rate limit", "requests per minute", "peak requests")
            )

            if not transient:
                if status in {401, 403} or "api key" in lowered or "authentication" in lowered:
                    raise GeminiRequestError(
                        "Gemini từ chối API key hoặc quyền truy cập. Kiểm tra GEMINI_API_KEY "
                        "và model trong Google AI Studio."
                    ) from error
                raise GeminiRequestError(f"{label} thất bại: {message[:700]}") from error

            if _is_permanent_quota(message):
                raise GeminiRequestError(
                    f"{label} đã chạm quota theo ngày/chi tiêu nên chờ vài giây sẽ không giải quyết được. "
                    "Kiểm tra dự án và billing trong Google AI Studio; chờ quota reset hoặc tăng tier/hạn mức. "
                    f"Chi tiết: {message[:350]}"
                ) from error

            if attempt >= MAX_RETRIES:
                break

            delay = _retry_after(error, message)
            if delay is None:
                delay = min(60.0, (2**attempt) + random.uniform(0.15, 0.9))
            if progress:
                progress(
                    f"Gemini đang giới hạn tần suất. Thử lại {attempt + 1}/{MAX_RETRIES} "
                    f"sau {delay:.1f} giây…"
                )
            time.sleep(delay)

    details = _message(last_error)[:350] if last_error else "Không có thông tin lỗi từ API."
    if any(token in details.lower() for token in ("spend-based", "spend rate limit", "rolling 10-minute")):
        guidance = (
            "Có thể đã chạm giới hạn chi tiêu theo cửa sổ thời gian; hãy chờ cửa sổ quota hồi lại và giảm "
            "số token/kích thước batch nếu lỗi lặp lại."
        )
    else:
        guidance = (
            "Hãy kiểm tra đúng project/model ở AI Studio → Rate limits, đặt RPM giới hạn trong .env thấp hơn "
            "hạn mức thật, hoặc chờ cửa sổ quota mở lại. Hạn mức Gemini được tính theo project, không theo API key."
        )
    raise GeminiRequestError(
        f"{label} vẫn gặp giới hạn tạm thời (RPM/quá tải) sau {MAX_RETRIES} lần retry. "
        f"Ứng dụng đã tự giãn nhịp và thử lại có giới hạn. {guidance} "
        f"Chi tiết: {details}"
    ) from last_error


class GeminiEmbeddings:
    """LangChain-compatible embedding adapter with batched, paced Gemini calls."""

    def __init__(
        self,
        api_key: str,
        *,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        rpm_limit: int = EMBEDDING_RPM_LIMIT,
        progress: ProgressCallback = None,
        meter: RequestMeter | None = None,
    ):
        self.client = make_client(api_key)
        self.batch_size = max(1, min(100, batch_size))
        self.rpm_limit = max(1, rpm_limit)
        self.progress = progress
        self.meter = meter

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = call_with_quota_handling(
                lambda: self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIMENSIONS,
                    ),
                ),
                purpose="embedding",
                model=EMBEDDING_MODEL,
                rpm=self.rpm_limit,
                label="Tạo embedding cho tài liệu",
                progress=self.progress,
                meter=self.meter,
                prompt=batch,
            )
            batch_vectors = [list(item.values) for item in (response.embeddings or [])]
            if len(batch_vectors) != len(batch):
                raise GeminiRequestError(
                    "Gemini trả về số embedding không khớp số chunk. Hãy thử nhập lại file."
                )
            vectors.extend(batch_vectors)
            if self.progress:
                done = min(start + len(batch), len(texts))
                self.progress(f"Đã tạo embedding cho {done}/{len(texts)} chunk…")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = call_with_quota_handling(
            lambda: self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=EMBEDDING_DIMENSIONS,
                ),
            ),
            purpose="embedding",
            model=EMBEDDING_MODEL,
            rpm=self.rpm_limit,
            label="Mã hóa câu hỏi để tìm tài liệu",
            progress=self.progress,
            meter=self.meter,
            prompt=text,
        )
        if not response.embeddings:
            raise GeminiRequestError("Gemini không trả về vector cho câu hỏi.")
        return list(response.embeddings[0].values)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple retrieval queries together to use one API request per batch."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = call_with_quota_handling(
                lambda: self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=EMBEDDING_DIMENSIONS,
                    ),
                ),
                purpose="embedding",
                model=EMBEDDING_MODEL,
                rpm=self.rpm_limit,
                label="Mã hóa nhóm câu hỏi tìm kiếm",
                progress=self.progress,
                meter=self.meter,
                prompt=batch,
            )
            batch_vectors = [list(item.values) for item in (response.embeddings or [])]
            if len(batch_vectors) != len(batch):
                raise GeminiRequestError("Gemini trả về số vector truy vấn không khớp số câu hỏi.")
            vectors.extend(batch_vectors)
        return vectors


def generate_text(
    api_key: str,
    prompt: str,
    *,
    rpm_limit: int = GENERATION_RPM_LIMIT,
    progress: ProgressCallback = None,
    output_tokens: int = 1200,
    meter: RequestMeter | None = None,
) -> str:
    client = make_client(api_key)
    response = call_with_quota_handling(
        lambda: client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=output_tokens,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        ),
        purpose="generation",
        model=GENERATION_MODEL,
        rpm=rpm_limit,
        label="Gemini tạo câu trả lời",
        progress=progress,
        meter=meter,
        prompt=prompt,
    )
    text = response.text
    if not text:
        raise GeminiRequestError("Gemini trả về nội dung trống. Hãy thử diễn đạt lại câu hỏi.")
    return text.strip()


def generate_answer_json(
    api_key: str,
    prompt: str,
    *,
    rpm_limit: int = GENERATION_RPM_LIMIT,
    progress: ProgressCallback = None,
    meter: RequestMeter | None = None,
) -> dict[str, Any]:
    client = make_client(api_key)
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "answer": types.Schema(type=types.Type.STRING),
            "suggestions": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
            ),
        },
        required=["answer", "suggestions"],
    )
    response = call_with_quota_handling(
        lambda: client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1600,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        ),
        purpose="generation",
        model=GENERATION_MODEL,
        rpm=rpm_limit,
        label="Gemini trả lời câu hỏi",
        progress=progress,
        meter=meter,
        prompt=prompt,
    )
    import json

    try:
        payload = json.loads(response.text or "{}")
        answer = str(payload.get("answer", "")).strip()
        suggestions = [str(item).strip() for item in payload.get("suggestions", []) if str(item).strip()]
        if not answer:
            raise ValueError("Missing answer")
        return {"answer": answer, "suggestions": suggestions[:3]}
    except (TypeError, ValueError):
        return {"answer": (response.text or "").strip(), "suggestions": []}


def generate_structured_json(
    api_key: str,
    prompt: str,
    schema: types.Schema,
    *,
    label: str,
    rpm_limit: int = GENERATION_RPM_LIMIT,
    progress: ProgressCallback = None,
    meter: RequestMeter | None = None,
    output_tokens: int = 400,
) -> dict[str, Any]:
    client = make_client(api_key)
    response = call_with_quota_handling(
        lambda: client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=output_tokens,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        ),
        purpose="generation",
        model=GENERATION_MODEL,
        rpm=rpm_limit,
        label=label,
        progress=progress,
        meter=meter,
        prompt=prompt,
    )
    try:
        payload = json.loads(response.text or "{}")
    except (TypeError, ValueError) as error:
        raise GeminiRequestError(f"{label}: Gemini trả về JSON không hợp lệ.") from error
    if not isinstance(payload, dict):
        raise GeminiRequestError(f"{label}: kết quả JSON không phải object.")
    return payload
