from __future__ import annotations

import re
import urllib.parse
import urllib.request
import wave
from io import BytesIO
from typing import Any

import streamlit as st
from google.genai import types

from .config import GENERATION_RPM_LIMIT
from .quota import GeminiRequestError, call_with_quota_handling, make_client

TRANSCRIPTION_MODEL = "gemini-3.5-transcribe"
TTS_MODEL = "gemini-3.1-flash-tts-preview"


def transcribe_vietnamese(audio: Any, api_key: str, *, rpm_limit: int = GENERATION_RPM_LIMIT) -> str:
    """Transcribe a short Streamlit microphone recording and remove its remote file."""
    payload = audio.getvalue()
    if not payload:
        raise ValueError("Bản ghi âm đang trống. Hãy thu âm lại.")
    if len(payload) > 15 * 1024 * 1024:
        raise ValueError("Bản ghi vượt quá 15 MB. Hãy thu âm một câu hỏi ngắn hơn.")
    client = make_client(api_key)
    remote_file = None
    try:
        remote_file = client.files.upload(
            file=BytesIO(payload),
            config={"mime_type": "audio/wav", "display_name": "voice-question.wav"},
        )
        response = call_with_quota_handling(
            lambda: client.interactions.create(
                model=TRANSCRIPTION_MODEL,
                input=[{"type": "audio", "uri": remote_file.uri, "mime_type": "audio/wav"}],
                generation_config={"transcription_config": {"language_codes": ["vi-VN"]}},
            ),
            purpose="generation",
            model=TRANSCRIPTION_MODEL,
            rpm=rpm_limit,
            label="Gemini nhận dạng giọng nói tiếng Việt",
        )
        transcript = str(getattr(response, "output_text", "") or "").strip()
        if not transcript:
            raise GeminiRequestError("Gemini không nhận ra lời nói. Hãy thử thu âm rõ hơn.")
        return transcript
    except GeminiRequestError:
        raise
    except Exception as error:
        raise GeminiRequestError(f"Không thể chuyển giọng nói thành văn bản: {str(error)[:500]}") from error
    finally:
        if remote_file is not None and getattr(remote_file, "name", None):
            try:
                client.files.delete(name=remote_file.name)
            except Exception:
                # A deletion failure must not hide a successful transcription.
                pass


def clean_tts_text(text: str) -> str:
    """Prepare clean, readable Vietnamese text by removing code blocks, citations, and markdown."""
    clean = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    clean = re.sub(r"\[S\d+\]", "", clean)
    clean = re.sub(r"\[C[^\]]+\]", "", clean)
    clean = re.sub(r"[#*_`>|~]", " ", clean)
    clean = re.sub(r"\bIII\b", "ba", clean)
    clean = re.sub(r"\bII\b", "hai", clean)
    clean = re.sub(r"\bIV\b", "bốn", clean)
    clean = re.sub(r"\bV\b", "năm", clean)
    clean = re.sub(r"\bI\b", "một", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def synthesize_vietnamese_fast(text: str) -> bytes:
    """Fast, zero-quota Vietnamese TTS via standard Python HTTP stream (~1-2s latency)."""
    clean = clean_tts_text(text)
    if not clean:
        raise ValueError("Không có nội dung để đọc.")

    parts: list[str] = []
    current = ""
    sentences = re.split(r"([.?!;\n]+)", clean)
    for i in range(0, len(sentences), 2):
        s = sentences[i]
        sep = sentences[i + 1] if i + 1 < len(sentences) else ""
        chunk = (s + sep).strip()
        if not chunk:
            continue
        if len(current) + len(chunk) + 1 <= 180:
            current = (current + " " + chunk).strip()
        else:
            if current:
                parts.append(current)
            if len(chunk) <= 180:
                current = chunk
            else:
                for word in chunk.split():
                    if len(current) + len(word) + 1 <= 180:
                        current = (current + " " + word).strip()
                    else:
                        if current:
                            parts.append(current)
                        current = word
    if current:
        parts.append(current)

    # Deliver up to 6 sentences (~1.5 minutes of spoken answer) to maintain rapid response
    parts = parts[:6]
    audio_combined = b""
    for part in parts:
        url = (
            "https://translate.google.com/translate_tts?ie=UTF-8&q="
            + urllib.parse.quote(part)
            + "&tl=vi&client=tw-ob"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=7) as resp:
            audio_combined += resp.read()

    if not audio_combined:
        raise ValueError("Không thể tạo dữ liệu âm thanh nhanh.")
    return audio_combined


def synthesize_vietnamese_gemini(
    text: str, api_key: str, *, rpm_limit: int = GENERATION_RPM_LIMIT
) -> bytes:
    """Fallback WAV audio generator using Gemini TTS."""
    clean_text = clean_tts_text(text)[:1500]
    if not clean_text:
        raise ValueError("Không có nội dung để đọc.")

    prompt = (
        "Đọc nguyên văn nội dung tiếng Việt sau bằng giọng rõ ràng, tự nhiên. "
        "Không dịch, không thêm lời dẫn hoặc nhận xét.\n\n" + clean_text
    )
    client = make_client(api_key)
    response = call_with_quota_handling(
        lambda: client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    language_code="vi-VN",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                    ),
                ),
            ),
        ),
        purpose="generation",
        model=TTS_MODEL,
        rpm=rpm_limit,
        label="Gemini tạo giọng đọc tiếng Việt",
        prompt=prompt,
    )

    pcm = None
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                pcm = inline_data.data
                break
        if pcm is not None:
            break
    if pcm is None:
        raise GeminiRequestError(
            "Gemini TTS không trả về dữ liệu âm thanh. Kiểm tra model hoặc quyền truy cập API."
        )

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(pcm)
    return output.getvalue()


def synthesize_vietnamese(
    text: str, api_key: str | None = None, *, rpm_limit: int = GENERATION_RPM_LIMIT
) -> bytes:
    """Synthesize Vietnamese audio with rapid neural stream first, falling back to Gemini TTS."""
    try:
        return synthesize_vietnamese_fast(text)
    except Exception:
        if api_key:
            return synthesize_vietnamese_gemini(text, api_key, rpm_limit=rpm_limit)
        raise


def render_tts_player(
    text: str, key: str, api_key: str | None = None, *, rpm_limit: int = GENERATION_RPM_LIMIT
) -> None:
    """Render a one-click audio player with instant generation and autoplay support."""
    cache_key = f"tts-audio-{key}"
    if cache_key not in st.session_state:
        if st.button(
            "🔊 Nghe câu trả lời",
            key=f"tts-generate-{key}",
            help="Tạo giọng đọc tiếng Việt siêu tốc (1-2s).",
        ):
            with st.spinner("Đang tạo giọng đọc tiếng Việt (~1-2 giây)…"):
                try:
                    st.session_state[cache_key] = synthesize_vietnamese(
                        text, api_key or None, rpm_limit=rpm_limit
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Chưa tạo được audio: {error}")

    audio_bytes = st.session_state.get(cache_key)
    if audio_bytes:
        audio_format = "audio/wav" if audio_bytes.startswith(b"RIFF") else "audio/mp3"
        st.audio(
            audio_bytes,
            format=audio_format,
            autoplay=True,
            alt="Audio tiếng Việt đọc câu trả lời",
        )
        st.caption("▶️ Đang phát giọng đọc tiếng Việt. Nếu trình duyệt chặn tự phát, hãy bấm nút Play (▶️).")
