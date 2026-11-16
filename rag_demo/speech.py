from __future__ import annotations

import re
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


def synthesize_vietnamese(text: str, api_key: str, *, rpm_limit: int = GENERATION_RPM_LIMIT) -> bytes:
    """Create a WAV audio response with Gemini TTS for Streamlit's native player."""
    clean_text = re.sub(r"```.*?```", "Khối mã.", text, flags=re.DOTALL)
    clean_text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean_text)
    clean_text = re.sub(r"[#*_`>|]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    if not clean_text:
        raise ValueError("Không có nội dung để đọc.")
    clean_text = clean_text[:12_000]

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
        raise GeminiRequestError("Gemini TTS không trả về dữ liệu âm thanh. Kiểm tra model hoặc quyền truy cập API.")

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(pcm)
    return output.getvalue()


def render_tts_player(text: str, key: str, api_key: str, *, rpm_limit: int = GENERATION_RPM_LIMIT) -> None:
    cache_key = f"tts-audio-{key}"
    if cache_key not in st.session_state:
        if not api_key:
            st.caption("Nhập Gemini API key để tạo audio tiếng Việt.")
        if st.button(
            "🔊 Tạo giọng đọc",
            key=f"tts-generate-{key}",
            disabled=not api_key,
            help="Tạo audio bằng Gemini TTS; lượt này dùng quota generation.",
        ):
            with st.spinner("Đang tạo audio tiếng Việt…"):
                try:
                    st.session_state[cache_key] = synthesize_vietnamese(
                        text, api_key, rpm_limit=rpm_limit
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Chưa tạo được audio: {error}")
    audio_bytes = st.session_state.get(cache_key)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav", alt="Audio tiếng Việt đọc câu trả lời")
