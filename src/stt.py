"""
Optional voice-input layer via Sarvam saaras:v3 speech-to-text.
Text queries skip this module entirely — it's only invoked when the
caller passes an audio file instead of a string.
"""
from __future__ import annotations
import requests

from .config import config

_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"


def transcribe(audio_file_path: str) -> str:
    """Sends an audio file to Sarvam saaras:v3 and returns the transcribed text."""
    if not config.SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not set in environment.")

    with open(audio_file_path, "rb") as f:
        files = {"file": f}
        data = {"model": config.SARVAM_MODEL}
        headers = {"api-subscription-key": config.SARVAM_API_KEY}
        resp = requests.post(_SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("transcript", "").strip()
