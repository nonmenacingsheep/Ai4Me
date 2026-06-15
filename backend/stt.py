"""
Speech-to-text for Aitha via faster-whisper, running locally on the GPU.

The renderer captures the mic, detects an utterance (voice-activity), and POSTs
the recorded audio to /api/stt. We transcribe it here and hand the text back, so
nothing ever leaves the machine and there's no API cost.

Model + device are configurable:
    AITHA_WHISPER_MODEL   default 'base.en'  (small.en is more accurate, ~480MB)
    AITHA_WHISPER_DEVICE  'cuda' / 'cpu'     (auto-detected if unset)
    AITHA_WHISPER_COMPUTE 'float16'/'int8'   (auto: float16 on GPU, int8 on CPU)
"""

import io
import os
import threading

MODEL_NAME = os.getenv("AITHA_WHISPER_MODEL", "base.en")

# Phrases Whisper loves to hallucinate from silence / noise — drop them outright.
_HALLUCINATIONS = {
    "", "thank you.", "thanks for watching!", "thank you for watching.",
    "thank you for watching!", "you", ".", "thanks for watching.",
    "please subscribe.", "bye.", "okay.", "so.",
}


class STTEngine:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _device_and_compute(self):
        device = os.getenv("AITHA_WHISPER_DEVICE")
        if not device:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = os.getenv("AITHA_WHISPER_COMPUTE") or (
            "float16" if device == "cuda" else "int8"
        )
        return device, compute

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel
                    device, compute = self._device_and_compute()
                    try:
                        self._model = WhisperModel(
                            MODEL_NAME, device=device, compute_type=compute
                        )
                        print(f"[stt] whisper '{MODEL_NAME}' on {device} ({compute})")
                    except Exception as e:
                        print(f"[stt] {device} load failed ({e}); falling back to CPU")
                        self._model = WhisperModel(
                            MODEL_NAME, device="cpu", compute_type="int8"
                        )
        return self._model

    def warm_up(self):
        try:
            self._get_model()
        except Exception as e:
            print(f"[stt] warm-up failed: {e}")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe a recorded utterance (any ffmpeg-decodable container, e.g.
        webm/opus from MediaRecorder). Returns cleaned text, or '' if it was just
        noise/silence."""
        if not audio_bytes:
            return ""
        try:
            model = self._get_model()
            segments, _info = model.transcribe(
                io.BytesIO(audio_bytes),
                language="en" if MODEL_NAME.endswith(".en") else None,
                vad_filter=True,  # drop leading/trailing silence the browser sent
                beam_size=1,      # fast; these are short conversational utterances
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            print(f"[stt] transcribe failed: {e}")
            return ""
        if text.lower().strip() in _HALLUCINATIONS:
            return ""
        return text


# Module-level singleton
engine = STTEngine()
