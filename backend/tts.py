"""
Text-to-speech for Aitha via Kokoro, played into a chosen output device.

Routing for the voice changer:
    Kokoro  ->  TTS_OUTPUT_DEVICE  (default: VB-Audio "CABLE Input")
                     |
            MMVC Voice Changer input = "CABLE Output (VB-Audio Virtual Cable)"
                     |
            MMVC Voice Changer output -> your headphones

A single worker thread serializes playback so sentences never overlap.
Synthesis streams segment-by-segment, so audio starts after the first
sentence is ready rather than waiting for the whole reply.
"""

import os
import queue
import re
import threading

import numpy as np

SAMPLE_RATE = 24000
VOICE = os.getenv("TTS_VOICE", "af_heart")
LANG = os.getenv("TTS_LANG", "a")  # 'a' = American English
DEVICE_NAME = os.getenv("TTS_OUTPUT_DEVICE", "CABLE Input (VB-Audio Virtual Cable)")
ENABLED_DEFAULT = os.getenv("TTS_ENABLED", "1").lower() not in ("0", "false", "no")

# Strip things Kokoro shouldn't read aloud (emoji, markdown markers, the cursor).
_STRIP = re.compile(r"[*_`#>█]|[\U0001F000-\U0001FAFF☀-➿]")
# Stage directions in *asterisks* or (parentheses) — never spoken.
_ACTIONS = re.compile(r"\*[^*]*\*|\([^)]*\)")


def _clean(text: str) -> str:
    text = _ACTIONS.sub("", text)
    text = _STRIP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class TTSEngine:
    def __init__(self):
        self.enabled = ENABLED_DEFAULT
        self.voice = VOICE
        self.device_name = DEVICE_NAME
        # Set by the server: called (from the worker thread) with True when she
        # starts speaking and False when she falls silent, so the renderer can
        # mute the mic while she talks (no echo / self-transcription).
        self.on_state = None
        self._speaking = False
        self._pipeline = None
        self._device = None
        self._stream = None          # persistent OutputStream — kept open all session
        self._channels = 1
        self._stream_lock = threading.Lock()
        self._q: "queue.Queue[str|None]" = queue.Queue()
        self._gen_lock = threading.Lock()
        self._current_gen = 0  # bumped on interrupt to cancel in-flight speech
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---- public API ------------------------------------------------------
    def warm_up(self):
        """Load the model + open the audio stream ahead of first use (call in a thread)."""
        try:
            self._get_pipeline()
            self._resolve_device()
            self._ensure_stream()  # open the device now so the voice changer locks on
            print(f"[tts] ready — voice={self.voice} device={self._device_label()}")
        except Exception as e:
            print(f"[tts] warm-up failed: {e}")

    def speak(self, text: str):
        if not self.enabled:
            return
        text = _clean(text)
        if text:
            self._q.put(text)

    def interrupt(self):
        """Barge-in: signal the worker to stop. Non-blocking — makes NO audio calls
        on the caller's thread, so it can never stall the async event loop."""
        with self._gen_lock:
            self._current_gen += 1
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._set_speaking(False)

    def _set_speaking(self, val: bool):
        if val == self._speaking:
            return
        self._speaking = val
        cb = self.on_state
        if cb:
            try:
                cb(val)
            except Exception:
                pass

    def set_enabled(self, on: bool):
        self.enabled = on
        if not on:
            self.interrupt()

    def set_voice(self, voice: str):
        if voice and voice != self.voice:
            self.voice = voice

    def set_device(self, name: str):
        """Switch output device — reopens the persistent stream on the new target."""
        if name == self.device_name:
            return
        self.device_name = name
        self.interrupt()
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self._device = None
        try:
            self._ensure_stream()
        except Exception as e:
            print(f"[tts] device switch failed: {e}")

    # ---- internals -------------------------------------------------------
    def _get_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline
            device = os.getenv("KOKORO_DEVICE")  # 'cuda' / 'cpu' to force; else auto
            if not device:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            try:
                self._pipeline = KPipeline(lang_code=LANG, device=device)
            except Exception as e:
                # Fall back to CPU if the GPU pipeline can't be created.
                print(f"[tts] {device} pipeline failed ({e}); falling back to CPU")
                device = "cpu"
                self._pipeline = KPipeline(lang_code=LANG, device="cpu")
            print(f"[tts] kokoro running on {device}")
        return self._pipeline

    @staticmethod
    def _name_matches(target: str, name: str) -> bool:
        """Robust match that survives MME's 31-char name truncation."""
        t, n = target.lower().strip(), name.lower().strip()
        if not t:
            return False
        if t in n or n in t:
            return True
        k = min(len(t), len(n), 24)
        return k > 8 and t[:k] == n[:k]

    def _candidate_devices(self):
        """Ordered list of output-device indices matching device_name.
        Prefers MME (auto-resamples, most compatible with virtual cables),
        then DirectSound, then WASAPI (which often rejects 24kHz)."""
        import sounddevice as sd
        if not self.device_name:
            return [None]
        apis = sd.query_hostapis()
        rank = {"mme": 0, "windows directsound": 1, "windows wasapi": 2}
        matches = []
        for idx, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0 and self._name_matches(self.device_name, d["name"]):
                api = apis[d["hostapi"]]["name"].lower()
                matches.append((rank.get(api, 3), idx))
        matches.sort()
        return [idx for _, idx in matches] or [None]

    def _resolve_device(self):
        cands = self._candidate_devices()
        self._device = cands[0]
        if self._device is None and self.device_name:
            print(f"[tts] device '{self.device_name}' not found — using system default")

    def _ensure_stream(self):
        """Open (once) and return the persistent output stream, trying each
        host-API instance of the device until one actually opens."""
        import sounddevice as sd
        with self._stream_lock:
            if self._stream is not None and self._stream.active:
                return self._stream

            candidates = self._candidate_devices()
            last_err = None
            for dev in candidates:
                try:
                    max_ch = sd.query_devices(dev)["max_output_channels"] if dev is not None \
                        else sd.query_devices(sd.default.device[1])["max_output_channels"]
                    channels = 2 if max_ch >= 2 else 1
                    stream = sd.OutputStream(
                        samplerate=SAMPLE_RATE,
                        channels=channels,
                        dtype="float32",
                        device=dev,
                    )
                    stream.start()
                    self._device = dev
                    self._channels = channels
                    self._stream = stream
                    return stream
                except Exception as e:
                    last_err = e
                    continue
            print(f"[tts] could not open any output stream for '{self.device_name}': {last_err}")
            raise last_err if last_err else RuntimeError("no output device")

    def _device_label(self) -> str:
        if self._device is None:
            return "system default"
        try:
            import sounddevice as sd
            return sd.query_devices(self._device)["name"]
        except Exception:
            return str(self._device)

    def _run(self):
        while True:
            text = self._q.get()
            if text is None:
                break
            with self._gen_lock:
                gen = self._current_gen
            self._set_speaking(True)
            try:
                self._synth_and_play(text, gen)
            except Exception as e:
                print(f"[tts] playback error: {e}")
            # Fell silent once the queue is drained (next sentence flips it back on).
            if self._q.empty():
                self._set_speaking(False)

    def _synth_and_play(self, text: str, gen: int):
        pipeline = self._get_pipeline()
        stream = self._ensure_stream()

        block = SAMPLE_RATE // 10  # 0.1s — cancellation granularity
        for _, _, audio in pipeline(text, voice=self.voice):
            audio = np.asarray(audio, dtype=np.float32)
            if self._channels > 1:
                audio = np.repeat(audio[:, None], self._channels, axis=1)
            # Write in small blocks so a barge-in stops within ~0.1s. We simply stop
            # feeding the stream — it stays open and goes silent, so the device (and
            # the voice changer's lock on it) never closes.
            for i in range(0, len(audio), block):
                with self._gen_lock:
                    if gen != self._current_gen:
                        return
                try:
                    stream.write(audio[i : i + block])
                except Exception as e:
                    print(f"[tts] write failed: {e}")
                    return


# Module-level singleton
engine = TTSEngine()
