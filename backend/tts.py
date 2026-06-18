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

import io
import os
import queue
import re
import shutil
import subprocess
import threading

import numpy as np

SAMPLE_RATE = 24000
VOICE = os.getenv("TTS_VOICE", "af_heart")
LANG = os.getenv("TTS_LANG", "a")  # 'a' = American English
DEVICE_NAME = os.getenv("TTS_OUTPUT_DEVICE", "CABLE Input (VB-Audio Virtual Cable)")
ENABLED_DEFAULT = os.getenv("TTS_ENABLED", "1").lower() not in ("0", "false", "no")

# ── Voice presence (#8): mood → voice + prosody ──────────────────────────────
# Each lever is independently toggleable from Behavior settings, so any of these
# can be switched off and she falls straight back to the plain base voice.
#
# Mood → which Kokoro voice she speaks in (only used when "mood_map" is on).
MOOD_VOICE = {
    "tender":  "af_aoede",
    "content": "af_heart",
    "warm":    "af_bella",
    "pouty":   "af_nicole",
    "clingy":  "af_sarah",
    "angsty":  "af_sky",
}
# Mood → prosody profile (only used when "prosody" is on). Kept subtle so it
# colors her tone without sounding like a pitch toy.
#   speed  — Kokoro synthesis tempo (1.0 = normal; <1 slower, >1 quicker)
#   pitch  — resample ratio applied to the buffer (<1 lower/heavier, >1 brighter)
#   gain   — output level (1.0 = normal)
MOOD_PROSODY = {
    "tender":  {"speed": 0.93, "pitch": 0.985, "gain": 0.80},
    "content": {"speed": 1.00, "pitch": 1.000, "gain": 1.00},
    "warm":    {"speed": 1.02, "pitch": 1.010, "gain": 1.00},
    "pouty":   {"speed": 0.97, "pitch": 0.990, "gain": 0.92},
    "clingy":  {"speed": 1.03, "pitch": 1.020, "gain": 1.00},
    "angsty":  {"speed": 0.95, "pitch": 0.975, "gain": 0.90},
}
# Whisper / late-night softening, layered on top of the mood profile.
# True whisper is unvoiced/breathy — Kokoro can't produce it, so when whisper mode
# is active we synthesize that one utterance with eSpeak NG's "whisper" voice
# variant instead (tiny, CPU-cheap, genuinely breathy). It goes out the same device
# → the voice changer normalizes the timbre, so it still sounds like her. If eSpeak
# isn't installed we fall back to just attenuating/slowing the normal Kokoro voice.
WHISPER_GAIN = 0.55          # fallback hush level when eSpeak isn't available
WHISPER_SPEED = 0.97         # fallback tempo when eSpeak isn't available
ESPEAK_WHISPER_VOICE = os.getenv("ESPEAK_WHISPER_VOICE", "en-us+whisperf")
ESPEAK_WHISPER_WPM = int(os.getenv("ESPEAK_WHISPER_WPM", "155"))   # slow, intimate
ESPEAK_WHISPER_PITCH = int(os.getenv("ESPEAK_WHISPER_PITCH", "35"))  # 0-99, low/soft
ESPEAK_WHISPER_GAIN = 0.85   # eSpeak runs hot; tame it a touch before the changer

# Located once; None means eSpeak NG isn't installed and we use the fallback hush.
_ESPEAK_BIN = shutil.which("espeak-ng") or shutil.which("espeak")
# Micro-pause lengths (seconds) inserted between segments for breath.
PAUSE_DEFAULT = 0.12
PAUSE_SENTENCE = 0.26   # after . ! ?
PAUSE_CLAUSE = 0.17     # after , ; : or an em-dash

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
        self.voice = VOICE          # the user-chosen base voice (her default tone)
        self.device_name = DEVICE_NAME
        # Voice-presence state (#8). mood/late are pushed by the server each context
        # poll; the flags are the Behavior-settings toggles. All default on except
        # whisper so the feature is live out of the box but never surprises with
        # silence.
        self.mood = "content"
        self._late = False
        self.expr_mood_map = True     # swap voice per mood
        self.expr_prosody = True      # speed / pitch / gain per mood
        self.expr_micro_pauses = True  # breath between sentences
        self.expr_whisper = False     # soften late at night
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
            whisper = f"eSpeak ({_ESPEAK_BIN})" if _ESPEAK_BIN else "fallback hush (eSpeak not found)"
            print(f"[tts] ready — voice={self.voice} device={self._device_label()} whisper={whisper}")
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

    def set_mood(self, mood: str, late: bool = False):
        """Push her current emotional state in (called from the context loop). Cheap;
        only affects the NEXT thing she synthesizes, so no interruption needed."""
        if mood in MOOD_VOICE:
            self.mood = mood
        self._late = bool(late)

    def set_expression(self, *, mood_map=None, prosody=None,
                       micro_pauses=None, whisper=None):
        """Update the voice-presence toggles from Behavior settings. Any arg left
        None is unchanged."""
        if mood_map is not None:
            self.expr_mood_map = bool(mood_map)
        if prosody is not None:
            self.expr_prosody = bool(prosody)
        if micro_pauses is not None:
            self.expr_micro_pauses = bool(micro_pauses)
        if whisper is not None:
            self.expr_whisper = bool(whisper)

    def _voice_for_mood(self) -> str:
        if self.expr_mood_map:
            return MOOD_VOICE.get(self.mood, self.voice)
        return self.voice

    def _prosody_for_mood(self) -> dict:
        """Resolved {speed, pitch, gain} for the current mood + whisper state."""
        p = {"speed": 1.0, "pitch": 1.0, "gain": 1.0}
        if self.expr_prosody:
            p.update(MOOD_PROSODY.get(self.mood, p))
        if self.expr_whisper and self._late:
            p["gain"] *= WHISPER_GAIN
            p["speed"] *= WHISPER_SPEED
        return p

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

    @staticmethod
    def _pitch_shift(audio: np.ndarray, ratio: float) -> np.ndarray:
        """Subtle pitch shaping via linear-interpolation resampling. ratio>1 raises
        (and slightly shortens), <1 lowers (and slightly lengthens) — the small
        coupled tempo change reads as natural emotional color, not an effect. No
        external DSP libs needed."""
        if abs(ratio - 1.0) < 1e-3 or len(audio) < 2:
            return audio
        n = len(audio)
        idx = np.arange(0, n, ratio, dtype=np.float32)
        idx = idx[idx < n - 1]
        return np.interp(idx, np.arange(n, dtype=np.float32), audio).astype(np.float32)

    @staticmethod
    def _pause_after(text: str) -> float:
        """How long a breath to leave after a segment, based on its trailing punctuation."""
        t = text.rstrip()
        if not t:
            return PAUSE_DEFAULT
        if t.endswith((".", "!", "?", "…")):
            return PAUSE_SENTENCE
        if t.endswith((",", ";", ":", "—", "-")):
            return PAUSE_CLAUSE
        return PAUSE_DEFAULT

    def _write_samples(self, stream, audio: np.ndarray, gen: int) -> bool:
        """Write mono float32 samples to the stream in small cancellable blocks.
        Returns False if a barge-in cancelled us mid-write."""
        block = SAMPLE_RATE // 10  # 0.1s — cancellation granularity
        if self._channels > 1:
            audio = np.repeat(audio[:, None], self._channels, axis=1)
        for i in range(0, len(audio), block):
            with self._gen_lock:
                if gen != self._current_gen:
                    return False
            try:
                stream.write(audio[i : i + block])
            except Exception as e:
                print(f"[tts] write failed: {e}")
                return False
        return True

    def _whisper_synth(self, text: str):
        """Render one utterance as a true breathy whisper via eSpeak NG. Returns mono
        float32 at SAMPLE_RATE, or None if eSpeak is missing / failed (caller then
        falls back to the attenuated Kokoro hush)."""
        if not _ESPEAK_BIN:
            return None
        try:
            proc = subprocess.run(
                [_ESPEAK_BIN, "-v", ESPEAK_WHISPER_VOICE,
                 "-s", str(ESPEAK_WHISPER_WPM), "-p", str(ESPEAK_WHISPER_PITCH),
                 "--stdout"],
                input=text.encode("utf-8"), capture_output=True, timeout=30,
            )
            if proc.returncode != 0 or not proc.stdout:
                return None
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
            if data.ndim > 1:                  # stereo → mono
                data = data.mean(axis=1)
            if sr != SAMPLE_RATE:              # match the open stream's rate
                data = self._pitch_shift(data, sr / SAMPLE_RATE)
            return np.clip(data * ESPEAK_WHISPER_GAIN, -1.0, 1.0)
        except Exception as e:
            print(f"[tts] whisper (espeak) failed: {e}")
            return None

    def _synth_and_play(self, text: str, gen: int):
        stream = self._ensure_stream()

        # Whisper mode (late-night + toggle): synthesize via eSpeak's breathy voice
        # rather than Kokoro, routed through the same device so the voice changer
        # keeps it sounding like her.
        if self.expr_whisper and self._late:
            wav = self._whisper_synth(text)
            if wav is not None:
                self._write_samples(stream, wav, gen)
                return
            # eSpeak unavailable — fall through to the attenuated Kokoro hush below
            # (WHISPER_GAIN/SPEED are already folded into the prosody profile).

        pipeline = self._get_pipeline()

        # Snapshot the expressive profile once per utterance so it stays consistent
        # even if mood updates mid-sentence.
        voice = self._voice_for_mood()
        pros = self._prosody_for_mood()
        gain = float(pros["gain"])
        pitch = float(pros["pitch"])
        pause = self.expr_micro_pauses

        for gs, _, audio in pipeline(text, voice=voice, speed=pros["speed"]):
            audio = np.asarray(audio, dtype=np.float32)
            if pitch != 1.0:
                audio = self._pitch_shift(audio, pitch)
            if gain != 1.0:
                audio = np.clip(audio * gain, -1.0, 1.0)
            # Write in small blocks so a barge-in stops within ~0.1s. We simply stop
            # feeding the stream — it stays open and goes silent, so the device (and
            # the voice changer's lock on it) never closes.
            if not self._write_samples(stream, audio, gen):
                return
            if pause:
                secs = self._pause_after(gs if isinstance(gs, str) else text)
                silence = np.zeros(int(SAMPLE_RATE * secs), dtype=np.float32)
                if not self._write_samples(stream, silence, gen):
                    return


# Module-level singleton
engine = TTSEngine()
