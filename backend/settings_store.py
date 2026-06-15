"""
Runtime-editable settings, persisted so they survive restarts.

Defaults come from environment (.env); a saved settings.json overrides them.
Lives next to the long-term memory in ~/.ai4me/.
"""

import json
import os

_HOME = os.path.expanduser("~")
_DIR = os.path.join(_HOME, ".ai4me")
PATH = os.path.join(_DIR, "settings.json")

# Voices Kokoro ships for American English (lang_code 'a').
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "af_aoede", "af_kore", "af_nova", "am_michael", "am_adam", "am_echo",
]


PRESETS = ["default", "sky", "warm", "moody", "magma", "hearth"]


def default_theme() -> dict:
    # preset = the big look; accent/bg/orb = optional fine-tune hex overrides on top.
    return {"preset": "default", "accent": None, "bg": None, "orb": None}


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in ("0", "false", "no")


def default_behavior() -> dict:
    """Her self-directed drives. Toggles are hard on/off; the *_freq values scale
    how eagerly each drive fires (she still decides) — lower = fewer tokens.
    heartbeat_seconds is how often she even considers acting."""
    return {
        "proactive": _env_flag("AITHA_PROACTIVE"),   # reaching out to him unprompted
        "journaling": True,                          # writing private journal entries
        "curiosity": _env_flag("AITHA_CURIOSITY"),   # going off to explore / pursue
        "speak_freq": 1.0,
        "journal_freq": 1.0,
        "curiosity_freq": 1.0,
        "heartbeat_seconds": int(os.getenv("AITHA_HEARTBEAT", "40")),
    }


def get_behavior() -> dict:
    """Behavior settings merged over defaults, with every value validated/clamped."""
    b = default_behavior()
    saved = load().get("behavior") or {}
    if isinstance(saved, dict):
        for k in b:
            if k in saved:
                b[k] = saved[k]
    for k in ("proactive", "journaling", "curiosity"):
        b[k] = bool(b[k])
    for k in ("speak_freq", "journal_freq", "curiosity_freq"):
        try:
            b[k] = max(0.0, min(3.0, float(b[k])))
        except (TypeError, ValueError):
            b[k] = 1.0
    try:
        b["heartbeat_seconds"] = max(15, min(600, int(b["heartbeat_seconds"])))
    except (TypeError, ValueError):
        b["heartbeat_seconds"] = 40
    return b


def save_behavior(beh: dict) -> dict:
    """Merge & persist behavior settings without disturbing the rest. Returns the
    validated result."""
    cur = get_behavior()
    if isinstance(beh, dict):
        for k in cur:
            if k in beh:
                cur[k] = beh[k]
    s = load()
    s["behavior"] = cur
    save(s)
    return get_behavior()


def defaults() -> dict:
    return {
        # Default to the cloud model; local Ollama models still appear if Ollama is
        # running (it is never auto-started — see README).
        "model": os.getenv("AITHA_MODEL", "deepseek-chat"),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        "tts_enabled": os.getenv("TTS_ENABLED", "1").lower() not in ("0", "false", "no"),
        "tts_voice": os.getenv("TTS_VOICE", "af_heart"),
        "tts_device": os.getenv("TTS_OUTPUT_DEVICE", "CABLE Input (VB-Audio Virtual Cable)"),
        "theme": default_theme(),
        "char_name": os.getenv("AITHA_NAME", "Aitha"),
        "behavior": default_behavior(),
    }


def get_theme() -> dict:
    t = default_theme()
    saved = load().get("theme") or {}
    if isinstance(saved, dict):
        t.update({k: saved.get(k, t[k]) for k in t})
    if t.get("preset") not in PRESETS:
        t["preset"] = "default"
    return t


def save_theme(theme: dict) -> dict:
    """Merge & persist the theme without disturbing other settings. Returns it."""
    cur = get_theme()
    for k in ("preset", "accent", "bg", "orb"):
        if k in theme:
            cur[k] = theme[k]
    if cur.get("preset") not in PRESETS:
        cur["preset"] = "default"
    s = load()
    s["theme"] = cur
    save(s)
    return cur


def load() -> dict:
    s = defaults()
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in s})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return s


def save(settings: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PATH)
    except OSError as e:
        print(f"[settings] save failed: {e}")


DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]


def list_ollama_models() -> list[str]:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


def list_models() -> list[str]:
    """Every cloud model whose provider key is set, then any local Ollama models."""
    try:
        from brain import cloud_models
        cloud = cloud_models()
    except Exception:
        cloud = []
    local = list_ollama_models()
    seen, out = set(), []
    for m in cloud + local:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def list_output_devices() -> list[str]:
    """Unique output device names, ordered for a dropdown."""
    try:
        import sounddevice as sd
        seen, out = set(), []
        for d in sd.query_devices():
            name = d["name"]
            if d["max_output_channels"] > 0 and name not in seen:
                seen.add(name)
                out.append(name)
        return out
    except Exception:
        return []
