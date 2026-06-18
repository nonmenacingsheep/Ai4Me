"""
Spotify integration for Aitha (full control needs a Premium account).

OAuth Authorization-Code flow: he connects once (Settings -> Connect Spotify),
which opens the consent page in his browser; the /spotify/callback route hands the
code here and we store the tokens in ~/.ai4me/spotify.json. Access tokens are
refreshed automatically. Everything else is a thin wrapper over the Web API.

Aitha uses the helpers below to see what's playing, pull his top tracks, search,
control playback, and build playlists.
"""

import base64
import os
import time
import urllib.parse

import httpx

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
TOKEN_PATH = os.path.join(_DIR, "spotify.json")

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:7823/spotify/callback").strip()

# Everything she needs: read taste/playback + control playback + manage playlists.
SCOPES = (
    "user-read-private "  # needed for the `product` field (Premium vs Free detection)
    "user-top-read user-read-playback-state user-modify-playback-state "
    "user-read-currently-playing user-read-recently-played "
    "playlist-read-private playlist-modify-private playlist-modify-public"
)

AUTH_BASE = "https://accounts.spotify.com"
API_BASE = "https://api.spotify.com"

_tokens: dict = {}   # {access_token, refresh_token, expires_at}


# ── token persistence ────────────────────────────────────────────────
def _load_tokens() -> dict:
    global _tokens
    if _tokens:
        return _tokens
    try:
        import json
        with open(TOKEN_PATH, "r", encoding="utf-8") as f:
            _tokens = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        _tokens = {}
    return _tokens


def _save_tokens(data: dict) -> None:
    global _tokens
    _tokens = data
    try:
        import json
        os.makedirs(_DIR, exist_ok=True)
        tmp = TOKEN_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, TOKEN_PATH)
    except OSError as e:
        print(f"[spotify] token save failed: {e}")


def disconnect() -> None:
    global _tokens
    _tokens = {}
    _reset_premium_cache()
    try:
        os.remove(TOKEN_PATH)
    except OSError:
        pass


# ── status ───────────────────────────────────────────────────────────
def is_configured() -> bool:
    return bool(CLIENT_ID and CLIENT_SECRET)


def is_connected() -> bool:
    return bool(_load_tokens().get("refresh_token"))


# ── OAuth ─────────────────────────────────────────────────────────────
def auth_url(state: str = "") -> str:
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        # Force the consent screen so a reconnect actually re-grants any newly added
        # scopes — otherwise Spotify silently reuses the prior (narrower) grant.
        "show_dialog": "true",
    }
    return f"{AUTH_BASE}/authorize?" + urllib.parse.urlencode(params)


def _basic_auth() -> str:
    return base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()


def exchange_code(code: str) -> bool:
    """Trade an auth code for tokens (called from the /spotify/callback route)."""
    try:
        r = httpx.post(
            f"{AUTH_BASE}/api/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Authorization": f"Basic {_basic_auth()}"},
            timeout=15,
        )
        r.raise_for_status()
        tok = r.json()
    except (httpx.HTTPError, ValueError) as e:
        print(f"[spotify] code exchange failed: {e}")
        return False
    _save_tokens({
        "access_token": tok.get("access_token", ""),
        "refresh_token": tok.get("refresh_token", ""),
        "expires_at": time.time() + tok.get("expires_in", 3600) - 60,
    })
    _reset_premium_cache()   # re-check tier for the freshly connected account
    print("[spotify] connected")
    return True


def _access_token() -> str | None:
    t = _load_tokens()
    if not t.get("refresh_token"):
        return None
    if t.get("access_token") and time.time() < t.get("expires_at", 0):
        return t["access_token"]
    # refresh
    try:
        r = httpx.post(
            f"{AUTH_BASE}/api/token",
            data={"grant_type": "refresh_token", "refresh_token": t["refresh_token"]},
            headers={"Authorization": f"Basic {_basic_auth()}"},
            timeout=15,
        )
        r.raise_for_status()
        tok = r.json()
    except (httpx.HTTPError, ValueError) as e:
        print(f"[spotify] token refresh failed: {e}")
        return None
    t["access_token"] = tok.get("access_token", t.get("access_token", ""))
    t["expires_at"] = time.time() + tok.get("expires_in", 3600) - 60
    if tok.get("refresh_token"):
        t["refresh_token"] = tok["refresh_token"]
    _save_tokens(t)
    return t["access_token"]


# ── Web API wrapper ───────────────────────────────────────────────────
def _api(method: str, endpoint: str, params: dict | None = None, json: dict | None = None):
    """Call the Web API. Returns parsed JSON (or {} for empty 2xx), or
    {'error': '...'} on failure so callers can speak the problem gracefully."""
    token = _access_token()
    if not token:
        return {"error": "not connected"}
    try:
        r = httpx.request(
            method, f"{API_BASE}/{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            params=params, json=json, timeout=15,
        )
    except httpx.HTTPError as e:
        return {"error": str(e)}
    if r.status_code == 204:
        return {}
    if r.status_code == 404:
        return {"error": "no active device — open Spotify on a device and start playing first"}
    if r.status_code == 403:
        # 403 covers two very different cases: Premium-only playback, and a token
        # that lacks the needed scope. Read Spotify's message instead of guessing.
        try:
            msg = (r.json().get("error", {}).get("message", "") or "").strip()
        except ValueError:
            msg = ""
        low = msg.lower()
        reconnect = ("Spotify needs re-authorizing for this — disconnect and reconnect Spotify "
                     "in Settings → General to grant playlist access.")
        if "scope" in low or "permission" in low:
            return {"error": reconnect}
        if "premium" in low:
            return {"error": "this needs Spotify Premium"}
        # Playlist endpoints don't require Premium — a 403 here (even a bare "Forbidden")
        # is a missing-scope problem, which a reconnect fixes.
        if "playlist" in endpoint:
            return {"error": reconnect}
        # Playback endpoints are the Premium-gated ones.
        if "v1/me/player" in endpoint:
            return {"error": "this needs Spotify Premium (and an active device)"}
        return {"error": msg or "Spotify refused that (403)"}
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        return {"error": f"{r.status_code} {msg}".strip()}
    try:
        return r.json()
    except ValueError:
        return {}


# ── reads ─────────────────────────────────────────────────────────────
def _fmt_track(t: dict) -> str:
    if not t:
        return ""
    artists = ", ".join(a["name"] for a in t.get("artists", []))
    return f"{t.get('name','')} by {artists}" if artists else t.get("name", "")


def now_playing() -> dict | None:
    data = _api("GET", "v1/me/player/currently-playing")
    if not data or data.get("error") or not data.get("item"):
        return None
    item = data["item"]
    return {
        "playing": data.get("is_playing", False),
        "track": item.get("name", ""),
        "artists": [a["name"] for a in item.get("artists", [])],
        "text": _fmt_track(item),
        "uri": item.get("uri", ""),
    }


def top_tracks(limit: int = 10, time_range: str = "long_term") -> list[dict]:
    data = _api("GET", "v1/me/top/tracks", params={"limit": limit, "time_range": time_range})
    return data.get("items", []) if isinstance(data, dict) else []


def top_artists(limit: int = 10, time_range: str = "long_term") -> list[dict]:
    data = _api("GET", "v1/me/top/artists", params={"limit": limit, "time_range": time_range})
    return data.get("items", []) if isinstance(data, dict) else []


def recently_played(limit: int = 10) -> list[dict]:
    data = _api("GET", "v1/me/player/recently-played", params={"limit": limit})
    return [it["track"] for it in data.get("items", [])] if isinstance(data, dict) else []


def search_tracks(query: str, limit: int = 5) -> list[dict]:
    # Feb-2026 API migration capped search results at 10 for development-mode apps.
    data = _api("GET", "v1/search", params={"q": query, "type": "track", "limit": min(limit, 10)})
    return data.get("tracks", {}).get("items", []) if isinstance(data, dict) else []


def my_playlists(limit: int = 20) -> list[dict]:
    data = _api("GET", "v1/me/playlists", params={"limit": limit})
    return data.get("items", []) if isinstance(data, dict) else []


# ── playback control ──────────────────────────────────────────────────
def play(query: str = "", uri: str = "") -> dict:
    """Resume (no args), play a specific uri, or search a query and play the top hit."""
    if query and not uri:
        hits = search_tracks(query, limit=1)
        if not hits:
            return {"error": f"couldn't find “{query}” on Spotify"}
        uri = hits[0]["uri"]
        body = {"uris": [uri]}
        res = _api("PUT", "v1/me/player/play", json=body)
        return {"ok": "error" not in res, "now": _fmt_track(hits[0]), **res}
    if uri:
        res = _api("PUT", "v1/me/player/play", json={"uris": [uri]})
        return {"ok": "error" not in res, **res}
    res = _api("PUT", "v1/me/player/play")     # resume
    return {"ok": "error" not in res, **res}


def play_playlist(name: str) -> dict:
    """Find one of his playlists by name and start playing it (context playback)."""
    name = (name or "").strip()
    if not name:
        return {"error": "which playlist?"}
    pls = my_playlists(limit=50)
    low = name.lower()
    match = (next((p for p in pls if p.get("name", "").lower() == low), None)
             or next((p for p in pls if low in p.get("name", "").lower()), None))
    if not match:
        return {"error": f"couldn't find a playlist called “{name}”"}
    res = _api("PUT", "v1/me/player/play", json={"context_uri": match["uri"]})
    return {"ok": "error" not in res, "now": f"playlist {match.get('name','')}", **res}


def pause() -> dict:
    res = _api("PUT", "v1/me/player/pause")
    return {"ok": "error" not in res, **res}


def next_track() -> dict:
    res = _api("POST", "v1/me/player/next")
    return {"ok": "error" not in res, **res}


def previous_track() -> dict:
    res = _api("POST", "v1/me/player/previous")
    return {"ok": "error" not in res, **res}


# ── playlists ─────────────────────────────────────────────────────────
def me() -> dict:
    """His Spotify profile. {} on error / not connected."""
    data = _api("GET", "v1/me")
    return data if isinstance(data, dict) and not data.get("error") else {}


def _me_id() -> str | None:
    return me().get("id") or None


# Cache the Premium check — the product tier rarely changes within a session,
# and we don't want a /v1/me round-trip on every prompt build.
_PREMIUM_CACHE: dict = {"checked": False, "premium": False}


def is_premium() -> bool:
    """True only if he's on Spotify Premium (playback control requires it)."""
    if not is_connected():
        return False
    if not _PREMIUM_CACHE["checked"]:
        _PREMIUM_CACHE["premium"] = me().get("product") == "premium"
        _PREMIUM_CACHE["checked"] = True
    return _PREMIUM_CACHE["premium"]


def _reset_premium_cache() -> None:
    _PREMIUM_CACHE["checked"] = False
    _PREMIUM_CACHE["premium"] = False


def create_playlist(name: str, track_uris: list[str], description: str = "",
                    public: bool = False) -> dict:
    """Create a playlist for him and add the given track URIs.
    Uses the /v1/me/playlists shorthand — the /v1/users/{id}/playlists form
    returns a bare 403 'Forbidden' on some accounts even with full scopes."""
    if not is_connected():
        return {"error": "not connected"}
    pl = _api("POST", "v1/me/playlists",
              json={"name": name or "Aitha's playlist", "description": description, "public": public})
    if pl.get("error") or not pl.get("id"):
        return {"error": pl.get("error", "couldn't create the playlist")}
    if track_uris:
        # Feb-2026 API migration renamed this from /tracks to /items; the old path
        # now 403s for development-mode apps.
        add = _api("POST", f"v1/playlists/{pl['id']}/items", json={"uris": track_uris})
        if add.get("error"):
            return {"error": f"created the playlist but couldn't add songs: {add['error']}",
                    "id": pl["id"], "name": pl.get("name", name), "count": 0}
    return {"ok": True, "id": pl["id"], "name": pl.get("name", name), "count": len(track_uris)}


def create_playlist_from_top(name: str = "My Top Tracks", description: str = "",
                            limit: int = 20, time_range: str = "long_term") -> dict:
    uris = [t["uri"] for t in top_tracks(limit=limit, time_range=time_range) if t.get("uri")]
    if not uris:
        return {"error": "couldn't read your top tracks"}
    return create_playlist(name, uris, description or "Your most-played, gathered by Aitha.")


def create_playlist_from_search(name: str, query: str, description: str = "",
                                limit: int = 10) -> dict:
    uris = [t["uri"] for t in search_tracks(query, limit=limit) if t.get("uri")]
    if not uris:
        return {"error": f"couldn't find tracks for “{query}”"}
    return create_playlist(name, uris, description or f"Songs for: {query}")


def add_tracks_to_playlist(name: str, tracks: list[str]) -> dict:
    """Add hand-picked songs to one of his EXISTING playlists (found by name).
    Each entry is searched individually (best as "song — artist")."""
    name = (name or "").strip()
    if not name:
        return {"error": "which playlist?"}
    pls = my_playlists(limit=50)
    low = name.lower()
    match = (next((p for p in pls if p.get("name", "").lower() == low), None)
             or next((p for p in pls if low in p.get("name", "").lower()), None))
    if not match:
        return {"error": f"couldn't find a playlist called “{name}” — make it first or check the name"}
    uris, missed = [], []
    for raw in tracks:
        q = (raw or "").strip()
        if not q:
            continue
        hits = search_tracks(q, limit=1)
        (uris if hits and hits[0].get("uri") else missed).append(
            hits[0]["uri"] if hits and hits[0].get("uri") else q)
    if not uris:
        return {"error": f"couldn't find any of those songs on Spotify"}
    add = _api("POST", f"v1/playlists/{match['id']}/items", json={"uris": uris})
    if add.get("error"):
        return {"error": add["error"], "name": match.get("name", name)}
    res = {"ok": True, "name": match.get("name", name), "count": len(uris)}
    if missed:
        res["missed"] = missed
    return res


def create_playlist_from_tracks(name: str, tracks: list[str], description: str = "") -> dict:
    """Curate a playlist from a hand-picked list of songs. Each entry is searched
    for individually (best as "song — artist"); the top hit for each is added.
    Entries that match nothing are skipped and reported back."""
    uris: list[str] = []
    missed: list[str] = []
    for raw in tracks:
        q = (raw or "").strip()
        if not q:
            continue
        hits = search_tracks(q, limit=1)
        if hits and hits[0].get("uri"):
            uris.append(hits[0]["uri"])
        else:
            missed.append(q)
    if not uris:
        return {"error": "couldn't find any of those songs on Spotify"}
    res = create_playlist(name, uris, description or "Hand-picked by Aitha.")
    if missed and not res.get("error"):
        res["missed"] = missed
    return res


# ── context digest for her prompt ─────────────────────────────────────
def status_summary() -> dict:
    """Light-weight status for the UI."""
    return {
        "configured": is_configured(),
        "connected": is_connected(),
        "premium": is_premium() if is_connected() else False,
        "now_playing": now_playing() if is_connected() else None,
    }


def digest() -> str:
    """What she should know about his music right now. '' when nothing useful."""
    if not is_configured() or not is_connected():
        return ""
    np = now_playing()
    if np:
        verb = "playing" if np["playing"] else "paused on"
        return f"His Spotify is connected. Right now it's {verb}: {np['text']}."
    return "His Spotify is connected (nothing playing right now)."
