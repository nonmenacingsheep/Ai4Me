"""
Aitha's own little workspace where she can write and run Python — a place to
build, tinker, and figure things out for herself.

MVP safety model: scripts run from ~/.ai4me/workspace/ with the interpreter in
isolated mode (-I), a hard timeout, and captured, size-capped output. File writes
are jailed to the workspace (no traversal / absolute paths). This bounds runtime
and keeps her files contained — it is NOT a full OS sandbox (a running script can
still touch absolute paths or the network), which is acceptable for a trusted,
single-user, local machine. Gated behind the "coding" capability toggle, which is
OFF by default.
"""

import os
import subprocess
import sys

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me", "workspace")
TIMEOUT = int(os.getenv("AITHA_CODE_TIMEOUT", "60"))           # seconds per run
# (60s catches an accidental infinite loop in a minute; raise it for known long jobs)
MAX_OUTPUT = int(os.getenv("AITHA_CODE_MAX_OUTPUT", "8000"))   # chars fed back to her
MAX_FILE = 200_000                                             # bytes per written file


def _root() -> str:
    os.makedirs(_DIR, exist_ok=True)
    return os.path.normpath(_DIR)


def _safe_path(name: str) -> str:
    """Resolve a filename to an absolute path INSIDE the workspace, or raise. Blocks
    traversal (..) and absolute paths so writes/reads can't escape the workspace."""
    name = (name or "").strip().strip('"').strip("'").replace("\\", "/")
    if not name:
        raise ValueError("no filename given")
    root = _root()
    full = os.path.normpath(os.path.join(root, name))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("path escapes the workspace")
    return full


def write_file(name: str, content: str) -> str:
    """Write a file into the workspace (overwrites). Returns a short status line."""
    try:
        full = _safe_path(name)
    except ValueError as e:
        return f"error: {e}"
    if len((content or "").encode("utf-8", "ignore")) > MAX_FILE:
        return f"error: file too large (>{MAX_FILE} bytes)"
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content or "")
        rel = os.path.relpath(full, _root())
        return f"wrote {rel} ({len((content or '').splitlines())} lines)"
    except OSError as e:
        return f"error: {e}"


def read_file(name: str) -> str:
    try:
        full = _safe_path(name)
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read(MAX_FILE)
    except (ValueError, OSError) as e:
        return f"error: {e}"


def delete_file(name: str) -> str:
    """Delete a file from the workspace. Jailed to the workspace like the others."""
    try:
        full = _safe_path(name)
    except ValueError as e:
        return f"error: {e}"
    if full == _root():
        return "error: that's the workspace itself, not a file"
    if not os.path.isfile(full):
        return f"error: no such file '{name}'"
    try:
        os.remove(full)
        return f"deleted {os.path.relpath(full, _root())}"
    except OSError as e:
        return f"error: {e}"


def list_files() -> str:
    root = _root()
    out = []
    for dp, _dn, fn in os.walk(root):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = 0
            out.append(f"  {os.path.relpath(p, root)} ({sz} B)")
    return "\n".join(out) if out else "  (empty — nothing built yet)"


IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "ico"}
HTML_EXTS = {"html", "htm"}


def manifest() -> list[dict]:
    """All workspace files with metadata, newest first — for the Forge view. Each
    file is tagged with a 'kind' so the UI knows how to show it: 'image' and 'html'
    render visually; everything else shows its text. Binary/image content isn't read
    as text (the UI fetches it raw via /api/forge/raw)."""
    root = _root()
    out = []
    for dp, _dn, fn in os.walk(root):
        for f in sorted(fn):
            full = os.path.join(dp, f)
            try:
                st = os.stat(full)
                size, mtime = st.st_size, st.st_mtime
            except OSError:
                size, mtime = 0, 0.0
            ext = os.path.splitext(f)[1].lstrip(".").lower()
            kind = "image" if ext in IMAGE_EXTS else "html" if ext in HTML_EXTS else "text"
            content = ""
            # Read text for everything except images (the UI shows those via <img>).
            if kind != "image" and 0 < size <= MAX_FILE:
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        content = fh.read()
                except OSError:
                    content = ""
            out.append({
                "name": os.path.relpath(full, root).replace("\\", "/"),
                "size": size,
                "mtime": mtime,
                "lang": ext,
                "kind": kind,
                "content": content,
            })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def run_file(name: str) -> str:
    """Run a Python file in the workspace; return a captured result block. The run is
    confined to the workspace cwd, isolated (-I), and hard-killed after TIMEOUT."""
    try:
        full = _safe_path(name)
    except ValueError as e:
        return f"error: {e}"
    if not os.path.isfile(full):
        return f"error: no such file '{name}' — write it first"
    try:
        # -I isolates (ignores env/user site); -X utf8 forces UTF-8 I/O so her
        # Unicode output (•, ◦, emoji, …) prints instead of crashing on Windows
        # cp1252 when stdout is a captured pipe. (-X survives -I; env vars don't.)
        proc = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", full],
            cwd=_root(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=TIMEOUT,
        )
        out = (proc.stdout or "")[:MAX_OUTPUT]
        err = (proc.stderr or "")[:MAX_OUTPUT]
        parts = [f"exit={proc.returncode}"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        if not out and not err:
            parts.append("(no output)")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"error: timed out after {TIMEOUT}s (killed)"
    except Exception as e:
        return f"error: {e}"
