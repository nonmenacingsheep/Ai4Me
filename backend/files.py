"""
Scoped, read-only local file access for Aitha.

He explicitly grants a set of folders ("roots"); she may list and read text files
INSIDE those folders and nowhere else. Every path is resolved to its real location
and checked against the granted roots, so symlinks or `..` can't escape the sandbox.
She can never write, move, or delete anything — this module only reads.
"""

import os

# Folders he's granted, as real (resolved) absolute paths. Set from settings at
# startup and whenever he changes them in Settings.
_roots: list[str] = []

# Files we'll actually read as text. Anything else is treated as binary/opaque.
_TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".csv", ".tsv", ".log", ".ini",
    ".cfg", ".conf", ".toml", ".yaml", ".yml", ".xml", ".html", ".htm", ".css",
    ".js", ".ts", ".jsx", ".tsx", ".py", ".rs", ".go", ".java", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".bat", ".ps1", ".sql", ".env",
    ".gitignore", ".lua", ".r", ".kt", ".swift", ".vue", ".svelte", ".tex",
}
_MAX_FILE_BYTES = 5_000_000        # don't even open files bigger than this
_READ_BUDGET = 200_000             # chars handed back to her per file


def set_roots(paths: list) -> list[str]:
    """Replace the granted folder set. Keeps only existing directories, resolved."""
    global _roots
    seen, out = set(), []
    for p in paths or []:
        if not p:
            continue
        try:
            rp = os.path.realpath(str(p))
        except OSError:
            continue
        key = os.path.normcase(rp)
        if key not in seen and os.path.isdir(rp):
            seen.add(key)
            out.append(rp)
    _roots = out
    return list(_roots)


def get_roots() -> list[str]:
    return list(_roots)


def _within_roots(path: str) -> str | None:
    """Resolve `path` and return it only if it lives inside a granted root."""
    if not path or not _roots:
        return None
    try:
        rp = os.path.realpath(str(path).strip().strip('"').strip("'"))
    except OSError:
        return None
    nrp = os.path.normcase(rp)
    for root in _roots:
        nroot = os.path.normcase(root)
        if nrp == nroot or nrp.startswith(nroot + os.sep):
            return rp
    return None


def list_dir(path: str) -> str:
    """A readable listing of a folder she's allowed to see."""
    rp = _within_roots(path)
    if not rp:
        return "(that folder isn't one he's shared with you)"
    if not os.path.isdir(rp):
        return "(not a folder)"
    rows = []
    try:
        names = sorted(os.listdir(rp), key=str.lower)
    except OSError as e:
        return f"(couldn't open the folder: {e})"
    for name in names[:300]:
        full = os.path.join(rp, name)
        try:
            if os.path.isdir(full):
                rows.append(f"[dir]  {name}/")
            else:
                rows.append(f"       {name}  ({os.path.getsize(full):,} bytes)")
        except OSError:
            continue
    if not rows:
        return "(empty folder)"
    if len(names) > 300:
        rows.append(f"…(+{len(names) - 300} more)")
    return "\n".join(rows)


def read_file(path: str, budget: int = _READ_BUDGET) -> str:
    """The text contents of a file she's allowed to read (text files only)."""
    rp = _within_roots(path)
    if not rp:
        return "(that file isn't in a folder he's shared with you)"
    if not os.path.isfile(rp):
        return "(not a file)"
    try:
        size = os.path.getsize(rp)
    except OSError as e:
        return f"(couldn't access the file: {e})"
    if size > _MAX_FILE_BYTES:
        return f"(that file is too large to read — {size:,} bytes)"
    ext = os.path.splitext(rp)[1].lower()
    try:
        with open(rp, "rb") as f:
            raw = f.read(budget + 1)
    except OSError as e:
        return f"(couldn't read the file: {e})"
    if b"\x00" in raw[:8192] or (ext and ext not in _TEXT_EXT and _looks_binary(raw)):
        return "(this looks like a binary file, not something I can read as text)"
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return "(couldn't decode this file as text)"
    if len(text) > budget:
        text = text[:budget] + "\n…(truncated)"
    return text or "(the file is empty)"


def _looks_binary(raw: bytes) -> bool:
    """Heuristic: lots of non-text bytes in the head ⇒ treat as binary."""
    if not raw:
        return False
    sample = raw[:4096]
    textish = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b <= 126)
    return textish / len(sample) < 0.7


def roots_digest(per_root_entries: int = 40) -> str:
    """A snapshot of the granted folders (and a shallow listing of each) so she
    knows what's available to look at. Empty string when nothing is shared."""
    if not _roots:
        return ""
    parts = [f"Folders he's shared with you ({len(_roots)}), read-only:"]
    for root in _roots:
        parts.append(f"• {root}")
        try:
            names = sorted(os.listdir(root), key=str.lower)
        except OSError:
            parts.append("    (couldn't list it)")
            continue
        for name in names[:per_root_entries]:
            full = os.path.join(root, name)
            try:
                marker = "[dir] " if os.path.isdir(full) else "      "
            except OSError:
                marker = "      "
            parts.append(f"    {marker}{name}")
        if len(names) > per_root_entries:
            parts.append(f"    …(+{len(names) - per_root_entries} more)")
    return "\n".join(parts)
