from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_groq_key() -> str | None:
    """Reads GROQ_API_KEY from .env on every call (so editing the file needs no restart), then from the environment."""
    for path in (Path.cwd() / ".env", PROJECT_ROOT / ".env"):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                if name.strip() == "GROQ_API_KEY" and value.strip().strip("\"'"):
                    return value.strip().strip("\"'")
    return (os.getenv("GROQ_API_KEY") or "").strip() or None
