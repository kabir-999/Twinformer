"""Questions like "translate the first slide" or "summarise page 5" name a position, not a topic: read that exact slide/page."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.qa import SourcePassage, clean_text
from app.rag.service import RagService

MAX_FILES = 3


@dataclass
class Location:
    position: str | int      # "first", "last" or a 1-based number
    unit: str                # "slide" or "page"


def parse_location(question: str) -> Location | None:
    q = question.lower()
    if re.search(r"\btitle slide\b", q):
        return Location("first", "slide")
    m = re.search(r"\b(first|1st|last|final)\s+(slide|page)\b", q)
    if m:
        return Location("last" if m.group(1) in ("last", "final") else "first", m.group(2))
    m = re.search(r"\b(slide|page)\s*(?:no\.?|number|#)?\s*(\d{1,4})\b", q)
    if m:
        return Location(int(m.group(2)), m.group(1))
    return None


def locate_sources(rag: RagService, location: Location, documents: list[str], keep_lines: bool = False) -> list[SourcePassage]:
    """The requested slide/page from each selected file that has that kind of unit (at most 3 files)."""
    sources: list[SourcePassage] = []
    for doc in (documents or rag.list_documents()):
        try:
            segments = [(label, text) for label, text in rag.segments(doc) if label.startswith(location.unit + " ")]
        except KeyError:
            continue
        if not segments:
            continue
        if location.position == "first":
            label, text = segments[0]
        elif location.position == "last":
            label, text = segments[-1]
        else:
            match = next(((l, t) for l, t in segments if l == f"{location.unit} {location.position}"), None)
            if match is None:
                continue
            label, text = match
        cleaned = clean_text(text, keep_lines=keep_lines)
        sources.append(SourcePassage(rank=len(sources) + 1, document=doc, location=label, score=1.0,
                                     text=cleaned or f"(This {location.unit} has no extractable text, for example because it is only an image.)"))
        if len(sources) >= MAX_FILES:
            break
    return sources
