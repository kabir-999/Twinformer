import os

os.environ.setdefault("TRANSFORMER_TWIN_DB", "/tmp/transformer_twin_test.sqlite3")

from app.chat.service import ChatRequest, Route, route_question, run_turn, translation_result
from app.rag.locator import Location, locate_sources, parse_location
from app.rag.qa import ArchAnswer, SourcePassage, ThreeWayAnswer, dedupe_repeats
from app.rag.translate import split_for_translation, target_language


def test_parse_location_understands_slides_and_pages():
    assert parse_location("translate first slide to hindi") == Location("first", "slide")
    assert parse_location("Summarise the LAST page") == Location("last", "page")
    assert parse_location("what is on slide 7?") == Location(7, "slide")
    assert parse_location("explain page no. 12") == Location(12, "page")
    assert parse_location("show the title slide") == Location("first", "slide")
    assert parse_location("what is dvfs?") is None


class FakeRag:
    def __init__(self):
        self.segs = {"Sem 7/a.pptx": [("slide 1", "Green Data Science\nfor Sustainability"), ("slide 2", "Second"), ("slide 3", "")],
                     "MBA/b.pdf": [("page 1", "Page one text"), ("page 2", "Page two")]}

    def list_documents(self):
        return list(self.segs)

    def segments(self, name):
        return self.segs[name]

    def retrieve(self, question, top_k, documents=None, **_):
        from app.rag.service import Citation
        return [Citation(rank=1, document="MBA/b.pdf", chunk_id="x", char_start=0, char_end=3, location="page 1", score=0.4, text="Fallback passage text.")]


def test_locate_sources_reads_the_exact_unit_from_matching_files_only():
    rag = FakeRag()
    first = locate_sources(rag, Location("first", "slide"), [])
    assert [(s.document, s.location, s.text) for s in first] == [("Sem 7/a.pptx", "slide 1", "Green Data Science for Sustainability")]
    assert locate_sources(rag, Location("last", "page"), ["MBA/b.pdf"])[0].text == "Page two"
    assert locate_sources(rag, Location(9, "slide"), []) == []
    empty = locate_sources(rag, Location(3, "slide"), ["Sem 7/a.pptx"])
    assert "no extractable text" in empty[0].text


def test_language_detection_and_routing():
    assert target_language("translate first slide to hindi") == ("hi", "Hindi")
    assert target_language("translate it into Tamil") == ("ta", "Tamil")
    assert target_language("translate this") is None
    hindi = route_question("translate first slide to hindi")
    assert (hindi.architectures, hindi.language_code) == (["encoder_decoder"], "hi") and "encoder-decoder" in hindi.reason
    tamil = route_question("translate first slide to tamil")
    assert (tamil.architectures, tamil.language_code) == (["encoder_decoder"], "ta")


def test_translator_choice_prefers_nllb_then_cached_marian():
    from app.rag.translate import NLLB, pick_translator
    assert pick_translator("hi", False, cached=lambda m: m == NLLB) == NLLB
    assert pick_translator("hi", True, cached=lambda m: False) == NLLB
    assert pick_translator("hi", False, cached=lambda m: m == "Helsinki-NLP/opus-mt-en-hi") == "Helsinki-NLP/opus-mt-en-hi"
    assert pick_translator("ta", False, cached=lambda m: False) == NLLB


def test_split_normalizes_caps_and_separates_fragments():
    from app.rag.translate import normalize_caps
    assert normalize_caps("B.TECH · DATA SCIENCE CSE DJS23") == "B.TECH · Data Science CSE DJS23"
    pieces = split_for_translation("B.TECH · DATA SCIENCE · AY 2026–27\nIntroduction to Green Data Science. Subject: Green Data Science")
    assert pieces == ["B.TECH", "Data Science", "AY 2026–27", "Introduction to Green Data Science.", "Subject: Green Data Science"]
    long = split_for_translation("word " * 200, limit=100)
    assert len(long) > 5 and all(len(p) <= 100 for p in long)


def test_keep_lines_preserves_slide_structure():
    from app.rag.qa import clean_text
    assert clean_text("Title\n\n  Line two\nEfﬁcient", keep_lines=True) == "Title\nLine two\nEfficient"
    assert clean_text("Title\nLine two") == "Title Line two"


def test_dedupe_helper():
    looped = "This technique can reduce energy use. This technique can reduce energy use. Something else. This technique can reduce energy use."
    assert dedupe_repeats(looped) == "This technique can reduce energy use. Something else."


class FakeCompare:
    def available_models(self):
        return {"defaults": {"decoder_only": "Qwen/Qwen2.5-1.5B-Instruct"}}

    def get_model(self, arch, model_id, allow):
        return object()


def test_translation_translates_the_located_slide_and_skips_the_encoder():
    route = route_question("translate first slide to hindi")
    seen = {}
    judged = {}

    def judge(question, sources, answers, key, kind):
        judged["kind"] = kind
        from app.rag.qa import Verdict, AnswerVerdict
        return Verdict(source="groq", judge_model="m", winner="encoder_decoder", summary="Marian is more faithful.", per_answer=[AnswerVerdict(architecture=a.architecture, score=80, comment="c") for a in answers], caveat="c")
    result = translation_result("translate first slide to hindi", route, FakeRag(), FakeCompare(), [], False, "key", run_all=True,
                                marian=lambda loaded, t: seen.setdefault("marian", t) and "हिंदी", decoder=lambda loaded, t, lang: seen.setdefault("decoder", (t, lang)) and "hindi?", judge=judge,
                                picker=lambda code, allow: "Helsinki-NLP/opus-mt-en-hi")
    assert seen["marian"] == "Green Data Science\nfor Sustainability" and seen["decoder"] == ("Green Data Science\nfor Sustainability", "Hindi")
    assert [a.architecture for a in result.answers] == ["encoder_decoder", "decoder_only"]
    assert result.answers[0].model_id == "Helsinki-NLP/opus-mt-en-hi" and result.answers[0].answer == "हिंदी"
    assert result.sources[0].location == "slide 1" and judged["kind"] == "translation" and result.verdicts[0].winner == "encoder_decoder"
    assert any("Encoder-only models cannot generate" in n for n in result.notes)


def test_translation_without_a_language_asks_for_one_and_without_key_says_quality_unchecked():
    ask = translation_result("translate first slide", route_question("translate first slide"), FakeRag(), FakeCompare(), [], False, None, run_all=True)
    assert "which language" in ask.answers[0].error
    route = route_question("translate first slide to french")
    nokey = translation_result("translate first slide to french", route, FakeRag(), FakeCompare(), [], False, None, run_all=True, marian=lambda l, t: "Science", decoder=lambda l, t, g: "Science des données", picker=lambda c, a: "m")
    assert not nokey.verdicts and any("GROQ_API_KEY" in n for n in nokey.notes)
    auto = translation_result("translate first slide to french", route, FakeRag(), FakeCompare(), [], False, None, run_all=False, marian=lambda l, t: "Science", decoder=lambda l, t, g: "x", picker=lambda c, a: "m")
    assert [a.architecture for a in auto.answers] == ["encoder_decoder"]


def test_run_turn_uses_the_located_slide_for_non_translation_questions():
    seen = {}

    def answer_fn(request, rag, compare, key, fixed_sources=None):
        seen["fixed"] = fixed_sources
        return ThreeWayAnswer(question=request.question, sources=fixed_sources or [], answers=[ArchAnswer(architecture="decoder_only", model_id="m", method="x", answer="ok")], verdicts=[], notes=[])
    turn = run_turn(ChatRequest(question="Summarise the first slide", mode="all"), None, FakeRag(), None, None, answer_fn=answer_fn)
    assert seen["fixed"][0].location == "slide 1" and turn.route is None
    turn = run_turn(ChatRequest(question="What is on slide 99?", mode="all"), None, FakeRag(), None, None, answer_fn=answer_fn)
    assert seen["fixed"] is None and any("Could not find a slide 99" in n for n in turn.result.notes)
