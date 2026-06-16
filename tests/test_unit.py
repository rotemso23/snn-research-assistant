"""
Unit tests for pure utility functions.

These tests require no API key, no ChromaDB, and no embedding model.
They test logic that is independent of any AI component.

Run with:
    pytest tests/test_unit.py -v
"""

import pytest
from langchain_core.documents import Document

from src.retriever import _is_hebrew_dominant
from src.generator import _extract_cited_sources, _build_context_block, generate
from src.agent_graph import route_after_router, route_after_grade, MAX_RETRIES, _build_context_block, generate
from src.graph import route_after_grading


# ── _is_hebrew_dominant ───────────────────────────────────────────────────────

class TestIsHebrewDominant:

    def test_empty_string_returns_false(self):
        """Empty input has no letters — should not crash or return True."""
        assert _is_hebrew_dominant("") is False

    def test_only_punctuation_returns_false(self):
        """Non-alphabetic characters only — letter count is zero, should return False."""
        assert _is_hebrew_dominant("123 !@# ...") is False

    def test_all_english_returns_false(self):
        assert _is_hebrew_dominant("Hello world") is False

    def test_all_hebrew_returns_true(self):
        assert _is_hebrew_dominant("שלום שלום שלום") is True

    def test_exactly_at_threshold_returns_false(self):
        """Ratio == threshold (0.2) should return False — the check is strictly >."""
        # 1 Hebrew letter out of 5 alphabetic = exactly 0.20
        # "aaaaש" → letters: a,a,a,a,ש → 1/5 = 0.20 → not > 0.20 → False
        assert _is_hebrew_dominant("aaaaש") is False

    def test_just_above_threshold_returns_true(self):
        """Ratio just above threshold (0.2) should return True."""
        # 2 Hebrew letters out of 5 alphabetic = 0.40 → True
        # "aaaשש" → letters: a,a,a,ש,ש → 2/5 = 0.40 → True
        assert _is_hebrew_dominant("aaaשש") is True

    def test_mixed_mostly_english_returns_false(self):
        """A single Hebrew word in an otherwise-English sentence should not trigger."""
        # "The word שלום appears here" — 4 Hebrew letters out of many English
        text = "The word שלום appears here"
        assert _is_hebrew_dominant(text) is False

    def test_mixed_mostly_hebrew_returns_true(self):
        """Mostly Hebrew text with a few English letters should return True."""
        text = "שלום hi שלום שלום"
        assert _is_hebrew_dominant(text) is True


# ── _extract_cited_sources ────────────────────────────────────────────────────

def _make_doc(filename: str) -> Document:
    """Helper: create a Document with the given filename as source metadata."""
    return Document(page_content="some content", metadata={"source": f"/papers/{filename}"})


class TestExtractCitedSources:

    def test_single_valid_citation_returned(self):
        chunks = [_make_doc("paper_a.pdf"), _make_doc("paper_b.pdf")]
        answer = "The LIF model was described in (paper_a.pdf)."
        assert _extract_cited_sources(answer, chunks) == ["paper_a.pdf"]

    def test_hallucinated_filename_excluded(self):
        """A filename that does not match any chunk source should be filtered out."""
        chunks = [_make_doc("paper_a.pdf")]
        answer = "See (paper_a.pdf) and also (invented_paper.pdf)."
        assert _extract_cited_sources(answer, chunks) == ["paper_a.pdf"]

    def test_duplicate_citation_deduplicated(self):
        """The same source cited twice should appear only once in the result."""
        chunks = [_make_doc("paper_a.pdf")]
        answer = "First (paper_a.pdf). Also confirmed in (paper_a.pdf)."
        assert _extract_cited_sources(answer, chunks) == ["paper_a.pdf"]

    def test_multiple_valid_citations_in_order(self):
        """Multiple valid citations should be returned in order of first appearance."""
        chunks = [_make_doc("paper_a.pdf"), _make_doc("paper_b.pdf")]
        answer = "Fact one (paper_b.pdf). Fact two (paper_a.pdf)."
        assert _extract_cited_sources(answer, chunks) == ["paper_b.pdf", "paper_a.pdf"]

    def test_no_citation_fallback_returns_all_sources(self):
        """When the model cites nothing, fall back to returning all chunk sources."""
        chunks = [_make_doc("paper_a.pdf"), _make_doc("paper_b.pdf")]
        answer = "The model uses spike-based communication."  # no (file.pdf) pattern
        result = _extract_cited_sources(answer, chunks)
        assert set(result) == {"paper_a.pdf", "paper_b.pdf"}

    def test_all_hallucinated_citations_trigger_fallback(self):
        """If every cited filename is hallucinated, fall back to all chunk sources."""
        chunks = [_make_doc("paper_a.pdf")]
        answer = "As shown in (made_up.pdf) and (also_fake.pdf)."
        result = _extract_cited_sources(answer, chunks)
        assert result == ["paper_a.pdf"]


# ── _build_context_block ──────────────────────────────────────────────────────

class TestBuildContextBlock:

    def test_empty_list_returns_empty_string(self):
        assert _build_context_block([]) == ""

    def test_single_chunk_formatted_correctly(self):
        doc = Document(page_content="Neurons fire spikes.", metadata={"source": "/papers/thesis.pdf"})
        result = _build_context_block([doc])
        assert result == "[1] Source: thesis.pdf\nNeurons fire spikes."

    def test_multiple_chunks_numbered_in_order(self):
        docs = [
            Document(page_content="First fact.", metadata={"source": "/papers/paper_a.pdf"}),
            Document(page_content="Second fact.", metadata={"source": "/papers/paper_b.pdf"}),
        ]
        result = _build_context_block(docs)
        assert "[1] Source: paper_a.pdf\nFirst fact." in result
        assert "[2] Source: paper_b.pdf\nSecond fact." in result

    def test_missing_source_metadata_uses_unknown(self):
        doc = Document(page_content="Some content.", metadata={})
        result = _build_context_block([doc])
        assert "Source: unknown" in result


# ── generate() with empty chunks ─────────────────────────────────────────────

class TestGenerateEmptyChunks:

    def test_returns_i_dont_know_with_no_api_call(self):
        """generate() must short-circuit when chunks=[] — no API key needed."""
        result = generate("What is a spiking neuron?", [])
        assert result["sources"] == []
        assert "don't know" in result["answer"].lower()


# ── route_after_router ────────────────────────────────────────────────────────

class TestRouteAfterRouter:

    def test_multi_part_routes_to_decompose(self):
        state = {"query_type": "multi_part"}
        assert route_after_router(state) == "decompose_node"

    def test_thesis_specific_routes_to_retrieve(self):
        state = {"query_type": "thesis_specific"}
        assert route_after_router(state) == "retrieve_rerank"

    def test_conceptual_snn_routes_to_retrieve(self):
        state = {"query_type": "conceptual_snn"}
        assert route_after_router(state) == "retrieve_rerank"

    def test_simple_factual_routes_to_retrieve(self):
        state = {"query_type": "simple_factual"}
        assert route_after_router(state) == "retrieve_rerank"


# ── route_after_grade ─────────────────────────────────────────────────────────

class TestRouteAfterGrade:

    def test_need_more_within_budget_retries_retrieval(self):
        """grade_node flagged insufficient context and retries remain → retrieve_rerank."""
        state = {"need_more": True, "retry_count": 1}
        assert route_after_grade(state) == "retrieve_rerank"

    def test_need_more_budget_exhausted_goes_to_answer(self):
        """retry_count exceeds MAX_RETRIES → answer_node regardless of need_more."""
        state = {"need_more": True, "retry_count": MAX_RETRIES + 1}
        assert route_after_grade(state) == "answer_node"

    def test_no_more_context_needed_goes_to_answer(self):
        """grade_node satisfied → answer_node directly."""
        state = {"need_more": False, "retry_count": 0}
        assert route_after_grade(state) == "answer_node"
