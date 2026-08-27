from app.retrieval.filters import MetadataFilters
from app.vectorstore.chroma_store import _chroma_requested_count


def test_no_filters_requests_candidate_pool() -> None:
    assert _chroma_requested_count(None, top_k=5) == max(5 * 4, 5)


def test_language_filter_only_requests_top_k() -> None:
    filters = MetadataFilters(language="python")
    assert _chroma_requested_count(filters, top_k=5) == 5


def test_chunk_type_filter_only_requests_top_k() -> None:
    filters = MetadataFilters(chunk_type="method")
    assert _chroma_requested_count(filters, top_k=5) == 5


def test_path_prefix_flag_requests_candidate_pool() -> None:
    filters = MetadataFilters(path_prefix="app")
    assert _chroma_requested_count(filters, top_k=5) == max(5 * 4, 5)


def test_language_and_path_prefix_requests_candidate_pool() -> None:
    # Regression: language + path_prefix used to request only top_k, then the
    # client-side path filter could silently return fewer than top_k results.
    filters = MetadataFilters(language="python", path_prefix="app")
    assert _chroma_requested_count(filters, top_k=5) == max(5 * 4, 5)