"""Edge-case sweep for the ingest pipeline.

These tests stress file discovery, parsing, and chunking with awkward
real-world inputs: empty files, binary bytes in supported extensions,
unicode paths and identifiers, deep nesting, ignored directories,
extremely long lines, CRLF endings, BOMs, and unusual Markdown.
"""
from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingest.chunker import create_chunks
from app.ingest.discover import discover_files


class TestTokenizer:
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_many(self, texts: list[str]) -> list[int]:
        return [self.count(text) for text in texts]


def _use_test_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr("app.ingest.splitter.get_tokenizer", lambda model_name: TestTokenizer())


def _chunk_types(chunks) -> set[str]:
    return {chunk.metadata["chunk_type"] for chunk in chunks}


def _splittable_settings() -> Settings:
    return Settings(max_chunk_tokens=20, chunk_overlap_tokens=4)


def test_empty_and_whitespace_only_files_do_not_crash(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "empty.py").write_bytes(b"")
    (tmp_path / "blank.md").write_bytes(b"   \n\n  ")
    (tmp_path / "blank.txt").write_text("   \n\t\n", encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())

    # Empty Python still yields a file metadata chunk; empty markdown/text yield nothing.
    assert _chunk_types(chunks) == {"file_metadata"}
    file_chunk = chunks[0]
    assert file_chunk.metadata["symbol"] == "empty.py"
    assert file_chunk.metadata["file_path"] == "empty.py"
    assert file_chunk.metadata["has_parse_errors"] is False


def test_empty_repository_produces_no_chunks(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    assert create_chunks(tmp_path, Settings()) == []


def test_binary_bytes_in_supported_extensions_do_not_crash(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "bin.py").write_bytes(b"\x00\x01\xff def foo():\n    pass\n")
    (tmp_path / "bin.md").write_bytes(b"\x00# Head\xff\nbody\n")
    (tmp_path / "bin.txt").write_bytes(b"\x00\xff binary data \x00")

    chunks = create_chunks(tmp_path, Settings())

    assert {chunk.metadata["file_path"] for chunk in chunks} == {"bin.py", "bin.md", "bin.txt"}
    assert any(
        chunk.metadata["file_path"] == "bin.py" and chunk.metadata["chunk_type"] == "function"
        for chunk in chunks
    )
    assert any(chunk.metadata["has_parse_errors"] for chunk in chunks)
    text_chunks = [chunk for chunk in chunks if chunk.metadata["file_path"] == "bin.txt"]
    assert len(text_chunks) == 1
    assert "binary data" in text_chunks[0].content


def test_ignored_dirs_are_excluded_and_deep_files_are_found(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.pyc", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__" / "c.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "node_modules").mkdir(parents=True)
    (tmp_path / "pkg" / "node_modules" / "j.txt").write_text("x", encoding="utf-8")
    (tmp_path / "nest" / ".venv").mkdir(parents=True)
    (tmp_path / "nest" / ".venv" / "lib.py").write_text("y = 1\n", encoding="utf-8")
    deep = tmp_path / "a" / "b" / "c" / "d" / "deep.py"
    deep.parent.mkdir(parents=True)
    deep.write_text("def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "README.MD").write_text("# Hi\n", encoding="utf-8")
    (tmp_path / "UPPER.TXT").write_text("hi", encoding="utf-8")
    (tmp_path / "skip.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all:", encoding="utf-8")

    discovered = [str(path.relative_to(tmp_path)) for path in discover_files(tmp_path, Settings())]
    assert discovered == ["README.MD", "UPPER.TXT", "a/b/c/d/deep.py"]

    chunks = create_chunks(tmp_path, Settings())
    symbols = {chunk.metadata["symbol"] for chunk in chunks}
    assert "helper" in symbols
    helper = next(chunk for chunk in chunks if chunk.metadata["symbol"] == "helper")
    # Module name is derived from the nested relative path.
    assert helper.metadata["module"] == "a.b.c.d.deep"


def test_unicode_paths_and_identifiers_are_indexed(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    source = tmp_path / "café" / "中文" / "módulo.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def 获取用户() -> str:\n    return \"ok\"\n",
        encoding="utf-8",
    )
    readme = tmp_path / "café" / "README 中文.md"
    readme.write_text("# 中文标题\n\n正文内容。\n", encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())

    by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}
    assert "获取用户" in by_symbol
    assert "中文标题" in by_symbol
    assert by_symbol["获取用户"].metadata["file_path"] == "café/中文/módulo.py"
    assert by_symbol["获取用户"].metadata["module"] == "café.中文.módulo"
    assert by_symbol["获取用户"].metadata["has_parse_errors"] is False
    assert by_symbol["中文标题"].metadata["parent_headings"] == ""


def test_markdown_heading_jumps_duplicates_and_empty_sections(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "guide.md").write_text(
        "\n".join(
            [
                "# Top",
                "",
                "intro text",
                "",
                "### Deep",
                "",
                "deep text",
                "",
                "# Top",
                "",
                "second top section",
                "",
                "## Empty",
                "",
                "## Next",
                "",
                "content under next",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("some plain notes without headings", encoding="utf-8")

    chunks = create_chunks(tmp_path, Settings())
    sections = [chunk for chunk in chunks if chunk.metadata["file_path"] == "guide.md"]

    top_chunks = [chunk for chunk in sections if chunk.metadata["symbol"] == "Top"]
    # Duplicate headings keep the same symbol but produce separate chunks.
    assert len(top_chunks) == 2
    assert top_chunks[0].metadata["start_line"] != top_chunks[1].metadata["start_line"]

    deep = next(chunk for chunk in sections if chunk.metadata["symbol"] == "Deep")
    assert deep.metadata["heading_level"] == 3
    assert deep.metadata["parent_headings"] == "Top"

    # A heading with no body still yields a section chunk whose content is only the heading line.
    empty_chunk = next(chunk for chunk in sections if chunk.metadata["symbol"] == "Empty")
    assert empty_chunk.content.strip() == "## Empty"
    next_section = next(chunk for chunk in sections if chunk.metadata["symbol"] == "Next")
    assert "content under next" in next_section.content

    text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text_file"]
    assert len(text_chunks) == 1


def test_oversized_single_line_never_splits_into_garbage(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    long_line = "# " + "word " * 400
    (tmp_path / "long.md").write_text(long_line, encoding="utf-8")
    long_txt = "x" * 2000
    (tmp_path / "long.txt").write_text(long_txt, encoding="utf-8")

    chunks = create_chunks(tmp_path, _splittable_settings())

    for chunk in chunks:
        assert chunk.metadata.get("is_chunk_part") is not True
    md_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "markdown_section"]
    txt_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text_file"]
    # A single physical line cannot be split, so the chunk is kept intact.
    assert len(md_chunks) == 1 and md_chunks[0].content == long_line
    assert len(txt_chunks) == 1 and txt_chunks[0].content == long_txt


def test_large_text_file_is_split_into_overlapping_parts(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    lines = [f"value {index}" for index in range(120)]
    (tmp_path / "big.txt").write_text("\n".join(lines), encoding="utf-8")

    chunks = create_chunks(tmp_path, _splittable_settings())
    parts = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text_file"]

    assert len(parts) > 1
    assert all(chunk.metadata["is_chunk_part"] for chunk in parts)
    indexes = [part.metadata["part_index"] for part in parts]
    assert indexes == list(range(1, len(parts) + 1))
    assert len({part.metadata["parent_chunk_id"] for part in parts}) == 1
    assert parts[0].metadata["start_line"] == 1
    assert parts[-1].metadata["end_line"] == len(lines)

    # Union of every part reconstructs the original body (overlap duplicates allowed).
    part_lines = [line for part in parts for line in part.content.splitlines()]
    assert set(part_lines) == set(lines)

    # Consecutive parts share overlapping source lines from the previous tail.
    first_of_second = parts[1].content.splitlines()[0]
    assert first_of_second in parts[0].content.splitlines()


def test_crlf_line_endings_are_handled(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "crlf.py").write_bytes(b"def bar():\r\n    return 1\r\n")
    (tmp_path / "crlf.txt").write_bytes(b"line one\r\nline two\r\n")

    chunks = create_chunks(tmp_path, Settings())
    by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}

    assert "bar" in by_symbol
    assert by_symbol["bar"].metadata["has_parse_errors"] is False
    assert "def bar():" in by_symbol["bar"].content
    assert by_symbol["bar"].metadata["signature"] == "def bar():"
    txt_chunk = next(chunk for chunk in chunks if chunk.metadata["file_path"] == "crlf.txt")
    assert txt_chunk.content.splitlines() == ["line one", "line two"]


def test_bom_prefixed_text_file_is_indexed(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "bom.txt").write_bytes(b"\xef\xbb\xbfhello\nworld\n")

    chunks = create_chunks(tmp_path, Settings())
    txt_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text_file"]
    assert len(txt_chunks) == 1
    assert "hello" in txt_chunks[0].content
    assert "world" in txt_chunks[0].content


def test_modern_python_syntax_parses_without_errors(tmp_path: Path, monkeypatch) -> None:
    _use_test_tokenizer(monkeypatch)
    (tmp_path / "modern.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "CONFIG: dict[str, list[int]] = {\"ports\": [80, 443]}",
                "",
                "def describe_status(code: int) -> str:",
                "    match code:",
                "        case 200:",
                "            return \"ok\"",
                "        case err if err >= 500:",
                "            return f\"server error: {err}\"",
                "        case _:",
                "            return \"unknown\"",
                "",
                "async def handler(payload: dict) -> str | None:",
                "    if (value := payload.get(\"name\")) is None:",
                "        return None",
                "",
                "    async def inner() -> str:",
                "        return f\"hi {value}\"",
                "",
                "    return await _run(inner)",
                "",
                "def _run(fn):",
                "    return fn()",
                "",
                "identity = lambda value: value",
                "",
                "if __name__ == \"__main__\":",
                "    print(describe_status(200))",
            ]
        ),
        encoding="utf-8",
    )

    chunks = create_chunks(tmp_path, Settings())
    by_symbol = {chunk.metadata["symbol"]: chunk for chunk in chunks}

    assert not any(chunk.metadata["has_parse_errors"] for chunk in chunks)
    assert "describe_status" in by_symbol
    assert "handler" in by_symbol
    assert by_symbol["handler"].metadata["is_async"] is True
    assert "_run" in by_symbol["handler"].metadata["calls"]
    assert "payload.get" in by_symbol["handler"].metadata["calls"]
    assert by_symbol["handler"].metadata["parent_symbol"] == ""