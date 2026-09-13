from __future__ import annotations

from enum import Enum


class ChunkType(str, Enum):
    CLASS = "class"
    ENUM = "enum"
    FILE_METADATA = "file_metadata"
    FUNCTION = "function"
    IMPORTS = "imports"
    MARKDOWN_SECTION = "markdown_section"
    METHOD = "method"
    PARSE_ERROR = "parse_error"
    TEXT_FILE = "text_file"
    TS_INTERFACE = "ts_interface"
    TYPE_ALIAS = "type_alias"
