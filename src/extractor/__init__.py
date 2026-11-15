"""Article Extractor package: boilerplate stripping, metadata parsing, and validation."""

from src.extractor.extractor import extract_and_store, extract_article
from src.extractor.schemas import (
    DEFAULT_MAX_LINK_TO_TEXT_RATIO,
    DEFAULT_MIN_WORD_COUNT,
    ExtractionResult,
)

__all__ = [
    "DEFAULT_MAX_LINK_TO_TEXT_RATIO",
    "DEFAULT_MIN_WORD_COUNT",
    "ExtractionResult",
    "extract_and_store",
    "extract_article",
]
