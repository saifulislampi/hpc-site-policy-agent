"""Persistent corpus construction and transient retrieval."""

from corpus.chunking import build_corpus_records
from corpus.retrieval import FIELD_QUERIES, LexicalRetriever
from corpus.store import CorpusSnapshot, CorpusStore

__all__ = [
    "build_corpus_records",
    "FIELD_QUERIES",
    "LexicalRetriever",
    "CorpusSnapshot",
    "CorpusStore",
]

