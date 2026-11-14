"""Small shared lexical helpers for bounded literature queries."""

from __future__ import annotations

import re


_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
    "model", "failure", "metric", "hook", "least", "db", "more", "than",
    "less", "minimum", "maximum", "greater", "above", "below", "over", "under",
})
_INTENT_VERBS = frozenset({
    "improve", "improving", "improvement", "optimize", "optimise", "optimization",
    "optimisation", "reduce", "reducing", "increase", "increasing", "achieve",
    "better", "please", "research", "find", "papers", "paper",
})


def query_terms(value: str, *, maximum: int, intent: bool = False) -> tuple[str, ...]:
    """Keep ordered English search terms; retain untranslated intent elsewhere.

    arXiv's English metadata cannot reliably match arbitrary Chinese sentences.
    Callers retain the original intent as provenance and use canonical model,
    diagnostic and mechanism terms when available; this helper does not invent
    a translation or silently turn the sentence into a mandatory phrase.
    """

    stop = _STOP_WORDS | (_INTENT_VERBS if intent else frozenset())
    tokens = re.findall(r"(?<![\w])[a-z0-9][a-z0-9-]*(?![\w])", value.lower().replace("_", " "))
    return tuple(dict.fromkeys(token for token in tokens if 1 < len(token) <= 32
                               and re.search(r"[a-z]", token) and token not in stop))[:maximum]


def arxiv_clause(terms: tuple[str, ...], *, operator: str = "OR") -> str:
    """Encode already bounded lexical terms with explicit field/boolean scope."""

    if operator not in {"AND", "OR"}:
        raise ValueError("LITERATURE_QUERY_OPERATOR_INVALID")
    return "(" + f" {operator} ".join(f'all:"{term}"' for term in terms) + ")" if terms else ""
