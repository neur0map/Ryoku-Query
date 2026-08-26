from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

RISKS = frozenset({"informational", "state-changing", "destructive"})
MIN_FUSED_MARGIN = 0.01
CARD_FIELDS = frozenset(
    {
        "id",
        "title",
        "examples",
        "keywords",
        "exact_terms",
        "answer",
        "risk",
        "docs_url",
        "source_hints",
        "clarifies_with",
    }
)


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class SupportCard:
    id: str
    title: str
    examples: tuple[str, ...]
    keywords: tuple[str, ...]
    exact_terms: tuple[str, ...]
    answer: str
    risk: str
    docs_url: str | None
    source_hints: tuple[str, ...]
    clarifies_with: tuple[str, ...]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _text(value, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{context} {field} must be a non-empty string")
    return value.strip()


def _strings(value, field: str, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CatalogError(f"{context} {field} must be a list of non-empty strings")
    cleaned = tuple(item.strip() for item in value)
    if len({normalize_text(item) for item in cleaned}) != len(cleaned):
        raise CatalogError(f"{context} {field} must contain distinct values")
    return cleaned


def _parse_card(row: object, index: int) -> SupportCard:
    context = f"card #{index}"
    if not isinstance(row, dict):
        raise CatalogError(f"{context} must be an object")
    unknown = sorted(set(row) - CARD_FIELDS)
    if unknown:
        raise CatalogError(f"{context} has unknown fields: {', '.join(unknown)}")
    missing = sorted(CARD_FIELDS - set(row))
    if missing:
        raise CatalogError(f"{context} is missing fields: {', '.join(missing)}")

    card_id = _text(row["id"], "id", context)
    context = f"card {card_id}"
    title = _text(row["title"], "title", context)
    examples = _strings(row["examples"], "examples", context)
    if len(examples) < 2:
        raise CatalogError(f"{context} examples must contain at least two values")
    keywords = _strings(row["keywords"], "keywords", context)
    exact_terms = _strings(row["exact_terms"], "exact_terms", context)
    answer = _text(row["answer"], "answer", context)
    risk = row["risk"]
    if risk not in RISKS:
        raise CatalogError(f"{context} risk must be one of {sorted(RISKS)}")
    docs_url = row["docs_url"]
    if docs_url is not None and (
        not isinstance(docs_url, str) or not docs_url.startswith("https://")
    ):
        raise CatalogError(f"{context} docs_url must be HTTPS or null")
    source_hints = _strings(row["source_hints"], "source_hints", context)
    clarifies_with = _strings(
        row["clarifies_with"], "clarifies_with", context
    )
    if card_id in clarifies_with:
        raise CatalogError(f"{context} clarifies_with cannot reference itself")
    return SupportCard(
        id=card_id,
        title=title,
        examples=examples,
        keywords=keywords,
        exact_terms=exact_terms,
        answer=answer,
        risk=risk,
        docs_url=docs_url,
        source_hints=source_hints,
        clarifies_with=clarifies_with,
    )


def load_support_cards(path: Path) -> list[SupportCard]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise CatalogError("support catalog must be a non-empty JSON list")
    cards = [_parse_card(row, index) for index, row in enumerate(raw)]
    by_id: dict[str, SupportCard] = {}
    for card in cards:
        if card.id in by_id:
            raise CatalogError(f"duplicate support card id: {card.id}")
        by_id[card.id] = card
    for card in cards:
        missing = sorted(set(card.clarifies_with) - set(by_id))
        if missing:
            raise CatalogError(
                f"card {card.id} clarifies_with references missing intents: "
                + ", ".join(missing)
            )
    return cards


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class RankedIntent:
    card: SupportCard
    fused_score: float
    exact_score: float
    lexical_score: float
    semantic_score: float
    channels: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalDecision:
    kind: str
    card: SupportCard | None = None
    alternatives: tuple[SupportCard, ...] = ()
    ranked: tuple[RankedIntent, ...] = ()


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+_.:/-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "should",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    }
)
_DANGER_PHRASES = (
    "delete all",
    "delete every",
    "do not care",
    "destructive",
    "erase",
    "factory defaults",
    "repartition",
    "reset the entire",
    "wipe",
    "without checking",
)
_DANGER_PATTERN = re.compile(
    r"\b(remove|delete|reset)\b.{0,50}\b(all|every|entire|whole)\b"
    r"|\b(all|every|entire|whole)\b.{0,50}\b(remove|delete|reset)\b"
    r"|\b(skip|ignore|bypass)\b.{0,50}\b(warning|check|confirmation|prompt)\b"
    r"|\bwithout\b.{0,30}\b(check|warning|confirmation)\b"
)
_AMBIGUITY_PHRASES = (
    "cannot tell whether",
    "do you mean",
    "don't know whether",
    "not sure",
    "unsure",
)
_AMBIGUITY_PATTERN = re.compile(
    r"\b(which|should|do you|are we|is this)\b.{0,100}\bor\b"
    r"|\bbetween\b.{0,100}\band\b"
)

def requires_safety_confirmation(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(
        _DANGER_PATTERN.search(normalized)
        or any(phrase in normalized for phrase in _DANGER_PHRASES)
    )


def _raw_tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(text)))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token for token in _raw_tokens(text) if token not in _STOPWORDS
    )


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _contains_phrase(query: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(query):
        return False
    width = len(phrase)
    return any(
        query[index : index + width] == phrase
        for index in range(len(query) - width + 1)
    )


def _ranks(scores: dict[str, float], ids: tuple[str, ...]) -> dict[str, int]:
    ordered = sorted(ids, key=lambda card_id: (-scores[card_id], card_id))
    ranks: dict[str, int] = {}
    previous = None
    rank = 0
    for index, card_id in enumerate(ordered, start=1):
        score = scores[card_id]
        if previous is None or score != previous:
            rank = index
            previous = score
        ranks[card_id] = rank
    return ranks


class SupportRetriever:
    def __init__(self, cards: list[SupportCard], encoder: Encoder):
        if not cards:
            raise ValueError("SupportRetriever requires at least one card")
        self.cards = tuple(cards)
        self.encoder = encoder
        self._ids = tuple(card.id for card in cards)
        self._documents = tuple(self._document_tokens(card) for card in cards)
        self._document_terms = tuple(
            frozenset(document) for document in self._documents
        )
        self._index_by_id = {
            card.id: index for index, card in enumerate(cards)
        }
        self._doc_lengths = tuple(len(document) for document in self._documents)
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        self._idf = self._build_idf()
        examples: list[str] = []
        self._example_ranges: list[tuple[int, int]] = []
        for card in cards:
            start = len(examples)
            examples.extend(card.examples)
            self._example_ranges.append((start, len(examples)))
        self._example_vectors = _normalize_rows(encoder.encode(examples))

    @staticmethod
    def _document_tokens(card: SupportCard) -> tuple[str, ...]:
        tokens: list[str] = []
        tokens.extend(_tokens(card.title) * 2)
        for keyword in card.keywords:
            tokens.extend(_tokens(keyword) * 2)
        for term in card.exact_terms:
            tokens.extend(_tokens(term) * 3)
        for hint in card.source_hints:
            tokens.extend(_tokens(hint))
        for example in card.examples:
            tokens.extend(_tokens(example))
        return tuple(tokens)

    def _build_idf(self) -> dict[str, float]:
        frequencies: dict[str, int] = {}
        for document in self._documents:
            for token in set(document):
                frequencies[token] = frequencies.get(token, 0) + 1
        count = len(self._documents)
        return {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in frequencies.items()
        }

    def _exact_scores(self, query: tuple[str, ...]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for card in self.cards:
            matches = [
                len(tokens)
                for term in card.exact_terms
                if _contains_phrase(query, tokens := _raw_tokens(term))
            ]
            scores[card.id] = float(max(matches, default=0))
        return scores

    def _lexical_scores(self, query: tuple[str, ...]) -> dict[str, float]:
        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        for card, document, length in zip(
            self.cards, self._documents, self._doc_lengths
        ):
            counts: dict[str, int] = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token in set(query):
                frequency = counts.get(token, 0)
                if frequency == 0:
                    continue
                denominator = frequency + k1 * (
                    1 - b + b * length / self._avg_doc_length
                )
                score += self._idf.get(token, 0.0) * (
                    frequency * (k1 + 1) / denominator
                )
            scores[card.id] = score
        return scores

    def _semantic_scores(self, query: str) -> dict[str, float]:
        vector = _normalize_rows(self.encoder.encode([query]))[0]
        similarities = self._example_vectors @ vector
        scores: dict[str, float] = {}
        for card, (start, end) in zip(self.cards, self._example_ranges):
            values = np.sort(similarities[start:end])
            scores[card.id] = float(np.mean(values[-min(2, len(values)) :]))
        return scores

    def retrieve(self, query: str) -> RetrievalDecision:
        normalized = normalize_text(query)
        exact_query_tokens = _raw_tokens(normalized)
        query_tokens = _tokens(normalized)
        if not query_tokens:
            return RetrievalDecision("no_match")

        exact = self._exact_scores(exact_query_tokens)
        lexical = self._lexical_scores(query_tokens)
        semantic = self._semantic_scores(normalized)
        exact_ranks = _ranks(exact, self._ids)
        lexical_ranks = _ranks(lexical, self._ids)
        semantic_ranks = _ranks(semantic, self._ids)
        ranked: list[RankedIntent] = []
        for card in self.cards:
            channels: list[str] = []
            fused = 0.0
            if exact[card.id] > 0:
                channels.append("exact")
                fused += 3.0 / (10 + exact_ranks[card.id])
            if lexical[card.id] > 0:
                channels.append("lexical")
                fused += 1.5 / (10 + lexical_ranks[card.id])
            if semantic[card.id] >= 0.25:
                channels.append("semantic")
                fused += 1.0 / (10 + semantic_ranks[card.id])
            ranked.append(
                RankedIntent(
                    card=card,
                    fused_score=fused,
                    exact_score=exact[card.id],
                    lexical_score=lexical[card.id],
                    semantic_score=semantic[card.id],
                    channels=tuple(channels),
                )
            )
        ranked.sort(key=lambda item: (-item.fused_score, item.card.id))
        top = ranked[0]
        snapshot = tuple(ranked[:5])
        unique_exact = top.exact_score > 0 and sum(
            item.exact_score == top.exact_score for item in ranked
        ) == 1
        dangerous = requires_safety_confirmation(normalized)
        if top.fused_score == 0 or "lexical" not in top.channels:
            return RetrievalDecision("no_match", ranked=snapshot)
        if dangerous:
            return RetrievalDecision(
                "clarify", alternatives=(top.card,), ranked=snapshot
            )
        related = next(
            (
                item
                for item in ranked[1:]
                if item.card.id in top.card.clarifies_with
                or top.card.id in item.card.clarifies_with
            ),
            None,
        )
        explicit_ambiguity = (
            any(
                phrase in f" {normalized} "
                for phrase in _AMBIGUITY_PHRASES
            )
            or (
                " or " in f" {normalized} "
                and len(query_tokens) <= 5
            )
            or bool(_AMBIGUITY_PATTERN.search(normalized))
        )
        if top.card.risk == "destructive":
            if explicit_ambiguity and related is not None:
                return RetrievalDecision(
                    "clarify",
                    alternatives=(top.card, related.card),
                    ranked=snapshot,
                )
            if unique_exact or (
                "ryoku" in query_tokens and top.semantic_score >= 0.5
            ):
                return RetrievalDecision(
                    "clarify", alternatives=(top.card,), ranked=snapshot
                )
            return RetrievalDecision("no_match", ranked=snapshot)
        if related is not None:
            margin = top.fused_score - related.fused_score
            lexical_ratio = top.lexical_score / max(
                related.lexical_score, 1e-9
            )
            if explicit_ambiguity or (
                not unique_exact
                and margin < 0.02
                and lexical_ratio < 1.7
                and abs(top.semantic_score - related.semantic_score) < 0.05
            ):
                return RetrievalDecision(
                    "clarify",
                    alternatives=(top.card, related.card),
                    ranked=snapshot,
                )
        if (
            not unique_exact
            and len(ranked) > 1
            and top.fused_score - ranked[1].fused_score < MIN_FUSED_MARGIN
        ):
            return RetrievalDecision("no_match", ranked=snapshot)
        if unique_exact:
            return RetrievalDecision("answer", card=top.card, ranked=snapshot)
        top_match_count = len(
            set(query_tokens)
            & self._document_terms[self._index_by_id[top.card.id]]
        )
        lexical_runner_up = max(
            (item.lexical_score for item in ranked[1:]),
            default=0.0,
        )
        domain_named = "ryoku" in query_tokens or any(
            self._idf.get(token, 0.0) >= 3.0 for token in query_tokens
        )
        semantic_floor = 0.25 if domain_named else 0.40
        if (
            top_match_count >= 2
            and top.semantic_score >= semantic_floor
            and top.lexical_score >= 3.0
            and (
                lexical_runner_up == 0
                or top.lexical_score >= lexical_runner_up * 1.2
            )
        ):
            return RetrievalDecision("answer", card=top.card, ranked=snapshot)
        if (
            "semantic" in top.channels
            and lexical_ranks[top.card.id] <= 5
            and semantic_ranks[top.card.id] <= 5
            and top.semantic_score >= max(0.35, semantic_floor)
        ):
            return RetrievalDecision("answer", card=top.card, ranked=snapshot)
        return RetrievalDecision("no_match", ranked=snapshot)
