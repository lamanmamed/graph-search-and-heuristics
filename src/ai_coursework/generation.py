"""Heuristic sentence completion over the same word-adjacency network.

This module does not call a language model. It searches transitions observed in the
source graph and uses small, interpretable linguistic signals to rank valid candidates.
"""

import heapq
import math


MAX_EXPANSIONS = 20_000
BEAM_WIDTH = 40
RARE_TOKEN = "<RARE>"

FUNCTION_WORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "from", "with",
    "and", "or", "but", "that", "this", "is", "are", "was", "were",
}
AUXILIARIES = {
    "be", "am", "is", "are", "was", "were", "been", "being", "do", "does",
    "did", "have", "has", "had", "will", "would", "should", "can", "could",
    "may", "might", "must",
}


def tokenize_text(text):
    """Tokenize prompts using the same lightweight punctuation convention."""
    text = text.replace("<CONTENT>", "content").replace("<content>", "content")
    text = text.replace(",", " , ").replace(".", " . ")
    return [token.strip().lower() for token in text.split() if token.strip()]


def simple_stem(word):
    """Apply a tiny suffix-based stemmer used only as a ranking feature."""
    word = word.lower()
    for suffix in ("ing", "ed", "ies", "s"):
        if not word.endswith(suffix):
            continue
        if suffix == "ies" and len(word) > 3:
            return word[:-3] + "y"
        if suffix == "s" and len(word) <= 3:
            continue
        return word[: -len(suffix)]
    return word


def guess_pos(word):
    """Guess a coarse part of speech from lexical and suffix rules."""
    word = word.lower()

    if word in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}:
        return "PRON"
    if word in {"the", "a", "an", "this", "that", "these", "those"}:
        return "DET"
    if word in {"in", "on", "at", "by", "for", "to", "from", "with", "of", "over", "under", "into"}:
        return "ADP"
    if word in {"and", "or", "but", "nor", "so", "yet"}:
        return "CONJ"
    if word.isdigit():
        return "NUM"
    if word.endswith("ly"):
        return "ADV"
    if word.endswith(("ous", "ful", "able", "ible", "al", "ive", "less", "ic")):
        return "ADJ"
    if word.endswith(("ing", "ed", "en", "ify", "ise", "ize")):
        return "V"
    return "N"


def pos_pair_score(left_pos, right_pos):
    """Score a small set of common part-of-speech transitions."""
    good_pairs = {
        ("DET", "N"): 1.0,
        ("ADJ", "N"): 1.0,
        ("N", "V"): 0.8,
        ("PRON", "V"): 1.0,
        ("V", "N"): 0.8,
        ("V", "DET"): 0.6,
        ("ADP", "DET"): 0.8,
        ("ADP", "N"): 0.7,
        ("ADV", "V"): 0.6,
        ("V", "ADV"): 0.4,
        ("N", "ADP"): 0.3,
        ("N", "CONJ"): 0.2,
    }
    if left_pos == right_pos:
        return 0.1
    return good_pairs.get((left_pos, right_pos), 0.0)


def char_overlap(left, right):
    """Return character-set Jaccard overlap between two words."""
    left_chars = set(left.lower())
    right_chars = set(right.lower())
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / len(left_chars | right_chars)


def _matrix_value(matrix, row, col):
    """Read either a NumPy-style or nested-list matrix."""
    try:
        return float(matrix[row, col])
    except (TypeError, IndexError):
        return float(matrix[row][col])


def _network_parts(text_network):
    nodes = text_network["nodes"]
    index = {token: i for i, token in enumerate(nodes)}
    return {
        "nodes": nodes,
        "index": index,
        "graph": text_network["graph"],
        "distance": text_network["distance_matrix"],
        "counts": text_network["count_matrix"],
        "token_counts": text_network["token_counts"],
        "rare_tokens": set(text_network["rare_tokens"]),
        "adjacency_counts": text_network["adjacency_counts"],
    }


def _neighbours(graph, token):
    try:
        return list(graph.neighbors(token))
    except Exception:
        return []


def _commonness(parts):
    """Combine normalized graph degree and token frequency into one hub score."""
    graph = parts["graph"]
    token_counts = parts["token_counts"]
    nodes = parts["nodes"]

    degrees = {}
    frequencies = {}
    for token in nodes:
        try:
            degrees[token] = float(graph.degree(token))
        except Exception:
            degrees[token] = 0.0
        frequencies[token] = float(token_counts.get(token, 0.0))

    max_degree = max(degrees.values(), default=1.0) or 1.0
    max_frequency = max(frequencies.values(), default=1.0) or 1.0

    return {
        token: 0.5 * (
            degrees[token] / max_degree
            + frequencies[token] / max_frequency
        )
        for token in nodes
    }


def _transition_strength(left, right, parts):
    """Measure how strongly two tokens are associated in the observed network."""
    index = parts["index"]
    if left not in index or right not in index:
        return 0.0

    undirected = _matrix_value(
        parts["counts"], index[left], index[right]
    )
    directed = float(parts["adjacency_counts"].get((left, right), 0.0))
    return undirected + 1.25 * directed


def _linguistic_bonus(left, right, recent_context, parts):
    """Combine interpretable local signals used to rank valid transitions."""
    bonus = 2.0 * pos_pair_score(guess_pos(left), guess_pos(right))
    bonus += 0.5 * char_overlap(left, right)

    if simple_stem(left) == simple_stem(right) and left != right:
        bonus += 0.6

    if recent_context:
        context_scores = [
            _transition_strength(context_token, right, parts)
            for context_token in recent_context[-3:]
        ]
        bonus += 0.25 * (sum(context_scores) / len(context_scores))

    return bonus


def _candidate_score(left, right, recent_context, parts, commonness):
    """Return a lower-is-better search cost for one valid transition."""
    index = parts["index"]
    if left not in index or right not in index:
        return math.inf

    base = _matrix_value(parts["distance"], index[left], index[right])
    hub_penalty = 2.0 * commonness.get(right, 0.0)
    function_penalty = 0.6 if right in FUNCTION_WORDS else 0.0
    bonus = _linguistic_bonus(left, right, recent_context, parts)

    return max(1e-6, base + hub_penalty + function_penalty - bonus)


def _can_end(tokens):
    """Check whether the generated span currently ends like a complete clause."""
    if not tokens:
        return False

    last = tokens[-1]
    if last in AUXILIARIES or last in {"it", "there", ",", "the", "a", "an"}:
        return False

    return guess_pos(last) in {"N", "V", "ADJ", "ADV", "NUM"}


def _can_begin(tokens):
    """Check whether the generated span starts like a plausible clause."""
    if not tokens:
        return False

    first = tokens[0]
    if first in AUXILIARIES or first in {",", "the", "a", "an"}:
        return False

    return guess_pos(first) in {"N", "V", "ADJ", "ADV", "PRON", "NUM"}


def _top_forward_candidates(token, used, recent, parts, commonness):
    candidates = []

    for neighbour in _neighbours(parts["graph"], token):
        if neighbour == RARE_TOKEN or neighbour in used:
            continue
        if neighbour in parts["rare_tokens"]:
            continue
        if not neighbour.isalpha() and neighbour not in {",", "."}:
            continue

        score = _candidate_score(token, neighbour, recent, parts, commonness)
        candidates.append((score, neighbour))

    candidates.sort()
    return candidates[:BEAM_WIDTH]


def _top_backward_candidates(token, used, recent, parts, commonness):
    candidates = []

    for predecessor in _neighbours(parts["graph"], token):
        if predecessor == RARE_TOKEN or predecessor in used:
            continue
        if predecessor in parts["rare_tokens"]:
            continue
        if not predecessor.isalpha() and predecessor not in {",", "."}:
            continue

        # Backward expansion still scores the real forward transition.
        score = _candidate_score(predecessor, token, recent, parts, commonness)
        candidates.append((score, predecessor))

    candidates.sort()
    return candidates[:BEAM_WIDTH]


def _resolve_anchor(token, parts):
    """Map an out-of-vocabulary prompt word to a simple lexical neighbour."""
    if token in parts["index"]:
        return token

    stem = simple_stem(token)
    best_token = None
    best_score = -1.0

    for candidate in parts["nodes"]:
        if candidate == RARE_TOKEN or not candidate.isalpha():
            continue

        score = char_overlap(token, candidate)
        if simple_stem(candidate) == stem:
            score += 1.0

        if score > best_score:
            best_token = candidate
            best_score = score

    return best_token


def complete_sentence(
    text_network,
    prompt="please believe my eyes <CONTENT>.",
):
    """Fill the placeholder by searching forward from the fixed prefix.

    Every generated transition must exist in the source graph. The heuristic scores
    only influence which valid candidates are explored first.
    """
    parts = _network_parts(text_network)
    commonness = _commonness(parts)
    tokens = tokenize_text(prompt)

    if "content" not in tokens:
        raise ValueError("Prompt must contain <CONTENT>.")

    split = tokens.index("content")
    prefix = tokens[:split]
    suffix = tokens[split + 1 :]

    if not prefix:
        raise ValueError("Forward completion needs a fixed prefix before <CONTENT>.")

    goal = "."
    anchor = _resolve_anchor(prefix[-1], parts)
    if anchor is None:
        return prefix + suffix

    used_fixed = set(prefix + suffix)
    queue = [(0.0, 0, anchor, tuple())]
    best_seen = {}
    expansions = 0

    while queue and expansions < MAX_EXPANSIONS:
        cost, _, current, generated = heapq.heappop(queue)
        expansions += 1

        if generated and _can_end(list(generated)):
            if goal in _neighbours(parts["graph"], current):
                return prefix + list(generated) + [goal]

        used = used_fixed | set(generated)
        recent = list(prefix[-3:]) + list(generated[-3:])

        for step_cost, candidate in _top_forward_candidates(
            current, used, recent, parts, commonness
        ):
            if candidate == ".":
                if generated and _can_end(list(generated)):
                    return prefix + list(generated) + ["."]
                continue

            new_generated = generated + (candidate,)
            new_cost = cost + step_cost
            state = (candidate, len(new_generated))

            if new_cost >= best_seen.get(state, math.inf):
                continue

            best_seen[state] = new_cost
            heapq.heappush(
                queue,
                (new_cost, len(new_generated), candidate, new_generated),
            )

    # A bounded search may fail to reach punctuation; keep the fixed text intact.
    return prefix + suffix


def start_sentence(
    text_network,
    prompt="two <CONTENT> can ask for a solution.",
):
    """Fill the placeholder by searching backward from the fixed suffix."""
    parts = _network_parts(text_network)
    commonness = _commonness(parts)
    tokens = tokenize_text(prompt)

    if "content" not in tokens:
        raise ValueError("Prompt must contain <CONTENT>.")

    split = tokens.index("content")
    prefix = tokens[:split]
    suffix = tokens[split + 1 :]

    if not suffix:
        raise ValueError("Backward completion needs a fixed suffix after <CONTENT>.")

    right_anchor = _resolve_anchor(suffix[0], parts)
    if right_anchor is None:
        return prefix + suffix

    used_fixed = set(prefix + suffix)
    queue = [(0.0, 0, right_anchor, tuple())]
    best_seen = {}
    expansions = 0

    while queue and expansions < MAX_EXPANSIONS:
        cost, _, current, generated = heapq.heappop(queue)
        expansions += 1

        if generated and _can_begin(list(generated)):
            if not prefix:
                return list(generated) + suffix

            left_anchor = _resolve_anchor(prefix[-1], parts)
            if left_anchor is not None:
                first_generated = generated[0]
                if float(
                    parts["adjacency_counts"].get(
                        (left_anchor, first_generated), 0.0
                    )
                ) > 0:
                    return prefix + list(generated) + suffix

        used = used_fixed | set(generated)
        recent = list(generated[:3]) + list(suffix[:3])

        for step_cost, predecessor in _top_backward_candidates(
            current, used, recent, parts, commonness
        ):
            if predecessor == ".":
                continue

            new_generated = (predecessor,) + generated
            new_cost = cost + step_cost
            state = (predecessor, len(new_generated))

            if new_cost >= best_seen.get(state, math.inf):
                continue

            best_seen[state] = new_cost
            heapq.heappush(
                queue,
                (new_cost, len(new_generated), predecessor, new_generated),
            )

    return prefix + suffix
