"""Graph-search algorithms for a word-adjacency network.

The functions in this module operate on the text_network dictionary produced by the
coursework preprocessing pipeline. Search is kept separate from preprocessing so the
algorithms can also be tested on small synthetic networks.
"""

import heapq
import math
import random
import time


RARE_TOKEN = "<RARE>"


# Longest simple path

def build_adjacency(text_network, rare_token_str=RARE_TOKEN, mode="undirected"):
    """Build a filtered adjacency list from observed token transitions.

    Rare tokens are excluded. Neighbours are ordered by increasing degree so the
    longest-path search explores sparse branches before highly connected hubs.
    """
    nodes = set(text_network.get("nodes", []))
    adjacency_counts = text_network.get("adjacency_counts", {})
    nodes.discard(rare_token_str)

    adjacency_sets = {node: set() for node in nodes}

    for (left, right), count in adjacency_counts.items():
        if count <= 0 or left not in nodes or right not in nodes:
            continue

        adjacency_sets[left].add(right)
        if mode == "undirected":
            adjacency_sets[right].add(left)

    degree = {node: len(neighbours) for node, neighbours in adjacency_sets.items()}
    adjacency = {
        node: sorted(neighbours, key=lambda n: (degree.get(n, 0), n))
        for node, neighbours in adjacency_sets.items()
    }
    return adjacency, degree


def bfs_reachable_count_excluding(adj, src, end, forbidden):
    """Count remaining reachable nodes and check whether the target is reachable."""
    if src in forbidden:
        return 0, src == end

    queue = [src]
    seen = {src}
    end_reachable = src == end

    for node in queue:
        for neighbour in adj.get(node, []):
            if neighbour in forbidden or neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
            if neighbour == end:
                end_reachable = True

    return len(seen), end_reachable


def astar_longest_simple_path(
    adj,
    start,
    end,
    max_expansions=500_000,
    prune_every=1,
):
    """Search for a long simple path using bounded branch-and-bound.

    The longest simple path problem is NP-hard, so exhaustive search is impractical
    on the coursework graph. A BFS over the remaining unvisited graph checks whether
    the target is still reachable and supplies an optimistic upper bound for pruning.
    """
    if start not in adj or end not in adj:
        return []
    if start == end:
        return [start]

    best_path = []
    best_length = -1
    expansions = 0
    stack = [(start, iter(adj[start]), [start], {start})]

    while stack:
        node, neighbours, path, visited = stack[-1]

        if node == end:
            length = len(path) - 1
            if length > best_length:
                best_length = length
                best_path = path.copy()
            stack.pop()
            continue

        if expansions % max(1, prune_every) == 0:
            forbidden = visited - {node}
            reachable_count, end_reachable = bfs_reachable_count_excluding(
                adj, node, end, forbidden
            )
            optimistic_length = len(path) - 1 + reachable_count - 1

            if not end_reachable or optimistic_length <= best_length:
                stack.pop()
                continue

        try:
            next_node = next(neighbours)
        except StopIteration:
            stack.pop()
            continue

        if next_node in visited:
            continue

        expansions += 1
        new_path = path + [next_node]
        new_visited = visited | {next_node}
        stack.append((next_node, iter(adj.get(next_node, [])), new_path, new_visited))

        if expansions >= max_expansions:
            break

    return best_path


def print_long_path(text_network, start_word="such", end_word="idea"):
    """Return the longest simple path found between two selected words."""
    adjacency, _ = build_adjacency(text_network)
    return astar_longest_simple_path(adjacency, start_word, end_word)


# Longest contiguous quote

def _is_rare(token, rare_tokens):
    return token not in {",", "."} and token in rare_tokens


def _valid_text_step(index, tokens, adjacency_counts, rare_tokens):
    """Check whether two consecutive token positions form a valid graph transition."""
    next_index = index + 1
    if next_index >= len(tokens):
        return False

    left, right = tokens[index], tokens[next_index]
    if _is_rare(left, rare_tokens) or _is_rare(right, rare_tokens):
        return False

    return adjacency_counts.get((left, right), 0) > 0


def lq_astar_longest_quote(tokens, adj_counts, rare_set, start_word, end_word):
    """Find the longest valid contiguous span between two endpoint words.

    A quote can only move from one token position to the next, so the positional
    search is effectively a set of linear A-star-style expansions from every valid
    occurrence of the start word.
    """
    start_positions = [i for i, token in enumerate(tokens) if token == start_word]
    if not start_positions:
        return []

    best_quote = []

    for start in start_positions:
        if _is_rare(tokens[start], rare_set):
            continue

        position = start
        while True:
            if tokens[position] == end_word:
                candidate = list(tokens[start : position + 1])
                if len(candidate) > len(best_quote):
                    best_quote = candidate

            if not _valid_text_step(position, tokens, adj_counts, rare_set):
                break
            position += 1

    return best_quote


def print_long_quote(text_network, start_word="perhaps", end_word="it"):
    """Return the longest valid contiguous quote for the selected endpoint words."""
    return lq_astar_longest_quote(
        text_network.get("original_tokens", []),
        text_network.get("adjacency_counts", {}),
        set(text_network.get("rare_tokens", set())),
        start_word,
        end_word,
    )


# Highest-cost simple path

def _build_weighted_undirected(nodes, adjacency_counts, distance_matrix):
    """Build a weighted undirected adjacency list and edge-cost lookup."""
    token_to_index = {token: i for i, token in enumerate(nodes)}
    adjacency = {token: [] for token in nodes}
    lookup = {token: {} for token in nodes}
    seen_pairs = set()

    for (left, right), count in adjacency_counts.items():
        if count <= 0 or left not in token_to_index or right not in token_to_index:
            continue

        pair = tuple(sorted((left, right)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        cost = float(distance_matrix[token_to_index[left]][token_to_index[right]])
        adjacency[left].append((right, cost))
        adjacency[right].append((left, cost))
        lookup[left][right] = cost
        lookup[right][left] = cost

    for neighbours in adjacency.values():
        neighbours.sort(key=lambda item: item[1], reverse=True)

    return adjacency, lookup


def _path_cost(path, edge_lookup):
    """Sum edge costs along a path."""
    total = 0.0
    for left, right in zip(path, path[1:]):
        if right not in edge_lookup.get(left, {}):
            return -math.inf
        total += edge_lookup[left][right]
    return total


def _connect_high_cost(
    adjacency,
    start,
    goal,
    forbidden,
    beam_width=8,
    max_expansions=2_000,
    top_k=8,
):
    """Use beam search to find a high-cost simple connection between two nodes."""
    if start == goal:
        return [start]

    beam = [(0.0, [start], {start})]
    best_path = None
    best_cost = -math.inf
    expansions = 0

    while beam and expansions < max_expansions:
        next_beam = []

        for cost, path, visited in beam:
            tail = path[-1]

            if tail == goal:
                if cost > best_cost:
                    best_cost = cost
                    best_path = path
                continue

            branches = 0
            for neighbour, edge_cost in adjacency.get(tail, []):
                if neighbour in forbidden or neighbour in visited:
                    continue

                next_beam.append(
                    (cost + edge_cost, path + [neighbour], visited | {neighbour})
                )
                branches += 1
                if branches >= top_k:
                    break

            expansions += 1
            if expansions >= max_expansions:
                break

        next_beam.sort(key=lambda state: state[0], reverse=True)
        beam = next_beam[:beam_width]

    for cost, path, _ in beam:
        if path[-1] == goal and cost > best_cost:
            best_cost = cost
            best_path = path

    return best_path


def _propose_path(path, adjacency, end_word):
    """Create an annealing proposal by branching from a random path prefix."""
    if len(path) <= 2:
        return path.copy()

    pivot_index = random.randint(0, len(path) - 2)
    prefix = path[: pivot_index + 1]
    pivot = prefix[-1]
    old_next = path[pivot_index + 1]

    choices = [
        neighbour
        for neighbour, _ in adjacency.get(pivot, [])
        if neighbour != old_next and neighbour not in prefix
    ]
    if not choices:
        return path.copy()

    branch = random.choice(choices[: min(8, len(choices))])
    connector = _connect_high_cost(
        adjacency,
        branch,
        end_word,
        forbidden=set(prefix) | {branch},
    )
    if connector is None:
        return path.copy()

    return prefix + connector


def print_expensive_path(text_network, start_word="floor", end_word="ministry"):
    """Search for a high-cost simple path with beam search and simulated annealing."""
    nodes = text_network.get("nodes", [])
    distance_matrix = text_network.get("distance_matrix")
    adjacency_counts = text_network.get("adjacency_counts", {})

    if not nodes or distance_matrix is None:
        return [], 0.0

    adjacency, lookup = _build_weighted_undirected(
        nodes, adjacency_counts, distance_matrix
    )
    if start_word not in adjacency or end_word not in adjacency:
        return [], 0.0

    initial = _connect_high_cost(adjacency, start_word, end_word, forbidden=set())
    if not initial:
        return [], 0.0

    current = initial
    current_cost = _path_cost(current, lookup)
    best = current.copy()
    best_cost = current_cost

    temperature = max(1.0, abs(current_cost) * 0.05)
    start_time = time.time()

    for _ in range(1_600):
        if time.time() - start_time > 180:
            break

        candidate = _propose_path(current, adjacency, end_word)
        candidate_cost = _path_cost(candidate, lookup)
        delta = candidate_cost - current_cost

        acceptance = delta >= 0
        if not acceptance and temperature > 0:
            acceptance = random.random() < math.exp(delta / temperature)

        if acceptance:
            current, current_cost = candidate, candidate_cost
            if current_cost > best_cost:
                best, best_cost = current.copy(), current_cost

        temperature *= 0.995

    return best, float(best_cost)


# Highest-cost contiguous quote

def _prepare_quote_costs(text_network):
    """Precompute valid transitions and prefix sums for contiguous quote scoring."""
    tokens = text_network.get("original_tokens", [])
    nodes = text_network.get("nodes", [])
    adjacency_counts = text_network.get("adjacency_counts", {})
    distance_matrix = text_network.get("distance_matrix")
    rare_tokens = set(text_network.get("rare_tokens", set()))

    if not tokens or not nodes or distance_matrix is None:
        return None

    index = {token: i for i, token in enumerate(nodes)}
    valid_token = [
        token in index and not _is_rare(token, rare_tokens)
        for token in tokens
    ]

    valid_edge = [False] * max(0, len(tokens) - 1)
    edge_cost = [0.0] * max(0, len(tokens) - 1)

    for i, (left, right) in enumerate(zip(tokens, tokens[1:])):
        if not (valid_token[i] and valid_token[i + 1]):
            continue
        if adjacency_counts.get((left, right), 0) <= 0:
            continue

        valid_edge[i] = True
        edge_cost[i] = float(distance_matrix[index[left]][index[right]])

    prefix = [0.0]
    for is_valid, cost in zip(valid_edge, edge_cost):
        prefix.append(prefix[-1] + (cost if is_valid else 0.0))

    return tokens, valid_token, valid_edge, prefix


def print_expensive_quote(text_network, start_word="perhaps", end_word="it"):
    """Return the highest-cost valid contiguous quote between two endpoint words."""
    prepared = _prepare_quote_costs(text_network)
    if prepared is None:
        return [], 0.0

    tokens, valid_token, valid_edge, prefix = prepared
    best_quote = []
    best_cost = -math.inf

    for start, token in enumerate(tokens):
        if token != start_word or not valid_token[start]:
            continue

        end = start
        while end < len(tokens):
            if not valid_token[end]:
                break

            if tokens[end] == end_word:
                cost = prefix[end] - prefix[start]
                if cost > best_cost:
                    best_cost = cost
                    best_quote = tokens[start : end + 1]

            if end >= len(valid_edge) or not valid_edge[end]:
                break
            end += 1

    if not best_quote:
        return [], 0.0

    return best_quote, float(best_cost)
