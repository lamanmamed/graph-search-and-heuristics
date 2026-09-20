import unittest

from ai_coursework.pathfinding import (
    astar_longest_simple_path,
    build_adjacency,
    lq_astar_longest_quote,
    print_expensive_quote,
)


class PathfindingTests(unittest.TestCase):
    def test_longest_simple_path_has_no_repeated_nodes(self):
        network = {
            "nodes": ["a", "b", "c", "d", "e"],
            "adjacency_counts": {
                ("a", "b"): 1,
                ("b", "c"): 1,
                ("c", "d"): 1,
                ("b", "e"): 1,
                ("e", "c"): 1,
            },
        }

        adjacency, _ = build_adjacency(network)
        path = astar_longest_simple_path(
            adjacency,
            "a",
            "d",
            max_expansions=10_000,
        )

        self.assertEqual(path[0], "a")
        self.assertEqual(path[-1], "d")
        self.assertEqual(len(path), len(set(path)))
        self.assertEqual(len(path) - 1, 4)

    def test_longest_quote_prefers_longer_contiguous_span(self):
        tokens = ["a", "b", "c", "d", "a", "b", "d"]
        adjacency = {
            ("a", "b"): 2,
            ("b", "c"): 1,
            ("c", "d"): 1,
            ("b", "d"): 1,
            ("d", "a"): 1,
        }

        quote = lq_astar_longest_quote(
            tokens,
            adjacency,
            set(),
            "a",
            "d",
        )

        self.assertEqual(
            quote,
            ["a", "b", "c", "d", "a", "b", "d"],
        )

    def test_expensive_quote_uses_prefix_sum_costs(self):
        nodes = ["a", "b", "c", "d"]
        distance_matrix = [
            [0, 2, 0, 0],
            [2, 0, 3, 1],
            [0, 3, 0, 4],
            [0, 1, 4, 0],
        ]
        network = {
            "original_tokens": ["a", "b", "d", "a", "b", "c", "d"],
            "nodes": nodes,
            "adjacency_counts": {
                ("a", "b"): 2,
                ("b", "d"): 1,
                ("d", "a"): 1,
                ("b", "c"): 1,
                ("c", "d"): 1,
            },
            "distance_matrix": distance_matrix,
            "rare_tokens": set(),
        }

        quote, cost = print_expensive_quote(network, "a", "d")

        self.assertEqual(
            quote,
            ["a", "b", "d", "a", "b", "c", "d"],
        )
        self.assertEqual(cost, 12.0)


if __name__ == "__main__":
    unittest.main()
