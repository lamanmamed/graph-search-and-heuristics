from ai_coursework.pathfinding import astar_longest_simple_path, build_adjacency


network = {
    "nodes": ["start", "quiet", "street", "old", "house", "end"],
    "adjacency_counts": {
        ("start", "quiet"): 1,
        ("quiet", "street"): 1,
        ("street", "end"): 1,
        ("quiet", "old"): 1,
        ("old", "house"): 1,
        ("house", "street"): 1,
    },
}

adj, _ = build_adjacency(network)
path = astar_longest_simple_path(adj, "start", "end")
print(" -> ".join(path))
