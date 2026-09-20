# Graph Search and Heuristics

A project exploring **graph search, heuristic design, optimisation, and constrained generation** on a word-adjacency network derived from text.

Each unique token is represented as a node and observed word-to-word transitions form graph edges. The project treats several tasks as search problems: finding long or high-cost paths, extracting endpoint-constrained sequences, and generating missing sentence spans while staying on the observed graph. The original coursework experiments used George Orwell's *Nineteen Eighty-Four* as the source text.

## What I implemented

### 1. Longest simple path

I implemented a depth-first **branch-and-bound** search for a long simple path between two words. Because the longest simple path problem is NP-hard, the search is bounded and guided rather than exhaustively enumerating every possibility.

Two ideas make the search practical:

- low-degree neighbors are explored first, which tends to follow sparse branches before entering highly connected hubs;
- a BFS over the remaining unvisited graph checks whether the target is still reachable and gives an optimistic upper bound for pruning.

In the coursework experiments, this improved the earlier greedy result from **444 edges to 464 edges**.

### 2. Longest contiguous quote

The original text is treated as a sequence of token positions. A quote is valid only when every adjacent pair is an observed graph transition and no collapsed rare token appears. An A*-style maximization searches over valid positions to recover the longest endpoint-constrained span.

### 3. Highest-cost path

For weighted paths, I combined **beam search** with **simulated annealing**. Beam search builds plausible high-cost connections, while simulated annealing occasionally accepts worse candidates to escape local maxima. Candidate paths always keep the fixed endpoints and avoid repeated nodes.

### 4. Highest-cost quote

For contiguous quotes, I precompute valid-token and valid-edge arrays plus prefix sums of edge costs. This makes the cost of any valid window constant-time to evaluate. The text is split into maximal valid segments, and endpoint positions are searched efficiently inside each segment.

### 5. Heuristic sentence completion

The final part uses A*-style search to fill a `<CONTENT>` span in either direction. The generator is constrained to transitions found in the source network and uses lightweight, interpretable signals instead of an external NLP model:

- coarse part-of-speech compatibility;
- directed bigram strength;
- local context co-occurrence;
- simple stemming and lexical overlap;
- penalties for graph hubs and extremely common words;
- checks for plausible sentence beginnings and endings.

The goal was not to build a general language model, but to explore how far **search + graph structure + hand-designed heuristics** can go in producing locally coherent text.

## What this project taught me

This coursework made the difference between an algorithm that is correct in theory and one that is usable under real computational limits very concrete. I learned how to design pruning bounds, choose expansion orders, and use approximate methods when exhaustive search is impractical.

It also showed me how different search strategies fit different objectives. Branch-and-bound was useful when I could construct a meaningful upper bound; beam search helped control branching; simulated annealing helped escape local optima; and prefix sums turned an expensive repeated calculation into a cheap lookup.

The heuristic-generation task was especially useful for understanding the trade-off between **hard constraints** and **soft preferences**. Graph edges enforce what is allowed, while linguistic scores only influence which valid option is explored first. That separation made the behavior easier to reason about and debug.

## Repository structure

```text
.
├── src/ai_coursework/
│   ├── __init__.py
│   ├── pathfinding.py       # longest/highest-cost path and quote searches
│   └── generation.py        # forward and backward heuristic generation
├── tests/
│   └── test_pathfinding.py  # small synthetic regression tests
├── examples/
│   └── synthetic_demo.py
├── pyproject.toml
├── .gitignore
└── README.md
```

## Running the project

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

The coursework search functions expect the preprocessed `text_network` dictionary used in the original assignment. The original novel text and university preprocessing/starter materials are intentionally not included in this portfolio repository.

The main public functions are:

```python
from ai_coursework import (
    complete_sentence,
    start_sentence,
    print_long_path,
    print_long_quote,
    print_expensive_path,
    print_expensive_quote,
)
```

## Notes on reproducibility

The pathfinding tests use small synthetic networks so the core search logic can be checked without distributing copyrighted source text. The heuristic generation functions additionally expect a graph object and the matrices created by the coursework preprocessing pipeline.

## Academic context

This repository is a cleaned portfolio version of work completed for my MSc Artificial Intelligence coursework. The code has been reorganized and documented for readability; the underlying search approaches are the ones developed for the assignment.
