"""Search algorithms from the Artificial Intelligence coursework project."""

from .generation import complete_sentence, start_sentence
from .pathfinding import (
    print_expensive_path,
    print_expensive_quote,
    print_long_path,
    print_long_quote,
)

__all__ = [
    "complete_sentence",
    "start_sentence",
    "print_long_path",
    "print_long_quote",
    "print_expensive_path",
    "print_expensive_quote",
]
