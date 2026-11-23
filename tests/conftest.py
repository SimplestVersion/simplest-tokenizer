"""Shared test helpers."""

import random

import pytest

WORDS = (
    "the and of to a in strawberry raspberry blueberry banana bookkeeper committee mississippi "
    "parallel letter tomorrow apple happen little better pepper bottle coffee seventeen there "
    "these their other mother brother whether another rather father"
).split()


@pytest.fixture
def small_text() -> str:
    """A few thousand made-up words: quick to train on, the same every time."""
    rng = random.Random(0)
    lines = []
    for _ in range(400):
        words = [rng.choice(WORDS) for _ in range(rng.randint(3, 12))]
        if rng.random() < 0.3:
            words[0] = words[0].capitalize()
        lines.append(" ".join(words) + rng.choice([".", ",", "!", "?", ""]))
    return "\n".join(lines)
