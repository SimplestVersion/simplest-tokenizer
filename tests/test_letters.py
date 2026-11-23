"""The letter-counting word set: right answers, right mix, same every time."""

import json
from collections import Counter
from pathlib import Path

import pytest

from dumbtok import letters

FAMOUS = [("strawberry", "r"), ("banana", "a")]


def build(text, seed=1):
    return letters.build_word_set(
        text, famous=FAMOUS, per_count=3, seed=seed, min_letters=6, max_letters=12,
        min_occurrences=5, common_letters="etaoinshrdlu", exclude={"tomorrow"},
    )  # fmt: skip


def test_true_counts_are_right_and_groups_are_balanced(small_text):
    items = build(small_text)
    for item in items:
        assert item.count == item.word.count(item.letter)
    sampled = [item for item in items if item.group == "sampled"]
    assert Counter(item.count_group for item in sampled) == {"0": 3, "1": 3, "2": 3, "3+": 3}


def test_famous_first_no_repeats_and_exclusions_respected(small_text):
    items = build(small_text)
    assert [(i.word, i.letter) for i in items[:2]] == FAMOUS
    words = [item.word for item in items]
    assert len(words) == len(set(words))
    assert "tomorrow" not in words


def test_same_seed_same_words(small_text):
    assert build(small_text, seed=1) == build(small_text, seed=1)
    assert build(small_text, seed=1) != build(small_text, seed=2)


def test_count_zero_uses_a_letter_that_is_not_there(small_text):
    for item in build(small_text):
        if item.count == 0:
            assert item.letter not in item.word


@pytest.mark.skipif(not Path("results/word_set.json").exists(), reason="run run_all.py first")
def test_the_real_word_set_file_has_right_answers():
    rows = json.loads(Path("results/word_set.json").read_text())["items"]
    for row in rows:
        assert row["count"] == row["word"].count(row["letter"]), row
        assert row["count_group"] == letters.count_group(row["count"])
