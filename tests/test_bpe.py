"""BPE: the fast trainer must learn exactly what the slow, obvious one learns,
and encoding must reproduce the splits seen during training."""

from collections import Counter

from dumbtok import bpe
from dumbtok.words import split_words


def slow_bpe(text: str, num_merges: int) -> tuple[list, dict]:
    """The obvious way: re-count every pair after every merge. Easy to trust."""
    word_counts = Counter(split_words(text))
    pieces = {word: list(word) for word in word_counts}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, count in word_counts.items():
            for pair in zip(pieces[word], pieces[word][1:]):
                pairs[pair] += count
        if not pairs:
            break
        best = bpe.most_common_pair(pairs)
        merges.append(best)
        pieces = {word: bpe.merge_pair(p, *best) for word, p in pieces.items()}
    return merges, pieces


def test_fast_training_learns_the_same_merges_as_slow_training(small_text):
    tok = bpe.train(small_text, 150)
    slow_merges, _ = slow_bpe(small_text, 150)
    assert tok.merges == slow_merges


def test_encoding_reproduces_the_training_splits(small_text):
    tok = bpe.train(small_text, 150)
    _, training_splits = slow_bpe(small_text, 150)
    for word, pieces in training_splits.items():
        assert bpe.encode_word(tok, word) == pieces, word


def test_encoding_with_fewer_merges_matches_training_with_fewer_merges(small_text):
    tok = bpe.train(small_text, 150)
    _, splits_at_40 = slow_bpe(small_text, 40)
    for word, pieces in splits_at_40.items():
        assert bpe.encode_word(tok, word, num_merges=40) == pieces, word


def test_zero_merges_means_single_characters(small_text):
    tok = bpe.train(small_text, 50)
    assert bpe.encode_word(tok, " strawberry", num_merges=0) == list(" strawberry")


def test_round_trip_and_no_unknowns_for_new_words(small_text):
    tok = bpe.train(small_text, 150)
    text = "Brother, the strawberries are better tomorrow."  # "strawberries" never appears in training
    tokens = bpe.encode(tok, text)
    assert "<UNK>" not in tokens
    assert bpe.decode(tok, bpe.ids(tok, tokens)) == text


def test_a_character_never_seen_becomes_unknown(small_text):
    tok = bpe.train(small_text, 50)
    assert "<UNK>" in bpe.encode(tok, " zebra")  # there is no "z" in the training text


def test_merge_pair_goes_left_to_right():
    assert bpe.merge_pair(list("aaa"), "a", "a") == ["aa", "a"]
    assert bpe.merge_pair(["b", "e", "r", "r", "y"], "r", "y") == ["b", "e", "r", "ry"]
