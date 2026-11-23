"""v2, the dictionary: words it has seen get their own id, anything else is <UNK>."""

from dumbtok import UNK, dictionary
from dumbtok.words import split_words


def test_words_keep_their_leading_space_and_nothing_is_lost():
    text = "How many r's are in strawberry?\n\nTwo  spaces."
    words = split_words(text)
    assert words[:3] == ["How", " many", " r"]
    assert " strawberry" in words
    assert "".join(words) == text


def test_unseen_words_become_unk():
    tok = dictionary.train("the cat sat on the mat")
    assert dictionary.encode(tok, "the dog sat") == ["the", UNK, " sat"]


def test_unknown_words_are_listed_with_counts():
    tok = dictionary.train("the cat sat")
    unknown = dictionary.unknown_words(tok, "the dog and the dog")
    assert unknown == {" dog": 2, " and": 1, " the": 1}  # " the" (with a space) was never seen


def test_max_vocab_keeps_the_most_common_words():
    tok = dictionary.train("a a a b b c", max_vocab=2)
    # " a" and " b" appear twice each; "a" and " c" once, so they are dropped.
    assert tok.vocab == [UNK, " a", " b"]
    assert dictionary.encode(tok, " c") == [UNK]


def test_ids_and_decode_round_trip():
    tok = dictionary.train("the cat sat")
    tokens = dictionary.encode(tok, "the cat sat")
    assert dictionary.decode(tok, dictionary.ids(tok, tokens)) == "the cat sat"
