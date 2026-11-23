"""v2, the dictionary: every word is a token.

Cut the text into words (see words.py) and give every distinct word a number.
Sequences get short: one token per word instead of one per letter.

Two problems show up straight away:

1. The dictionary is huge. Every spelling, every capitalisation, every name is
   its own entry (" The", " the", "The" are three tokens).
2. A word it has never seen becomes <UNK>, "unknown". A new book is full of them.

And "strawberry" is now a single number. The letters are gone.
"""

from collections import Counter
from dataclasses import dataclass

from dumbtok import UNK
from dumbtok.words import split_words


@dataclass
class DictionaryTokenizer:
    vocab: list[str]  # token id -> word. Id 0 is <UNK>.
    id_of: dict[str, int]  # word -> token id
    counts: Counter  # how often each word appeared in the training text


def train(text: str, max_vocab: int | None = None) -> DictionaryTokenizer:
    """Give every word in the training text a number.

    max_vocab: keep only this many of the most common words (plus <UNK>);
    rarer words will become <UNK>. None keeps them all.
    """
    counts = Counter(split_words(text))
    # Most common first; ties in alphabetical order, so the result never changes.
    words = sorted(counts, key=lambda word: (-counts[word], word))
    if max_vocab is not None:
        words = words[:max_vocab]
    vocab = [UNK] + words
    return DictionaryTokenizer(vocab=vocab, id_of={word: i for i, word in enumerate(vocab)}, counts=counts)


def encode(tok: DictionaryTokenizer, text: str) -> list[str]:
    """One token per word. Words not in the dictionary become <UNK>."""
    return [word if word in tok.id_of else UNK for word in split_words(text)]


def ids(tok: DictionaryTokenizer, tokens: list[str]) -> list[int]:
    return [tok.id_of[token] for token in tokens]


def decode(tok: DictionaryTokenizer, token_ids: list[int]) -> str:
    return "".join(tok.vocab[i] for i in token_ids)


def unknown_words(tok: DictionaryTokenizer, text: str) -> Counter:
    """Which words in this text are not in the dictionary, and how often."""
    return Counter(word for word in split_words(text) if word not in tok.id_of)
