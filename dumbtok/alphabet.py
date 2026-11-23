"""v1, the alphabet: every character is a token.

The dumbest tokenizer there is. The vocabulary is just every character that
appears in the training text: letters, digits, punctuation, spaces, newlines.

It can count the r's in "strawberry" perfectly, because the r's are right there
as tokens. The catch: every character is a separate token, so a book becomes a
very long sequence, and a model has to take one step per token.
"""

from dataclasses import dataclass

from dumbtok import UNK


@dataclass
class AlphabetTokenizer:
    vocab: list[str]  # token id -> token. Id 0 is <UNK>.
    id_of: dict[str, int]  # token -> token id


def train(text: str) -> AlphabetTokenizer:
    """The vocabulary is every distinct character in the text, sorted."""
    vocab = [UNK] + sorted(set(text))
    return AlphabetTokenizer(vocab=vocab, id_of={token: i for i, token in enumerate(vocab)})


def encode(tok: AlphabetTokenizer, text: str) -> list[str]:
    """One token per character. A character never seen in training becomes <UNK>."""
    return [char if char in tok.id_of else UNK for char in text]


def ids(tok: AlphabetTokenizer, tokens: list[str]) -> list[int]:
    return [tok.id_of[token] for token in tokens]


def decode(tok: AlphabetTokenizer, token_ids: list[int]) -> str:
    return "".join(tok.vocab[i] for i in token_ids)


def count_letter(tokens: list[str], letter: str) -> int:
    """With one token per letter, counting a letter is just counting tokens."""
    return sum(1 for token in tokens if token == letter)
