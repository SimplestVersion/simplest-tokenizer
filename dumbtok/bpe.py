"""v3, byte-pair encoding (BPE): the compromise real models use.

Start from v1: every character is a token. Then repeat:

    1. Count every pair of neighbouring tokens in the training text.
    2. Glue the most common pair into one new token ("t" + "h" -> "th").
    3. Write that merge down.

After a few thousand merges, common words are one or two tokens, rare words
are a few pieces, and no word is ever unknown: anything can fall back to
single letters. The price is that the letters disappear inside the pieces:
" strawberry" becomes something like " straw" + "ber" + "ry".

This is "word-internal" BPE: we first cut the text into words (words.py) and
never merge across two words. GPT-2 and most models since do the same.

To encode new text, we replay the merges, in the order they were learned.
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from dumbtok import UNK
from dumbtok.words import split_words

Pair = tuple[str, str]


@dataclass
class BPETokenizer:
    alphabet: list[str]  # the starting tokens: every character in the training text
    merges: list[Pair]  # every merge, in the order it was learned
    merge_counts: list[int]  # how often each pair occurred at the moment it was merged
    rank: dict[Pair, int]  # merge -> its position in `merges` (0 = learned first)
    vocab: list[str]  # token id -> token: <UNK>, the alphabet, then one new token per merge
    id_of: dict[str, int]  # token -> token id
    snapshots: list[dict]  # tokens and characters per token at each checkpoint
    training_seconds: float


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def merge_pair(pieces: list[str], a: str, b: str) -> list[str]:
    """Glue every "a" followed by "b" into one piece "ab", left to right.

    merge_pair(["b", "e", "r", "r", "y"], "r", "y")  ->  ["b", "e", "r", "ry"]
    """
    merged = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(pieces[i])
            i += 1
    return merged


def train(text: str, num_merges: int, checkpoints: list[int] = ()) -> BPETokenizer:
    """Learn `num_merges` merges from the text.

    checkpoints: after these numbers of merges, record how many tokens the
    training text takes (0 = before any merge).
    """
    started = time.perf_counter()

    # We never merge across words, so we only need each distinct word and how
    # often it appears. 3 million words of text become about 50,000 distinct words.
    word_counts = Counter(split_words(text))
    alphabet = sorted({char for word in word_counts for char in word})

    # Every word starts out as single characters.
    pieces = {word: list(word) for word in word_counts}

    # pair_counts[("r", "y")] = how often "r" is followed by "y" in the whole text.
    # words_with_pair[("r", "y")] = the words in which that happens.
    pair_counts: Counter = Counter()
    words_with_pair: dict[Pair, set[str]] = defaultdict(set)
    for word in word_counts:
        add_pairs(word, pieces[word], word_counts[word], pair_counts, words_with_pair)

    total_characters = sum(len(word) * count for word, count in word_counts.items())
    merges: list[Pair] = []
    merge_counts: list[int] = []
    snapshots: list[dict] = []

    def take_snapshot() -> None:
        tokens = sum(len(pieces[word]) * count for word, count in word_counts.items())
        snapshots.append(
            {
                "merges": len(merges),
                "vocab_size": 1 + len(alphabet) + len(merges),  # +1 for <UNK>
                "tokens": tokens,
                "chars_per_token": total_characters / tokens,
                "seconds": time.perf_counter() - started,
            }
        )

    if 0 in checkpoints:
        take_snapshot()

    while len(merges) < num_merges and pair_counts:
        best = most_common_pair(pair_counts)
        merges.append(best)
        merge_counts.append(pair_counts[best])

        # THE TRICK THAT MAKES THIS FAST.
        # The slow way: after every merge, re-count the pairs in every word.
        # That's 50,000 words, 8,000 times over (the first version took 14 minutes).
        # But a merge only changes the words that contain the merged pair, and
        # only their pair counts change. So for just those words: subtract
        # their old pairs, merge, and add their new pairs back.
        a, b = best
        for word in list(words_with_pair[best]):
            count = word_counts[word]
            remove_pairs(word, pieces[word], count, pair_counts, words_with_pair)
            pieces[word] = merge_pair(pieces[word], a, b)
            add_pairs(word, pieces[word], count, pair_counts, words_with_pair)

        if len(merges) in checkpoints:
            take_snapshot()

    vocab = [UNK] + alphabet + [a + b for a, b in merges]
    # (A merged token can be spelled the same as an earlier one, e.g. "the" from
    # "t"+"he" and from "th"+"e". It keeps its first id.)
    id_of: dict[str, int] = {}
    for i, token in enumerate(vocab):
        id_of.setdefault(token, i)
    return BPETokenizer(
        alphabet=alphabet,
        merges=merges,
        merge_counts=merge_counts,
        rank={pair: i for i, pair in enumerate(merges)},
        vocab=vocab,
        id_of=id_of,
        snapshots=snapshots,
        training_seconds=time.perf_counter() - started,
    )


def most_common_pair(pair_counts: Counter) -> Pair:
    """The pair with the highest count. Ties go to the alphabetically first pair,
    so training gives the same merges every time."""
    highest = max(pair_counts.values())
    return min(pair for pair, count in pair_counts.items() if count == highest)


def add_pairs(word: str, word_pieces: list[str], count: int, pair_counts: Counter, words_with_pair: dict) -> None:
    """Count the neighbouring pairs in one word (which appears `count` times)."""
    for pair in zip(word_pieces, word_pieces[1:]):
        pair_counts[pair] += count
        words_with_pair[pair].add(word)


def remove_pairs(word: str, word_pieces: list[str], count: int, pair_counts: Counter, words_with_pair: dict) -> None:
    """Undo add_pairs for one word."""
    for pair in zip(word_pieces, word_pieces[1:]):
        pair_counts[pair] -= count
        if pair_counts[pair] == 0:
            del pair_counts[pair]
        words_with_pair[pair].discard(word)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def merge_history(tok: BPETokenizer, word: str, num_merges: int | None = None) -> list[tuple[int, list[str]]]:
    """Apply the learned merges to one word, in the order they were learned.

    Returns every stage the word goes through, as (merge number, pieces):

        [(0, ['s','t','r','a','w','b','e','r','r','y']), (12, ['s','t','r','a','w','b','er','r','y']), ...]

    The last entry is the final split, and its merge number is the step at which
    the word reached it.

    num_merges: use only the first N merges (None = all of them).
    TRY THIS: merge_history(tok, " strawberry", 100) shows the word as a
    tokenizer with only 100 merges would see it.

    This gives exactly the same result as looping over every merge in order and
    applying it if it fits. It's just faster: instead of trying thousands of
    merges that can't apply, it jumps to the next one that can.
    """
    limit = len(tok.merges) if num_merges is None else min(num_merges, len(tok.merges))
    # Start from single characters. A character never seen in training is <UNK>:
    # the only way BPE can fail. (GPT-2 avoids even that by starting from the 256
    # possible bytes instead of characters.)
    pieces = [char if char in tok.id_of else UNK for char in word]
    history = [(0, pieces)]
    next_rank = 0  # merges before this one have already had their chance
    while True:
        # Ranks of the merges (not yet passed) that fit some neighbouring pair now.
        fitting = [tok.rank[pair] for pair in zip(pieces, pieces[1:]) if tok.rank.get(pair, -1) >= next_rank]
        if not fitting:
            break
        rank = min(fitting)  # the earliest learned of them is the next to apply
        if rank >= limit:
            break
        a, b = tok.merges[rank]
        pieces = merge_pair(pieces, a, b)
        history.append((rank + 1, pieces))  # merge numbers count from 1
        next_rank = rank + 1
    return history


def encode_word(tok: BPETokenizer, word: str, num_merges: int | None = None) -> list[str]:
    """Split one word into tokens."""
    return merge_history(tok, word, num_merges)[-1][1]


def encode(tok: BPETokenizer, text: str, num_merges: int | None = None) -> list[str]:
    """Split text into words, then each word into tokens."""
    seen: dict[str, list[str]] = {}  # most words repeat, so remember each one's split
    tokens = []
    for word in split_words(text):
        if word not in seen:
            seen[word] = encode_word(tok, word, num_merges)
        tokens.extend(seen[word])
    return tokens


def ids(tok: BPETokenizer, tokens: list[str]) -> list[int]:
    return [tok.id_of[token] for token in tokens]


def decode(tok: BPETokenizer, token_ids: list[int]) -> str:
    return "".join(tok.vocab[i] for i in token_ids)
