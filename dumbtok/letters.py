"""Build the letter-counting test: which words, and which letter to ask about.

Two groups of questions:

* The famous 8: strawberry/r and friends, the words everyone tries.
* 96 words sampled from our books, chosen so that the right answers are
  spread evenly: 24 questions where the letter appears 0 times, 24 where it
  appears once, 24 twice, and 24 three times or more. (If every answer were 1,
  a model that always says "1" would look brilliant.)

The true counts come from Python's str.count, never typed by hand.
The sample is random but seeded: the same seed always gives the same words.
"""

import random
from collections import Counter
from dataclasses import asdict, dataclass

from dumbtok.words import split_words

# The four groups of true counts. "3+" means three or more.
COUNT_GROUPS = ["0", "1", "2", "3+"]


@dataclass
class Item:
    word: str
    letter: str
    count: int  # the right answer: word.count(letter)
    group: str  # "famous" or "sampled"

    @property
    def count_group(self) -> str:
        return count_group(self.count)


def count_group(count: int) -> str:
    return "3+" if count >= 3 else str(count)


def candidate_words(text: str, min_letters: int, max_letters: int, min_occurrences: int) -> list[str]:
    """Plain lower-case English words from the books that are common enough to be fair game.

    Only a-z (no accents, no apostrophes, no capitals), min_letters..max_letters
    long, appearing at least min_occurrences times.
    """
    counts = Counter(word.strip() for word in split_words(text))
    return sorted(
        word
        for word, n in counts.items()
        if n >= min_occurrences
        and min_letters <= len(word) <= max_letters
        and word.isascii()
        and word.isalpha()
        and word.islower()
    )


def letters_with_count(word: str, group: str, common_letters: str) -> list[str]:
    """Letters we could ask about so that the true count falls in this group."""
    if group == "0":
        # A common letter that isn't in the word: "how many e's in 'scrutiny'?"
        return sorted(set(common_letters) - set(word))
    return sorted(letter for letter in set(word) if count_group(word.count(letter)) == group)


def build_word_set(
    text: str,
    famous: list[tuple[str, str]],
    per_count: int,
    seed: int,
    min_letters: int,
    max_letters: int,
    min_occurrences: int,
    common_letters: str,
    exclude: set[str] = frozenset(),
) -> list[Item]:
    """The famous words, then `per_count` sampled words for each count group."""
    rng = random.Random(seed)
    items = [Item(word, letter, word.count(letter), "famous") for word, letter in famous]

    used = {word for word, _ in famous} | set(exclude)
    pool = [word for word in candidate_words(text, min_letters, max_letters, min_occurrences) if word not in used]
    rng.shuffle(pool)

    # Fill the rarest group first (3+), so the common groups don't use up the
    # words that could have served it. Each word is used at most once.
    for group in ["3+", "2", "1", "0"]:
        chosen = 0
        for word in pool:
            if chosen == per_count:
                break
            if word in used:
                continue
            options = letters_with_count(word, group, common_letters)
            if options:
                letter = rng.choice(options)
                items.append(Item(word, letter, word.count(letter), "sampled"))
                used.add(word)
                chosen += 1
        if chosen < per_count:
            raise ValueError(f"Only found {chosen} words for count group {group}; use more books or fewer words.")
    return items


def to_json(items: list[Item]) -> list[dict]:
    return [{**asdict(item), "count_group": item.count_group} for item in items]


def from_json(rows: list[dict]) -> list[Item]:
    return [Item(row["word"], row["letter"], row["count"], row["group"]) for row in rows]


def spaced(word: str) -> str:
    """'strawberry' -> 's t r a w b e r r y'. Each letter becomes its own token."""
    return " ".join(word)
