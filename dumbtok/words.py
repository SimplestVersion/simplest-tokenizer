"""Cut text into words. Used by v2 (the dictionary) and v3 (BPE).

Both tokenizers start from the same words, so their numbers are comparable.

The rule, the same one GPT-2 uses (simplified):

* A word is a run of letters, a run of digits, or a run of punctuation.
* A single space in front of a word sticks to the word: " strawberry".
  So "strawberry" at the start of a line and " strawberry" in the middle of a
  sentence are two different words. Real tokenizers do exactly this, which is
  why " strawberry" can be one token while "strawberry" is three.
* Any other whitespace (newlines, double spaces) becomes its own word.

Nothing is thrown away: joining the words back together gives the original
text, character for character.
"""

import re

WORD_PATTERN = re.compile(
    r"""
      \ ?[^\W\d_]+         # letters, with the space in front (if any):   " strawberry"
    | \ ?\d+               # digits, with the space in front:             " 1952"
    | \ ?(?:[^\w\s]|_)+    # punctuation and symbols:                     "?", " --", "_"
    | \s+                  # whatever whitespace is left:                 "\n\n", "  "
    """,
    re.VERBOSE,
)
# ([^\W\d_] is Python's way of saying "any letter, in any alphabet":
#  not a non-word character, not a digit, not an underscore.)
# TRY THIS: delete the three "\ ?". Spaces become words of their own, and the
# training books go from about 2.1 million words to about 3.6 million.


def split_words(text: str) -> list[str]:
    """Cut text into words, keeping a single leading space on each word."""
    return WORD_PATTERN.findall(text)
