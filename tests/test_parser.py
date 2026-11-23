"""Reading the count out of a model's reply. Every reply here is a real one,
copied from the spike or the pilot runs (Qwen3-0.6B, SmolLM2-360M, OLMo-2-1B)."""

import pytest

from dumbtok.count_test import check_steps, parse_count, spelled_correctly


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("The letter 'r' appears **once** in \"str", 1),
        ("The letter 's' appears **3 times** in \"", 3),
        ('The word "strawberry" contains the letter **"r"** once. \n\nAnswer: **1**', 1),
        ('The letter "a" appears **2 times** in the word "banana".', 2),
        ('The letter "r" appears 3 times in the word "strawberry".', 3),
        ('The letter "a" appears **three times**.\n\nAnswer: **3**', 3),
        ("The letter 'r' appears **twice** in the", 2),
        ('The letter "l" appears 1 time in the word "parallel".', 1),
        ("1", 1),
        # The first number wins, even if a different one comes later.
        ('The letter "s" appears 10 times in the word "mississippi".', 10),
        # No number at all.
        ('The letter \'b\' appears in the word "b l', None),
    ],
)
def test_plain_takes_the_first_number(reply, expected):
    assert parse_count(reply, "plain") == expected


def test_number_words_must_be_whole_words():
    assert parse_count("Someone counted: 2", "plain") == 2  # not the "one" in "someone"
    assert parse_count("It is done, zero.", "plain") == 0


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("t\no <-\nm\no <-\nr\nr\no <-\nw\n3 lines are marked.\nAnswer: 3", 3),
        ("s\nt\nr\na <-\nb\nr\na <-\nr\nAnswer: 2", 2),
        ("r <-\ns\nt\nr\n<-\nb\ne\nr\n<-\ne\nr\nAnswer: 3", 3),
        ("p <-\nb  \no  \no  \nk  \ne  \np  \ne  \nr  \ne  \nn  \ne  \n3 lines are marked.  \nAnswer: 3", 3),
        # Copied the instruction literally: no number after "Answer:".
        ("r <-\na <-\ns <-\np <-\np <-\ne <-\nr <-\nb <-\ne <-\nr <-\nAnswer: N", None),
        # Ran away and never wrote "Answer:".
        ("l <-\na\np <-\nl <-\nl <-\nl <-\nl <-\nl <-", None),
        ("a <-\na <-\na <-\nn\nn\nn", None),
        # Only the last "Answer:" counts.
        ("Answer: 2\nWait, let me recount.\nAnswer: 3", 3),
    ],
)
def test_spell_then_count_takes_the_number_after_the_last_answer(reply, expected):
    assert parse_count(reply, "spell_then_count") == expected


def test_thinking_reads_only_after_the_thinking():
    reply = '<think>\nS-T-R-A-W-B-E-R-R-Y. That is 10 letters, and 2 of them... no, 3 r\'s.\n</think>\n\nThe letter "r" appears **3** times.'
    assert parse_count(reply, "thinking") == 3
    assert parse_count("<think>\nOkay, let's see. 1, 2, ", "thinking") is None  # still thinking when it was cut off
    assert parse_count("<think>\nhmm 2\n</think>\n\nAnswer: 4", "thinking") == 4


def test_spelling_must_be_letter_by_letter():
    assert spelled_correctly("s t r a w b e r r y", "strawberry")
    assert spelled_correctly("S-T-R-A-W-B-E-R-R-Y", "strawberry")
    assert not spelled_correctly("strawberry", "strawberry")  # repeating the word isn't spelling it
    assert not spelled_correctly("s t r a w b e r r o y", "strawberry")
    assert not spelled_correctly("missisissippi", "mississippi")


def test_check_steps_sees_whether_the_word_was_written_out_right():
    # Qwen3-1.7B, strawberry: copied the example's "t", dropped an r, marked 2, answered 3.
    lucky = check_steps("t  \ns  \nt  \nr <-\na  \nw  \nb  \ne  \nr <-\ny  \nAnswer: 3", "strawberry")
    assert lucky == {"wrote_the_word": False, "marks": 2}
    # Qwen3-1.7B, bedside/e: did exactly what the example showed.
    assert check_steps("b  \ne <-\nd  \ns  \ni  \nd  \ne <-\nAnswer: 2", "bedside") == {"wrote_the_word": True, "marks": 2}
    # Qwen3-0.6B, baboon/a: wrote the word right but marked an "o" too.
    assert check_steps("b  \na <-\nb  \no <-\no  \nn  \nAnswer: 2", "baboon") == {"wrote_the_word": True, "marks": 2}
