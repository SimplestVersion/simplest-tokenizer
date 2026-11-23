"""Every knob in the project, in one place.

Change a value here, re-run `uv run python run_all.py` (or the notebook), and
see what happens. Each setting says what it does and what to expect if you
change it. Look for `# TRY THIS:` for the most fun ones.
"""

from pathlib import Path

# Where things live. `data/` holds the downloaded books (never committed);
# `results/` holds every number the episode quotes, as JSON.
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# The books we train our tokenizers on
# ---------------------------------------------------------------------------

# Project Gutenberg book ids. All public domain. Downloaded once into data/.
# TRY THIS: keep only the first two books. BPE training gets about 5x faster,
# the dictionary (v2) shrinks from about 55,000 words to about 11,000, and its
# <UNK> rate on the held-out book goes from about 1% to about 7%.
TRAINING_BOOKS = [
    11,  # Alice's Adventures in Wonderland
    84,  # Frankenstein
    1342,  # Pride and Prejudice
    1661,  # The Adventures of Sherlock Holmes
    98,  # A Tale of Two Cities
    2701,  # Moby Dick
    76,  # Adventures of Huckleberry Finn
    16,  # Peter Pan
    174,  # The Picture of Dorian Gray
    345,  # Dracula
    2600,  # War and Peace (the big one: about a third of all the text)
]

# A book the tokenizers never see during training. We use it to ask: how well
# does a tokenizer cope with text it has never met?
# TRY THIS: use 2600 (War and Peace) here, remove it from TRAINING_BOOKS and
# add 1952 there instead. Russian names everywhere (Natásha, Rostóv, Kutúzov):
# v2's <UNK> rate goes from about 1% to about 4%. And a surprise: even BPE gets
# <UNK>s now, about 5,800 of them, because the translation marks stress with
# accented letters (ó, í, ú) that the other books never use. A character BPE
# never saw is the one thing it can't fall back on. Real tokenizers avoid this
# by starting from the 256 possible bytes instead of characters.
HELD_OUT_BOOK = 1952  # The Yellow Wallpaper

# Where to download from. This is a Project Gutenberg mirror; the main site
# sometimes serves a "down for maintenance" page with a success status code,
# so corpus.py checks that what arrives really is a Gutenberg book.
GUTENBERG_URL = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"


# ---------------------------------------------------------------------------
# Things we push through every tokenizer
# ---------------------------------------------------------------------------

# The question from the cold open.
EXAMPLE_PROMPT = "How many r's are in strawberry?"

# Words we follow through every tokenizer, with and without a leading space.
# TRY THIS: add your own name, or a long made-up word like "flibbertigibbet".
PROBE_WORDS = ["strawberry", "raspberry", "blueberry", "Mississippi", "unbelievable", "bookkeeper"]


# ---------------------------------------------------------------------------
# v2: the dictionary tokenizer
# ---------------------------------------------------------------------------

# Keep only the N most common words; everything else becomes <UNK>.
# None means "keep every word we ever saw".
# TRY THIS: set DICTIONARY_MAX_VOCAB = 5000 and watch the <UNK> rate on the
# held-out book climb from about 1% to about 8%.
DICTIONARY_MAX_VOCAB = None


# ---------------------------------------------------------------------------
# v3: byte-pair encoding (BPE)
# ---------------------------------------------------------------------------

# How many merges to learn. Vocabulary size = number of distinct characters
# + number of merges (+1 for <UNK>).
# TRY THIS: set NUM_MERGES = 50 and watch "strawberry" stay in tiny pieces.
# TRY THIS: set NUM_MERGES = 30000. Training takes about 5 times longer.
# " Mississippi" becomes a single token; " strawberry" only gets down to
# " straw" + "berry" (it is too rare in these books to become one).
NUM_MERGES = 8000

# After how many merges we stop to take a snapshot: characters per token, and
# how each probe word is split at that point. Anything above NUM_MERGES is ignored.
BPE_CHECKPOINTS = [0, 50, 100, 250, 500, 1000, 2000, 4000, 8000]

# How many of the first merges to list in the results.
FIRST_MERGES_TO_SHOW = 20


# ---------------------------------------------------------------------------
# Real tokenizers, from real models
# ---------------------------------------------------------------------------

# OpenAI's tokenizers, via the `tiktoken` library. Name -> which model uses it.
TIKTOKEN_ENCODINGS = {
    "o200k_base": "GPT-4o",
    "cl100k_base": "GPT-4",
    "gpt2": "GPT-2",
}

# Open models on Hugging Face. We only download their tokenizer.json (a few MB),
# not the model weights.
# TRY THIS: add "deepseek-ai/DeepSeek-V3" or "openai-community/gpt2".
HF_TOKENIZERS = [
    "Qwen/Qwen3-0.6B",
    "allenai/OLMo-2-0425-1B",
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceTB/SmolLM2-135M",
    "microsoft/Phi-4-mini-instruct",
]


# ---------------------------------------------------------------------------
# The letter-counting test
# ---------------------------------------------------------------------------

# The random seed for picking test words. Same seed = same words, every time.
# TRY THIS: change the seed to get a different 96 words. Do the scores hold up?
SEED = 2026

# The eight famous words (and the letter we ask about) that everyone tries.
FAMOUS_WORDS = [
    ("strawberry", "r"),
    ("raspberry", "r"),
    ("blueberry", "b"),
    ("mississippi", "s"),
    ("bookkeeper", "e"),
    ("committee", "t"),
    ("banana", "a"),
    ("parallel", "l"),
]

# How many extra words to sample from the books for each true count:
# 0 (the letter isn't there), 1, 2, and 3 or more. 24 each = 96 words.
# TRY THIS: set it to 4 for a quick 24-question test (8 famous + 16 sampled).
SAMPLED_PER_COUNT = 24

# Which book words are allowed into the sample.
SAMPLE_MIN_LETTERS = 6
SAMPLE_MAX_LETTERS = 12
SAMPLE_MIN_OCCURRENCES = 5  # a word must appear at least this often in the training books

# For a "count is 0" question we ask about a letter that isn't in the word,
# picked from these common letters so the question sounds plausible.
COMMON_LETTERS = "etaoinshrdlu"

# The models we test: a size ladder in one family (Qwen3), plus two others.
# All are Apache-2.0. The 4B model's weights alone take about 8 GB in bfloat16.
# TRY THIS: add "Qwen/Qwen3-8B" if you have a 24 GB GPU.
MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
]

# Exact Hugging Face commits, so that anyone re-running this gets the same files.
# TRY THIS: replace a commit with "main" to use the latest version.
HF_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "a10cc1512eabd3dde888204e902eca88bddb4951",
    "HuggingFaceTB/SmolLM2-135M": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "allenai/OLMo-2-0425-1B-Instruct": "48d788eca847d4d7548f375ad03d3c9312f6139e",
    "allenai/OLMo-2-0425-1B": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
    "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    "microsoft/Phi-4-mini-instruct": "cfbefacb99257ffa30c83adab238a50856ac3083",
}

# The ways we ask. "thinking" only runs on Qwen3 models (they have a switch for it).
# TRY THIS: remove "thinking" to make the full run about 3 times faster.
CONDITIONS = ["plain", "spaced", "spell_then_count", "thinking"]

# The exact prompts. {letter} and {word} get filled in.
# In "spaced", {word} is written with spaces between the letters: "s t r a w b e r r y",
# so that every letter becomes its own token.
PLAIN_PROMPT = 'How many times does the letter "{letter}" appear in the word "{word}"? Answer with just the number.'

# "spell_then_count" shows the model one worked example first (a short fake
# conversation), then asks the real question in the same words.
SPELL_THEN_COUNT_PROMPT = (
    'Count how many times the letter "{letter}" appears in the word "{word}". '
    "First write the letters of the word one per line. "
    'Put " <-" after every line that is the letter "{letter}". '
    'Then count the marked lines. On the last line, write "Answer:" followed by that number.'
)
# The worked example. Its word is never in the test set (letters.py makes sure).
# It doesn't start with the letter we count, so the model can't just copy the
# first line, and it has a double letter ("rr") that isn't counted.
SPELL_THEN_COUNT_EXAMPLE_WORD = "tomorrow"
SPELL_THEN_COUNT_EXAMPLE_LETTER = "o"
SPELL_THEN_COUNT_EXAMPLE_ANSWER = "t\no <-\nm\no <-\nr\nr\no <-\nw\n3 lines are marked.\nAnswer: 3"
# TRY THIS: number the marks ("o <- 1", "o <- 2", ...) so that counting becomes
# "add one each time". Small models may find that easier.

# The spelling check: can the model even spell the word?
SPELL_PROMPT = 'Spell the word "{word}" with one letter separated by spaces. Reply with only the letters.'

# Longest reply we allow, in tokens. Long enough for a 12-letter word written
# one letter per line, with marks and a count, plus some chatter.
# TRY THIS: set it to 8 and watch "spell_then_count" fall apart (it gets cut off).
MAX_NEW_TOKENS = 256
# In "thinking" mode, Qwen3 writes its reasoning first, which takes many tokens.
THINKING_MAX_NEW_TOKENS = 2048

# How many questions go through the GPU at once. Bigger is faster but uses more
# memory. Answers can differ very slightly with batch size (rounding in bfloat16).
# TRY THIS: set both to 1 if you run out of GPU memory.
BATCH_SIZE = 32
THINKING_BATCH_SIZE = 8

# Run on GPU when there is one; otherwise CPU (fine for the 0.6B model, slow for 4B).
# "auto", "cuda" or "cpu".
DEVICE = "auto"


# ---------------------------------------------------------------------------
# The quick run (`uv run python run_all.py --quick`)
# ---------------------------------------------------------------------------

# A small version of everything, for checking that the code runs. Its results
# go to results/quick/ so they never overwrite the real ones.
QUICK = {
    "TRAINING_BOOKS": [11, 16],
    "NUM_MERGES": 500,
    "SAMPLED_PER_COUNT": 2,
    "MODELS": ["Qwen/Qwen3-0.6B"],
    "CONDITIONS": ["plain", "spell_then_count"],
}
