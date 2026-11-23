# simplest-tokenizer

This is the companion repo to the [**Simplest Version**](https://www.youtube.com/@SimplestVersion) episode "Why AI can't count the r's in 'strawberry'". We start with the dumbest tokenizer we can build (every character is a token), move to a slightly less dumb one (every word is a token), and finish with the compromise real models use (byte-pair encoding), all trained on 11 old novels from Project Gutenberg. Then we compare our toy tokenizers against 8 real ones and run a 104-question letter-counting test on five small open models to see where token blindness stops and counting itself breaks down.

## What's in here

If you just want to play with the sliders and type your own words, start with `notebook.py`, which walks through the whole episode interactively. The tokenizers and the letter-counting test live in `dumbtok/`, every tweakable setting sits in `config.py`, and `run_all.py` runs the entire pipeline in order and writes the numbers to `results/`. You can inspect `results/` directly without running a single line of code, because every number from the episode is already saved there.

```
config.py           every setting you might want to change, in one place
notebook.py         the episode, step by step, with sliders and text boxes (marimo)
run_all.py          runs everything in order and writes results/
dumbtok/
  corpus.py         downloads the books from Project Gutenberg
  words.py          cuts text into words (used by v2 and v3)
  alphabet.py       v1: every character is a token
  dictionary.py     v2: every word is a token
  bpe.py            v3: byte-pair encoding
  real.py           real tokenizers: GPT-4o, GPT-4, GPT-2, Qwen3, OLMo-2, Mistral, SmolLM2, Phi-4-mini
  letters.py        picks the words for the letter-counting test
  count_test.py     asks small open models to count letters
  hosted_test.py    the same test against a hosted model (any OpenAI-compatible API)
  results.py        writes results/*.json
results/            every number from the episode, as JSON
tests/              a few checks for the parts most likely to be quietly wrong
```

## Quick start

You'll need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and an internet connection the first time you run anything, because the code downloads the Project Gutenberg novels and the real tokenizers on first launch.

Open the notebook:

```bash
uv run marimo edit notebook.py
```

Without PyTorch, a download of several GB (everything except the live model button works):

```bash
uv run --no-group models marimo edit notebook.py
```

## Reproduce the numbers

Every number quoted in the episode comes from `results/`, and `run_all.py` rebuilds the whole folder from scratch. A full run took about 15 minutes on an RTX 4060 Ti (plus about 16 GB of model downloads the first time), while `--skip-models` skips the language models and finishes in under a minute on any laptop.

```bash
uv run python run_all.py                            # everything
uv run python run_all.py --skip-models              # just the tokenizers, no GPU needed
uv run python run_all.py --models Qwen/Qwen3-0.6B   # the counting test for one model only
uv run python run_all.py --quick                    # a tiny version of everything, into results/quick/
```

Check that the tricky parts still work:

```bash
uv run pytest
```

Save the notebook as a web page:

```bash
uv run marimo export html notebook.py -o notebook.html
```

## Things to try

Every tweakable setting lives in `config.py` with `# TRY THIS` comments next to the best knobs. Edit a value, re-run the notebook or `run_all.py`, and see what changes.

- **`NUM_MERGES`** (default `8000`). Drop it to `50` and `"strawberry"` stays in tiny pieces. Raise it to `30000` and training takes about 5 times longer, `" Mississippi"` collapses into one token, and `" strawberry"` still stops at `" straw"` plus `"berry"` because it's too rare in these books.
- **`TRAINING_BOOKS`** (default 11 novels). Keep only the first two books and BPE trains about 5 times faster, while the v2 dictionary shrinks from about 55,000 entries to about 11,000 and its `<UNK>` rate on the held-out book climbs from about 1% to about 7%.
- **`HELD_OUT_BOOK`** (default `1952`, *The Yellow Wallpaper*). Move `2600` (*War and Peace*) out of `TRAINING_BOOKS` and put it here. Unfamiliar Russian names push v2's `<UNK>` rate from about 1% to about 4%. Even BPE gets unknowns this time, because the translation uses accented letters the other books never do, and a character BPE has never seen is the one thing it can't fall back on.
- **`DICTIONARY_MAX_VOCAB`** (default `None`). Cap it at `5000` (or use the dropdown in `notebook.py`) and watch v2's `<UNK>` rate on the held-out book rise from about 1% to about 8%.
- **`SEED`** (default `2026`). Change it to sample a fresh set of 96 words from the books and check whether the letter-counting scores hold up.
- **`MODELS`** and **`CONDITIONS`**. Add `"Qwen/Qwen3-8B"` to `MODELS` if you have a 24 GB GPU, or remove `"thinking"` from `CONDITIONS` to make the full run about 3 times faster.

## The results files

Each JSON file records the exact settings that produced it alongside the output.

- `corpus.json` holds the 11 training novels and *The Yellow Wallpaper*, with word counts, character counts, and file hashes for the setup section.
- `v1_alphabet.json` backs v1 (the alphabet) with its vocabulary, token counts, prompt split, and strawberry r count.
- `v2_dictionary.json` backs v2 (the dictionary) with its vocabulary size, token counts, `<UNK>` rates, and most common unknown words on the held-out book.
- `v3_bpe.json` backs v3 (byte-pair encoding) with the training time, first merges, checkpoint stats, full merge list, and how each probe word shrinks across 8,000 merges.
- `cold_open.json` holds each model's reply to the episode's opening question, "How many r's are in strawberry?" (`uv run python -m dumbtok.cold_open` remakes it).
- `real_tokenizers.json` backs the real-tokenizer comparison with vocabulary sizes, licences, prompt splits, and probe-word variants across all 8 real tokenizers.
- `word_set.json` holds the 104 letter-counting questions (the 8 famous words plus 96 words sampled from the novels) and their count groups.
- `count_test.json` (built from one file per model in `count_test/`) backs the letter-counting section with every model's accuracy, spelling check, and raw replies.
- `summary.txt` is the short plain-text scoreboard and results summary printed at the end of `run_all.py`.
- `hosted_test.json` appears if you run the hosted-model script below, holding its `plain` and `spaced` results.

## Testing a hosted model (ChatGPT and friends)

The five local models above are small (360M to 4B parameters) and say nothing on their own about big hosted chatbots. If you want to run the same `plain` and `spaced` questions against any OpenAI-compatible API, pass your settings through environment variables so your API key is never written to disk. Expect 208 requests per model (104 questions in each condition).

```bash
DT_BASE_URL=https://api.openai.com/v1 DT_API_KEY=sk-... DT_MODELS=gpt-4o-mini \
  uv run python -m dumbtok.hosted_test
```

## Licences

The code and notebook are MIT licensed (see `LICENSE`).

The 12 books are in the public domain in the US via Project Gutenberg. They're downloaded at run time by `dumbtok/corpus.py` and aren't bundled in this repository.

The models and tokenizers carry their own licences. Qwen3, OLMo-2, SmolLM2, and Mistral 7B are released under Apache-2.0. Phi-4-mini and `tiktoken` (`o200k_base` and `cl100k_base`) are MIT, and GPT-2's vocabulary (`gpt2`) is Modified MIT. See `results/real_tokenizers.json` for the exact pinned commits and licence sources.

## Credits and further reading

If you want to read further on why language models struggle with letter counting, start with two papers. Fu et al. (2024), *"Why Do Large Language Models Struggle to Count Letters?"* (arXiv 2412.18626), tested 8 models across 10,000 words and found that models often recognise the letters in a word while still miscounting repeated ones (even GPT-4o miscounted 17% of words). Cosma et al. (2025), *"The Strawberry Problem"* (EMNLP 2025), looks at how tokenization slows down learning spellings because ordinary text almost never spells words out.

If you want to take the tokenizer code further than our toy BPE, Andrej Karpathy's `minbpe` repository is a great next step into byte-level BPE, and his nanochat discussion note *"Guide: counting r in strawberry"* (GitHub discussion #164) shows how adding step-by-step spelling and running-count examples teaches a small model to count letters.
