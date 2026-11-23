"""Run the whole project, in the order of the episode, and write results/*.json.

    uv run python run_all.py                          everything (about 15 min on an RTX 4060 Ti)
    uv run python run_all.py --skip-models            just the tokenizers (about a minute, no GPU)
    uv run python run_all.py --models Qwen/Qwen3-0.6B only this model in the counting test
    uv run python run_all.py --quick                  a tiny version of everything, into results/quick/

The tour:

    1. corpus      download 11 public-domain books to learn from, plus 1 to test on
    2. v1          the alphabet: every character is a token
    3. v2          the dictionary: every word is a token
    4. v3          byte-pair encoding: start from letters, merge common pairs
    5. real        what GPT-4o, Qwen3 and friends do to the same words
    6. word set    pick the words for the letter-counting test
    7. count test  ask small open models to count letters

Every setting comes from config.py.
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import config
from dumbtok import UNK, alphabet, bpe, corpus, dictionary, letters, real, results

# ---------------------------------------------------------------------------
# Settings: config.py, plus the --quick overrides
# ---------------------------------------------------------------------------


def load_settings(quick: bool) -> dict:
    """All the knobs this run uses, as a plain dict (it's saved in every results file)."""
    names = [
        "TRAINING_BOOKS", "HELD_OUT_BOOK", "GUTENBERG_URL", "EXAMPLE_PROMPT", "PROBE_WORDS",
        "DICTIONARY_MAX_VOCAB", "NUM_MERGES", "BPE_CHECKPOINTS", "FIRST_MERGES_TO_SHOW",
        "TIKTOKEN_ENCODINGS", "HF_TOKENIZERS", "HF_REVISIONS", "SEED", "FAMOUS_WORDS",
        "SAMPLED_PER_COUNT", "SAMPLE_MIN_LETTERS", "SAMPLE_MAX_LETTERS", "SAMPLE_MIN_OCCURRENCES",
        "COMMON_LETTERS", "MODELS", "CONDITIONS", "PLAIN_PROMPT", "SPELL_THEN_COUNT_PROMPT",
        "SPELL_THEN_COUNT_EXAMPLE_WORD", "SPELL_THEN_COUNT_EXAMPLE_LETTER", "SPELL_THEN_COUNT_EXAMPLE_ANSWER",
        "SPELL_PROMPT", "MAX_NEW_TOKENS", "THINKING_MAX_NEW_TOKENS", "BATCH_SIZE", "THINKING_BATCH_SIZE", "DEVICE",
    ]  # fmt: skip
    settings = {name: getattr(config, name) for name in names}
    if quick:
        settings.update(config.QUICK)
    settings["BPE_CHECKPOINTS"] = [c for c in settings["BPE_CHECKPOINTS"] if c <= settings["NUM_MERGES"]]
    return settings


def pick(settings: dict, *names: str) -> dict:
    """Just the settings one step uses, to record in its results file."""
    return {name: settings[name] for name in names}


def scoreboard_row(data: dict, held_out_characters: int, strawberry: list[str]) -> dict:
    """The numbers we compare across v1, v2 and v3 (the final scoreboard)."""
    return {
        "vocab_size": data["vocab_size"],
        "training_tokens": data["training_tokens"],
        "held_out_tokens": data["held_out_tokens"],
        "held_out_chars_per_token": held_out_characters / data["held_out_tokens"],
        "held_out_unk_rate": data["held_out_unk_rate"],
        "strawberry_tokens": strawberry,
    }


def with_and_without_space(words: list[str]) -> list[str]:
    """["strawberry"] -> ["strawberry", " strawberry"]"""
    return [form for word in words for form in (word, " " + word)]


# ---------------------------------------------------------------------------
# 1. The books
# ---------------------------------------------------------------------------


def step_corpus(s: dict, out: Path) -> dict:
    started = time.perf_counter()
    training = corpus.load_books(s["TRAINING_BOOKS"], config.DATA_DIR, s["GUTENBERG_URL"])
    held_out = corpus.load_book(s["HELD_OUT_BOOK"], config.DATA_DIR, s["GUTENBERG_URL"])
    training_text = corpus.join_books(training)
    data = {
        "training_books": corpus.describe(training),
        "held_out_book": corpus.describe([held_out])[0],
        "training_characters": len(training_text),
        "training_words": len(training_text.split()),
        "held_out_characters": len(held_out.text),
        "held_out_words": len(held_out.text.split()),
    }
    results.write(out / "corpus.json", data, pick(s, "TRAINING_BOOKS", "HELD_OUT_BOOK", "GUTENBERG_URL"), time.perf_counter() - started)
    print(f"  {len(training)} training books, {data['training_words']:,} words; held out: {held_out.title}")
    return {"training_text": training_text, "held_out_text": held_out.text}


# ---------------------------------------------------------------------------
# 2. v1, the alphabet
# ---------------------------------------------------------------------------


def step_alphabet(s: dict, out: Path, training_text: str, held_out_text: str) -> dict:
    started = time.perf_counter()
    tok = alphabet.train(training_text)
    held_out_tokens = alphabet.encode(tok, held_out_text)
    prompt_tokens = alphabet.encode(tok, s["EXAMPLE_PROMPT"])
    strawberry = alphabet.encode(tok, "strawberry")
    data = {
        "vocab_size": len(tok.vocab),  # includes <UNK>
        "vocab": tok.vocab,
        "training_tokens": len(alphabet.encode(tok, training_text)),
        "held_out_tokens": len(held_out_tokens),
        "held_out_unk_tokens": held_out_tokens.count(UNK),
        "held_out_unk_rate": held_out_tokens.count(UNK) / len(held_out_tokens),
        "held_out_unknown_characters": sorted(set(held_out_text) - set(tok.vocab)),
        "prompt": {"text": s["EXAMPLE_PROMPT"], "tokens": prompt_tokens, "ids": alphabet.ids(tok, prompt_tokens)},
        # The one thing v1 is great at: the r's are right there.
        "strawberry_tokens": strawberry,
        "strawberry_r_count": alphabet.count_letter(strawberry, "r"),
        "probes": {w: alphabet.encode(tok, w) for w in with_and_without_space(s["PROBE_WORDS"])},
    }
    data["scoreboard"] = scoreboard_row(data, len(held_out_text), data["probes"][" strawberry"])
    results.write(out / "v1_alphabet.json", data, pick(s, "TRAINING_BOOKS", "HELD_OUT_BOOK", "EXAMPLE_PROMPT"), time.perf_counter() - started)
    print(f"  vocab {data['vocab_size']}, training text {data['training_tokens']:,} tokens, r's in strawberry: {data['strawberry_r_count']}")
    return data


# ---------------------------------------------------------------------------
# 3. v2, the dictionary
# ---------------------------------------------------------------------------


def step_dictionary(s: dict, out: Path, training_text: str, held_out_text: str) -> dict:
    started = time.perf_counter()
    tok = dictionary.train(training_text, s["DICTIONARY_MAX_VOCAB"])
    held_out_tokens = dictionary.encode(tok, held_out_text)
    unknown = dictionary.unknown_words(tok, held_out_text)
    # Just the word tokens (not punctuation or whitespace). <UNK>s count as words
    # (in the held-out book, every unknown is a word).
    word_tokens = [w for w in held_out_tokens if w == UNK or any(c.isalpha() for c in w)]
    prompt_tokens = dictionary.encode(tok, s["EXAMPLE_PROMPT"])
    data = {
        "vocab_size": len(tok.vocab),  # includes <UNK>
        "distinct_words_in_training": len(tok.counts),
        "training_tokens": len(dictionary.encode(tok, training_text)),
        "held_out_tokens": len(held_out_tokens),
        "held_out_unk_tokens": held_out_tokens.count(UNK),
        # Share of all held-out tokens (words, punctuation and whitespace) that are <UNK>.
        "held_out_unk_rate": held_out_tokens.count(UNK) / len(held_out_tokens),
        # The same, as a share of word tokens only.
        "held_out_unk_rate_among_words": word_tokens.count(UNK) / len(word_tokens),
        "held_out_distinct_unknown_words": len(unknown),
        "held_out_most_common_unknown_words": unknown.most_common(15),
        "most_common_words": [(w, tok.counts[w]) for w in tok.vocab[1:21]],
        "words_seen_once": sum(1 for c in tok.counts.values() if c == 1),
        "prompt": {"text": s["EXAMPLE_PROMPT"], "tokens": prompt_tokens, "ids": dictionary.ids(tok, prompt_tokens)},
        "probes": {
            w: {"tokens": dictionary.encode(tok, w), "ids": dictionary.ids(tok, dictionary.encode(tok, w)), "count_in_training": tok.counts.get(w, 0)}
            for w in with_and_without_space(s["PROBE_WORDS"])
        },
    }
    data["scoreboard"] = scoreboard_row(data, len(held_out_text), data["probes"][" strawberry"]["tokens"])
    results.write(
        out / "v2_dictionary.json", data, pick(s, "TRAINING_BOOKS", "HELD_OUT_BOOK", "DICTIONARY_MAX_VOCAB", "EXAMPLE_PROMPT"), time.perf_counter() - started
    )
    print(f"  vocab {data['vocab_size']:,}, training text {data['training_tokens']:,} tokens, held-out <UNK> rate {data['held_out_unk_rate']:.2%}")
    return data


# ---------------------------------------------------------------------------
# 4. v3, byte-pair encoding
# ---------------------------------------------------------------------------


def step_bpe(s: dict, out: Path, training_text: str, held_out_text: str) -> dict:
    started = time.perf_counter()
    tok = bpe.train(training_text, s["NUM_MERGES"], s["BPE_CHECKPOINTS"])
    print(f"  trained {len(tok.merges):,} merges in {tok.training_seconds:.1f}s")

    probes = with_and_without_space(s["PROBE_WORDS"])
    checkpoints = []
    for snapshot in tok.snapshots:
        k = snapshot["merges"]
        held_out_tokens = bpe.encode(tok, held_out_text, num_merges=k)
        checkpoints.append(
            {
                **snapshot,
                "held_out_tokens": len(held_out_tokens),
                "held_out_chars_per_token": len(held_out_text) / len(held_out_tokens),
                "probes": {w: bpe.encode_word(tok, w, num_merges=k) for w in probes},
            }
        )

    training_tokens = bpe.encode(tok, training_text)
    held_out_tokens = bpe.encode(tok, held_out_text)
    prompt_tokens = bpe.encode(tok, s["EXAMPLE_PROMPT"])
    journeys = {}
    for w in probes:
        history = bpe.merge_history(tok, w)
        journeys[w] = {
            "final_tokens": history[-1][1],
            "final_split_reached_at_merge": history[-1][0],
            "steps": [{"merge": step, "tokens": pieces} for step, pieces in history],
        }
    data = {
        "num_merges": len(tok.merges),
        "alphabet_size": len(tok.alphabet),
        "vocab_size": len(tok.vocab),  # <UNK> + alphabet + one token per merge
        "training_seconds": tok.training_seconds,
        "training_tokens": len(training_tokens),
        "training_chars_per_token": len(training_text) / len(training_tokens),
        "held_out_tokens": len(held_out_tokens),
        "held_out_chars_per_token": len(held_out_text) / len(held_out_tokens),
        "held_out_unk_tokens": held_out_tokens.count(UNK),
        "held_out_unk_rate": held_out_tokens.count(UNK) / len(held_out_tokens),
        "held_out_unknown_characters": sorted(set(held_out_text) - set(tok.alphabet)),
        "held_out_round_trip_ok": bpe.decode(tok, bpe.ids(tok, held_out_tokens)) == held_out_text,
        "first_merges": [
            {"merge": i + 1, "pair": list(pair), "new_token": "".join(pair), "count": count}
            for i, (pair, count) in enumerate(zip(tok.merges[: s["FIRST_MERGES_TO_SHOW"]], tok.merge_counts))
        ],
        "checkpoints": checkpoints,
        "probe_journeys": journeys,
        "prompt": {"text": s["EXAMPLE_PROMPT"], "tokens": prompt_tokens, "ids": bpe.ids(tok, prompt_tokens)},
        # The whole merge list, in order, so the tokenizer can be rebuilt without training.
        "alphabet": tok.alphabet,
        "merges": [list(pair) for pair in tok.merges],
        "merge_counts": tok.merge_counts,
    }
    data["scoreboard"] = scoreboard_row(data, len(held_out_text), journeys[" strawberry"]["final_tokens"])
    results.write(
        out / "v3_bpe.json", data, pick(s, "TRAINING_BOOKS", "HELD_OUT_BOOK", "NUM_MERGES", "BPE_CHECKPOINTS", "EXAMPLE_PROMPT"), time.perf_counter() - started
    )
    print(f"  vocab {data['vocab_size']:,}, training text {data['training_tokens']:,} tokens, held-out <UNK> rate {data['held_out_unk_rate']:.2%}")
    return data


# ---------------------------------------------------------------------------
# 5. Real tokenizers
# ---------------------------------------------------------------------------


def step_real(s: dict, out: Path) -> dict:
    started = time.perf_counter()
    tokenizers = real.load_all(s["TIKTOKEN_ENCODINGS"], s["HF_TOKENIZERS"], s["HF_REVISIONS"])
    data = {"tokenizers": real.run(tokenizers, s["PROBE_WORDS"], s["EXAMPLE_PROMPT"])}
    results.write(
        out / "real_tokenizers.json",
        data,
        pick(s, "TIKTOKEN_ENCODINGS", "HF_TOKENIZERS", "HF_REVISIONS", "PROBE_WORDS", "EXAMPLE_PROMPT"),
        time.perf_counter() - started,
    )
    for t in data["tokenizers"]:
        split = next(v for v in t["probes"]["strawberry"] if v["text"] == " strawberry")
        print(f"  {t['name']:<32} vocab {t['vocab_size']:>7,}   ' strawberry' -> {split['tokens']} {split['ids']}")
    return data


# ---------------------------------------------------------------------------
# 6. The word set
# ---------------------------------------------------------------------------


def step_word_set(s: dict, out: Path, training_text: str) -> dict:
    started = time.perf_counter()
    items = letters.build_word_set(
        training_text,
        famous=s["FAMOUS_WORDS"],
        per_count=s["SAMPLED_PER_COUNT"],
        seed=s["SEED"],
        min_letters=s["SAMPLE_MIN_LETTERS"],
        max_letters=s["SAMPLE_MAX_LETTERS"],
        min_occurrences=s["SAMPLE_MIN_OCCURRENCES"],
        common_letters=s["COMMON_LETTERS"],
        exclude={s["SPELL_THEN_COUNT_EXAMPLE_WORD"]},
    )
    rows = letters.to_json(items)
    data = {
        "items": rows,
        "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "n": len(items),
        "by_count_group": dict(Counter(item.count_group for item in items)),
        "by_group": dict(Counter(item.group for item in items)),
    }
    results.write(
        out / "word_set.json",
        data,
        pick(s, "TRAINING_BOOKS", "SEED", "FAMOUS_WORDS", "SAMPLED_PER_COUNT", "SAMPLE_MIN_LETTERS", "SAMPLE_MAX_LETTERS",
             "SAMPLE_MIN_OCCURRENCES", "COMMON_LETTERS", "SPELL_THEN_COUNT_EXAMPLE_WORD"),
        time.perf_counter() - started,
    )  # fmt: skip
    print(f"  {data['n']} questions; true counts {data['by_count_group']}")
    return data


# ---------------------------------------------------------------------------
# 7. The counting test
# ---------------------------------------------------------------------------


def model_file(out: Path, model: str) -> Path:
    return out / "count_test" / (model.replace("/", "__") + ".json")


def step_count_test(s: dict, out: Path, word_set: dict, models: list[str]) -> dict:
    # Imported here so that everything above works without torch installed.
    from dumbtok import count_test

    items = letters.from_json(word_set["items"])
    for model in models:
        print(f"  {model}")
        started = time.perf_counter()
        result = count_test.run_model(
            model,
            items,
            s["CONDITIONS"],
            revision=s["HF_REVISIONS"].get(model, "main"),
            device=s["DEVICE"],
            max_new_tokens=s["MAX_NEW_TOKENS"],
            thinking_max_new_tokens=s["THINKING_MAX_NEW_TOKENS"],
            batch_size=s["BATCH_SIZE"],
            thinking_batch_size=s["THINKING_BATCH_SIZE"],
        )
        result["word_set_sha256"] = word_set["sha256"]
        # One file per model, so a crash (or `--models`) never loses the others.
        results.write(
            model_file(out, model),
            result,
            pick(s, "CONDITIONS", "PLAIN_PROMPT", "SPELL_THEN_COUNT_PROMPT", "SPELL_THEN_COUNT_EXAMPLE_WORD",
                 "SPELL_THEN_COUNT_EXAMPLE_LETTER", "SPELL_THEN_COUNT_EXAMPLE_ANSWER", "SPELL_PROMPT",
                 "MAX_NEW_TOKENS", "THINKING_MAX_NEW_TOKENS", "BATCH_SIZE", "THINKING_BATCH_SIZE", "DEVICE"),
            time.perf_counter() - started,
        )  # fmt: skip
    return combine_count_tests(s, out, word_set)


def combine_count_tests(s: dict, out: Path, word_set: dict) -> dict:
    """Merge every per-model file (for the current word set) into count_test.json."""
    from dumbtok import count_test

    started = time.perf_counter()
    per_model = {}
    files = sorted((out / "count_test").glob("*.json")) if (out / "count_test").exists() else []
    for path in files:
        result = results.read(path)
        if result.get("word_set_sha256") != word_set["sha256"]:
            print(f"  skipping {path.name}: it was run on a different word set")
            continue
        per_model[result["model"]] = result
    # Keep the order of config.MODELS, then anything else.
    order = [m for m in s["MODELS"] if m in per_model] + sorted(m for m in per_model if m not in s["MODELS"])
    summaries = {m: count_test.summarise(per_model[m]) for m in order}
    data = {
        "word_set_sha256": word_set["sha256"],
        "conditions": s["CONDITIONS"],
        "table": {m: {c: v["accuracy"] for c, v in summaries[m]["conditions"].items()} | {"spelling": summaries[m]["spelling_accuracy"]} for m in order},
        "strawberry": {m: summaries[m]["strawberry"] for m in order},
        "models": {
            m: {
                "info": per_model[m]["info"],
                "settings": per_model[m]["settings"],
                "seconds": per_model[m]["seconds"],
                "summary": summaries[m],
                "example_prompts": per_model[m]["example_prompts"],
                "rows": per_model[m]["rows"],
            }
            for m in order
        },
    }
    results.write(out / "count_test.json", data, pick(s, "MODELS", "CONDITIONS"), time.perf_counter() - started)
    return data


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def summary_lines(v1: dict, v2: dict, v3: dict, real_data: dict, count: dict | None, timings: dict) -> list[str]:
    """The short human summary printed at the end (and saved to summary.txt)."""
    lines = ["", "SCOREBOARD (the held-out book was never seen in training)", ""]
    lines.append(f"{'':<15}{'vocab':>8}{'tokens, training':>18}{'tokens, held out':>18}{'chars/token':>13}{'<UNK> rate':>12}   ' strawberry'")
    for name, d in [("v1 alphabet", v1), ("v2 dictionary", v2), ("v3 BPE", v3)]:
        row = d["scoreboard"]
        lines.append(
            f"{name:<15}{row['vocab_size']:>8,}{row['training_tokens']:>18,}{row['held_out_tokens']:>18,}"
            f"{row['held_out_chars_per_token']:>13.2f}{row['held_out_unk_rate']:>12.2%}   {row['strawberry_tokens']}"
        )
    journey = v3["probe_journeys"][" strawberry"]
    lines.append("")
    lines.append(f"BPE: {v3['num_merges']:,} merges in {v3['training_seconds']:.1f}s; ' strawberry' reached {journey['final_tokens']} at merge {journey['final_split_reached_at_merge']:,}")

    lines.append("")
    lines.append(f"{'REAL TOKENIZERS':<41}{repr(' strawberry'):<30}{repr('strawberry')}")
    for t in real_data["tokenizers"]:
        by_text = {v["text"]: v for v in t["probes"]["strawberry"]}
        lines.append(f"  {t['name']:<39}{str(by_text[' strawberry']['tokens']):<30}{by_text['strawberry']['tokens']}")

    if count and count["models"]:
        from dumbtok import count_test

        lines.append("")
        lines.append("LETTER COUNTING (correct / asked)")
        lines += count_test.table_lines({m: v["summary"] for m, v in count["models"].items()}, count["conditions"])
        lines.append("")
        lines.append("'strawberry', plain:")
        for m, replies in count["strawberry"].items():
            lines.append(f"  {m:<38} {replies.get('plain', '')[:100]!r}")

    lines.append("")
    lines.append("WALL TIME PER STEP")
    for name, seconds in timings.items():
        lines.append(f"  {name:<12} {seconds:8.1f}s")
    lines.append(f"  {'total':<12} {sum(timings.values()):8.1f}s")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-models", action="store_true", help="skip the letter-counting test (no GPU, no big downloads)")
    parser.add_argument("--models", nargs="+", metavar="MODEL", help="run the counting test only for these models")
    parser.add_argument("--quick", action="store_true", help="a tiny version of everything; results go to results/quick/")
    args = parser.parse_args()

    s = load_settings(args.quick)
    out = config.RESULTS_DIR / "quick" if args.quick else config.RESULTS_DIR
    timings: dict[str, float] = {}

    def step(name: str, function, *arguments):
        print(f"\n[{name}]")
        started = time.perf_counter()
        value = function(*arguments)
        timings[name] = time.perf_counter() - started
        return value

    texts = step("corpus", step_corpus, s, out)
    v1 = step("v1", step_alphabet, s, out, texts["training_text"], texts["held_out_text"])
    v2 = step("v2", step_dictionary, s, out, texts["training_text"], texts["held_out_text"])
    v3 = step("v3", step_bpe, s, out, texts["training_text"], texts["held_out_text"])
    real_data = step("real", step_real, s, out)
    word_set = step("word set", step_word_set, s, out, texts["training_text"])
    count = None
    if not args.skip_models:
        models = args.models or s["MODELS"]
        count = step("count test", step_count_test, s, out, word_set, models)

    lines = summary_lines(v1, v2, v3, real_data, count, timings)
    print("\n".join(lines))
    (out / "summary.txt").write_text("\n".join(lines).lstrip("\n") + "\n")
    print(f"\nResults are in {out}/")


if __name__ == "__main__":
    main()
