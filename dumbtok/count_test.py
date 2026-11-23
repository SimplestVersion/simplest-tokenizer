"""Ask small open models to count letters, and check their answers.

For every model, every test word, and every way of asking ("condition"):

    plain             How many times does the letter "r" appear in the word "strawberry"?
    spaced            ... in the word "s t r a w b e r r y"?   (one token per letter)
    spell_then_count  after one worked example: write the letters one per line,
                      mark the r's, count the marks, say "Answer: N"
    thinking          plain, but Qwen3 may think out loud first (Qwen3 only)
    spelling          not a count: "Spell the word ..." (can it even spell it?)

Decoding is greedy (always the most likely next token), so every run gives the
same replies. Every raw reply is saved, so any score can be checked by hand.

torch and transformers are imported inside the functions that need them, so
the rest of this file (the answer parser) works without them.
"""

import gc
import importlib.metadata
import re
import time

import config
from dumbtok.letters import COUNT_GROUPS, Item, spaced

# ---------------------------------------------------------------------------
# The prompts
# ---------------------------------------------------------------------------


def is_qwen3(model_name: str) -> bool:
    """Qwen3 models have a switch for "thinking mode"; the others don't."""
    return model_name.startswith("Qwen/Qwen3")


def conditions_for(model_name: str, conditions: list[str]) -> list[str]:
    """The conditions this model takes part in (thinking is Qwen3 only)."""
    return [c for c in conditions if c != "thinking" or is_qwen3(model_name)]


def messages_for(condition: str, item: Item) -> list[dict]:
    """The chat messages for one question. The exact wording lives in config.py."""
    if condition in ("plain", "thinking"):
        question = config.PLAIN_PROMPT.format(letter=item.letter, word=item.word)
        return [{"role": "user", "content": question}]
    if condition == "spaced":
        question = config.PLAIN_PROMPT.format(letter=item.letter, word=spaced(item.word))
        return [{"role": "user", "content": question}]
    if condition == "spell_then_count":
        # One worked example as a short fake conversation, then the real question.
        example_question = config.SPELL_THEN_COUNT_PROMPT.format(
            letter=config.SPELL_THEN_COUNT_EXAMPLE_LETTER, word=config.SPELL_THEN_COUNT_EXAMPLE_WORD
        )
        question = config.SPELL_THEN_COUNT_PROMPT.format(letter=item.letter, word=item.word)
        return [
            {"role": "user", "content": example_question},
            {"role": "assistant", "content": config.SPELL_THEN_COUNT_EXAMPLE_ANSWER},
            {"role": "user", "content": question},
        ]
    if condition == "spelling":
        return [{"role": "user", "content": config.SPELL_PROMPT.format(word=item.word)}]
    raise ValueError(f"unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Reading the answers
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "once": 1, "twice": 2,
}  # fmt: skip
# A number written in digits, or one of the words above, as a whole word
# ("one" counts, the "one" inside "someone" doesn't).
NUMBER_PATTERN = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)


def first_number(text: str) -> int | None:
    """The first number in the text, in digits or words: "**once**" -> 1, "3 times" -> 3."""
    match = NUMBER_PATTERN.search(text)
    if match is None:
        return None
    found = match.group(1).lower()
    return int(found) if found.isdigit() else NUMBER_WORDS[found]


def parse_count(reply: str, condition: str) -> int | None:
    """Read the model's count from its reply. None means we couldn't find one.

    The rule (decided before running the test):
    * thinking: only the part after "</think>" counts. No "</think>" (it ran
      out of tokens while thinking) means no answer.
    * spell_then_count and thinking: the number after the last "Answer:".
      For spell_then_count, no "Answer:" means no answer.
    * otherwise: the first number in the reply.
    """
    if condition == "thinking":
        if "</think>" not in reply:
            return None
        reply = reply.rsplit("</think>", 1)[1]
    position = reply.lower().rfind("answer:")
    if condition == "spell_then_count" and position == -1:
        return None
    if condition in ("spell_then_count", "thinking") and position != -1:
        return first_number(reply[position + len("answer:") :])
    return first_number(reply)


def check_steps(reply: str, word: str) -> dict:
    """For spell_then_count: did the model really do the steps it was shown?

    wrote_the_word: the one-letter lines spell the word exactly.
    marks: how many " <-" marks it wrote (before "Answer:").
    A right answer with the wrong letters written out was a lucky guess.
    """
    position = reply.lower().rfind("answer:")
    work = reply if position == -1 else reply[:position]
    letter_lines = [line.strip() for line in work.splitlines() if re.fullmatch(r"\s*[A-Za-z]\s*(<-)?\s*", line)]
    written = "".join(line[0].lower() for line in letter_lines)
    return {"wrote_the_word": written == word, "marks": work.count("<-")}


def spelled_correctly(reply: str, word: str) -> bool:
    """Did the reply spell the word out, letter by letter, exactly?

    The reply must be the word's letters as separate pieces, in order, and
    nothing else. Case and separators don't matter:
    "s t r a w b e r r y", "S-T-R-A-W-B-E-R-R-Y" and "s, t, r, ..." all pass.
    Just repeating "strawberry" doesn't: copying the word isn't spelling it.
    """
    return re.findall(r"[a-z]+", reply.lower()) == list(word)


# ---------------------------------------------------------------------------
# Running a model
# ---------------------------------------------------------------------------


def pick_device(setting: str = "auto") -> str:
    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


def load_model(name: str, revision: str, device: str):
    """Download (once) and load a model and its tokenizer."""
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_info = HfApi().model_info(name, revision=revision)
    commit = model_info.sha
    tokenizer = AutoTokenizer.from_pretrained(name, revision=commit)
    # We ask many questions at once (a batch). Shorter prompts get padded on the
    # left, so every reply starts right after its own prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bfloat16 halves the memory on a GPU; on a CPU, float32 is faster and exact.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, revision=commit, dtype=dtype).to(device)
    model.eval()
    info = {
        "hf_commit": commit,
        "licence": (model_info.card_data or {}).get("license"),  # read from the model card
        "licence_source_url": f"https://huggingface.co/{name}/blob/{commit}/README.md",
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return model, tokenizer, info


def render_prompt(tokenizer, messages: list[dict], thinking: bool | None) -> str:
    """Turn chat messages into the exact text the model reads (its "chat template").

    thinking: for Qwen3, switch thinking mode on or off. None for other models.
    """
    extra = {} if thinking is None else {"enable_thinking": thinking}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[dict]:
    """Greedy replies to already-rendered prompts, a batch at a time."""
    import torch
    from transformers import GenerationConfig

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    settings = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: always pick the most likely token
        eos_token_id=list(eos_ids),
        pad_token_id=tokenizer.pad_token_id,
    )
    replies = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        # The chat template already added any special start token, so don't add another.
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, generation_config=settings)
        for row in output[:, inputs["input_ids"].shape[1] :].tolist():
            # The reply ends at the first end-of-text token; no such token means
            # it was still going when it hit max_new_tokens.
            ends = [i for i, token in enumerate(row) if token in eos_ids]
            length = ends[0] + 1 if ends else len(row)
            replies.append(
                {
                    "reply": tokenizer.decode(row[:length], skip_special_tokens=True).strip(),
                    "new_tokens": length,
                    "hit_cap": not ends,
                }
            )
    return replies


def run_model(
    name: str,
    items: list[Item],
    conditions: list[str],
    revision: str = "main",
    device: str = "auto",
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    thinking_max_new_tokens: int = config.THINKING_MAX_NEW_TOKENS,
    batch_size: int = config.BATCH_SIZE,
    thinking_batch_size: int = config.THINKING_BATCH_SIZE,
) -> dict:
    """Ask one model every question, in every condition, plus the spelling probe."""
    import torch

    device = pick_device(device)
    started = time.perf_counter()
    model, tokenizer, info = load_model(name, revision, device)
    info["load_seconds"] = time.perf_counter() - started
    info["torch"] = importlib.metadata.version("torch")
    info["transformers"] = importlib.metadata.version("transformers")

    rows, seconds, example_prompts = [], {}, {}
    for condition in conditions_for(name, conditions) + ["spelling"]:
        condition_started = time.perf_counter()
        thinking = (condition == "thinking") if is_qwen3(name) else None
        prompts = [render_prompt(tokenizer, messages_for(condition, item), thinking) for item in items]
        example_prompts[condition] = prompts[0]
        if condition == "thinking":
            replies = generate(model, tokenizer, prompts, thinking_max_new_tokens, thinking_batch_size)
        else:
            replies = generate(model, tokenizer, prompts, max_new_tokens, batch_size)
        for item, reply in zip(items, replies):
            row = {
                "condition": condition,
                "word": item.word,
                "letter": item.letter,
                "truth": item.count,
                "count_group": item.count_group,
                "group": item.group,
                **reply,
            }
            if condition == "spelling":
                row["spelled_ok"] = spelled_correctly(reply["reply"], item.word)
                # Stricter still: exactly "s t r a w b e r r y", as asked. Kept for reference.
                row["spelled_exactly_as_asked"] = reply["reply"] == spaced(item.word)
            else:
                row["answer"] = parse_count(reply["reply"], condition)
                row["correct"] = row["answer"] == item.count
            rows.append(row)
        seconds[condition] = time.perf_counter() - condition_started
        print(f"    {condition:<17} {seconds[condition]:6.1f}s  {describe(rows, condition)}", flush=True)

    # Free the GPU before the next model.
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": name,
        "info": info,
        "settings": {
            "max_new_tokens": max_new_tokens,
            "thinking_max_new_tokens": thinking_max_new_tokens,
            "batch_size": batch_size,
            "thinking_batch_size": thinking_batch_size,
            "greedy": True,
        },
        "seconds": seconds,
        "example_prompts": example_prompts,
        "rows": rows,
    }


def describe(rows: list[dict], condition: str) -> str:
    """One line of progress, like "41/104 correct, 2 unparsed"."""
    these = [r for r in rows if r["condition"] == condition]
    if condition == "spelling":
        return f"{sum(r['spelled_ok'] for r in these)}/{len(these)} spelled right"
    unparsed = sum(r["answer"] is None for r in these)
    return f"{sum(r['correct'] for r in these)}/{len(these)} correct, {unparsed} unparsed"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def accuracy(rows: list[dict]) -> dict:
    """Share of correct answers, plus the raw counts behind it."""
    correct = sum(r["correct"] for r in rows)
    return {
        "accuracy": correct / len(rows) if rows else None,
        "correct": correct,
        "n": len(rows),
        "unparsed": sum(r["answer"] is None for r in rows),
        "hit_cap": sum(r["hit_cap"] for r in rows),
    }


def summarise(model_result: dict) -> dict:
    """Scores for one model: per condition, split by true count and by word group."""
    rows = model_result["rows"]
    conditions = [c for c in model_result["seconds"] if c != "spelling"]
    summary = {"conditions": {}}
    for condition in conditions:
        these = [r for r in rows if r["condition"] == condition]
        summary["conditions"][condition] = {
            **accuracy(these),
            "by_true_count": {g: accuracy([r for r in these if r["count_group"] == g]) for g in COUNT_GROUPS},
            "by_group": {g: accuracy([r for r in these if r["group"] == g]) for g in ("famous", "sampled")},
        }

    spelling = {r["word"]: r for r in rows if r["condition"] == "spelling"}
    summary["spelling_accuracy"] = sum(r["spelled_ok"] for r in spelling.values()) / len(spelling)
    summary["spelling_hit_cap"] = sum(r["hit_cap"] for r in spelling.values())

    # The twist: words it spells right but still miscounts (plain condition).
    if "plain" in conditions:
        plain = [r for r in rows if r["condition"] == "plain"]
        spelled = [r for r in plain if spelling[r["word"]]["spelled_ok"]]
        twist = [r["word"] for r in spelled if not r["correct"]]
        summary["spells_right_but_miscounts_plain"] = {
            "words": len(twist),
            "share_of_all_words": len(twist) / len(plain),
            "share_of_words_it_spells_right": len(twist) / len(spelled) if spelled else None,
            "examples": twist[:10],
        }

    # Did spell_then_count work because the model followed the steps, or by luck?
    if "spell_then_count" in conditions:
        these = [r for r in rows if r["condition"] == "spell_then_count"]
        steps = [check_steps(r["reply"], r["word"]) for r in these]
        followed = [r for r, s in zip(these, steps) if s["wrote_the_word"]]
        not_followed = [r for r, s in zip(these, steps) if not s["wrote_the_word"]]
        parsed = [(r, s) for r, s in zip(these, steps) if r["answer"] is not None]
        summary["spell_then_count_steps"] = {
            "wrote_the_word_correctly": len(followed) / len(these),
            "accuracy_when_word_written_correctly": accuracy(followed)["accuracy"],
            "accuracy_when_not": accuracy(not_followed)["accuracy"],
            "answer_equals_its_own_marks": sum(r["answer"] == s["marks"] for r, s in parsed) / len(parsed) if parsed else None,
        }

    summary["strawberry"] = {r["condition"]: r["reply"] for r in rows if r["word"] == "strawberry"}
    return summary


def table_lines(summaries: dict[str, dict], conditions: list[str]) -> list[str]:
    """Accuracy per model and condition as a small text table, then the same
    split by true count (0, 1, 2, 3+) and by word group (famous 8 or sampled)."""
    header = f"{'model':<38}" + "".join(f"{c:>18}" for c in conditions) + f"{'spelling':>10}"
    lines = [header, "-" * len(header)]
    for model, summary in summaries.items():
        cells = []
        for condition in conditions:
            score = summary["conditions"].get(condition)
            cells.append(f"{score['correct']:>4}/{score['n']:<3} {score['accuracy']:5.0%}" if score else "-")
        lines.append(f"{model:<38}" + "".join(f"{cell:>18}" for cell in cells) + f"{summary['spelling_accuracy']:>10.0%}")

    lines += ["", f"{'split (correct / asked)':<56}" + "".join(f"{g:>8}" for g in ["0", "1", "2", "3+", "famous", "sampled"])]
    for model, summary in summaries.items():
        for condition, score in summary["conditions"].items():
            parts = list(score["by_true_count"].values()) + list(score["by_group"].values())
            lines.append(f"{model:<38}{condition:<18}" + "".join(f"{p['correct']:>4}/{p['n']:<3}" for p in parts))
    return lines
