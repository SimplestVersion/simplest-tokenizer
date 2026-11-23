"""Ask the episode's opening question, word for word, to every local model.

The counting test (count_test.py) uses a stricter wording ("answer with just
the number"). This asks the casual version a person would type. Greedy
decoding, one question at a time, Qwen3 thinking off.

TRY THIS: change QUESTION below, or ask("Qwen/Qwen3-4B", enable_thinking=True).

Writes results/cold_open.json. Run from the repo root:
    uv run python -m dumbtok.cold_open
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from dumbtok import results
from dumbtok.count_test import pick_device

# The question exactly as the episode's cold open shows it.
QUESTION = config.EXAMPLE_PROMPT  # "How many r's are in strawberry?"


def ask(model_name: str, enable_thinking: bool = False, max_new_tokens: int = 256) -> dict:
    device = pick_device(config.DEVICE)
    revision = config.HF_REVISIONS.get(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.bfloat16).to(device)
    extra = {"enable_thinking": enable_thinking} if "Qwen3" in model_name else {}
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}], add_generation_prompt=True, return_tensors="pt", return_dict=True, **extra
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"model": model_name, "revision": revision, "thinking": enable_thinking, "reply": reply}


def run() -> list[dict]:
    started = time.perf_counter()
    replies = [ask(m) for m in config.MODELS]
    for r in replies:
        print(f"{r['model']:42s} {r['reply'][:120]!r}")
    results.write(config.RESULTS_DIR / "cold_open.json", {"question": QUESTION, "replies": replies},
                  settings={"MODELS": config.MODELS, "greedy": True, "dtype": "bfloat16", "batch_size": 1},
                  seconds=time.perf_counter() - started)
    return replies


if __name__ == "__main__":
    run()
