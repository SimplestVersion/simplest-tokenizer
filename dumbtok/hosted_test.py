"""The same letter-counting test, against a hosted model (ChatGPT-style APIs).

Works with any "OpenAI-compatible" chat endpoint: OpenAI itself, OpenRouter,
a LiteLLM proxy, a local llama.cpp or vLLM server, and so on.

Everything about the endpoint comes from environment variables, never from
files in this repo:

    DT_BASE_URL   the API's base URL, e.g. https://api.openai.com/v1
    DT_API_KEY    your API key (sent as "Authorization: Bearer ...")
    DT_MODELS     comma-separated model names, e.g. gpt-4o-mini,gpt-4.1

Run it with:

    DT_BASE_URL=... DT_API_KEY=... DT_MODELS=... uv run python -m dumbtok.hosted_test

It asks the `plain` and `spaced` questions for every word in
results/word_set.json and writes results/hosted_test.json. Hosted models
change without notice, so the date and the model name each API reports back
are saved with every answer.
"""

import json
import os
import time
import urllib.error
import urllib.request

import config
from dumbtok import results
from dumbtok.count_test import accuracy, messages_for, parse_count
from dumbtok.letters import COUNT_GROUPS, from_json

CONDITIONS = ["plain", "spaced"]
# Upper limit on reply length, in tokens. Some newer models think before they
# answer and need room for that.
MAX_TOKENS = 4096


def settings_from_environment() -> tuple[str, str, list[str]]:
    """Read DT_BASE_URL, DT_API_KEY and DT_MODELS, or explain what's missing."""
    base_url = os.environ.get("DT_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("DT_API_KEY", "")
    models = [m.strip() for m in os.environ.get("DT_MODELS", "").split(",") if m.strip()]
    missing = [name for name, value in [("DT_BASE_URL", base_url), ("DT_API_KEY", api_key), ("DT_MODELS", models)] if not value]
    if missing:
        raise SystemExit(
            f"Missing environment variable(s): {', '.join(missing)}.\n"
            "Set DT_BASE_URL (e.g. https://api.openai.com/v1), DT_API_KEY and DT_MODELS "
            "(comma-separated model names), then run again."
        )
    return base_url, api_key, models


def request_body(model: str, messages: list[dict]) -> bytes:
    """Build the JSON request. Temperature 0 means "always pick the likeliest word".

    Some newer models (Claude Opus 5.5, for one) refuse any temperature except
    their default. TRY THIS: set DT_TEMPERATURE=none to leave it out entirely.
    """
    body = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS}
    temperature = os.environ.get("DT_TEMPERATURE", "0")
    if temperature.lower() != "none":
        body["temperature"] = float(temperature)
    return json.dumps(body).encode()


def ask(base_url: str, api_key: str, model: str, messages: list[dict], retries: int = 4) -> dict:
    """Send one chat request. Retries a few times if the server is busy."""
    body = request_body(model, messages)
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            # 429 = too many requests, 5xx = server trouble: wait and try again.
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(2**attempt * 5)
                continue
            raise SystemExit(f"{model}: HTTP {error.code} from {base_url}: {error.read()[:500]!r}") from error
        except urllib.error.URLError as error:
            if attempt < retries:
                time.sleep(2**attempt * 5)
                continue
            raise SystemExit(f"Could not reach {base_url}: {error.reason}") from error
    raise AssertionError("unreachable")


def run(results_dir=config.RESULTS_DIR) -> dict:
    base_url, api_key, models = settings_from_environment()
    word_set_file = results_dir / "word_set.json"
    if not word_set_file.exists():
        raise SystemExit(f"{word_set_file} is missing. Run `uv run python run_all.py --skip-models` first.")
    items = from_json(json.loads(word_set_file.read_text())["items"])

    started = time.perf_counter()
    output = {"endpoint": base_url, "conditions": CONDITIONS, "max_tokens": MAX_TOKENS, "models": {}}
    for model in models:
        print(f"{model}:")
        rows, seconds = [], {}
        for condition in CONDITIONS:
            condition_started = time.perf_counter()
            for item in items:
                response = ask(base_url, api_key, model, messages_for(condition, item))
                choice = response["choices"][0]
                reply = (choice["message"].get("content") or "").strip()
                answer = parse_count(reply, condition)
                rows.append(
                    {
                        "condition": condition,
                        "word": item.word,
                        "letter": item.letter,
                        "truth": item.count,
                        "count_group": item.count_group,
                        "group": item.group,
                        "reply": reply,
                        "answer": answer,
                        "correct": answer == item.count,
                        "hit_cap": choice.get("finish_reason") == "length",
                        "model_reported": response.get("model"),
                    }
                )
            seconds[condition] = time.perf_counter() - condition_started
            these = [r for r in rows if r["condition"] == condition]
            print(f"    {condition:<8} {sum(r['correct'] for r in these)}/{len(these)} correct")
        output["models"][model] = {
            "seconds": seconds,
            "summary": {
                c: {
                    **accuracy([r for r in rows if r["condition"] == c]),
                    "by_true_count": {
                        g: accuracy([r for r in rows if r["condition"] == c and r["count_group"] == g]) for g in COUNT_GROUPS
                    },
                }
                for c in CONDITIONS
            },
            "strawberry": {r["condition"]: r["reply"] for r in rows if r["word"] == "strawberry"},
            "rows": rows,
        }
    # The API key is never written anywhere; only the endpoint URL is recorded.
    results.write(
        results_dir / "hosted_test.json",
        output,
        settings={"conditions": CONDITIONS, "max_tokens": MAX_TOKENS, "models": models},
        seconds=time.perf_counter() - started,
    )
    return output


if __name__ == "__main__":
    run()
