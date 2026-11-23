import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Why AI can't count the r's in strawberry")

with app.setup:
    # Everything the notebook uses. The tokenizers live in dumbtok/, the knobs in config.py.
    import html
    import json

    import marimo as mo

    import config
    from dumbtok import UNK, alphabet, bpe, corpus, dictionary, real


@app.function
def visible(token: str) -> str:
    """Show a space as ␣ and a newline as ↵, so you can see where they go."""
    return token.replace(" ", "␣").replace("\n", "↵")


@app.function
def token_boxes(tokens: list[str], ids: list[int] | None = None) -> mo.Html:
    """Draw tokens as boxes (with their ids underneath, if given)."""
    boxes = []
    for i, token in enumerate(tokens):
        shown = html.escape(visible(token))
        number = f"<div style='font-size:0.7em;color:#666'>{ids[i]}</div>" if ids else ""
        boxes.append(
            "<div style='display:inline-block;border:2px solid #111;border-radius:4px;"
            "padding:2px 6px;margin:2px;font-family:monospace;text-align:center'>"
            f"{shown}{number}</div>"
        )
    return mo.Html("".join(boxes))


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Why AI can't count the r's in strawberry

    This is the companion notebook to the **Simplest Version** episode "Why AI can't count the r's in 'strawberry'". We build three tokenizers from scratch, see how real models split the same words, and run a letter-counting test on five small open models. Every number in the episode comes from the JSON files in `results/`, and everything below runs live so you can change the knobs and watch the numbers move.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## What the model actually sees

    A language model never reads letters. Before a prompt reaches the model, a tokenizer cuts it into chunks called tokens and turns each one into a number. Below is our question after GPT-4o's tokenizer (`o200k_base`) finishes with it. The model gets eight numbers, and `" strawberry"` (with the space before it) is the single id `101830`. There's no `r` sitting inside that number to count, so the model only knows how strawberry is spelled if it happened to pick that up as a fact during training.
    """)
    return


@app.cell
def _():
    gpt4o = real.load_tiktoken("o200k_base", "GPT-4o")
    prompt_in_gpt4o = real.tokenize(gpt4o, config.EXAMPLE_PROMPT)
    token_boxes(prompt_in_gpt4o["tokens"], prompt_in_gpt4o["ids"])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The books we learn from

    To see why real tokenizers work this way, we'll build our own from scratch and train them on 11 public-domain novels from Project Gutenberg, about 1.6 million words in total (1,644,054 words and 9,141,555 characters). We also set aside one extra book, *The Yellow Wallpaper* (6,085 words, 31,497 characters), that none of our tokenizers ever see during training. That held-out book is our reality check for how well each tokenizer handles words it has never met. You can swap the book list or the held-out book in `config.py` (try making *War and Peace* the held-out book: its Russian names stress-test v2, and its accented letters even trip up v3).
    """)
    return


@app.cell
def _():
    books = corpus.load_books(config.TRAINING_BOOKS, config.DATA_DIR, config.GUTENBERG_URL)
    held_out_book = corpus.load_book(config.HELD_OUT_BOOK, config.DATA_DIR, config.GUTENBERG_URL)
    training_text = corpus.join_books(books)
    held_out_text = held_out_book.text
    mo.ui.table(
        [{"book": b.title, "words": len(b.text.split()), "role": "training"} for b in books]
        + [{"book": held_out_book.title, "words": len(held_out_text.split()), "role": "held out"}],
        selection=None,
    )
    return held_out_text, training_text


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## v1, the alphabet

    The simplest tokenizer we could build gives every character its own token, which in Python is basically `list(text)`. Its vocabulary is tiny at 128 entries (every distinct character in the 11 novels, plus `<UNK>`), and counting r's is exact by construction because every letter sits in its own box. The catch is length. The training books turn into 9,141,555 tokens, *The Yellow Wallpaper* takes 31,497, and our one-sentence prompt takes 31. Because a model takes one step per token and every new token looks back at every earlier one, sequences that long mean more work and less fits in the model's context.
    """)
    return


@app.cell
def _(held_out_text, training_text):
    tok_v1 = alphabet.train(training_text)
    _prompt = alphabet.encode(tok_v1, config.EXAMPLE_PROMPT)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{len(tok_v1.vocab):,}", label="vocabulary"),
                    mo.stat(f"{len(training_text):,}", label="tokens, training books"),
                    mo.stat(f"{len(held_out_text):,}", label="tokens, held-out book"),
                ]
            ),
            token_boxes(_prompt, alphabet.ids(tok_v1, _prompt)),
        ]
    )
    return (tok_v1,)


@app.cell
def _(tok_v1):
    # With one token per letter, counting r's is just counting tokens.
    alphabet.count_letter(alphabet.encode(tok_v1, "strawberry"), "r")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## v2, the dictionary

    To shorten those sequences, v2 gives every word its own token, splitting on spaces and punctuation and gluing a single leading space onto each word (the same rule v3 will use). That cuts *The Yellow Wallpaper* down to 7,831 tokens and the training books to 2,137,445, but it brings three new headaches. The vocabulary blows up to 54,622 entries, `" strawberry"` disappears into one opaque number (`29224`), and any unseen word becomes `<UNK>`. On *The Yellow Wallpaper*, 1.35% of tokens turn into `<UNK>`, including `" wallpaper"` 12 times in a book called *The Yellow Wallpaper* (and the straight apostrophe in our prompt becomes `<UNK>` too, because the novels use curly ones). Pick a smaller cap in the dropdown below, or change `DICTIONARY_MAX_VOCAB` in `config.py`, and watch the `<UNK>` rate climb.
    """)
    return


@app.cell
def _():
    # TRY THIS: keep fewer words and watch the <UNK> rate climb.
    max_vocab = mo.ui.dropdown(
        options={"every word": None, "50,000": 50000, "20,000": 20000, "5,000": 5000, "1,000": 1000},
        value="every word",
        label="Dictionary size: keep the most common",
    )
    max_vocab
    return (max_vocab,)


@app.cell
def _(held_out_text, max_vocab, training_text):
    tok_v2 = dictionary.train(training_text, max_vocab.value)
    _held_out = dictionary.encode(tok_v2, held_out_text)
    _unknown = dictionary.unknown_words(tok_v2, held_out_text)
    _prompt = dictionary.encode(tok_v2, config.EXAMPLE_PROMPT)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{len(tok_v2.vocab):,}", label="vocabulary"),
                    mo.stat(f"{len(dictionary.encode(tok_v2, training_text)):,}", label="tokens, training books"),
                    mo.stat(f"{len(_held_out):,}", label="tokens, held-out book"),
                    mo.stat(f"{_held_out.count(UNK) / len(_held_out):.2%}", label="<UNK>, held-out book"),
                ]
            ),
            token_boxes(_prompt, dictionary.ids(tok_v2, _prompt)),
            mo.md("**Most common unknown words in the held-out book:** " + ", ".join(f"`{w!r}` ×{n}" for w, n in _unknown.most_common(12))),
        ]
    )
    return (tok_v2,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## v3, the compromise (byte-pair encoding)

    We want short sequences without a giant dictionary or unknown words, and byte-pair encoding (BPE) is the compromise real models use. Start from single characters, find the most common neighbouring pair inside words, glue it into one new token, and repeat. In seconds on one CPU core, 8,000 merges give us a vocabulary of 8,128 that brings *The Yellow Wallpaper* down to 8,737 tokens (3.61 characters per token) with zero `<UNK>` tokens, because any unfamiliar word can still fall back to single letters (as long as each letter is one our books contain). Early merges grab the most common English chunks like `␣`+`t`, `h`+`e`, and `␣t`+`he`, and by merge 5,019 `" strawberry"` has shrunk from 11 individual characters down to `␣straw | ber | ry`. Drag the slider below to watch pieces snap together as merges grow, or type your own word into the text box (try leaving off the leading space, or bump `NUM_MERGES` in `config.py`).
    """)
    return


@app.cell
def _(training_text):
    # Learns config.NUM_MERGES merges (8,000 by default): about 10-20 seconds.
    tok_v3 = bpe.train(training_text, config.NUM_MERGES, config.BPE_CHECKPOINTS)
    mo.md(f"Learned **{len(tok_v3.merges):,} merges** in **{tok_v3.training_seconds:.1f} s**.")
    return (tok_v3,)


@app.cell
def _():
    merges_slider = mo.ui.slider(0, config.NUM_MERGES, step=10, value=config.NUM_MERGES, label="Merges", show_value=True, full_width=True)
    word_box = mo.ui.text(value=" strawberry", label="Try your own word (a leading space matters!)")
    mo.vstack([merges_slider, word_box])
    return merges_slider, word_box


@app.cell
def _(held_out_text, merges_slider, tok_v3, word_box):
    _k = merges_slider.value
    _held_out = bpe.encode(tok_v3, held_out_text, num_merges=_k)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{1 + len(tok_v3.alphabet) + _k:,}", label="vocabulary"),
                    mo.stat(f"{len(held_out_text) / len(_held_out):.2f}", label="characters per token, held-out book"),
                    mo.stat(f"{_held_out.count(UNK)}", label="<UNK> tokens, held-out book"),
                ]
            ),
            mo.md("**␣strawberry**"),
            token_boxes(bpe.encode_word(tok_v3, " strawberry", _k)),
            mo.md(f"**{visible(word_box.value)}**"),
            token_boxes(bpe.encode(tok_v3, word_box.value, _k)),
        ]
    )
    return


@app.cell
def _(tok_v3):
    # The first 20 merges BPE learned, with how often each pair occurred.
    mo.ui.table(
        [
            {"merge": i + 1, "pair": f"{visible(a)} + {visible(b)}", "new token": visible(a + b), "count": n}
            for i, ((a, b), n) in enumerate(zip(tok_v3.merges[:20], tok_v3.merge_counts))
        ],
        selection=None,
    )
    return


@app.cell
def _(tok_v3, word_box):
    # Every step your word goes through, from letters to its final split.
    # (BPE never merges across words, so type a single word here.)
    mo.ui.table(
        [
            {"after merge": step, "pieces": " | ".join(visible(p) for p in pieces), "tokens": len(pieces)}
            for step, pieces in bpe.merge_history(tok_v3, word_box.value)
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The real tokenizers

    Our BPE never merged `" strawberry"` all the way to one token because the word only appeared twice in our 11 novels. Real tokenizers use the same merge idea with much bigger vocabularies, from 32,000 in Mistral 7B up to about 200,000 in GPT-4o and Phi-4-mini. (Most of them merge bytes rather than characters, which is how they avoid `<UNK>` entirely.) In 7 of the 8 real tokenizers we tested (GPT-4o, GPT-4, GPT-2, Qwen3, OLMo-2, SmolLM2, and Phi-4-mini), `" strawberry"` is a single token, with Mistral 7B as the lone holdout (`␣straw | berry`). Change the word box above to compare your own text across all 11 tokenizers side by side, or use the dropdown below to see how a leading space or capital letters change the split. Drop the space and GPT-4o breaks `"strawberry"` into `st | raw | berry`, and in all caps every one of the eight splits it into 4 or 5 pieces.
    """)
    return


@app.cell
def _():
    # Downloads each tokenizer once (a few MB each), then works offline.
    real_tokenizers = real.load_all(config.TIKTOKEN_ENCODINGS, config.HF_TOKENIZERS, config.HF_REVISIONS)
    return (real_tokenizers,)


@app.cell
def _(real_tokenizers, tok_v1, tok_v2, tok_v3, word_box):
    # Your word through every tokenizer, ours and the real ones, side by side.
    _word = word_box.value
    _rows = [
        {"tokenizer": "v1 alphabet (ours)", "pieces": alphabet.encode(tok_v1, _word)},
        {"tokenizer": "v2 dictionary (ours)", "pieces": dictionary.encode(tok_v2, _word)},
        {"tokenizer": "v3 BPE (ours)", "pieces": bpe.encode(tok_v3, _word)},
    ] + [{"tokenizer": f"{t.name} ({t.used_by})", "pieces": real.tokenize(t, _word)["tokens"]} for t in real_tokenizers]
    mo.ui.table(
        [{"tokenizer": r["tokenizer"], "tokens": len(r["pieces"]), "pieces": " | ".join(visible(p) for p in r["pieces"])} for r in _rows],
        selection=None,
        page_size=12,
    )
    return


@app.cell
def _(real_tokenizers):
    pick_tokenizer = mo.ui.dropdown(options={t.name: t for t in real_tokenizers}, value="o200k_base", label="Tokenizer")
    pick_tokenizer
    return (pick_tokenizer,)


@app.cell
def _(pick_tokenizer):
    # Every probe word, in every spelling, through the tokenizer you picked.
    _rows = []
    for _word in config.PROBE_WORDS:
        for _variant in real.variants(_word):
            _split = real.tokenize(pick_tokenizer.value, _variant)
            _rows.append({"text": visible(_variant), "tokens": " | ".join(visible(p) for p in _split["tokens"]), "ids": str(_split["ids"])})
    mo.ui.table(_rows, selection=None, page_size=12)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The letter-counting test

    To test whether token blindness is the whole story, we built 104 questions (the 8 famous ones plus 96 words sampled from the novels, balanced across true counts of 0, 1, 2, and 3 or more) and asked five small open models, from 360M to 4B parameters, in four ways. We tried a `plain` question, a `spaced` version where every letter is its own token (`s t r a w b e r r y`), a `spell_then_count` prompt with one worked example, and Qwen3's built-in `thinking` mode, plus a separate `spelling` check. Asked plainly, every one of these models scored 35 out of 104 or worse, and spacing the letters out helped some models a lot while hurting OLMo 2 1B. The twist shows up when you pick `Qwen/Qwen3-4B` in the dropdown below. It spelled 102 of the 104 words right, yet still miscounted 90 of those 102 when asked plainly (it spelled strawberry cleanly and guessed 7 r's), which tells us counting repeated letters in one go is a separate problem from knowing the spelling. Keep the scope in mind, since these numbers are only for five small models using greedy decoding and one wording per condition, and they don't say anything on their own about big chatbots.
    """)
    return


@app.cell
def _():
    _path = config.RESULTS_DIR / "count_test.json"
    count_results = json.loads(_path.read_text()) if _path.exists() else None
    mo.stop(count_results is None, mo.callout(mo.md("No `results/count_test.json` yet. Run `uv run python run_all.py` first."), kind="warn"))
    mo.ui.table(
        [
            {"model": m} | {c: f"{v['correct']}/{v['n']} ({v['accuracy']:.0%})" for c, v in r["summary"]["conditions"].items()} | {"spelling": f"{r['summary']['spelling_accuracy']:.0%}"}
            for m, r in count_results["models"].items()
        ],
        selection=None,
    )
    return (count_results,)


@app.cell
def _(count_results):
    pick_model = mo.ui.dropdown(options=list(count_results["models"]), value=list(count_results["models"])[0], label="Model")
    pick_model
    return (pick_model,)


@app.cell
def _(count_results, pick_model):
    # Accuracy split by the true count (0, 1, 2, 3+) and by word group, for the model you picked.
    _summary = count_results["models"][pick_model.value]["summary"]
    _rows = []
    for _condition, _score in _summary["conditions"].items():
        _row = {"condition": _condition}
        _row |= {f"count {g}": f"{s['accuracy']:.0%}" for g, s in _score["by_true_count"].items()}
        _row |= {g: f"{s['accuracy']:.0%}" for g, s in _score["by_group"].items()}
        _row["unparsed"] = _score["unparsed"]
        _rows.append(_row)
    _twist = _summary.get("spells_right_but_miscounts_plain", {})
    _steps = _summary.get("spell_then_count_steps", {})
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                f"Spelling probe: **{_summary['spelling_accuracy']:.0%}** spelled right. "
                f"Spells right but miscounts (plain): **{_twist.get('words', 0)}** words.  \n"
                f"In spell_then_count it wrote the word's letters out correctly in **{_steps.get('wrote_the_word_correctly', 0):.0%}** of replies."
            ),
            mo.md("Its answers for strawberry:"),
            mo.ui.table([{"condition": c, "reply": r} for c, r in _summary["strawberry"].items()], selection=None, wrapped_columns=["reply"]),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Try it yourself

    Click the button below to run our smallest Qwen3 model (`Qwen/Qwen3-0.6B`) live on the 8 famous words, in both the `plain` and `spell_then_count` conditions. You'll need the `models` dependency group installed (`uv sync --group models`) for PyTorch. It uses a GPU if you have one and falls back to CPU otherwise, which takes about a minute.
    """)
    return


@app.cell
def _():
    run_live = mo.ui.run_button(label="Ask Qwen3-0.6B the famous 8 now")
    run_live
    return (run_live,)


@app.cell
def _(run_live):
    mo.stop(not run_live.value, mo.md("_Click the button to run the model._"))
    import importlib.util

    mo.stop(importlib.util.find_spec("torch") is None, mo.md("This needs torch: run `uv sync --group models` first."))
    from dumbtok import count_test, letters

    _items = [letters.Item(w, l, w.count(l), "famous") for w, l in config.FAMOUS_WORDS]
    _model = "Qwen/Qwen3-0.6B"
    _result = count_test.run_model(_model, _items, ["plain", "spell_then_count"], config.HF_REVISIONS[_model], config.DEVICE)
    mo.ui.table(
        [{k: r[k] for k in ("condition", "word", "letter", "truth", "reply")} | {"answer": r.get("answer"), "right": r.get("correct", r.get("spelled_ok"))} for r in _result["rows"]],
        selection=None,
        wrapped_columns=["reply"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The fixes

    Once you see why these models miss when asked plainly, two real fixes stand out. The first is to make the model go one letter at a time, out loud. In the table above, `spell_then_count` lifted Qwen3 4B from 12 to 84 out of 104, moved Qwen3 1.7B from 15 to 43, and nudged Qwen3 0.6B from 24 to 31, while hurting SmolLM2 360M (22 down to 8) and OLMo 2 1B (35 down to 19) because the smallest models couldn't follow the template. Turning on Qwen3's built-in `thinking` mode let all three sizes talk through the letters on their own, jumping to 80, 96, and 103 out of 104. The second fix is even simpler, which is to hand the job to code. One line of Python is always right, and you can test it on your word from above with the letter box below.
    """)
    return


@app.cell
def _():
    letter_box = mo.ui.text(value="r", max_length=1, label="Letter")
    letter_box
    return (letter_box,)


@app.cell
def _(letter_box, word_box):
    # The fix that never fails: count in code.
    word_box.value.strip().count(letter_box.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The scoreboard

    Here are all three tokenizers side by side on *The Yellow Wallpaper*, the book none of them saw during training. v1 keeps every letter in plain sight with a vocabulary of 128 and zero unknowns, at the cost of 31,497 tokens. v2 packs the book down to 7,831 tokens (4.02 characters per token), but needs a 54,622-entry dictionary, loses 1.35% of tokens to `<UNK>`, and hides `" strawberry"` behind one number. v3 BPE lands in the middle with 8,128 entries in its vocabulary, 8,737 tokens for the book (3.61 characters per token), and zero unknowns, splitting `" strawberry"` into `␣straw | ber | ry`.

    Turns out the dumbest tokenizer was the only one that could actually see the letters.
    """)
    return


@app.cell
def _(held_out_text, tok_v1, tok_v2, tok_v3):
    _rows = []
    for _name, _vocab, _tokens, _strawberry in [
        ("v1 alphabet", len(tok_v1.vocab), alphabet.encode(tok_v1, held_out_text), alphabet.encode(tok_v1, " strawberry")),
        ("v2 dictionary", len(tok_v2.vocab), dictionary.encode(tok_v2, held_out_text), dictionary.encode(tok_v2, " strawberry")),
        ("v3 BPE", len(tok_v3.vocab), bpe.encode(tok_v3, held_out_text), bpe.encode(tok_v3, " strawberry")),
    ]:
        _rows.append(
            {
                "tokenizer": _name,
                "vocabulary": _vocab,
                "tokens, held-out book": len(_tokens),
                "characters per token": round(len(held_out_text) / len(_tokens), 2),
                "<UNK> rate": f"{_tokens.count(UNK) / len(_tokens):.2%}",
                "␣strawberry": " | ".join(visible(p) for p in _strawberry),
                "letters visible?": all(len(p) == 1 for p in _strawberry if p.strip()),
            }
        )
    mo.ui.table(_rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
