"""dumbtok: three home-made tokenizers, the real ones, and a letter-counting test.

One idea per file:

    corpus.py      download the books
    words.py       cut text into words (shared by v2 and v3)
    alphabet.py    v1: every character is a token
    dictionary.py  v2: every word is a token
    bpe.py         v3: byte-pair encoding, the compromise real models use
    real.py        what real tokenizers (GPT-4o, Qwen3, ...) do to the same words
    letters.py     build the letter-counting test set
    count_test.py  ask small open models to count letters
    hosted_test.py the same test against a hosted model (any OpenAI-style API)
    results.py     write results/*.json with a header saying how they were made

Every tokenizer has the same two main functions, so you can swap them:

    tok = train(text, ...)          # learn a vocabulary from text
    pieces = encode(tok, "hello")   # cut text into token strings

and also `ids(tok, pieces)` (token strings -> numbers) and `decode(tok, ids)`.
"""

# The token we use for "I have never seen this before".
UNK = "<UNK>"
