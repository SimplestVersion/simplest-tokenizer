"""What real tokenizers do to the same words.

We push the probe words and the example prompt through tokenizers from real
models:

* OpenAI's (GPT-4o, GPT-4, GPT-2), with the `tiktoken` library.
* Open models on Hugging Face (Qwen3, OLMo-2, Mistral, SmolLM2, Phi-4-mini),
  with the `tokenizers` library. We download only each model's tokenizer.json
  (a few MB), not the model.

For each one we record exactly which files were used (commit and SHA-256), so
the numbers can be checked later even if the tokenizers change.
"""

import hashlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass, field

import tiktoken
import tiktoken.load
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

# The files behind each tiktoken encoding (from tiktoken_ext/openai_public.py).
TIKTOKEN_FILES = {
    "o200k_base": ["https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"],
    "cl100k_base": ["https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"],
    "gpt2": [
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    ],
}

# Licences for the tiktoken encodings. There's no model card, so these were
# read by hand from the licence files at these URLs (2026-11-21).
TIKTOKEN_LICENCES = {
    "o200k_base": {
        "licence": "MIT",
        "note": "Encoding file distributed by OpenAI with the tiktoken library, which is MIT-licensed.",
        "source_url": "https://github.com/openai/tiktoken/blob/main/LICENSE",
    },
    "cl100k_base": {
        "licence": "MIT",
        "note": "Encoding file distributed by OpenAI with the tiktoken library, which is MIT-licensed.",
        "source_url": "https://github.com/openai/tiktoken/blob/main/LICENSE",
    },
    "gpt2": {
        "licence": "Modified MIT",
        "note": "GPT-2's vocabulary comes from the openai/gpt-2 release (\"Modified MIT License\"); tiktoken itself is MIT.",
        "source_url": "https://github.com/openai/gpt-2/blob/master/LICENSE",
    },
}


@dataclass
class RealTokenizer:
    name: str  # "o200k_base" or "Qwen/Qwen3-0.6B"
    used_by: str  # which model(s) use it, for people
    encode_ids: Callable[[str], list[int]]  # text -> token ids
    raw_token: Callable[[int], str]  # token id -> the tokenizer's own spelling of it
    vocab_size: int
    info: dict = field(default_factory=dict)  # library version, commit, file hashes, licence


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_tiktoken(name: str, used_by: str) -> RealTokenizer:
    """Load one of OpenAI's encodings."""
    enc = tiktoken.get_encoding(name)
    files = [{"url": url, "sha256": sha256_of_bytes(tiktoken.load.read_file_cached(url))} for url in TIKTOKEN_FILES[name]]
    return RealTokenizer(
        name=name,
        used_by=used_by,
        encode_ids=enc.encode_ordinary,  # "ordinary": treat "<|endoftext|>" etc. as plain text
        raw_token=lambda i: enc.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace"),
        vocab_size=enc.n_vocab,
        info={
            "library": f"tiktoken {importlib.metadata.version('tiktoken')}",
            "files": files,
            "licence": TIKTOKEN_LICENCES[name],
        },
    )


def load_hugging_face(repo: str, revision: str = "main") -> RealTokenizer:
    """Download one model's tokenizer.json from Hugging Face and load it."""
    model_info = HfApi().model_info(repo, revision=revision)
    path = hf_hub_download(repo, "tokenizer.json", revision=model_info.sha)
    tok = Tokenizer.from_file(path)
    with open(path, "rb") as f:
        tokenizer_json_sha256 = sha256_of_bytes(f.read())
    return RealTokenizer(
        name=repo,
        used_by=repo.split("/")[1],
        encode_ids=lambda text: tok.encode(text, add_special_tokens=False).ids,
        raw_token=tok.id_to_token,
        vocab_size=tok.get_vocab_size(with_added_tokens=True),
        info={
            "library": f"tokenizers {importlib.metadata.version('tokenizers')}",
            "hf_commit": model_info.sha,
            "tokenizer_json_sha256": tokenizer_json_sha256,
            "licence": hugging_face_licence(model_info),
        },
    )


def hugging_face_licence(model_info) -> dict:
    """Read the licence from the model card (the `license:` line at its top)."""
    repo, commit = model_info.id, model_info.sha
    card = model_info.card_data or {}
    files = [sibling.rfilename for sibling in model_info.siblings or []]
    licence_files = [f for f in files if f.upper().startswith("LICEN")]
    return {
        "licence": card.get("license"),
        "source_url": f"https://huggingface.co/{repo}/blob/{commit}/README.md",
        "licence_file_urls": [f"https://huggingface.co/{repo}/blob/{commit}/{f}" for f in licence_files],
    }


def readable(raw_token: str) -> str:
    """Show a token the way a person would read it.

    Tokenizers spell spaces in odd ways inside their vocabularies: byte-level
    ones (GPT-2 style) write a space as "Ġ" and a newline as "Ċ"; SentencePiece
    ones (Mistral) write a space as "▁". We turn those back into real spaces.
    """
    return raw_token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")


def tokenize(tok: RealTokenizer, text: str) -> dict:
    """Split text with a real tokenizer; return the pieces and their ids."""
    token_ids = tok.encode_ids(text)
    raw = [tok.raw_token(i) for i in token_ids]
    return {"text": text, "tokens": [readable(r) for r in raw], "ids": token_ids, "raw_tokens": raw}


def variants(word: str) -> list[str]:
    """The ways we write each probe word: lower case, Capitalised, ALL CAPS,
    each with and without a leading space."""
    forms = [word.lower(), word.capitalize(), word.upper()]
    out = []
    for form in forms:
        for text in (form, " " + form):
            if text not in out:
                out.append(text)
    return out


def load_all(tiktoken_encodings: dict[str, str], hf_repos: list[str], revisions: dict[str, str]) -> list[RealTokenizer]:
    tokenizers = [load_tiktoken(name, used_by) for name, used_by in tiktoken_encodings.items()]
    tokenizers += [load_hugging_face(repo, revisions.get(repo, "main")) for repo in hf_repos]
    return tokenizers


def run(tokenizers: list[RealTokenizer], probe_words: list[str], prompt: str) -> list[dict]:
    """Every probe word (in every variant) and the prompt, through every tokenizer."""
    results = []
    for tok in tokenizers:
        results.append(
            {
                "name": tok.name,
                "used_by": tok.used_by,
                "vocab_size": tok.vocab_size,
                **tok.info,
                "prompt": tokenize(tok, prompt),
                "probes": {word: [tokenize(tok, text) for text in variants(word)] for word in probe_words},
            }
        )
    return results
