"""Download public-domain books from Project Gutenberg, and clean them.

Every book on Gutenberg starts with a header and ends with a footer of legal
text. We keep only what's between the "*** START OF" and "*** END OF" lines.

The books are not part of this repo: they are downloaded into data/ the first
time you run anything. The SHA-256 of each downloaded file is recorded in
results/corpus.json, so you can check you trained on exactly the same text.
"""

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Book:
    id: int
    title: str
    text: str  # the book itself, without the Gutenberg header and footer
    path: Path  # where the downloaded file lives
    sha256: str  # fingerprint of the downloaded file


def download(book_id: int, data_dir: Path, url_template: str) -> Path:
    """Download one book into data_dir, unless it's already there."""
    path = data_dir / f"pg{book_id}.txt"
    if path.exists():
        return path
    url = url_template.format(id=book_id)
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "dumbest-tokenizer/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    # Don't trust the status code: a "site under maintenance" page also says 200 OK.
    # Check the content instead.
    if not looks_like_a_gutenberg_book(raw):
        raise RuntimeError(f"{url} did not return a Project Gutenberg book. First bytes: {raw[:200]!r}")
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def looks_like_a_gutenberg_book(raw: bytes) -> bool:
    """True if the bytes look like a Gutenberg plain-text book, not an error page."""
    head = raw[:5000].decode("utf-8", errors="replace")
    return "Project Gutenberg" in head and b"*** START OF" in raw and b"*** END OF" in raw


def strip_header_and_footer(text: str) -> str:
    """Keep only the book: the lines between "*** START OF ..." and "*** END OF ..."."""
    start = text.index("*** START OF")
    start = text.index("\n", start) + 1  # skip the rest of the START line
    end = text.index("*** END OF", start)
    return text[start:end].strip()


def find_title(text: str) -> str:
    """Read the "Title: ..." line from the Gutenberg header."""
    match = re.search(r"^Title:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else "?"


def load_book(book_id: int, data_dir: Path, url_template: str) -> Book:
    """Download (if needed) and clean one book."""
    path = download(book_id, data_dir, url_template)
    raw = path.read_bytes()
    # utf-8-sig drops the invisible "byte order mark" some files start with.
    # Windows line endings (\r\n) become plain \n.
    full_text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    return Book(
        id=book_id,
        title=find_title(full_text),
        text=strip_header_and_footer(full_text),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_books(book_ids: list[int], data_dir: Path, url_template: str) -> list[Book]:
    return [load_book(book_id, data_dir, url_template) for book_id in book_ids]


def join_books(books: list[Book]) -> str:
    """All books as one long text, separated by blank lines."""
    return "\n\n".join(book.text for book in books)


def describe(books: list[Book]) -> list[dict]:
    """A small JSON-friendly summary of each book, for results/corpus.json."""
    return [
        {
            "id": book.id,
            "title": book.title,
            "file": book.path.name,
            "sha256": book.sha256,
            "characters": len(book.text),
            "words": len(book.text.split()),
        }
        for book in books
    ]
