#!/usr/bin/env python3
"""
Step 1 — Extract text from PDFs and EPUBs.

Uses pdfplumber (best for scientific papers) with a pypdf fallback.
EPUB support via ebooklib + BeautifulSoup.

Usage:
    python extract_text.py --input-dir ./pdfs
    python extract_text.py --input-dir ./pdfs ./ebooks --output-dir ./extracted_text
"""

import re
import argparse
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_pdfplumber(path):
    """Primary PDF extractor — handles tables and scientific layouts well."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)


def extract_pypdf(path):
    """Fallback PDF extractor."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


def extract_pdf(path):
    """Try pdfplumber first, fall back to pypdf."""
    try:
        return extract_pdfplumber(path)
    except Exception:
        return extract_pypdf(path)


def extract_epub(path):
    """Extract text from an EPUB file."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            if text.strip():
                chapters.append(text.strip())
    return "\n\n".join(chapters)


def extract(path):
    """Route to the correct extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix in (".pdf",):
        return extract_pdf(path)
    elif suffix in (".epub",):
        return extract_epub(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text):
    """Remove common PDF artefacts."""
    # Rejoin words broken by hyphenation at line ends
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove standalone page numbers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Strip leading / trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUPPORTED = {".pdf", ".epub"}


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs / EPUBs")
    parser.add_argument("--input-dir", nargs="+", default=["./pdfs"],
                        help="Directories to scan (default: ./pdfs)")
    parser.add_argument("--output-dir", default="./extracted_text",
                        help="Where to write .txt files (default: ./extracted_text)")
    parser.add_argument("--min-length", type=int, default=200,
                        help="Drop files shorter than this many characters (default: 200)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- discover files ---
    all_files = []
    for d in args.input_dir:
        p = Path(d)
        if not p.exists():
            print(f"⚠️  Directory not found: {d}")
            continue
        found = [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED]
        all_files.extend(found)
        print(f"📁 {d}: {len(found)} file(s)")

    print(f"\n📚 Total files to extract: {len(all_files)}\n")

    # --- extract ---
    success = failed = skipped = 0
    total_chars = 0

    for src in tqdm(all_files, desc="Extracting"):
        try:
            raw = extract(src)
            text = clean_text(raw)

            if len(text) < args.min_length:
                skipped += 1
                continue

            # collision-safe output name
            out = output_dir / f"{src.stem}.txt"
            i = 1
            while out.exists():
                out = output_dir / f"{src.stem}_{i}.txt"
                i += 1

            out.write_text(text, encoding="utf-8")
            total_chars += len(text)
            success += 1

        except Exception as e:
            print(f"\n❌ {src.name}: {str(e)[:80]}")
            failed += 1

    # --- summary ---
    print(f"\n{'=' * 50}")
    print(f"  ✅ Extracted : {success}")
    print(f"  ❌ Failed    : {failed}")
    print(f"  ⏭️  Skipped   : {skipped}  (< {args.min_length} chars)")
    print(f"  📊 Total text: {total_chars / 1_000_000:.2f} MB  →  {output_dir}")
    print(f"{'=' * 50}")
    print(f"\n  Next step:  python prepare_dataset.py --input-dir {output_dir}")


if __name__ == "__main__":
    main()
