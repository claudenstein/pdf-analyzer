#!/usr/bin/env python3
"""
Step 1 — Extract text from PDFs and EPUBs.

Uses Docling (ML-based layout analysis, best for scientific papers)
with a pdfplumber fallback for any files Docling cannot handle.
EPUB support via ebooklib + BeautifulSoup.

Usage:
    python extract_text.py --input-dir ./pdfs
    python extract_text.py --input-dir ./pdfs ./ebooks --output-dir ./extracted_text
"""

import re
import argparse
from pathlib import Path
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def build_converter(device: str = "cuda") -> DocumentConverter:
    """Create a DocumentConverter with explicit device placement.

    device   "cuda"  → RTX 3070 (or any NVIDIA GPU)
             "cpu"   → no GPU required
             "auto"  → Docling picks the best available
    """
    print(f"🤖 Initialising Docling on {device} …  (may download models on first run)")

    pdf_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            # cuda_use_flash_attention2=True,   # extra speed on Ampere+
            #                                   # needs: pip install flash-attn
        ),
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )


def extract_docling(path, converter: DocumentConverter):
    """Primary PDF extractor — ML layout analysis, structured Markdown output."""
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def extract_pdfplumber(path):
    """Fallback PDF extractor — rule-based, no ML dependency at runtime."""
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)


def extract_pdf(path, converter: DocumentConverter):
    """Try Docling first, fall back to pdfplumber."""
    try:
        return extract_docling(path, converter)
    except Exception:
        return extract_pdfplumber(path)


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


def extract(path, converter: DocumentConverter):
    """Route to the correct extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, converter)
    elif suffix == ".epub":
        return extract_epub(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text):
    """Light cleanup pass.

    Docling already handles hyphenation, page numbers, and layout ordering,
    so these regexes mostly matter for the pdfplumber fallback path.
    Running them on Docling output is harmless — they simply won't match.
    """
    # Rejoin words broken by hyphenation at line ends (pdfplumber fallback)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove standalone page numbers (pdfplumber fallback)
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"],
                        help="Inference device for Docling layout models (default: cuda)")
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

    # --- build Docling converter once (model load is expensive) ---
    converter = build_converter(args.device)

    # --- extract ---
    success = failed = skipped = 0
    total_chars = 0

    for src in tqdm(all_files, desc="Extracting"):
        try:
            raw = extract(src, converter)
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
