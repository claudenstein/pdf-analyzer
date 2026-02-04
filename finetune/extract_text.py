#!/usr/bin/env python3
"""
Step 1 — Extract text from PDFs and EPUBs.

Uses Docling (ML-based layout analysis, best for scientific papers)
with a pdfplumber fallback for any files Docling cannot handle.
EPUB support via ebooklib + BeautifulSoup.

Features:
    • Logging   — every attempt is written to extract.log before the
                  extraction runs, so a segfault / crash still leaves a trail.
    • Resume    — processed files are tracked in .resume.json.  Restart the
                  script and it picks up exactly where it left off.
    • Crash recovery — if a file caused a segfault on the previous run it is
                  automatically retried with the pdfplumber fallback.

Usage:
    python extract_text.py --input-dir ./pdfs
    python extract_text.py --input-dir ./pdfs --device cuda
    python extract_text.py --input-dir ./pdfs --no-resume          # start fresh
"""

import re
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(log_path: Path):
    """Console + file logging.  File is created inside output_dir."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path)),
        ],
    )


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

class ResumeState:
    """Persist extraction progress to disk after every file.

    Layout of .resume.json:
        {
            "done":             { "<src path>": "<output path>", … },
            "failed":           { "<src path>": "<reason>",       … },
            "last_attempted":   "<src path>" | null
        }

    ``last_attempted`` is written *before* extraction starts.  If the
    process crashes (e.g. segfault) it is never cleared, so on the next
    run we know exactly which file was the culprit.
    """

    def __init__(self, path: Path):
        self.path = path
        self.state = self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                return self._empty()
        return self._empty()

    @staticmethod
    def _empty():
        return {"done": {}, "failed": {}, "last_attempted": None}

    def _save(self):
        self.path.write_text(json.dumps(self.state, indent=2))

    # -- transitions --------------------------------------------------------
    def mark_starting(self, src: Path):
        """Written to disk *before* we touch the file."""
        self.state["last_attempted"] = str(src)
        self._save()

    def mark_done(self, src: Path, out: Path):
        self.state["done"][str(src)] = str(out)
        self.state["last_attempted"] = None
        self._save()

    def mark_failed(self, src: Path, reason: str):
        self.state["failed"][str(src)] = reason
        self.state["last_attempted"] = None
        self._save()

    # -- queries ------------------------------------------------------------
    def is_done(self, src: Path) -> bool:
        return str(src) in self.state["done"]

    def is_failed(self, src: Path) -> bool:
        return str(src) in self.state["failed"]

    def crashed_file(self) -> str | None:
        """Return the path that was in-flight when the last run died."""
        la = self.state.get("last_attempted")
        if la and la not in self.state["done"] and la not in self.state["failed"]:
            return la
        return None

    @property
    def done_count(self) -> int:
        return len(self.state["done"])

    @property
    def failed_count(self) -> int:
        return len(self.state["failed"])


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def build_converter(device: str = "cuda") -> DocumentConverter:
    """Create a DocumentConverter with explicit device placement.

    device   "cuda"  → RTX 3070 (or any NVIDIA GPU)
             "cpu"   → no GPU required
             "auto"  → Docling picks the best available
    """
    logger.info(f"Initialising Docling on {device} …")

    # --- GPU sanity check (only when CUDA is requested) -----------------------
    if device in ("cuda", "auto"):
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name  = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1_073_741_824
                logger.info(f"GPU detected : {gpu_name} ({total_mem:.1f} GB VRAM)")
            else:
                logger.warning("torch.cuda.is_available() → False  "
                               "— Docling will fall back to CPU")
        except ImportError:
            logger.warning("torch not installed – cannot verify GPU availability")

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
        logger.warning(f"{path.name}: Docling raised — falling back to pdfplumber")
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
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED = {".pdf", ".epub"}


def _collision_safe(output_dir: Path, stem: str) -> Path:
    """Return a path that does not collide with an existing file."""
    out = output_dir / f"{stem}.txt"
    i = 1
    while out.exists():
        out = output_dir / f"{stem}_{i}.txt"
        i += 1
    return out


def _try_pdfplumber_recovery(src: Path, output_dir: Path, min_length: int,
                             state: ResumeState):
    """Last-resort fallback for a file that caused a crash."""
    logger.warning(f"Attempting pdfplumber recovery for {src.name} …")
    try:
        text = clean_text(extract_pdfplumber(src))
        if len(text) < min_length:
            state.mark_failed(src, "crash + pdfplumber fallback too short")
            logger.warning(f"{src.name}: recovered text too short, skipped")
            return
        out = _collision_safe(output_dir, f"{src.stem}_recovered")
        out.write_text(text, encoding="utf-8")
        state.mark_done(src, out)
        logger.info(f"{src.name}: recovered via pdfplumber → {out.name}")
    except Exception as e:
        state.mark_failed(src, f"crash + pdfplumber failed: {e}")
        logger.error(f"{src.name}: pdfplumber recovery also failed — {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore previous progress and start from scratch")
    args = parser.parse_args()

    # --- directories & logging ---------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "extract.log")

    # --- resume state ------------------------------------------------------
    resume_path = output_dir / ".resume.json"
    if args.no_resume and resume_path.exists():
        logger.info("--no-resume: clearing previous state")
        resume_path.unlink()
    state = ResumeState(resume_path)

    # --- crash recovery ----------------------------------------------------
    crashed = state.crashed_file()
    if crashed:
        crashed_path = Path(crashed)
        logger.warning(f"Previous run crashed on {crashed_path.name}")
        _try_pdfplumber_recovery(crashed_path, output_dir, args.min_length, state)

    # --- discover files ----------------------------------------------------
    all_files = []
    for d in args.input_dir:
        p = Path(d)
        if not p.exists():
            logger.warning(f"Directory not found: {d}")
            continue
        found = [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED]
        all_files.extend(found)
        logger.info(f"{d}: {len(found)} file(s)")

    # --- filter already-processed ------------------------------------------
    pending = []
    for src in all_files:
        if state.is_done(src):
            continue
        if state.is_failed(src):
            continue
        pending.append(src)

    logger.info(
        f"Total: {len(all_files)} | "
        f"done: {state.done_count} | "
        f"failed: {state.failed_count} | "
        f"pending: {len(pending)}"
    )

    if not pending:
        logger.info("Nothing to do — all files already processed.")
        print("\n  ✅ All files already extracted.  Use --no-resume to re-run.")
        return

    # --- build converter once ----------------------------------------------
    converter = build_converter(args.device)

    # --- extract -----------------------------------------------------------
    total_chars = 0

    for src in tqdm(pending, desc="Extracting"):
        state.mark_starting(src)          # ← written BEFORE we touch the file
        logger.info(f"Starting: {src}")   #    a crash here still leaves a trail

        try:
            raw  = extract(src, converter)
            text = clean_text(raw)

            if len(text) < args.min_length:
                state.mark_failed(src, f"too short ({len(text)} chars)")
                logger.warning(f"{src.name}: only {len(text)} chars, skipped")
                continue

            out = _collision_safe(output_dir, src.stem)
            out.write_text(text, encoding="utf-8")
            total_chars += len(text)

            state.mark_done(src, out)
            logger.info(f"OK  {src.name}  →  {out.name}  ({len(text):,} chars)")

        except Exception as e:
            state.mark_failed(src, str(e))
            logger.error(f"FAIL {src.name}: {str(e)[:120]}")

    # --- summary -----------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  ✅ Done   : {state.done_count}")
    print(f"  ❌ Failed : {state.failed_count}")
    print(f"  📊 Text   : {total_chars / 1_000_000:.2f} MB  →  {output_dir}")
    print(f"  📝 Log    : {output_dir / 'extract.log'}")
    print(f"  💾 State  : {resume_path}")
    print(f"{'=' * 55}")
    print(f"\n  Next step:  python prepare_dataset.py --input-dir {output_dir}")


if __name__ == "__main__":
    main()
