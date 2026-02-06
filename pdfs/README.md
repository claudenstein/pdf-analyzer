# PDFs Directory

Place your PDF files in this directory for analysis.

## Usage

1. Copy or move your PDF files here:
   ```bash
   cp /path/to/your/documents/*.pdf ./pdfs/
   ```

2. Run the analyzer:
   ```bash
   python pdf_analyzer_pro.py
   ```

## Organization

You can organize PDFs in subdirectories - the scanner will find them recursively:

```
pdfs/
├── research/
│   ├── paper1.pdf
│   └── paper2.pdf
├── reports/
│   └── annual_report.pdf
└── misc/
    └── document.pdf
```

## Supported Files

- `.pdf` and `.PDF` extensions
- Any size (timeout protection included)
- Multiple pages
- Scanned PDFs (OCR not included, text-based only)

The analyzer will automatically detect and skip duplicate files based on content hash.
