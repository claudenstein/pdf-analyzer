# PDF Analyzer Pro

A powerful PDF analysis system using RAG (Retrieval-Augmented Generation) with Mistral 7B LLM. This tool allows you to analyze multiple PDF documents, build a searchable vector database, and ask questions about the content using natural language.

## Overview

PDF Analyzer Pro uses advanced AI techniques to help you extract insights from large collections of PDF documents. It builds a semantic search index using vector embeddings and leverages local LLMs to provide accurate, context-aware answers with source citations.

## Features

- **Multiple PDF Directory Support**: Scan multiple directories recursively for comprehensive document coverage
- **Hash-based Duplicate Detection**: Automatically detect and skip duplicate PDFs using MD5 hashing
- **Incremental Updates**: Add new PDFs without rebuilding the entire database
- **Configurable Retrieval**: Fine-tune chunk sizes and retrieval parameters for optimal results
- **Interactive Q&A**: Ask questions about your PDF collection in natural language
- **GPU Acceleration**: Uses CUDA for 10-100x faster embeddings generation
- **Timeout Protection**: Handles problematic PDFs with configurable timeouts
- **Progress Tracking**: Visual progress bars and detailed statistics
- **Source Attribution**: Get answers with specific page citations from source documents

## Requirements

- **Python 3.8+**
- **CUDA-capable GPU** (optional but recommended for performance)
- **Ollama** with Mistral model installed
- **Disk Space**: Sufficient space for vector database (approx 2-3x total PDF size)
- **RAM**: Minimum 8GB, 16GB+ recommended for large collections

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/claudenstein/pdf-analyzer.git
cd pdf-analyzer
```

### 2. Install Python dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install and setup Ollama

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Mistral model
ollama pull mistral

# Verify installation
ollama list
```

For Windows, download from [ollama.com](https://ollama.com)

## Quick Start

### Basic Usage

```bash
# Analyze PDFs in the default ./pdfs directory
python pdf_analyzer_pro.py

# Specify a custom PDF directory
python pdf_analyzer_pro.py --pdf-dir /path/to/your/pdfs

# Multiple directories
python pdf_analyzer_pro.py --pdf-dir ./pdfs /home/docs /research/papers
```

### First Run Workflow

1. **Create a `pdfs` directory and add your PDF files:**
   ```bash
   mkdir pdfs
   cp /path/to/your/*.pdf pdfs/
   ```

2. **Run the analyzer:**
   ```bash
   python pdf_analyzer_pro.py
   ```

3. **The system will:**
   - Scan for PDFs recursively
   - Calculate hashes for duplicate detection
   - Extract text from each PDF
   - Create vector embeddings
   - Build a ChromaDB vector database
   - Launch interactive Q&A mode

4. **Ask questions:**
   ```
   🔍 Your question: What are the main topics discussed?
   🔍 Your question: Summarize the key findings
   🔍 Your question: quit
   ```

## Command-Line Flags

### Complete Flag Reference

```bash
python pdf_analyzer_pro.py [OPTIONS]
```

#### `--pdf-dir DIR [DIR ...]`
**Description:** Specifies one or more directories to scan for PDF files. The scanner will recursively search all subdirectories.

**Default:** `./pdfs`

**Type:** String (multiple values accepted)

**Examples:**
```bash
# Single directory
python pdf_analyzer_pro.py --pdf-dir ./documents

# Multiple directories
python pdf_analyzer_pro.py --pdf-dir ./research ./reports ./papers

# Absolute paths
python pdf_analyzer_pro.py --pdf-dir /home/user/Documents /mnt/storage/archives
```

**Use Cases:**
- Organizing PDFs by topic across different directories
- Scanning multiple mounted drives or network shares
- Processing different document collections separately

---

#### `--db-dir DIR`
**Description:** Specifies where to store the ChromaDB vector database and metadata files.

**Default:** `./chroma_db`

**Type:** String

**Examples:**
```bash
# Custom database location
python pdf_analyzer_pro.py --db-dir /mnt/ssd/vectordb

# Project-specific database
python pdf_analyzer_pro.py --db-dir ./research_db
```

**Use Cases:**
- Storing database on faster SSD for better performance
- Maintaining separate databases for different projects
- Using network storage for shared team access

**Notes:**
- Database directory will be created if it doesn't exist
- Contains `processed_files.json` for tracking indexed PDFs

---

#### `--top-k N`
**Description:** Number of text chunks to retrieve from the vector database for each query. Higher values provide more context but may introduce noise.

**Default:** `5`

**Type:** Integer

**Range:** `1-50` (recommended: 3-10)

**Examples:**
```bash
# Retrieve more context for complex questions
python pdf_analyzer_pro.py --top-k 10

# Faster, more focused answers
python pdf_analyzer_pro.py --top-k 3
```

**Use Cases:**
- **Low values (3-5)**: Specific fact-finding, faster responses
- **Medium values (5-10)**: Balanced context for general questions
- **High values (10-20)**: Complex analysis requiring broad context

**Can also be set per-query in interactive mode:**
```
🔍 Your question: top_k:10 What are all the recommendations?
```

---

#### `--chunk-size N`
**Description:** Size of text chunks (in characters) for splitting documents. Affects granularity of retrieval and context window.

**Default:** `1000`

**Type:** Integer

**Range:** `200-2000` (recommended: 500-1500)

**Examples:**
```bash
# Smaller chunks for precise retrieval
python pdf_analyzer_pro.py --chunk-size 500

# Larger chunks for more context per chunk
python pdf_analyzer_pro.py --chunk-size 1500
```

**Use Cases:**
- **Small chunks (500-800)**: Technical documents, precise citations
- **Medium chunks (1000-1200)**: General documents, balanced approach
- **Large chunks (1500-2000)**: Narrative documents, broad context

**Trade-offs:**
- Smaller chunks: More precise matching, less context per chunk
- Larger chunks: More context per retrieval, potentially less precise

---

#### `--chunk-overlap N`
**Description:** Number of characters to overlap between consecutive chunks. Helps maintain context across chunk boundaries.

**Default:** `200`

**Type:** Integer

**Range:** `0-500` (recommended: 10-20% of chunk size)

**Examples:**
```bash
# More overlap for better context continuity
python pdf_analyzer_pro.py --chunk-size 1000 --chunk-overlap 300

# Less overlap for distinct chunks
python pdf_analyzer_pro.py --chunk-size 1000 --chunk-overlap 100
```

**Use Cases:**
- Higher overlap: Documents with long sentences or paragraphs
- Lower overlap: Highly structured documents, saving storage space

**Recommended ratios:**
- 10-15%: Minimum overlap for context preservation
- 20-25%: Standard overlap for most use cases
- 30%+: Maximum overlap for critical context continuity

---

#### `--model MODEL`
**Description:** Specifies which Ollama LLM model to use for generating answers.

**Default:** `mistral`

**Type:** String

**Examples:**
```bash
# Use Llama 2
python pdf_analyzer_pro.py --model llama2

# Use Llama 3
python pdf_analyzer_pro.py --model llama3

# Use smaller/faster model
python pdf_analyzer_pro.py --model tinyllama

# Use larger model for better quality
python pdf_analyzer_pro.py --model mixtral
```

**Available Models** (must be pulled first with `ollama pull <model>`):
- `mistral`: Balanced performance and quality (default)
- `llama2`: Alternative general-purpose model
- `llama3`: Latest Llama version, improved reasoning
- `mixtral`: Larger model, better quality, slower
- `tinyllama`: Fastest, lower quality
- `codellama`: Optimized for code-related PDFs

**Use Cases:**
- Technical documents: `codellama` or `mixtral`
- Quick answers: `tinyllama` or `mistral`
- Best quality: `mixtral` or `llama3`
- Balanced: `mistral` (default)

---

#### `--temperature FLOAT`
**Description:** Controls randomness in LLM responses. Lower values = more deterministic, higher values = more creative.

**Default:** `0.1`

**Type:** Float

**Range:** `0.0-1.0`

**Examples:**
```bash
# Completely deterministic (same answer every time)
python pdf_analyzer_pro.py --temperature 0.0

# Slightly more varied responses
python pdf_analyzer_pro.py --temperature 0.3

# More creative/diverse answers
python pdf_analyzer_pro.py --temperature 0.7
```

**Use Cases:**
- **0.0-0.2**: Factual extraction, consistent answers, legal/medical docs
- **0.2-0.5**: Balanced creativity and consistency
- **0.5-1.0**: Creative summaries, brainstorming, exploratory analysis

**Recommendations:**
- Keep low (0.1-0.2) for factual question-answering
- Increase (0.3-0.5) for summarization and synthesis
- Avoid high values (>0.7) for accuracy-critical applications

---

#### `--timeout SECONDS`
**Description:** Maximum time (in seconds) to wait when processing each PDF file before automatically skipping it.

**Default:** `30`

**Type:** Integer

**Range:** `10-300` (recommended: 30-60)

**Examples:**
```bash
# Longer timeout for large/complex PDFs
python pdf_analyzer_pro.py --timeout 60

# Shorter timeout to skip problematic files faster
python pdf_analyzer_pro.py --timeout 15
```

**Use Cases:**
- Large PDFs (>100 pages): 60-120 seconds
- Standard PDFs (<50 pages): 30 seconds (default)
- Quick scanning: 15-20 seconds

**Failure scenarios:**
- Corrupted PDFs
- Password-protected PDFs
- Scanned images without text layer
- PDFs with complex encoding

---

#### `--rebuild`
**Description:** Forces complete rebuild of the vector database from scratch, ignoring existing database.

**Default:** `False` (flag not set)

**Type:** Boolean flag (no value needed)

**Examples:**
```bash
# Rebuild entire database
python pdf_analyzer_pro.py --rebuild

# Rebuild with custom settings
python pdf_analyzer_pro.py --rebuild --chunk-size 1500 --chunk-overlap 300
```

**Use Cases:**
- Changing chunk size or overlap settings
- Fixing corrupted database
- Starting fresh after major PDF collection changes
- Testing different chunking strategies

**Warning:** This will delete existing database and reprocess all PDFs!

---

#### `--stats`
**Description:** Displays database statistics and exits without launching interactive mode.

**Default:** `False` (flag not set)

**Type:** Boolean flag (no value needed)

**Examples:**
```bash
# View database statistics
python pdf_analyzer_pro.py --stats
```

**Output includes:**
- Total number of indexed PDFs
- Total storage size
- PDFs grouped by directory
- Metadata information

**Use Cases:**
- Quick database overview
- Monitoring database growth
- Verifying PDF indexing status
- Auditing document collection

---

## Usage Examples

### Example 1: Basic First-Time Setup

```bash
# 1. Add PDFs to default directory
mkdir pdfs
cp ~/Documents/*.pdf pdfs/

# 2. Run with defaults
python pdf_analyzer_pro.py

# Output:
# 🔬 PDF ANALYZER PRO - RAG System with Mistral 7B
# ================================================================
# 📁 Watching directories: ./pdfs
# 💾 Database: ./chroma_db
# 🎯 Retrieval: top-5 chunks
# ✂️  Chunking: 1000 chars (overlap: 200)
# ================================================================
```

### Example 2: Research Paper Analysis

```bash
# Optimized for academic papers with precise citations
python pdf_analyzer_pro.py \
  --pdf-dir ./research_papers \
  --db-dir ./research_db \
  --chunk-size 800 \
  --chunk-overlap 150 \
  --top-k 8 \
  --model mistral \
  --temperature 0.1
```

### Example 3: Large Document Collection

```bash
# Process multiple directories with longer timeouts
python pdf_analyzer_pro.py \
  --pdf-dir ./legal ./contracts ./reports \
  --db-dir /mnt/ssd/legal_db \
  --timeout 90 \
  --chunk-size 1200
```

### Example 4: Quick Exploration Mode

```bash
# Fast processing with smaller chunks and quick timeouts
python pdf_analyzer_pro.py \
  --pdf-dir ./docs \
  --chunk-size 500 \
  --top-k 3 \
  --timeout 20 \
  --model tinyllama
```

### Example 5: Incremental Update

```bash
# First run - builds initial database
python pdf_analyzer_pro.py --pdf-dir ./pdfs

# Later - add new PDFs to ./pdfs directory
cp ~/new_documents/*.pdf ./pdfs/

# Second run - only processes new PDFs
python pdf_analyzer_pro.py --pdf-dir ./pdfs
# (System automatically detects new files and offers incremental update)
```

### Example 6: Database Maintenance

```bash
# Check what's in the database
python pdf_analyzer_pro.py --stats

# Rebuild with better settings
python pdf_analyzer_pro.py --rebuild --chunk-size 1200 --chunk-overlap 240
```

### Example 7: High-Quality Analysis

```bash
# Maximum quality for important documents
python pdf_analyzer_pro.py \
  --pdf-dir ./critical_docs \
  --model mixtral \
  --chunk-size 1500 \
  --chunk-overlap 400 \
  --top-k 10 \
  --temperature 0.05
```

## Interactive Mode Commands

Once running, you can use these commands:

```bash
# Standard question
What are the main conclusions?

# Custom retrieval amount
top_k:10 Summarize all methodology sections

# Exit commands
quit
exit
q
```

## How It Works

### Architecture Overview

1. **PDF Scanning**: Recursively scans specified directories for `.pdf` and `.PDF` files
2. **Duplicate Detection**: Calculates MD5 hash of each file to identify duplicates
3. **Text Extraction**: Uses PyPDF to extract text content from each page
4. **Document Chunking**: Splits text into overlapping chunks using RecursiveCharacterTextSplitter
5. **Embedding Generation**: Creates 384-dimensional vectors using sentence-transformers (all-MiniLM-L6-v2)
6. **Vector Storage**: Stores embeddings in ChromaDB for efficient similarity search
7. **Query Processing**:
   - Converts question to embedding
   - Finds most similar chunks via cosine similarity
   - Retrieves top-k chunks
8. **Answer Generation**: Sends context + question to Mistral LLM for answer generation
9. **Source Attribution**: Returns answer with page citations from source PDFs

### Data Flow

```
PDF Files → Text Extraction → Chunking → Embeddings → ChromaDB
                                                          ↓
User Question → Embedding → Similarity Search → Top-K Chunks
                                                          ↓
                                         Context + Question → Mistral LLM → Answer
```

## Project Structure

```
pdf-analyzer/
├── pdf_analyzer_pro.py       # Main application (842 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation (this file)
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
├── pdfs/                    # Default PDF directory
│   └── README.md            # Usage instructions
└── chroma_db/               # Vector database (auto-generated)
    ├── chroma.sqlite3       # SQLite database
    ├── *.bin                # Vector index files
    └── processed_files.json # Metadata tracking
```

## Troubleshooting

### GPU Not Detected

If you don't have a CUDA GPU, edit `pdf_analyzer_pro.py` lines with GPU references:

```python
# Change from:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

# To:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
```

### Ollama Connection Issues

Ensure Ollama service is running:

```bash
# Start Ollama server
ollama serve

# In another terminal, verify it's working
ollama list
```

### PDF Processing Failures

Common issues and solutions:

**Password-protected PDFs:**
```bash
# Unlock PDFs first using qpdf
qpdf --decrypt --password=yourpassword input.pdf output.pdf
```

**Scanned PDFs (no text layer):**
```bash
# Use OCR to add text layer (requires tesseract)
ocrmypdf input.pdf output.pdf
```

**Timeout errors:**
```bash
# Increase timeout for large files
python pdf_analyzer_pro.py --timeout 120
```

### Memory Issues

If you run out of memory:

```bash
# Reduce chunk size to decrease memory usage
python pdf_analyzer_pro.py --chunk-size 500 --top-k 3

# Process fewer PDFs at a time
# Split your PDFs into smaller batches
```

### Database Corruption

If the database becomes corrupted:

```bash
# Backup metadata
cp chroma_db/processed_files.json processed_files.backup

# Delete and rebuild
rm -rf chroma_db
python pdf_analyzer_pro.py --rebuild
```

### Poor Answer Quality

Try adjusting these parameters:

```bash
# More context
python pdf_analyzer_pro.py --top-k 10 --chunk-size 1500

# More deterministic answers
python pdf_analyzer_pro.py --temperature 0.0

# Better model
ollama pull mixtral
python pdf_analyzer_pro.py --model mixtral
```

## Performance Tips

### 1. Hardware Optimization
- **SSD Storage**: Store database on SSD for 5-10x faster retrieval
- **GPU Usage**: CUDA GPU provides 10-100x faster embedding generation
- **RAM**: More RAM allows larger batch processing

### 2. Configuration Optimization
- **Chunk Size**: Larger chunks = fewer total chunks = faster indexing
- **Top-K**: Lower values = faster query responses
- **Incremental Updates**: Much faster than rebuilding

### 3. Workflow Optimization
- **Batch Processing**: Process similar documents together
- **Organized Directories**: Group PDFs by topic for easier management
- **Regular Stats Checks**: Monitor database growth with `--stats`

### 4. Model Selection
- **Development**: Use `tinyllama` for fast iteration
- **Production**: Use `mistral` for balanced performance
- **Critical**: Use `mixtral` for best quality

## Performance Benchmarks

Approximate performance on consumer hardware (RTX 3080, 32GB RAM):

| Operation | PDFs | Pages | Time (GPU) | Time (CPU) |
|-----------|------|-------|------------|------------|
| Initial indexing | 10 | 100 | ~2 min | ~15 min |
| Initial indexing | 100 | 1000 | ~15 min | ~2 hours |
| Incremental (10 new) | 10 | 100 | ~2 min | ~15 min |
| Query response | - | - | ~3-5 sec | ~3-5 sec |
| Database load | - | - | ~1 sec | ~1 sec |

## Advanced Usage

### Custom Embedding Models

Edit `pdf_analyzer_pro.py` to use different embedding models:

```python
# Larger, more accurate model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'}
)

# Multilingual model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}
)
```

### Programmatic Usage

```python
from pathlib import Path
from pdf_analyzer_pro import PDFAnalyzerPro

# Create analyzer
analyzer = PDFAnalyzerPro(
    pdf_dirs=['./pdfs'],
    db_dir='./chroma_db',
    chunk_size=1000,
    chunk_overlap=200
)

# Load or create database
analyzer.load_and_process_pdfs(incremental=False)

# Setup LLM
analyzer.setup_llm(temperature=0.1, model='mistral')

# Query
answer = analyzer.query("What are the main findings?", top_k=5)
print(answer)
```

## FAQ

**Q: Can I use this with non-English PDFs?**
A: Yes, but you should switch to a multilingual embedding model (see Advanced Usage).

**Q: How much disk space does the database use?**
A: Typically 2-3x the total size of your PDFs.

**Q: Can I share the database with my team?**
A: Yes, the database is portable. Just share the `chroma_db` directory.

**Q: Does this work with scanned PDFs?**
A: Only if they have an OCR text layer. Use `ocrmypdf` to add one if needed.

**Q: Can I use cloud LLMs instead of Ollama?**
A: Yes, but you'll need to modify the code to use OpenAI or Anthropic APIs.

**Q: Is my data sent to external servers?**
A: No, everything runs locally (PDFs, embeddings, LLM).

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for other document formats (DOCX, TXT, HTML)
- [ ] Implement OCR for scanned PDFs
- [ ] Add web interface
- [ ] Support for cloud LLM APIs
- [ ] Multi-language support
- [ ] Export functionality for Q&A sessions
- [ ] Conversation history and context

Please submit issues and pull requests on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain** - Document processing framework ([langchain.com](https://langchain.com/))
- **Sentence Transformers** - Embedding models ([sbert.net](https://www.sbert.net/))
- **ChromaDB** - Vector database ([trychroma.com](https://www.trychroma.com/))
- **Ollama** - Local LLM inference ([ollama.com](https://ollama.com/))
- **PyPDF** - PDF parsing library
- **Mistral AI** - LLM model

## Support

For issues, questions, or contributions:
- **GitHub Issues**: [github.com/claudenstein/pdf-analyzer/issues](https://github.com/claudenstein/pdf-analyzer/issues)
- **Discussions**: [github.com/claudenstein/pdf-analyzer/discussions](https://github.com/claudenstein/pdf-analyzer/discussions)

---

**Made with ❤️ for the open-source community**
