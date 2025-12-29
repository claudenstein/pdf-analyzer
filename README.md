# PDF Analyzer Pro

A powerful PDF analysis system using RAG (Retrieval-Augmented Generation) with Mistral 7B LLM. This tool allows you to analyze multiple PDF documents, build a searchable vector database, and ask questions about the content.

## Features

- **Multiple PDF Directory Support**: Scan multiple directories recursively
- **Hash-based Duplicate Detection**: Automatically detect and skip duplicate PDFs
- **Incremental Updates**: Add new PDFs without rebuilding the entire database
- **Configurable Retrieval**: Adjust chunk sizes and retrieval parameters
- **Interactive Q&A**: Ask questions about your PDF collection
- **GPU Acceleration**: Uses CUDA for faster embeddings
- **Timeout Protection**: Handles problematic PDFs with configurable timeouts
- **Progress Tracking**: Visual progress bars and detailed statistics

## Requirements

- Python 3.8+
- CUDA-capable GPU (for optimal performance)
- Ollama with Mistral model installed

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/claudenstein/pdf-analyzer.git
cd pdf-analyzer
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and setup Ollama

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Mistral model
ollama pull mistral
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

### First Run

1. Create a `pdfs` directory and add your PDF files:
   ```bash
   mkdir pdfs
   cp /path/to/your/*.pdf pdfs/
   ```

2. Run the analyzer:
   ```bash
   python pdf_analyzer_pro.py
   ```

3. The system will:
   - Scan for PDFs
   - Process and extract text
   - Create embeddings
   - Build a vector database
   - Launch interactive Q&A mode

### Interactive Mode

Once the database is built, you can ask questions:

```
🔍 Your question: What are the main topics discussed?
🔍 Your question: top_k:10 Summarize the key findings
🔍 Your question: quit
```

## Command-line Options

```bash
usage: pdf_analyzer_pro.py [OPTIONS]

Options:
  --pdf-dir DIR [DIR ...]   PDF directories to scan (default: ./pdfs)
  --db-dir DIR              Vector database directory (default: ./chroma_db)
  --top-k N                 Number of chunks to retrieve (default: 5)
  --chunk-size N            Text chunk size (default: 1000)
  --chunk-overlap N         Text chunk overlap (default: 200)
  --model MODEL             LLM model to use (default: mistral)
  --temperature FLOAT       LLM temperature (default: 0.1)
  --timeout SECONDS         Timeout per PDF file (default: 30)
  --rebuild                 Rebuild database from scratch
  --stats                   Show database statistics and exit
```

## Usage Examples

### View Database Statistics

```bash
python pdf_analyzer_pro.py --stats
```

### Rebuild Database from Scratch

```bash
python pdf_analyzer_pro.py --rebuild
```

### Custom Configuration

```bash
python pdf_analyzer_pro.py \
  --pdf-dir ./research ./papers \
  --chunk-size 1500 \
  --chunk-overlap 300 \
  --top-k 10 \
  --model mistral
```

### Using a Different LLM Model

```bash
# First pull the model
ollama pull llama2

# Use it with the analyzer
python pdf_analyzer_pro.py --model llama2
```

## How It Works

1. **PDF Scanning**: Recursively scans specified directories for PDF files
2. **Duplicate Detection**: Uses MD5 hashing to identify duplicate files
3. **Text Extraction**: Extracts text from PDFs using PyPDF
4. **Chunking**: Splits text into overlapping chunks for better context
5. **Embeddings**: Creates vector embeddings using HuggingFace models
6. **Vector Database**: Stores embeddings in ChromaDB for fast retrieval
7. **RAG Pipeline**: Retrieves relevant chunks and generates answers using Mistral

## Project Structure

```
pdf-analyzer/
├── pdf_analyzer_pro.py    # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
├── pdfs/                 # Default PDF directory (create this)
└── chroma_db/            # Vector database (auto-generated)
    └── processed_files.json
```

## Troubleshooting

### GPU Not Detected

If you don't have a CUDA GPU, modify the embedding setup in `pdf_analyzer_pro.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Change 'cuda' to 'cpu'
)
```

### Ollama Connection Issues

Make sure Ollama is running:

```bash
ollama serve
```

### PDF Processing Timeout

For large PDFs, increase the timeout:

```bash
python pdf_analyzer_pro.py --timeout 60
```

### Memory Issues

If you run out of memory:
- Reduce chunk size: `--chunk-size 500`
- Reduce top-k: `--top-k 3`
- Process fewer PDFs at once

## Performance Tips

1. **Use SSD**: Store the vector database on an SSD for faster retrieval
2. **GPU Acceleration**: Use CUDA for 10-100x faster embeddings
3. **Batch Processing**: Process PDFs in batches if you have many files
4. **Incremental Updates**: Use incremental mode instead of rebuilding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Embeddings from [Sentence Transformers](https://www.sbert.net/)
- Vector database powered by [ChromaDB](https://www.trychroma.com/)
- LLM inference via [Ollama](https://ollama.com/)
