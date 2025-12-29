#!/usr/bin/env python3
"""
PDF Analysis System PRO - RAG + Mistral 7B
Features:
- Multiple PDF directory support
- Hash-based duplicate detection
- Command-line arguments
- Configurable retrieval parameters
- Scalable for large document collections
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os
import sys
import hashlib
import json
import argparse
from pathlib import Path
from tqdm import tqdm

class PDFAnalyzerPro:
    def __init__(self, pdf_dirs, db_dir="./chroma_db", metadata_file="./chroma_db/processed_files.json",
                 chunk_size=1000, chunk_overlap=200):
        self.pdf_dirs = [Path(d) for d in pdf_dirs]
        self.db_dir = Path(db_dir)
        self.metadata_file = Path(metadata_file)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectordb = None
        self.llm = None

    def calculate_file_hash(self, filepath):
        """Calculate MD5 hash of a file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"⚠️  Could not hash {filepath}: {e}")
            return None

    def get_processed_metadata(self):
        """Get metadata of already processed PDFs (hash-based)"""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_processed_metadata(self, metadata):
        """Save metadata of processed PDFs"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def find_all_pdfs(self):
        """Find all PDFs in specified directories (recursively)"""
        all_pdfs = []

        print("\n🔍 Scanning directories for PDFs...")
        for pdf_dir in self.pdf_dirs:
            if not pdf_dir.exists():
                print(f"⚠️  Directory not found: {pdf_dir}")
                continue

            # Find PDFs recursively
            pdfs = list(pdf_dir.rglob("*.pdf")) + list(pdf_dir.rglob("*.PDF"))
            all_pdfs.extend(pdfs)
            print(f"   📁 {pdf_dir}: found {len(pdfs)} PDF(s)")

        return all_pdfs

    def analyze_pdfs(self):
        """Analyze PDFs and detect new/duplicate/changed files"""
        all_pdfs = self.find_all_pdfs()

        if not all_pdfs:
            return [], {}, {}

        print(f"\n📊 Total PDFs found: {len(all_pdfs)}")

        # Get existing metadata
        processed_metadata = self.get_processed_metadata()
        processed_hashes = {v['hash']: k for k, v in processed_metadata.items() if 'hash' in v}

        # Analyze current PDFs - only hash files not already in database
        current_files = {}
        new_files = []
        duplicate_files = []

        print("🔐 Analyzing files...")

        for pdf_path in tqdm(all_pdfs, desc="Checking files"):
            pdf_str = str(pdf_path)

            # If file is already in metadata with same path, reuse its hash
            if pdf_str in processed_metadata:
                existing_hash = processed_metadata[pdf_str].get('hash')
                current_size = pdf_path.stat().st_size
                stored_size = processed_metadata[pdf_str].get('size', 0)

                # Only rehash if file size changed
                if existing_hash and current_size == stored_size:
                    current_files[pdf_str] = processed_metadata[pdf_str]
                    continue

            # New file or changed file - compute hash
            pdf_hash = self.calculate_file_hash(pdf_path)
            if not pdf_hash:
                continue

            # Check if this hash already exists (different path = duplicate)
            if pdf_hash in processed_hashes:
                old_path = processed_hashes[pdf_hash]
                if old_path != pdf_str:
                    duplicate_files.append((pdf_str, old_path))
                else:
                    # Same file, same path - already processed
                    current_files[pdf_str] = {
                        'hash': pdf_hash,
                        'size': pdf_path.stat().st_size,
                        'modified': pdf_path.stat().st_mtime
                    }
            else:
                # New file
                new_files.append(pdf_path)
                current_files[pdf_str] = {
                    'hash': pdf_hash,
                    'size': pdf_path.stat().st_size,
                    'modified': pdf_path.stat().st_mtime
                }

        return new_files, current_files, duplicate_files

    def load_and_process_pdfs(self, incremental=False, timeout_seconds=30):
        """Load PDFs and create/update vector database"""

        # Analyze PDFs
        new_files, current_files, duplicates = self.analyze_pdfs()

        if duplicates:
            print(f"\n🔍 Found {len(duplicates)} duplicate(s):")
            for new_path, old_path in duplicates[:5]:
                print(f"   ⚠️  {Path(new_path).name}")
                print(f"      Already in DB as: {old_path}")
            if len(duplicates) > 5:
                print(f"   ... and {len(duplicates) - 5} more")

        # Decide which PDFs to process
        if incremental and new_files:
            pdf_files = new_files
            print(f"\n📊 Database Status:")
            print(f"   Total PDFs found: {len(current_files)}")
            print(f"   Already in database: {len(current_files) - len(new_files)}")
            print(f"   Duplicates (skipped): {len(duplicates)}")
            print(f"   NEW PDFs to add: {len(new_files)}")
        elif incremental and not new_files:
            print(f"\n✅ All {len(current_files)} PDFs are already in the database!")
            print("   Nothing new to process.")
            return True
        else:
            pdf_files = [Path(p) for p in current_files.keys() if not any(str(Path(p)) == dup[0] for dup in duplicates)]
            print(f"\n📊 Processing {len(pdf_files)} PDF(s) (excluding {len(duplicates)} duplicates)")

        if not pdf_files:
            print("   No files to process.")
            return True

        # Show sample of files being processed
        print(f"\n📄 Files to process:")
        for pdf in pdf_files[:10]:
            print(f"   - {pdf.name}")
        if len(pdf_files) > 10:
            print(f"   ... and {len(pdf_files) - 10} more")

        # Load PDFs with individual progress and timeout
        import multiprocessing
        from multiprocessing import Process, Queue

        def load_pdf_with_timeout(pdf_path, result_queue, timeout):
            """Load PDF in a separate process with timeout"""
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = []
                for doc in loader.lazy_load():
                    docs.append(doc)
                result_queue.put(('success', docs, len(docs)))
            except Exception as e:
                result_queue.put(('error', str(e), None))

        documents = []
        successful_files = []
        failed = []
        skipped = []

        print(f"\n📖 Loading PDF contents (timeout: {timeout_seconds}s per file)...")
        print("    Press Ctrl+C during a file load to skip that file\n")

        for i, pdf_path in enumerate(pdf_files, 1):
            pdf_name = pdf_path.name
            print(f"\n[{i}/{len(pdf_files)}] 📄 {pdf_name}")

            try:
                print(f"    Loading pages", end="", flush=True)

                # Create a queue for result
                result_queue = Queue()

                # Start loading in a separate process
                process = Process(target=load_pdf_with_timeout,
                                args=(pdf_path, result_queue, timeout_seconds))
                process.start()

                # Wait for completion with timeout
                process.join(timeout=timeout_seconds)

                if process.is_alive():
                    # Timeout occurred
                    process.terminate()
                    process.join()
                    print(f"\n    ⏱️  TIMEOUT after {timeout_seconds}s - Auto-skipping")
                    skipped.append((str(pdf_path), f"Timeout after {timeout_seconds}s"))
                    continue

                # Check if we got a result
                if not result_queue.empty():
                    status, data, page_count = result_queue.get()

                    if status == 'success':
                        documents.extend(data)
                        successful_files.append(str(pdf_path))
                        print(f" ✅ ({page_count} pages)")
                    else:
                        # Error occurred
                        failed.append((str(pdf_path), data))
                        print(f"\n    ❌ FAILED: {data[:50]}")
                else:
                    # Process ended but no result
                    skipped.append((str(pdf_path), "No result from loader"))
                    print(f"\n    ⏭️  Skipped (no result)")

            except KeyboardInterrupt:
                print(f"\n    ⏭️  Skipped by user (Ctrl+C)")
                skipped.append((str(pdf_path), "Skipped by user"))
                choice = input(f"    Continue with next file? (y/n): ").strip().lower()
                if choice != 'y' and choice != '':
                    print("\n🛑 Processing cancelled")
                    break

            except Exception as e:
                failed.append((str(pdf_path), str(e)))
                print(f"\n    ❌ FAILED: {str(e)[:50]}")
                continue

        # Summary
        print(f"\n" + "="*60)
        print(f"📊 Processing Summary:")
        print(f"   ✅ Successful: {len(successful_files)}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   ⏭️  Skipped: {len(skipped)}")
        print("="*60)

        if failed:
            print(f"\n⚠️  Failed files:")
            for pdf_path, error in failed[:5]:
                print(f"   - {Path(pdf_path).name}: {error[:60]}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more")

        if skipped:
            print(f"\n⏭️  Skipped files:")
            for pdf_path, reason in skipped[:5]:
                print(f"   - {Path(pdf_path).name}: {reason}")
            if len(skipped) > 5:
                print(f"   ... and {len(skipped) - 5} more")

        if not documents:
            if incremental:
                print(f"\n⚠️  No new documents loaded, but existing database is intact.")
                return True
            else:
                print(f"\n❌ No documents could be loaded!")
                return False

        print(f"\n✅ Total pages loaded: {len(documents)}")

        # Split into chunks
        print(f"✂️  Splitting into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        print(f"✅ Created {len(texts)} text chunks")

        # Create embeddings
        print("🚀 Creating embeddings (using GPU)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )

        # Create or update vector database
        if incremental and self.vectordb is not None:
            print("➕ Adding new documents to existing database...")
            self.vectordb.add_documents(texts)
            print(f"✅ Added {len(texts)} chunks to existing database")
        else:
            print("💾 Creating vector database...")
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=str(self.db_dir)
            )
            print(f"✅ Vector database created and saved to {self.db_dir}")

        # Update metadata (only successful files)
        processed_metadata = self.get_processed_metadata()
        for filepath in successful_files:
            if filepath in current_files:
                processed_metadata[filepath] = current_files[filepath]

        self.save_processed_metadata(processed_metadata)
        print(f"📝 Updated metadata: {len(processed_metadata)} PDFs tracked")

        return True

    def load_existing_db(self):
        """Load existing vector database"""
        if not self.db_dir.exists():
            print(f"❌ Database directory {self.db_dir} doesn't exist!")
            return False

        print("📂 Loading existing vector database...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )

        self.vectordb = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=embeddings
        )
        print("✅ Database loaded successfully")
        return True

    def setup_llm(self, temperature=0.1, model="mistral"):
        """Setup the LLM"""
        print(f"🤖 Setting up {model} LLM...")
        self.llm = Ollama(model=model, temperature=temperature)
        print("✅ LLM ready")
        return True

    def query(self, question, top_k=5):
        """Ask a question about the PDFs"""
        if self.vectordb is None or self.llm is None:
            print("❌ System not set up properly!")
            return None

        print(f"\n💬 Question: {question}")
        print(f"\n🔍 Searching for relevant content (top {top_k} chunks)...")

        # Retrieve relevant documents
        docs = self.vectordb.similarity_search(question, k=top_k)

        if not docs:
            print("❌ No relevant documents found!")
            return None

        print(f"✅ Found {len(docs)} relevant chunks")

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt = f"""Based on the following context from PDF documents, please answer the question.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        print("\n🤔 Thinking...\n")

        # Get response from LLM
        response = self.llm.invoke(prompt)

        print("\n\n📚 Sources used:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            print(f"  {i}. {Path(source).name} (Page {page})")

        return response

    def interactive_mode(self, default_top_k=5):
        """Interactive question-answering mode"""
        print("\n" + "="*60)
        print("📖 PDF ANALYSIS PRO - Interactive Mode")
        print("="*60)
        print("Type your questions (or 'quit' to exit)")
        print(f"Use 'top_k:N <question>' to retrieve N chunks (default: {default_top_k})")
        print("Example: 'top_k:10 What are the main findings?'")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("\n🔍 Your question: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Parse top_k if specified
                top_k = default_top_k
                question = user_input

                if user_input.startswith('top_k:'):
                    try:
                        parts = user_input.split(' ', 1)
                        top_k = int(parts[0].split(':')[1])
                        question = parts[1] if len(parts) > 1 else ""
                        if not question:
                            print("❌ Please provide a question after top_k:N")
                            continue
                    except (ValueError, IndexError):
                        print("❌ Invalid top_k format. Use: top_k:N <question>")
                        continue

                response = self.query(question, top_k=top_k)
                if response:
                    print(f"\n💡 Answer: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

    def show_statistics(self):
        """Show database statistics"""
        metadata = self.get_processed_metadata()

        print("\n" + "="*60)
        print("📊 DATABASE STATISTICS")
        print("="*60)
        print(f"Total PDFs in database: {len(metadata)}")

        if metadata:
            total_size = sum(v.get('size', 0) for v in metadata.values())
            print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

            # Group by directory
            dirs = {}
            for path in metadata.keys():
                dir_path = str(Path(path).parent)
                dirs[dir_path] = dirs.get(dir_path, 0) + 1

            print(f"\nPDFs by directory:")
            for dir_path, count in sorted(dirs.items(), key=lambda x: -x[1]):
                print(f"   {count:3d} files in {dir_path}")

        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='PDF Analyzer Pro - RAG System with Mistral 7B',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory
  python pdf_analyzer_pro.py --pdf-dir ./pdfs

  # Multiple directories
  python pdf_analyzer_pro.py --pdf-dir ./pdfs /home/docs /research/papers

  # Custom settings
  python pdf_analyzer_pro.py --pdf-dir ./pdfs --top-k 10 --chunk-size 1500

  # Different database location
  python pdf_analyzer_pro.py --pdf-dir ./pdfs --db-dir /mnt/storage/vectordb
        """
    )

    parser.add_argument('--pdf-dir', nargs='+', default=['./pdfs'],
                        help='PDF directory/directories to scan (default: ./pdfs)')
    parser.add_argument('--db-dir', default='./chroma_db',
                        help='Vector database directory (default: ./chroma_db)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of chunks to retrieve (default: 5)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Text chunk size (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                        help='Text chunk overlap (default: 200)')
    parser.add_argument('--model', default='mistral',
                        help='LLM model to use (default: mistral)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='LLM temperature (default: 0.1)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout per PDF file in seconds (default: 30)')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild database from scratch')
    parser.add_argument('--stats', action='store_true',
                        help='Show database statistics and exit')

    args = parser.parse_args()

    # Create analyzer
    analyzer = PDFAnalyzerPro(
        pdf_dirs=args.pdf_dir,
        db_dir=args.db_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    print("\n" + "="*60)
    print("🔬 PDF ANALYZER PRO - RAG System with Mistral 7B")
    print("="*60)
    print(f"📁 Watching directories: {', '.join(args.pdf_dir)}")
    print(f"💾 Database: {args.db_dir}")
    print(f"🎯 Retrieval: top-{args.top_k} chunks")
    print(f"✂️  Chunking: {args.chunk_size} chars (overlap: {args.chunk_overlap})")
    print("="*60 + "\n")

    # Show stats and exit if requested
    if args.stats:
        analyzer.show_statistics()
        return

    # Check if database exists
    db_exists = Path(args.db_dir).exists() and any(Path(args.db_dir).iterdir())

    if db_exists and not args.rebuild:
        print("Found existing database!")

        # Check for new PDFs
        new_files, current_files, duplicates = analyzer.analyze_pdfs()

        if new_files:
            print(f"\n🆕 Detected {len(new_files)} new PDF(s) not in database!")
            print("\nOptions:")
            print("  1. Add new PDFs to existing database (incremental update)")
            print("  2. Use existing database only (ignore new PDFs)")
            print("  3. Rebuild entire database from scratch")
            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == '1':
                print("\n➕ INCREMENTAL UPDATE MODE")
                if not analyzer.load_existing_db():
                    return
                if not analyzer.load_and_process_pdfs(incremental=True, timeout_seconds=args.timeout):
                    return
            elif choice == '2':
                print("\n📂 Using existing database (new PDFs ignored)")
                if not analyzer.load_existing_db():
                    return
            elif choice == '3':
                print("\n🔄 Rebuilding entire database from scratch...")
                if not analyzer.load_and_process_pdfs(incremental=False, timeout_seconds=args.timeout):
                    return
            else:
                print("Invalid choice. Exiting.")
                return
        else:
            print(f"✅ Database is up to date ({len(current_files)} PDFs)")
            if not analyzer.load_existing_db():
                return
    else:
        if args.rebuild:
            print("🔄 Rebuild mode: Processing all PDFs from scratch...")
        else:
            print("No existing database found. Processing all PDFs...")
        if not analyzer.load_and_process_pdfs(incremental=False, timeout_seconds=args.timeout):
            return

    # Setup LLM
    if not analyzer.setup_llm(temperature=args.temperature, model=args.model):
        return

    # Show stats
    analyzer.show_statistics()

    # Interactive mode
    analyzer.interactive_mode(default_top_k=args.top_k)

if __name__ == "__main__":
    main()
