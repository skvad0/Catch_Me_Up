# Catch Me Up - Book Q&A System

A RAG (Retrieval-Augmented Generation) based command-line tool that lets you ask questions about EPUB books and get contextual summaries.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/catch-me-up.git
cd catch-me-up
```

2. Create and activate a virtual environment (recommended):

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull a model:
```bash
# Pull a model (choose based on your system):
ollama pull llama3.2:1b      # Recommended (1.3GB, 3GB RAM)
ollama pull tinyllama:1.1b   # Fastest (637MB, 2GB RAM)
ollama pull qwen2.5:0.5b     # Smallest (397MB, 2GB RAM)
```

## Usage

### Basic Usage

```bash
python main.py yourbook.epub
```

### With Options

```bash
# Use a different model
python main.py yourbook.epub --model tinyllama:1.1b

# Specify output directory
python main.py yourbook.epub --output-dir ./my_books

# Retrieve more pages per query
python main.py yourbook.epub --top-k 10
```

### Command-Line Options

```
usage: main.py [-h] [-m MODEL] [-o OUTPUT_DIR] [-k TOP_K] book

positional arguments:
  book                  Path to the EPUB file

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Ollama model to use (default: llama3.2:1b)
                        Choices: tinyllama:1.1b, llama3.2:1b, qwen2.5:0.5b, qwen2.5:1.5b, qwen2.5:3b
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for pages and database (default: current directory)
  -k TOP_K, --top-k TOP_K
                        Number of pages to retrieve for each query (default: 5)
```

## Interactive Menu

Once started, you'll see an interactive menu:

```
1. Ask a question about the book
2. Summarize pages by position range
3. Summarize last N pages before position
4. Exit
```

## How It Works

1. **Processing**: The EPUB is parsed into semantic pages (paragraphs) with metadata (chapter, section, position)
2. **Indexing**: Pages are embedded and stored in a ChromaDB vector database
3. **Querying**: 
   - Your question is embedded
   - Most relevant pages are retrieved via similarity search
   - Context is passed to the LLM
   - LLM generates an answer based only on the retrieved context
```
