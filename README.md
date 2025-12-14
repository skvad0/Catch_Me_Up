# Catch Me Up - Book Q&A System

A RAG (Retrieval-Augmented Generation) based command-line tool that lets you ask questions about EPUB books and get contextual summaries.

## Features

- 📚 Ask natural language questions about any EPUB book
- 🎯 Get answers with source citations (chapter, section, position)
- 📝 Summarize specific chapter ranges or positions
- 🚀 Fast local inference using Ollama
- 💾 Persistent vector database for quick subsequent queries
- 🖥️ Works on low-end PCs with small models

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- 2-4GB RAM (depending on model choice)

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

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
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

4. Install and start Ollama, then pull a model:
```bash
# Install Ollama from https://ollama.ai/

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

# Retrieve more chunks per query
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
                        Output directory for chunks and database (default: current directory)
  -k TOP_K, --top-k TOP_K
                        Number of chunks to retrieve for each query (default: 5)
```

## Interactive Menu

Once started, you'll see an interactive menu:

```
1. Ask a question about the book
2. Summarize chunks by position range
3. Summarize last N chunks before position
4. Exit
```

### Example Questions

- "What happened to Pluto?"
- "Who discovered Eris?"
- "Explain the IAU definition of a planet"
- "What did the author think about the controversy?"

## Model Recommendations

| Model | Size | RAM | Speed | Quality | Best For |
|-------|------|-----|-------|---------|----------|
| tinyllama:1.1b | 637MB | 2GB | ⚡⚡⚡ | ⭐⭐ | Very low-end PCs |
| llama3.2:1b | 1.3GB | 3GB | ⚡⚡ | ⭐⭐⭐ | Most users (recommended) |
| qwen2.5:0.5b | 397MB | 2GB | ⚡⚡⚡ | ⭐⭐ | Minimal systems |
| qwen2.5:1.5b | 1GB | 3GB | ⚡⚡ | ⭐⭐⭐ | Good balance |
| qwen2.5:3b | 2GB | 4GB | ⚡ | ⭐⭐⭐⭐ | Better quality answers |

## How It Works

1. **Processing**: The EPUB is parsed into semantic chunks (paragraphs) with metadata (chapter, section, position)
2. **Indexing**: Chunks are embedded and stored in a ChromaDB vector database
3. **Querying**: 
   - Your question is embedded
   - Most relevant chunks are retrieved via similarity search
   - Context is passed to the LLM
   - LLM generates an answer based only on the retrieved context

## Project Structure

```
catch-me-up/
├── main.py              # CLI entry point and main logic
├── indexing.py          # Vector DB and embedding setup
├── query.py             # Query engine and retrieval functions
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Troubleshooting

### Ollama Timeout Errors

If you get timeout errors, the model might be too large for your system:
- Try a smaller model: `ollama pull tinyllama:1.1b`
- Increase timeout in `indexing.py` (currently 300s)
- Ensure Ollama is running: `ollama serve`

### Empty Responses

- Check debug output for retrieval issues
- Verify chunks were created: look for `*_chunks.jsonl` file
- Lower similarity cutoff if no matches found

### Low-End PC Performance

- Use `tinyllama:1.1b` or `qwen2.5:0.5b`
- Reduce `--top-k` to 3 for faster queries
- Close other applications to free RAM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/)
- Powered by [Ollama](https://ollama.ai/)
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
