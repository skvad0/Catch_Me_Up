import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import re
import os
import argparse
import sys
from indexing import initialize_components, create_storage_context_and_index, load_pages_from_jsonl, add_nodes_to_index
from query import load_existing_index, create_query_engine, query_book, get_pages_by_position, get_current_position_context, summarize_page_range
from llama_index.core import Settings

PARAGRAPHS_PER_PAGE = 10  # Number of paragraphs that make up one "page"

# Clean and Parse
def clean_text(text):
    """
    Clean text by removing excessive whitespace and unwanted artifacts
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
    text = re.sub(r'\s+', ' ', text)  # Clean up spaces again
    return text


# Extract, Parse, Clean, and Enrich with Metadata
def process_epub_to_pages(book, book_title=None, author=None):
    """
    Process EPUB and create pages with metadata for RAG system.
    Each page represents a paragraph with its contextual metadata.
    """
    # Try to extract metadata from EPUB if not provided
    if book_title is None:
        book_title = book.get_metadata('DC', 'title')
        book_title = book_title[0][0] if book_title else 'Unknown'
    
    if author is None:
        author = book.get_metadata('DC', 'creator')
        author = author[0][0] if author else 'Unknown'
    
    pages = []
    paragraph_id = 0
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')
            
            current_chapter = None
            current_section = None
            
            for element in soup.find_all(['h1', 'h2', 'p']):
                if element.name == 'h1':
                    current_chapter = element.get_text(strip=True)
                    current_section = None  # Reset section when new chapter starts
                    
                elif element.name == 'h2':
                    current_section = element.get_text(strip=True)
                    
                elif element.name == 'p':
                    text = element.get_text(strip=True)
                    
                    if text:  # Only process non-empty paragraphs
                        cleaned_text = clean_text(text)
                        
                        if cleaned_text:  # Only add if text remains after cleaning
                            paragraph = {
                                'id': paragraph_id,
                                'text': cleaned_text,
                                'metadata': {
                                    'source_file': item.get_name(),
                                    'chapter': current_chapter,
                                    'section': current_section,
                                    'book_title': book_title,
                                    'author': author,
                                    'position': paragraph_id
                                }
                            }
                            
                            pages.append(paragraph)
                            paragraph_id += 1
    
    return pages


# Format and Write as JSONL
def write_pages_to_jsonl(pages, output_filename='book_pages.jsonl'):
    """
    Write pages to JSONL file (one JSON object per line)
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        for page in pages:
            json.dump(page, f, ensure_ascii=False)
            f.write('\n')


def setup_system(book_path, pages_file, chroma_db_path, model_name):
    """
    Initial setup: Process EPUB and create index if not already done.
    """
    if not os.path.exists(pages_file):
        print("Processing EPUB file...")
        book = epub.read_epub(book_path)
        pages = process_epub_to_pages(book)
        print(f"Created {len(pages)} pages")
        
        print("Writing pages to JSONL...")
        write_pages_to_jsonl(pages, pages_file)
        print(f"Saved to {pages_file}")
    else:
        print(f"Pages file already exists: {pages_file}")
    
    if not os.path.exists(chroma_db_path):
        print(f"\nInitializing components (Ollama with {model_name})...")
        chroma_collection = initialize_components(model_name=model_name)
        
        print("Creating index...")
        storage_context, index = create_storage_context_and_index(chroma_collection)
        
        print("Loading pages and creating nodes...")
        nodes = load_pages_from_jsonl(pages_file)
        
        print(f"Indexing {len(nodes)} nodes...")
        add_nodes_to_index(index, nodes)
        print("Index created successfully!")
    else:
        print("Index already exists")


def display_menu():
    """
    Display the main menu.
    """
    print("\n" + "=" * 80)
    print("CATCH ME UP - Book Q&A System")
    print("=" * 80)
    print("\n1. Ask a question about the book")
    print("2. Summarize pages by position range (e.g., pages 5-10)")
    print("3. Summarize last N pages before current position (e.g., last 3 pages before page 15)")
    print("4. Exit")
    print("\nEnter your choice (1-4): ", end="")


def handle_question(query_engine):
    """
    Handle a user's question about the book.
    """
    question = input("\nEnter your question: ")
    
    print("\nProcessing...")
    print("1. RETRIEVE: Finding relevant pages...")
    print("2. AUGMENT: Adding context to prompt...")
    print("3. GENERATE: Generating answer...")
    
    print(f"[DEBUG] About to call query_book with question: '{question}'")
    try:
        response = query_book(query_engine, question)
        print(f"[DEBUG] Returned from query_book")
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {str(e)}")
        if "timeout" in str(e).lower():
            print("\nThe LLM timed out. This could mean:")
            print("  1. The model is too large/slow for your system")
            print("  2. Ollama service is overloaded or hung")
            print("  3. Try a smaller model: ollama pull qwen2.5:0.5b")
            print("  4. Or try: ollama pull mistral:7b (or another available Mistral tag)")
        return
    
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(response.response)
    
    if hasattr(response, 'source_nodes') and response.source_nodes:
        print("\n" + "=" * 80)
        print("SOURCES:")
        print("=" * 80)
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\nSource {i}:")
            print(f"  Score: {node.score:.4f}")
            print(f"  Chapter: {node.metadata.get('chapter', 'Unknown')}")
            print(f"  Section: {node.metadata.get('section', 'N/A')}")
            print(f"  Position: {node.metadata.get('position', 'N/A')}")
            print(f"  Preview: {node.text[:150]}...")
    else:
        print("\nNo source pages were retrieved. This might indicate:")
        print("- The similarity threshold is too high")
        print("- The question doesn't match indexed content")
        print("- There's an issue with the embeddings")


def handle_range_summary(pages_file):
    """
    Summarize a specific range of pages.
    """
    try:
        start_page = int(input("\nEnter start page number: "))
        end_page = int(input("Enter end page number: "))
        
        # Convert pages to paragraph positions
        start_pos = start_page * PARAGRAPHS_PER_PAGE
        end_pos = (end_page + 1) * PARAGRAPHS_PER_PAGE - 1
        
        print(f"\nRetrieving pages {start_page} to {end_page} (paragraphs {start_pos}-{end_pos})...")
        pages = get_pages_by_position(pages_file, start_pos, end_pos)
        
        if not pages:
            print("No pages found in that range.")
            return
        
        print(f"Found {len(pages)} pages across {end_page - start_page + 1} pages")
        print("\nGenerating summary...")
        
        summary = summarize_page_range(Settings.llm, pages)
        
        print("\n" + "=" * 80)
        print(f"SUMMARY (Pages {start_page}-{end_page}):")
        print("=" * 80)
        print(summary)
        
    except ValueError:
        print("Invalid input. Please enter numbers.")


def handle_context_summary(pages_file):
    """
    Summarize N pages before a current page position.
    """
    try:
        current_page = int(input("\nEnter current page number: "))
        num_pages = int(input("How many pages before this position to summarize? "))
        
        # Convert pages to paragraph positions
        start_page = max(0, current_page - num_pages)
        end_page = current_page - 1
        
        start_pos = start_page * PARAGRAPHS_PER_PAGE
        end_pos = (end_page + 1) * PARAGRAPHS_PER_PAGE - 1
        
        print(f"\nRetrieving {num_pages} pages before page {current_page}...")
        print(f"  (Pages {start_page}-{end_page}, paragraphs {start_pos}-{end_pos})")
        pages = get_pages_by_position(pages_file, start_pos, end_pos)
        
        if not pages:
            print("No pages found.")
            return
        
        print(f"\nFound {len(pages)} pages across {num_pages} pages")
        
        # Group paragraphs by page for display
        page_groups = {}
        for page in pages:
            page_pos = page.get('id', page['metadata'].get('position', 'N/A'))
            page_num = page_pos // PARAGRAPHS_PER_PAGE
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(page)
        
        # Display page summary
        for page_num in sorted(page_groups.keys()):
            page_items = page_groups[page_num]
            chapters = set(c['metadata'].get('chapter') for c in page_items if c['metadata'].get('chapter'))
            chapter_str = ', '.join(chapters) if chapters else 'Unknown'
            print(f"  - Page {page_num}: {len(page_items)} paragraphs ({chapter_str})")
        
        print("\nGenerating summary...")
        summary = summarize_page_range(Settings.llm, pages)
        
        print("\n" + "=" * 80)
        print(f"SUMMARY (Pages {start_page}-{end_page}):")
        print("=" * 80)
        print(summary)
        
    except ValueError:
        print("Invalid input. Please enter numbers.")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Catch Me Up - Book Q&A System using RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py mybook.epub
  python main.py mybook.epub --model tinyllama:1.1b
  python main.py mybook.epub --output-dir ./books --top-k 10
  
Models (sorted by speed):
  tinyllama:1.1b   - Fastest, 637MB, 2GB RAM (basic quality)
  llama3.2:1b      - Recommended, 1.3GB, 3GB RAM (good quality)
  qwen2.5:0.5b     - Smallest, 397MB, 2GB RAM (basic quality)
  qwen2.5:1.5b     - Balanced, 1GB, 3GB RAM (good quality)
  qwen2.5:3b       - Best quality, 2GB, 4GB RAM (slower)
        '''
    )
    
    parser.add_argument('book', type=str, help='Path to the EPUB file')
    parser.add_argument('-m', '--model', type=str, default='llama3.2:1b',
                        choices=['tinyllama:1.1b', 'llama3.2:1b', 'qwen2.5:0.5b', 'qwen2.5:1.5b', 'qwen2.5:3b'],
                        help='Ollama model to use (default: llama3.2:1b)')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                        help='Output directory for pages and database (default: current directory)')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                        help='Number of pages to retrieve for each query (default: 5)')
    
    args = parser.parse_args()
    
    # Validate book path
    if not os.path.exists(args.book):
        parser.error(f"EPUB file not found: {args.book}")
    
    if not args.book.lower().endswith('.epub'):
        parser.error("File must be an EPUB (.epub)")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args


def main():
    """
    Main application loop.
    """
    args = parse_arguments()
    
    # Generate file paths based on book name and output directory
    book_name = os.path.splitext(os.path.basename(args.book))[0]
    pages_file = os.path.join(args.output_dir, f"{book_name}_pages.jsonl")
    chroma_db_path = os.path.join(args.output_dir, 'chroma_db')
    
    print("=" * 80)
    print("INITIALIZING CATCH ME UP SYSTEM")
    print("=" * 80)
    print(f"Book: {args.book}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top-K retrieval: {args.top_k}")
    print("=" * 80)
    
    setup_system(args.book, pages_file, chroma_db_path, args.model)
    
    print("\nInitializing query components...")
    initialize_components(model_name=args.model)
    index = load_existing_index()
    query_engine = create_query_engine(index, top_k=args.top_k, similarity_cutoff=0.0)
    
    print("\nSystem ready!")
    
    while True:
        display_menu()
        choice = input().strip()
        
        if choice == '1':
            handle_question(query_engine)
        elif choice == '2':
            handle_range_summary(pages_file)
        elif choice == '3':
            handle_context_summary(pages_file)
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()




