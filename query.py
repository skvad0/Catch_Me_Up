import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import QueryBundle
import json


def load_existing_index(collection_name="book_chunks", persist_directory="./chroma_db"):
    """
    Load the existing index from ChromaDB.
    """
    print(f"[DEBUG] Loading index from {persist_directory}, collection: {collection_name}")
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.get_collection(collection_name)
    
    print(f"[DEBUG] Collection count: {chroma_collection.count()}")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    print(f"[DEBUG] Index loaded successfully")
    return index


def retrieve_relevant_chunks(index, query, top_k=5, similarity_cutoff=0.7):
    """
    Retrieve: Find the most relevant chunks from the database.
    
    Args:
        index: The VectorStoreIndex
        query: The user's question
        top_k: Number of top chunks to retrieve
        similarity_cutoff: Minimum similarity score (0-1)
    
    Returns:
        List of retrieved nodes with their scores
    """
    print(f"[DEBUG] retrieve_relevant_chunks called with query: '{query}'")
    print(f"[DEBUG] top_k={top_k}, similarity_cutoff={similarity_cutoff}")
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    
    retrieved_nodes = retriever.retrieve(query)
    print(f"[DEBUG] Retrieved {len(retrieved_nodes)} nodes")
    for i, node in enumerate(retrieved_nodes[:3]):  # Show first 3
        print(f"[DEBUG] Node {i}: score={node.score:.4f}, text_preview={node.text[:100]}...")
    
    return retrieved_nodes


def create_query_engine(index, top_k=5, similarity_cutoff=0.7):
    """
    Create a query engine that handles Retrieve, Augment, and Generate.
    
    The query engine will:
    1. Retrieve the most relevant chunks
    2. Augment the prompt with retrieved context
    3. Generate an answer using the LLM
    """
    print(f"[DEBUG] Creating query engine with top_k={top_k}, similarity_cutoff={similarity_cutoff}")
    
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
    )
    
    print(f"[DEBUG] Query engine created successfully")
    return query_engine


def query_book(query_engine, question):
    """
    Query the book with a question.
    
    This performs:
    1. Retrieve: Gets relevant chunks from the vector database
    2. Augment: Adds the chunks as context to the prompt
    3. Generate: LLM generates answer based only on the context
    
    Args:
        query_engine: The configured query engine
        question: User's question
    
    Returns:
        Response object with answer and source information
    """
    print(f"[DEBUG] query_book called with question: '{question}'")
    response = query_engine.query(question)
    print(f"[DEBUG] Response received")
    print(f"[DEBUG] Response type: {type(response)}")
    print(f"[DEBUG] Response.response: '{response.response}'")
    print(f"[DEBUG] Response has {len(response.source_nodes) if hasattr(response, 'source_nodes') else 0} source nodes")
    return response


def display_response(response):
    """
    Display the response with sources.
    """
    print("Answer:")
    print("-" * 80)
    print(response.response)
    print("\n" + "=" * 80)
    print("Sources:")
    print("=" * 80)
    
    for i, node in enumerate(response.source_nodes, 1):
        print(f"\nSource {i}:")
        print(f"Similarity Score: {node.score:.4f}")
        print(f"Chapter: {node.metadata.get('chapter', 'Unknown')}")
        print(f"Section: {node.metadata.get('section', 'N/A')}")
        print(f"Text Preview: {node.text[:200]}...")
        print("-" * 80)


def get_chunks_by_position(jsonl_file, start_position, end_position):
    """
    Retrieve chunks by their position range.
    Useful for "summarize the last 5 pages" type queries.
    
    Args:
        jsonl_file: Path to the JSONL file with chunks
        start_position: Starting chunk position (inclusive)
        end_position: Ending chunk position (inclusive)
    
    Returns:
        List of chunks in the specified range
    """
    chunks = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunk_pos = chunk['metadata'].get('chunk_position', chunk['id'])
            
            if start_position <= chunk_pos <= end_position:
                chunks.append(chunk)
    
    return chunks


def summarize_chunk_range(llm, chunks, custom_prompt=None):
    """
    Summarize a range of chunks using the LLM.
    Chunks are treated as a cohesive unit (like pages in a book).
    
    Args:
        llm: The LLM instance (from Settings.llm)
        chunks: List of chunk dictionaries (representing one or more pages)
        custom_prompt: Optional custom prompt template
    
    Returns:
        Summary generated by the LLM
    """
    # Combine all chunk texts into a single narrative
    combined_text = "\n\n".join([chunk['text'] for chunk in chunks])
    
    # Get metadata for context
    chapters = set(chunk['metadata'].get('chapter') for chunk in chunks if chunk['metadata'].get('chapter'))
    chapter_info = f" from chapter(s): {', '.join(chapters)}" if chapters else ""
    
    # Extract book title and author from chunk metadata
    book_title = chunks[0]['metadata'].get('book_title', 'Unknown') if chunks else 'Unknown'
    author = chunks[0]['metadata'].get('author', 'Unknown') if chunks else 'Unknown'
    book_context = f' the book "{book_title}" by {author}' if book_title != 'Unknown' else ' this book'
    
    if custom_prompt:
        prompt = custom_prompt.format(text=combined_text)
    else:
        prompt = f"""You are summarizing a section of{book_context}{chapter_info}.

Provide a comprehensive but concise summary of the following text. Focus on:
- Main events or discoveries described
- Key concepts or ideas presented
- Important characters or entities mentioned
- The narrative flow and progression

Text to summarize:

{combined_text}

Summary:"""
    
    response = llm.complete(prompt)
    return response.text


def get_current_position_context(jsonl_file, current_position, num_chunks_before=5):
    """
    Get context around a current reading position.
    For example, if you're at position 297, get the previous 5 chunks.
    
    Args:
        jsonl_file: Path to the JSONL file
        current_position: Current reading position (chunk number)
        num_chunks_before: Number of chunks to retrieve before current position
    
    Returns:
        List of chunks
    """
    start_pos = max(0, current_position - num_chunks_before)
    end_pos = current_position - 1
    
    return get_chunks_by_position(jsonl_file, start_pos, end_pos)
