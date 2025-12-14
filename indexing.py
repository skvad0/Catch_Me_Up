import json
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def initialize_components(collection_name="book_pages", persist_directory="./chroma_db"):
    """
    Initialize Vector DB (ChromaDB), Embedding Model, and LLM.
    Returns the chroma collection and configured settings.
    
    Using mistral:latest - a fast model suitable for Q&A
    """
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    
    embed_model = OllamaEmbedding(
        model_name="mistral:latest",
        base_url="http://localhost:11434",
        request_timeout=300.0
    )
    llm = Ollama(
        model="mistral:latest",
        base_url="http://localhost:11434",
        request_timeout=300.0
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    return chroma_collection


def create_storage_context_and_index(chroma_collection):
    """
    Create a StorageContext and VectorStoreIndex using ChromaDB.
    """
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)
    
    return storage_context, index


def load_pages_from_jsonl(jsonl_file):
    """
    Load pages from JSONL file and convert them to LlamaIndex Nodes.
    """
    nodes = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            page = json.loads(line)
            
            node = TextNode(
                text=page['text'],
                metadata=page['metadata'],
                id_=str(page['id'])
            )
            
            nodes.append(node)
    
    return nodes


def add_nodes_to_index(index, nodes):
    """
    Add nodes to the index. The framework will automatically:
    - Convert text to vector embeddings
    - Store them in ChromaDB
    """
    index.insert_nodes(nodes)
    return index
