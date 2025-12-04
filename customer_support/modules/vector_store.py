import os
import sys
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
import shutil

# Add project root to Python path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__)) # customer_support/modules/
project_root = os.path.dirname(os.path.dirname(current_dir)) # smart_customer_support/

sys.path.insert(0, project_root)

# Import Config
from customer_support.modules.config import Config

class VectorStoreManager:
    """Manages vector store creation, saving, and loading."""
    
    def __init__(self):
        print(f'Loading embedding model: {Config.EMBEDDING_MODEL}')
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        print('Embedding model loaded')
        
    def create_vector_store(self, chunks: List[str]) -> FAISS:
        """
        Create FAISS vector store from text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            FAISS vector store
        """
        print("Creating vector store from chunks...")
        
        # Create FAISS index from texts
        
        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
        
        print(f'Vector store created with {len(chunks)} chunks')
        
        return vector_store
    
    # Save vector store
    def save_vector_store(self, vector_store: FAISS, save_path: str = None):
        """
        Save vector store to disk.
        
        Args:
            vector_store: FAISS vector store
            save_path: Path to save the vector store
        """
        
        if save_path is None:
            save_path = Config.VECTOR_STORE_PATH
            
        print(f'Saving vector store to: {save_path}')
        vector_store.save_local(save_path)
        print('Vector store saved successfully')
        
    # Load vector store
    def load_vector_store(self, load_path: str = None) -> FAISS:
        """
        Load vector store from disk.
        
        Args:
            load_path: Path to load the vector store from
            
        Returns:
            FAISS vector store
        """
        
        if load_path is None:
            load_path = Config.VECTOR_STORE_PATH   
        
        print(f'Loading vector store from: {load_path}')
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f'Vector store not found at: {load_path}')
        
        vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True # 
        )
        
        print('Vector store loaded successfully')
        return vector_store
    
    # Create retriever
    def create_retriever(self, vector_store: FAISS = None) -> VectorStoreRetriever:
        """
        Create a retriever from vector store.
        
        Args:
            vector_store: FAISS vector store (optional, loads if not provided)
            
        Returns:
            VectorStoreRetriever
        """
        
        # Load vector store
        if vector_store is None:
            vector_store = self.load_vector_store()
        
        print('Creating retriever.....')
        
        retirever = vector_store.as_retriever(
            search_type=Config.RETRIEVAL_TYPE,
            search_kwargs={'k': Config.RETRIEVAL_K}
        )   
    
        print(f'Retriever created with k= {Config.RETRIEVAL_K}')
        
        return retirever
    
    
# Test the vector store manager
if __name__=='__main__':
    print('Testing Vector Store Manager...')
    
    try:
        # Import here to avoid circular import
        from customer_support.modules.document_processor import DocumentProcessor
        
        #Create sample chunks
        processor = DocumentProcessor()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        pdf_path = os.path.join(project_root, 'data', 'safebank-manual.pdf')
        
        chunks = processor.process_pdf_file(pdf_path)
        
        # Create and save vector store
        vs_manager = VectorStoreManager()
        
        # Create vector store
        vector_store = vs_manager.create_vector_store(chunks) # Use all chunks
        
        # Save vector store
        vs_manager.save_vector_store(vector_store, 'test_faiss_index')
        
        # Load vector store
        loaded_vs = vs_manager.load_vector_store('test_faiss_index')
        
        # Create retriever
        retriever = vs_manager.create_retriever(loaded_vs)
        
        #Test retrieval
        test_query = 'How to change password?'
        results  = retriever.invoke(test_query)
        
        print(f"\nTest query: '{test_query}'")
        print(f'Retrieved {len(results)} documents')
        print(f'First result snippets: {results[0].page_content}')
        
        # Cleanup test directory
        if os.path.exists('test_faiss_index'):
            shutil.rmtree('test_faiss_index')
            print('\nCleaned up test direcotry')
            
    except Exception as e:
        print(f'Error in vector store test: {e}')