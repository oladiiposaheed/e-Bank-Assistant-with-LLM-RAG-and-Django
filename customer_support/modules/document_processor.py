import os
import sys
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) # customer_support/modules
modules_dir = current_dir # customer_support/modules/
customer_support_dir = os.path.dirname(modules_dir) # customer_support/
project_root = os.path.dirname(customer_support_dir) # smart_customer_support/

# Add project root to Python path
sys.path.insert(0, project_root)

# Import Config - use absolute import
from customer_support.modules.config import Config
    
class DocumentProcessor:
    """Handles document for loading and text splitting."""
    
    def __init__(self):
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = Config.CHUNK_SIZE,
            chunk_overlap = Config.CHUNK_OVERLAP
        )
        
    def load_pdf_documents(self, file_path: str) -> List[Document]:
        
        """
        Load documents from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        print(f'Loading PDF from: {file_path}')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'PDF file not found: {file_path}')
    
        # Load PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        print(f'Loaded {len(documents)} pages from PDF')
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[str]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of text chunks
        """
        print('Splitting documents into chunks...')
        
        # Extract text from documents
        all_texts = [doc.page_content for doc in documents]
        combined_text = '\n'.join(all_texts)
        
        #Split the text
        chunks = self.text_splitter.split_text(combined_text)
        
        print(f'Created {len(chunks)} text chunks')
        return chunks
    
    def process_pdf_file(self, pdf_path: str) -> List[str]:
        """
        Complete pipeline: Load PDF and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text chunks
        """
        # Load documents
        documents = self.load_pdf_documents(pdf_path)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Display sample chunk
        if chunks:
            print(f'\n Sample chunk (first 300 characters):')
            print(chunks[0][:300] + '...')
            
        return chunks
    
# Test the document processor
if __name__=='__main__':
    
    print("Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct PDF path
    #pdf_path = os.path.join(Config.DATA_PATH, Config.PDF_FILE)
    pdf_path = os.path.join(project_root, 'data', 'safebank-manual.pdf')
    
    try:
        chunks = processor.process_pdf_file(pdf_path)
        print(f'\n Document processing completed successfully!')
    
    except Exception as e:
        print(f'Error processing document: {e}')
        