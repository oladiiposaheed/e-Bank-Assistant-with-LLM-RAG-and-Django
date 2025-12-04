import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Model Configuration
    MODEL_NAME = 'llama-3.3-70b-versatile'
    MODEL_TEMPERATURE = 0.7
    EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Vector Store
    VECTOR_STORE_PATH = 'faiss_index_custom'
    
    # Retrieval Configuration
    RETRIEVAL_K = 6
    RETRIEVAL_TYPE = 'similarity'
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return True


# Test the config
if __name__ == "__main__":
    try:
        Config.validate_config()
        print("Configuration loaded successfully!")
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Embedding Model: {Config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Configuration error: {e}")