import os
import sys
from typing import Dict, List, Optional

# Setup imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from customer_support.modules.config import Config
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain


class RAGPipeline:
    """RAG pipeline for customer queries."""
    
    def __init__(self, retriever: VectorStoreRetriever):
        print("Initializing RAG Pipeline...")
        self.retriever = retriever
        self.llm = self._init_llm()
        self.chain = self._create_chain()
        print("RAG Pipeline ready")
    
    def _init_llm(self) -> ChatGroq:
        """Initialize LLM with Groq."""
        print(f"Loading LLM: {Config.MODEL_NAME}")
        return ChatGroq(
            model=Config.MODEL_NAME,
            temperature=Config.MODEL_TEMPERATURE,
            api_key=Config.GROQ_API_KEY
        )
    
    def _create_chain(self):
        """Create main RAG chain with chat history."""
        print("Building RAG chain...")
        
        # Context reformulation
        context_prompt = ChatPromptTemplate.from_messages([
            ('system', "Reformulate question as standalone if needed."),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])
        
        # History-aware retriever
        history_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=context_prompt
        )
        
        # Answer generation
        qa_prompt = ChatPromptTemplate.from_messages([
            ('system', "Answer using context. Be helpful and concise."),
            MessagesPlaceholder('chat_history'),
            ('human', 'Q: {input}\n\nContext: {context}')
        ])
        
        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Final chain
        return create_retrieval_chain(history_retriever, qa_chain)
    
    def _create_simple_chain(self):
        """Simple chain without chat history."""
        prompt = ChatPromptTemplate.from_messages([
            ('system', "Answer using context."),
            ('human', 'Q: {input}\n\nContext: {context}')
        ])
        
        return (
            {'context': self.retriever, 'input': RunnablePassthrough()}
            | prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def query(self, question: str, chat_history: Optional[List] = None) -> Dict:
        """Process user query."""
        chat_history = chat_history or []
        print(f"Query: {question[:40]}...")
        
        try:
            result = self.chain.invoke({
                'input': question, 
                'chat_history': chat_history
            })
            
            answer = result.get('answer', '')
            context = result.get('context', [])
            
            # Format sources
            sources = []
            for doc in context[:2]:
                sources.append({
                    'content': doc.page_content[:150] + '...',
                    'metadata': doc.metadata
                })
            
            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'source_count': len(context)
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                'question': question,
                'answer': self.simple_query(question),
                'sources': [],
                'source_count': 0
            }
    
    def simple_query(self, question: str) -> str:
        """Simple query without history."""
        try:
            return self._create_simple_chain().invoke(question)
        except Exception as e:
            return f"Error: {str(e)}"


def main():
    """Test the pipeline."""
    print("=" * 50)
    print("Testing RAG Pipeline")
    print("=" * 50)
    
    try:
        from customer_support.modules.vector_store import VectorStoreManager
        from customer_support.modules.document_processor import DocumentProcessor
        
        # Setup
        processor = DocumentProcessor()
        vs_manager = VectorStoreManager()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        pdf_path = os.path.join(project_root, 'data', 'safebank-manual.pdf')
        
        # Process PDF
        chunks = processor.process_pdf_file(pdf_path)
        vector_store = vs_manager.create_vector_store(chunks[:30])
        retriever = vs_manager.create_retriever(vector_store)
        
        # Initialize pipeline
        rag = RAGPipeline(retriever)
        
        # Test queries
        tests = [
            ("How to change password?", []),
            ("Security features?", [
                HumanMessage(content="What services?"),
                AIMessage(content="Banking services")
            ]),
            ("Contact support?", [])
        ]
        
        for i, (question, history) in enumerate(tests, 1):
            print(f"\nTest {i}: {question}")
            result = rag.query(question, history)
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Sources: {result['source_count']}")
        
        print("\n✅ Test completed")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()