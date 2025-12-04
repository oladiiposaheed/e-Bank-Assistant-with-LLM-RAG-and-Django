from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import QueryForm
from .models import QueryHistory
import sys
import os

# Add the project root to Python path
# Get the absolute path to smart_customer_support
current_file = os.path.abspath(__file__)  # web_app/views.py
djrag_project_dir = os.path.dirname(os.path.dirname(current_file))  # djrag_project/
smart_customer_support_dir = os.path.dirname(djrag_project_dir)  # smart_customer_support/

# Add both paths to sys.path
sys.path.insert(0, smart_customer_support_dir)  # For customer_support module
sys.path.insert(0, djrag_project_dir)  # For Django project

def index(request):
    """Home page."""
    return render(request, 'web_app/index.html')

def query_view(request):
    """Handle user queries."""
    if request.method == 'POST':
        form = QueryForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            
            try:
                # Import RAG pipeline
                from customer_support.modules.rag_pipeline import RAGPipeline
                from customer_support.modules.vector_store import VectorStoreManager
                from customer_support.modules.document_processor import DocumentProcessor
                
                # Initialize pipeline
                processor = DocumentProcessor()
                vs_manager = VectorStoreManager()
                
                # Load PDF - adjust path for Django
                pdf_path = os.path.join(smart_customer_support_dir, 'data', 'safebank-manual.pdf')
                
                chunks = processor.process_pdf_file(pdf_path)
                vector_store = vs_manager.create_vector_store(chunks[:30])  # Limit for faster response
                retriever = vs_manager.create_retriever(vector_store)
                rag = RAGPipeline(retriever)
                
                # Get answer
                result = rag.query(question)
                
                # Save to database
                QueryHistory.objects.create(
                    question=question,
                    answer=result['answer'][:1000]
                )
                
                return render(request, 'web_app/query.html', {
                    'form': form,
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'source_count': result['source_count']
                })
                
            except Exception as e:
                error_msg = str(e)
                print(f"DEBUG - Error: {error_msg}")
                print(f"DEBUG - sys.path: {sys.path}")
                print(f"DEBUG - Current dir: {os.getcwd()}")
                
                return render(request, 'web_app/query.html', {
                    'form': form,
                    'error': f"Error: {error_msg}",
                    'question': question
                })
    
    else:
        form = QueryForm()
    
    return render(request, 'web_app/query.html', {'form': form})