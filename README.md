🏦 SafeBank RAG Assistant

A high performance Retrieval Augmented Generation (RAG) system built with Django, LangChain, FAISS, and Groq LLMs, designed to answer questions from the SafeBank Manual with accuracy, speed, and reliability.
________________________________________

📌 Overview

SafeBank RAG Assistant transforms the SafeBank Manual PDF into an intelligent, searchable knowledge engine.

It uses:

•	LangChain for chunking, embeddings, and retrieval

•	FAISS for fast vector similarity search

•	Groq LLMs for ultra fast inference

•	Django + Bootstrap for a clean, responsive UI

This project demonstrates a production ready RAG pipeline suitable for banking, compliance, customer support, and internal knowledge systems.

________________________________________

🚀 Features

•	Upload and index PDF documents

•	Automatic text extraction using PyPDF + PyMuPDF

•	Smart chunking with LangChain text splitters

•	Embedding generation using HuggingFace or Groq models

•	Vector search powered by FAISS

•	Grounded LLM responses using Groq

•	Django admin panel for managing documents

•	Clean Bootstrap UI for chat and document management

________________________________________

🧠 Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS Index → Retrieval → Groq LLM → Final Answer

🧪 How It Works

✅ PDF Processing

•	Extracts text using PyPDF + PyMuPDF

•	Cleans and normalizes content

•	Splits into overlapping chunks

✅ Embeddings

•	Uses HuggingFace or Groq embedding models

•	Converts chunks into vector representations

✅ Vector Search

•	FAISS performs fast similarity search

•	Retrieves the most relevant chunks

✅ LLM Response

•	Groq LLM generates grounded answers

•	Ensures responses stay aligned with the SafeBank Manual
