# 😎 Brokade AI — Personal Document Analyser

A RAG (Retrieval-Augmented Generation) powered document chatbot 
that lets you upload and chat with your documents in real time.

## Features
-  Supports PDF, DOCX, and TXT file uploads
-  Semantic search using FAISS + Sentence Transformers
-  Multi-turn conversation with chat history
-  Multi-file upload and indexing in one session

## Tech Stack
Python · Streamlit · LangChain · FAISS · Sentence Transformers · Groq API · PyMuPDF

## Run Locally
1. Clone the repo
2. Create a `.env` file and add your Groq API key:
   GROQ_API_KEY=your_key_here
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   streamlit run app.py
