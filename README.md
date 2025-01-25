# Conversational App for Report Analysis

This repository contains an **AI-powered conversational application** designed to help users extract, analyze, and query information from **PDFs** and **HTML files**. It uses **Generative AI (Google Gemini)** and **semantic search (FAISS)** to provide detailed answers to user queries, complete with **source citations**.

---

## Features

### 📄 Document Ingestion
- **PDF Parsing**:
  - Extracts text using **pdfplumber**.
  - Uses **OCR fallback** for scanned documents via **Tesseract**.
- **HTML Parsing**:
  - Extracts visible text from HTML files using **BeautifulSoup**.
- **Chunking**:
  - Splits extracted text into manageable chunks with overlap for efficient semantic retrieval.

### 🔍 Semantic Search
- Built on **FAISS** for fast and efficient vector-based search.
- Embedding generation powered by **SentenceTransformer** (`all-MiniLM-L6-v2`).
- Retrieves the most relevant document chunks for each query.

### 🤖 Generative AI (Gemini)
- Integrates **Google Gemini** for conversational AI responses.
- Answers queries using retrieved document chunks as context.
- Provides **source citations** (e.g., document name and page number).

### 🧠 Multi-Turn Conversations
- Tracks conversation history to enable **context-aware follow-up questions**.
- Dynamically builds prompts based on prior user input and retrieved information.

### ⚡ Streamlit Interface
- Interactive and user-friendly interface:
  - Upload **PDF** and **HTML files** for analysis.
  - Ask questions and view responses.
  - Displays the complete conversation history for clarity.

---

## How It Works

1. **Upload Documents**:
   - Place your **PDF** and **HTML** files in the `documents` folder.
   - The app extracts, processes, and indexes text for semantic retrieval.

2. **Semantic Retrieval**:
   - Uses **FAISS** and **SentenceTransformer** embeddings to retrieve the most relevant document chunks for a given query.

3. **Conversational Queries**:
   - Enter your questions in the Streamlit interface.
   - The app retrieves relevant document chunks, builds a conversational prompt, and generates answers using **Gemini**.

4. **Citations**:
   - Every response includes citations (source filename and page number).

---

## Setup and Installation

### Prerequisites
1. **Python 3.9+**
2. **Tesseract OCR**:
   - **Ubuntu**: `sudo apt install tesseract-ocr`
   - **Mac**: `brew install tesseract`
   - **Windows**: [Download Tesseract](https://github.com/tesseract-ocr/tesseract)
3. **Google Gemini API Key**:
   - Obtain an API key from Google Generative AI.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conversational-report-analysis.git
   cd conversational-report-analysis
