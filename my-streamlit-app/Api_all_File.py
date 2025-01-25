import os
import sys
import subprocess
import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TESSDATA_PREFIX"] = "/usr/share/tessdata"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

import google.generativeai as genai

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyBq0ByvTKMLd_qBmHboGAGrFcizR41WQyA"
genai.configure(api_key=GEMINI_API_KEY)

#########################
# Helper Functions
#########################

def chunk_text_with_overlap(text, max_length=1000, overlap=200):
    """
    Splits the input text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

def extract_text_from_pdf(pdf_path, ocr_threshold=300):
    """
    Extracts text from PDFs using pdfplumber, with OCR fallback for scanned pages.
    """
    page_texts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if len(text.strip()) < ocr_threshold:
                    pil_im = page.to_image(resolution=300).original.convert("RGB")
                    text = pytesseract.image_to_string(pil_im).strip()
                page_texts.append((i, text))
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return page_texts

def extract_text_from_html(html_path):
    """
    Extracts visible text from HTML files using BeautifulSoup.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n").strip()
            return text
    except Exception as e:
        print(f"Error processing HTML file {html_path}: {e}")
        return ""

#########################
# Data Ingestion
#########################

def ingest_documents(documents_folder="documents", max_length=1000, overlap=200, ocr_threshold=5):
    """
    Processes PDF and HTML files in the specified folder, extracting text and chunking it.
    """
    docs = []
    for filename in os.listdir(documents_folder):
        file_path = os.path.join(documents_folder, filename)
        if not os.path.isfile(file_path):
            continue

        # Handle PDF files
        if filename.lower().endswith(".pdf"):
            page_texts = extract_text_from_pdf(file_path, ocr_threshold=ocr_threshold)
            for page_num, page_text in page_texts:
                if page_text.strip():
                    chunks = chunk_text_with_overlap(page_text, max_length=max_length, overlap=overlap)
                    for chunk in chunks:
                        docs.append({
                            "text": chunk,
                            "metadata": {
                                "source": filename,
                                "page": page_num,
                                "chunk_type": "pdf_page"
                            }
                        })

        # Handle HTML files
        elif filename.lower().endswith((".html", ".htm")):
            text = extract_text_from_html(file_path)
            if text:
                chunks = chunk_text_with_overlap(text, max_length=max_length, overlap=overlap)
                for chunk in chunks:
                    docs.append({
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "page": "N/A",
                            "chunk_type": "html_body"
                        }
                    })

        else:
            print(f"Skipping unsupported file: {filename}")

    return docs

#########################
# Vector Store and Prompt Building
#########################

def create_vectorstore(chunks):
    """
    Creates a FAISS vector store from the given chunks using SentenceTransformer embeddings.
    """
    doc_objs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    if not doc_objs:
        print("Warning: No document chunks created.")
        return None
    embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=doc_objs, embedding=embedding)
    return vectorstore

def build_prompt(conversation_history, retrieved_docs):
    """
    Combines conversation history with retrieved document chunks into a prompt.
    """
    conv_strs = [f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in conversation_history]
    conversation_str = "\n".join(conv_strs)

    context_strs = [
        f"(SEMANTIC CONTEXT) [Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]"
        f":\n{doc.page_content.strip()}\n"
        for doc in retrieved_docs
    ]
    semantic_context_str = "\n".join(context_strs)

    prompt = f"""You are a helpful assistant in extracting info from pdfs and html pages.

Here is the conversation so far:
{conversation_str}

Below is additional SEMANTIC CONTEXT:
{semantic_context_str}

The user's latest message is the last 'User:' in the conversation above.
Please answer their question using the context if relevant, citing (Source, Page) where appropriate.
Answer:
"""
    return prompt

#########################
# Streamlit Application
#########################

def main():
    st.title("Conversational App for Report Analysis (Inmeta Interview)")
    st.write("""
    This app extracts text from PDFs and HTML files, chunks it for semantic retrieval, 
    and enables multi-turn conversations by document context.
    """)

    if st.button("Close App"):
        sys.exit(0)

    # 1. Ingest documents
    if "docs" not in st.session_state:
        with st.spinner("Extracting and chunking documents..."):
            st.session_state["docs"] = ingest_documents("documents", max_length=500, overlap=200, ocr_threshold=300)

# 2. Build or load the FAISS vector store.
    if "vectorstore" not in st.session_state:
        with st.spinner("Creating vector store..."):
            vectorstore = create_vectorstore(st.session_state["docs"])
            if vectorstore is None:
                st.error("No document chunks were created from the PDFs.")
                return
            st.session_state["vectorstore"] = vectorstore
    
    # 3. Initialize conversation history.
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []
    
    # 4. Set up the Gemini model.
    # Replace this with your actual generative model configuration.
    if "gemini_model" not in st.session_state:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        st.session_state["gemini_model"] = gemini_model

    # 4. Handle user input
    user_input = st.text_input("Your message:")
    if user_input:
        st.session_state["conversation"].append({"role": "user", "content": user_input})
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 9})
        retrieved_docs = retriever.get_relevant_documents(user_input)
        prompt = build_prompt(st.session_state["conversation"], retrieved_docs)
        response = st.session_state["gemini_model"].generate_content(prompt)
        assistant_reply = response.text
        st.session_state["conversation"].append({"role": "assistant", "content": assistant_reply})

    # Display conversation
    st.write("---")
    st.subheader("Conversation History")
    for msg in st.session_state["conversation"]:
        if msg["role"] == "user":
            st.write(f"**User**: {msg['content']}")
        else:
            st.write(f"**Assistant**: {msg['content']}")

if __name__ == "__main__":
    main()
