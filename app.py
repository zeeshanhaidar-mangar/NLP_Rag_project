"""
DocuMind AI - Intelligent Document Q&A Assistant
SIMPLIFIED VERSION - Uses OpenAI-compatible API format
"""

import streamlit as st
import os
from datetime import datetime
import re

# Core imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from PyPDF2 import PdfReader
    import google.generativeai as genai
    from docx import Document as DocxDocument
    import pandas as pd
    import pdfplumber
    import requests
except ImportError as e:
    st.error(f"⏳ Installing dependencies... Refresh in 1 minute.")
    st.stop()

# Page config
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Session state
def init_session_state():
    defaults = {
        'initialized': False,
        'embedding_model': None,
        'document_store': [],
        'metadata_store': [],
        'faiss_index': None,
        'question_history': [],
        'api_configured': False,
        'api_key': '',
        'stats': {
            'total_documents': 0,
            'total_chunks': 0,
            'total_queries': 0,
            'avg_confidence': 0.0
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Clean text
def clean_text(text):
    if not text:
        return ""
    text = text.replace('\x00', '')
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Chunk text
def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        start += size - overlap
    return chunks

# Process PDF
def process_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        if text.strip():
            return text, "success", len(reader.pages)
    except:
        pass
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text, "success", len(pdf.pages)
    except Exception as e:
        return "", f"error: {e}", 0

# Process DOCX
def process_docx(file):
    try:
        doc = DocxDocument(file)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text, "success", len(doc.paragraphs)
    except Exception as e:
        return "", f"error: {e}", 0

# Process TXT
def process_txt(file):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            file.seek(0)
            text = file.read().decode(enc)
            return text, "success", len(text)
        except:
            continue
    return "", "error: encoding", 0

# Process CSV
def process_csv(file):
    try:
        df = pd.read_csv(file)
        text = f"CSV: {len(df)} rows\nColumns: {', '.join(df.columns)}\n\n"
        text += df.to_string()
        return text, "success", len(df)
    except Exception as e:
        return "", f"error: {e}", 0

# Main processor
def process_document(file):
    ext = file.name.split('.')[-1].lower()
    processors = {
        'pdf': process_pdf,
        'docx': process_docx,
        'txt': process_txt,
        'csv': process_csv
    }
    processor = processors.get(ext)
    if processor:
        return processor(file)
    return "", f"Unsupported: {ext}", 0

# Test API with direct REST call
def test_api_key(api_key):
    """Test API key using direct REST API call"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": "Hello"
                }]
            }]
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result:
                return True, "Success", result['candidates'][0]['content']['parts'][0]['text']
        
        # Handle errors
        if response.status_code == 400:
            return False, "Invalid API key format", None
        elif response.status_code == 403:
            return False, "API not enabled or billing issue", None
        elif response.status_code == 429:
            return False, "Quota exceeded", None
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            return False, f"Error {response.status_code}: {error_msg}", None
            
    except requests.exceptions.Timeout:
        return False, "Request timeout - check internet connection", None
    except Exception as e:
        return False, f"Connection error: {str(e)}", None

# Generate summary using REST API
def generate_summary(text, filename):
    try:
        if not st.session_state.api_configured:
            return "API not configured"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={st.session_state.api_key}"
        
        data = {
            "contents": [{
                "parts": [{
                    "text": f"Summarize in 2-3 sentences:\n{text[:2000]}"
                }]
            }]
        }
        
        response = requests.post(url, json=data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        return "Summary unavailable"
    except:
        return "Summary unavailable"

# Search
def search(query, k=5):
    if not st.session_state.faiss_index or not st.session_state.document_store:
        return [], [], []
    
    q_emb = st.session_state.embedding_model.encode([query]).astype('float32')
    distances, indices = st.session_state.faiss_index.search(q_emb, k)
    
    max_dist = np.max(distances[0]) if len(distances[0]) > 0 else 1
    similarities = 1 - (distances[0] / (max_dist + 1e-6))
    
    chunks = [st.session_state.document_store[i] for i in indices[0] if i < len(st.session_state.document_store)]
    metadata = [st.session_state.metadata_store[i] for i in indices[0] if i < len(st.session_state.metadata_store)]
    scores = [float(s) for s in similarities]
    
    return chunks, metadata, scores

# Generate answer using REST API
def generate_answer(query, chunks, metadata):
    try:
        if not st.session_state.api_configured:
            return "⚠️ Configure API key in sidebar", 0
        
        if not chunks:
            return "❌ No relevant info found. Upload documents first.", 0
        
        context = "\n\n".join([f"[{m['source']}]\n{c}" for c, m in zip(chunks, metadata)])
        
        prompt = f"""Answer based ONLY on context. Cite sources.

Context:
{context}

Question: {query}

Answer:"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={st.session_state.api_key}"
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['candidates'][0]['content']['parts'][0]['text']
            confidence = np.mean([m.get('score', 0) for m in metadata]) * 100
            return answer, confidence
        else:
            return f"Error generating answer: {response.status_code}", 0
            
    except Exception as e:
        return f"Error: {str(e)}", 0

# Sidebar
with st.sidebar:
    st.markdown("# 📚 DocuMind AI")
    st.markdown("---")
    
    st.markdown("### 🔑 API Configuration")
    with st.expander("ℹ️ How to get API key"):
        st.markdown("""
        **Method 1: Google AI Studio (Recommended)**
        1. Visit: https://aistudio.google.com/app/apikey
        2. Click "Create API key"
        3. Select "Create API key in new project"
        4. Copy the key
        
        **Method 2: Google Cloud Console**
        1. Go to: https://console.cloud.google.com/
        2. Enable "Generative Language API"
        3. Create credentials → API key
        """)
    
    api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=st.session_state.api_key,
        placeholder="AIzaSy..."
    )
    
    if api_key and len(api_key) > 20:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔌 Test & Configure", type="primary"):
                with st.spinner("Testing API key..."):
                    # Use REST API instead of SDK
                    success, message, response_text = test_api_key(api_key)
                    
                    if success:
                        st.session_state.api_key = api_key
                        st.session_state.api_configured = True
                        st.success("✅ API key working!")
                        st.info(f"Test response: {response_text[:50]}...")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {message}")
                        
                        # Specific troubleshooting
                        if "not enabled" in message or "403" in message:
                            st.warning("**Fix:**")
                            st.markdown("1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
                            st.markdown("2. Click **ENABLE**")
                            st.markdown("3. Wait 2 minutes")
                            st.markdown("4. Try again")
                        elif "Invalid" in message or "400" in message:
                            st.warning("**Fix:**")
                            st.markdown("1. Get NEW key from: https://aistudio.google.com/app/apikey")
                            st.markdown("2. Click 'Create API key in NEW project'")
                            st.markdown("3. Copy FULL key")
                        elif "Quota" in message:
                            st.warning("Wait 10 minutes or enable billing")
        
        with col2:
            st.metric("Len", len(api_key))
    
    if st.session_state.api_configured:
        st.success("✅ Connected")
        if st.button("🔄 Reset"):
            st.session_state.api_key = ''
            st.session_state.api_configured = False
            st.rerun()
    
    st.markdown("---")
    page = st.radio("📍 Navigation", ["🏠 Home", "📤 Upload", "❓ Ask", "📊 Stats"])
    
    st.markdown("---")
    st.markdown("### 📈 Stats")
    st.metric("Docs", st.session_state.stats['total_documents'])
    st.metric("Chunks", st.session_state.stats['total_chunks'])
    st.metric("Queries", st.session_state.stats['total_queries'])

# Initialize model
if not st.session_state.initialized:
    with st.spinner("🔄 Loading..."):
        st.session_state.embedding_model = load_model()
        st.session_state.initialized = True

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<div class="main-header">📚 DocuMind AI</div>', unsafe_allow_html=True)
    st.markdown("### *Intelligent Document Q&A Assistant*")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📤 Upload")
        st.write("• PDF, DOCX, TXT, CSV")
        st.write("• Auto extraction")
        st.write("• Smart chunking")
    
    with col2:
        st.markdown("#### 🤖 AI Answers")
        st.write("• Natural language")
        st.write("• Source citations")
        st.write("• Confidence scores")
    
    with col3:
        st.markdown("#### 📊 Analytics")
        st.write("• Query history")
        st.write("• Statistics")
        st.write("• Performance")
    
    st.markdown("---")
    
    st.info("👈 **Start:** Configure API key in sidebar, then upload documents!")

# UPLOAD PAGE
elif page == "📤 Upload":
    st.title("📤 Upload Documents")
    
    files = st.file_uploader("Choose files", type=['pdf', 'docx', 'txt', 'csv'], accept_multiple_files=True)
    
    if st.button("🚀 Process", type="primary", disabled=not files):
        if not files:
            st.error("No files selected")
        else:
            progress = st.progress(0)
            
            for i, file in enumerate(files):
                with st.expander(f"📄 {file.name}"):
                    status = st.empty()
                    status.info("Processing...")
                    
                    try:
                        text, result, info = process_document(file)
                        
                        if "error" in result:
                            status.error(f"❌ {result}")
                            continue
                        
                        text = clean_text(text)
                        
                        if len(text) < 100:
                            status.warning("Too short")
                            continue
                        
                        summary = generate_summary(text, file.name)
                        chunks = chunk_text(text)
                        
                        if not chunks:
                            status.warning("No chunks")
                            continue
                        
                        embeddings = st.session_state.embedding_model.encode(chunks).astype('float32')
                        
                        if st.session_state.faiss_index is None:
                            st.session_state.faiss_index = faiss.IndexFlatL2(384)
                        
                        st.session_state.faiss_index.add(embeddings)
                        st.session_state.document_store.extend(chunks)
                        
                        metadata = [{'source': file.name, 'score': 0} for _ in chunks]
                        st.session_state.metadata_store.extend(metadata)
                        
                        st.session_state.stats['total_documents'] += 1
                        st.session_state.stats['total_chunks'] += len(chunks)
                        
                        status.success("✅ Success!")
                        st.write(f"Chunks: {len(chunks)}")
                        st.write(f"Summary: {summary}")
                    
                    except Exception as e:
                        status.error(f"Error: {e}")
                
                progress.progress((i + 1) / len(files))
            
            st.balloons()

# ASK PAGE
elif page == "❓ Ask":
    st.title("❓ Ask Questions")
    
    if not st.session_state.api_configured:
        st.error("⚠️ Configure API key first")
        st.stop()
    
    if not st.session_state.document_store:
        st.warning("⚠️ Upload documents first")
        st.stop()
    
    question = st.text_area("Your Question:", placeholder="What would you like to know?")
    
    if st.button("🔍 Get Answer", type="primary"):
        if len(question) < 3:
            st.error("Question too short")
        else:
            with st.spinner("Thinking..."):
                chunks, metadata, scores = search(question, k=5)
                
                if not chunks:
                    st.warning("No relevant info found")
                    st.stop()
                
                for i, score in enumerate(scores):
                    if i < len(metadata):
                        metadata[i]['score'] = score
                
                answer, confidence = generate_answer(question, chunks, metadata)
                
                st.session_state.question_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'question': question,
                    'answer': answer,
                    'confidence': confidence
                })
                
                st.session_state.stats['total_queries'] += 1
                
                st.markdown("---")
                st.markdown("### 💬 Answer")
                
                if confidence >= 70:
                    st.success(f"🟢 {confidence:.1f}%")
                elif confidence >= 40:
                    st.warning(f"🟡 {confidence:.1f}%")
                else:
                    st.error(f"🔴 {confidence:.1f}%")
                
                st.write(answer)
                
                st.markdown("### 📚 Sources")
                sources = list(set([m['source'] for m in metadata]))
                for i, s in enumerate(sources, 1):
                    st.write(f"{i}. {s}")

# STATS PAGE
elif page == "📊 Stats":
    st.title("📊 Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", st.session_state.stats['total_documents'])
    col2.metric("Chunks", st.session_state.stats['total_chunks'])
    col3.metric("Queries", st.session_state.stats['total_queries'])
    
    st.markdown("---")
    st.markdown("### 📜 History")
    
    if st.session_state.question_history:
        for i, entry in enumerate(reversed(st.session_state.question_history[-10:]), 1):
            with st.expander(f"Q{i}: {entry['question'][:50]}..."):
                st.write(f"**Time:** {entry['timestamp']}")
                st.write(f"**Confidence:** {entry['confidence']:.1f}%")
                st.write(f"**Answer:** {entry['answer'][:300]}...")
    else:
        st.info("No questions yet")

st.markdown("---")
st.caption("DocuMind AI © 2024")
