import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import pandas as pd

# -------------------
# App Config
# -------------------
st.set_page_config(page_title="AI PDF Analyzer", layout="wide")

# -------------------
# Sidebar
# -------------------
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to:", ["Home", "About"])

if tab == "About":
    st.title("About AI PDF Analyzer")
    st.write("""
    This project demonstrates AI capabilities:
    - Summarizes research papers or PDFs
    - Extracts key points
    - Tags main topics
    - Fully interactive web app
    Built using Python, Streamlit, and Hugging Face Transformers.
    """)
    st.stop()

# -------------------
# Load AI Model
# -------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -------------------
# Helper Functions
# -------------------
def chunk_text(text, max_words=500, overlap=100):
    """
    Splits text into chunks of max_words with optional overlap.
    """
    words = text.split()
    start = 0
    chunks = []
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap
    return chunks

def extract_key_points(text, num_points=10):
    sentences = text.split(". ")
    key_points = sentences[:num_points]  # first N sentences as simple key points
    return key_points

def extract_topics(text, top_n=10):
    words = text.lower().split()
    freq = pd.Series(words).value_counts().head(top_n)
    return freq

# -------------------
# File Upload
# -------------------
st.title("📄 AI PDF / Research Paper Analyzer")

uploaded_file = st.file_uploader("Upload a PDF to analyze", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
    st.success("Text extracted! ✅")

    # -------------------
    # Tabs
    # -------------------
    tabs = st.tabs(["Summary", "Key Points", "Topics"])

    # --- Summary Tab ---
    with tabs[0]:
        st.subheader("📑 Summary")
        with st.spinner("Generating detailed summary..."):
            chunks = chunk_text(text, max_words=500, overlap=100)
            chunk_summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=400, min_length=200, do_sample=False)
                chunk_summaries.append(summary[0]['summary_text'])
            summary_text = " ".join(chunk_summaries)
        st.write(summary_text)
        st.download_button("💾 Download Summary", summary_text, file_name="summary.txt")

    # --- Key Points Tab ---
    with tabs[1]:
        st.subheader("🔑 Key Points")
        key_points = extract_key_points(text, num_points=15)
        for idx, point in enumerate(key_points, 1):
            with st.expander(f"Point {idx}"):
                st.write(point)
        st.download_button("💾 Download Key Points", "\n".join(key_points), file_name="key_points.txt")

    # --- Topics Tab ---
    with tabs[2]:
        st.subheader(" Topics (Most Frequent Words)")
        freq = extract_topics(text, top_n=10)
        st.bar_chart(freq)




