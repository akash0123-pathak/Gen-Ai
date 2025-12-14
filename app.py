import streamlit as st
import pdfplumber
import os
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# ==================== MODELS ====================
# CPU-compatible embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Free public QA model (Flan-T5 large, CPU)
qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1  # -1 forces CPU
)

# ==================== FILE STORAGE ====================
DB_FILE = "faiss_vectors.index"
CHUNKS_FILE = "pdf_chunks.pkl"

# ==================== STREAMLIT APP ====================
st.title("📘 PDF RAG Chatbot (CPU-Compatible & Free)")
st.caption("Upload a PDF → Ask Questions → AI answers based on your document")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Utility Functions --------------------
def save_vector_db(index, chunks):
    faiss.write_index(index, DB_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_db():
    if os.path.exists(DB_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(DB_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None

# -------------------- PDF Upload --------------------
uploaded_pdf = st.file_uploader("📄 Upload PDF File", type="pdf")

if uploaded_pdf:
    text = ""
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = embedding_model.encode(chunks)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save FAISS index and chunks
    save_vector_db(index, chunks)
    st.success("PDF processed successfully! Now ask your questions below 👇")

# -------------------- QA --------------------
query = st.text_input("Ask a question from the uploaded PDF")

if st.button("Get Answer"):
    index, chunks = load_vector_db()
    
    if not index:
        st.error("⚠ Please upload and process a PDF first!")
    else:
        # Embed the query
        q_embed = embedding_model.encode([query])
        _, I = index.search(np.array(q_embed).astype("float32"), k=3)

        # Retrieve top chunks
        retrieved = " ".join([chunks[i] for i in I[0]])

        # Construct prompt for QA model
        prompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"
        response = qa_model(prompt)[0]["generated_text"]

        # Append to chat history
        st.session_state.history.append((query, response))

        st.subheader("🧠 AI Answer:")
        st.write(response)

# -------------------- Show Chat History --------------------
if st.session_state.history:
    st.write("---")
    st.subheader("💬 Chat History")
    for q, a in st.session_state.history:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
