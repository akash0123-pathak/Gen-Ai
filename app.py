# app.py - Single-file RAG PDF Chatbot with TinyLlama support
import os
import tempfile
from typing import List, Optional

import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------
# USER PROVIDED HUGGINGFACE TOKEN
# ---------------------------
# Put your token here (you asked to embed it in the file)
HF_TOKEN = "  "
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ---------------------------
# CONFIG
# ---------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = "./persist"   # internal only; not shown to user
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Model options (sidebar). TinyLlama included as requested.
MODEL_OPTIONS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "bigscience/bloom-560m",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2"
]

# ---------------------------
# UTILITIES
# ---------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = getattr(uploaded_file, "name", ".pdf")
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(suffix)[1] or ".pdf")
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(uploaded_file.read())
    return path

# ---------------------------
# PDF -> chunks
# ---------------------------
def load_pdf_and_split(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    return split_docs

# ---------------------------
# Embeddings + FAISS
# ---------------------------
def get_embeddings(model_name: str = EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)

def create_faiss_store(docs: List[Document], embeddings, persist_dir: str = DEFAULT_PERSIST_DIR):
    vs = FAISS.from_documents(docs, embeddings)
    ensure_dir(persist_dir)
    vs.save_local(persist_dir)
    return vs

def load_faiss_store(embeddings, persist_dir: str = DEFAULT_PERSIST_DIR):
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

# ---------------------------
# LLM loader (loads tokenizer + model with token, returns a transformers pipeline)
# ---------------------------
def load_text_generation_pipeline(model_name: str):
    """
    Load tokenizer and model using the HF token (from env var).
    Return a transformers text-generation pipeline object.
    """
    # pass token when downloading model files
    # prefer token argument for newer HF versions but use use_auth_token for compatibility
    tok_kwargs = {"use_auth_token": HF_TOKEN}
    model_kwargs = {"use_auth_token": HF_TOKEN}

    # When GPU is available, allow device_map for larger models
    if torch.cuda.is_available():
        model_kwargs.update({"device_map": "auto"})
    else:
        # try to reduce memory usage on CPU
        model_kwargs.update({"low_cpu_mem_usage": True})

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **model_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        top_p=0.92,
        temperature=0.7,
        repetition_penalty=1.05,
        max_new_tokens=256,   # keep answers short
    )

    return pipe

# ---------------------------
# Prompt and helpers
# ---------------------------
BASE_PROMPT = """Use ONLY the CONTEXT below to answer the QUESTION concisely.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS FOR THE MODEL (DO NOT SHOW IN THE ANSWER):
- Provide a complete answer based strictly on the context.
- Write a clear, coherent explanation.
- Use proper grammar and punctuation.
- Do not use emojis, slang, memes, abbreviations, or misspellings.
- Do not mention that you are following rules or instructions.
- Adjust the length of the answer according to the question.
- If the context does not contain the answer, respond with: "I do not have enough information."
"""



def shorten_context(text: str, max_chars: int = 3000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # keep the most recent portion
    return text[-max_chars:]

# ---------------------------
# Retrieval + generation
# ---------------------------
def answer_query(query: str, model_name: str, top_k: int, chat_history: List[dict]):
    # embeddings / vectorstore
    embeddings = get_embeddings()
    retrieved_docs = []
    context_parts = []

    if os.path.exists(DEFAULT_PERSIST_DIR):
        try:
            vs = load_faiss_store(embeddings, DEFAULT_PERSIST_DIR)
            retrieved_docs = vs.similarity_search(query, k=top_k)
            # take snippets (limit length) from retrieved docs
            for d in retrieved_docs:
                snippet = d.page_content.strip().replace("\n", " ")
                context_parts.append(snippet[:800])  # per-chunk truncation
        except Exception as e:
            print("FAISS load/search failed:", e)
            retrieved_docs = []

    # include recent chat history (last 3 turns) as additional context lines
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # store as dicts of q/a, take last few entries
        lines = []
        for t in recent:
            q = t.get("query", "")
            a = t.get("answer", "")
            # keep short
            lines.append(f"previous Q: {q}")
            lines.append(f"previous A: {a}")
        history_text = "\n".join(lines)[:1000]
        if history_text:
            context_parts.insert(0, history_text)

    # combined context and shorten
    combined_context = "\n\n".join(context_parts)
    combined_context = shorten_context(combined_context, max_chars=3000)

    # build prompt
    prompt_text = BASE_PROMPT.format(context=combined_context, question=query)

    # load pipeline (lazy)
    pipe = load_text_generation_pipeline(model_name)

    # generate
    out = pipe(prompt_text, max_new_tokens=128)
    # pipeline returns list of dicts with 'generated_text' or 'text' depending on tokenizer
    generated = ""
    if out and isinstance(out, list):
        first = out[0]
        generated = first.get("generated_text") or first.get("text") or str(first)
    else:
        generated = str(out)

    # cleanup generated text: remove the prompt if echoed (some models repeat prompt)
    if isinstance(generated, str) and prompt_text.strip() in generated:
        # try to cut after the prompt
        idx = generated.find(prompt_text.strip())
        if idx != -1:
            generated = generated[idx + len(prompt_text.strip()):].strip()

    # final short enforcement
    if len(generated) > 800:
        generated = generated[:797].rstrip() + "..."

    return generated.strip(), retrieved_docs

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
    st.title("📄 RAG PDF Chatbot (TinyLlama ready)")

    # sidebar controls
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("Choose model", MODEL_OPTIONS, index=0)
        top_k = st.number_input("Retriever top_k", min_value=1, max_value=10, value=4)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=5000, value=DEFAULT_CHUNK_SIZE)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=DEFAULT_CHUNK_OVERLAP)
        st.markdown("---")
        st.markdown("Note: large models may require GPU and more memory")

    ensure_dir(DEFAULT_PERSIST_DIR)

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of dicts {query, answer}

    st.info("Upload a PDF to build the knowledge base or leave empty to use the model without documents")

    upload = st.file_uploader("Upload PDF", type=["pdf"])
    if upload:
        st.info("Processing PDF. This may take a while for the first time.")
        tmp = save_uploaded_file_to_temp(upload)
        try:
            docs = load_pdf_and_split(tmp, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            embeddings = get_embeddings()
            create_faiss_store(docs, embeddings, persist_dir=DEFAULT_PERSIST_DIR)
            st.success(f"Processed PDF. Chunks: {len(docs)}")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    query = st.text_input("Ask a question about the uploaded PDF")

    if st.button("Send") and query:
        with st.spinner("Retrieving and generating answer..."):
            answer, sources = answer_query(query, model_choice, int(top_k), st.session_state.chat_history)

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Retrieved snippets")
            for i, s in enumerate(sources, 1):
                snippet = s.page_content.replace("\n", " ").strip()
                st.markdown(f"**{i}.** {snippet[:400]}{'...' if len(snippet) > 400 else ''}")

        # record chat history
        st.session_state.chat_history.append({"query": query, "answer": answer})

    # show chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat history (this session)")
        for turn in reversed(st.session_state.chat_history):
            st.write(f"**Q:** {turn['query']}")
            st.write(f"**A:** {turn['answer']}")
            st.write("---")

if __name__ == "__main__":
    main()
