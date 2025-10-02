import streamlit as st
from pypdf import PdfReader
import pdfplumber
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import json

# Connect to local DeepSeek LLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

st.set_page_config(page_title="GB LLM with RAG", layout="wide")
st.title("🤖 Chat with GB Corp AI + RAG")
 
# --- Utility Functions ---
def chunk_text(text, max_words=800):
    """Split text into chunks of max_words words"""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def table_to_markdown(table):
    """Convert a pdfplumber table to markdown"""
    md_table = "| " + " | ".join(table[0]) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
    for row in table[1:]:
        md_table += "| " + " | ".join(cell if cell else "" for cell in row) + " |\n"
    return md_table

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📄 Upload a PDF (Document or People Knowledge)", type="pdf")
max_tokens = st.sidebar.slider("Max tokens per reply", 128, 1024, 256, step=64)
chunk_size = st.sidebar.slider("Words per chunk", 200, 2000, 800, step=100)

# --- Setup Vector DB (Chroma) ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("docs")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Process Uploaded PDF ---
pdf_text = ""
pdf_tables = []
if uploaded_file is not None:
    # Extract plain text
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    # Extract tables (if any)
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                pdf_tables.append(table_to_markdown(table))

    # Store chunks + tables in Chroma
    st.sidebar.info("Indexing document into vector DB...")
    try:
        chroma_client.delete_collection("docs")
    except:
        pass
    collection = chroma_client.create_collection("docs")

    # Add plain text chunks
    for i, chunk in enumerate(chunk_text(pdf_text, max_words=chunk_size)):
        emb = embedder.encode(chunk).tolist()
        collection.add(documents=[chunk], embeddings=[emb], ids=[f"text-{i}"])

    # Add extracted tables
    for j, table_md in enumerate(pdf_tables):
        emb = embedder.encode(table_md).tolist()
        collection.add(documents=[table_md], embeddings=[emb], ids=[f"table-{j}"])

    st.sidebar.success("✅ PDF indexed into ChromaDB (text + tables or people knowledge)")

# --- Chat History for non-PDF mode ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "system", "content": "You are a helpful assistant for general Q&A."}
    ]

for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask me anything (or request data extraction)..."):
    st.chat_message("user").write(prompt)

    if uploaded_file:  # RAG mode (PDF uploaded)
        with st.spinner("🔎 Retrieving relevant chunks..."):
            query_emb = embedder.encode(prompt).tolist()
            results = collection.query(query_embeddings=[query_emb], n_results=5)
            context = "\n\n".join(results["documents"][0])

        with st.spinner("🧠 Thinking with RAG..."):
            messages = [
                {"role": "system", "content": (
                    "You are an AI assistant that extracts structured data from documents. "
                    "If the PDF is about people/org structure, answer org-related questions. "
                    "If it's a normal document, extract facts, key-values, or tables. "
                    "Use JSON for structured data and Markdown for tables."
                )},
                {"role": "user", "content": f"Question: {prompt}\n\nRelevant document parts:\n{context}"}
            ]
            response = client.chat.completions.create(
                model="deepseek-7b",
                messages=messages,
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content
            st.chat_message("assistant").write(reply)

    else:  # Normal Chat mode (no PDF uploaded)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.spinner("🧠 Thinking..."):
            response = client.chat.completions.create(
                model="deepseek-7b",
                messages=st.session_state["chat_history"],
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)
