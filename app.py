# app.py
# Streamlit RAG-powered Research Paper Assistant
# Improved visualization, caching, and correct arXiv parsing

import os
import hashlib
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from litellm import completion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from dotenv import load_dotenv
from huggingface_hub import login

# Streamlit configuration must come first
st.set_page_config(page_title="RAG Research Assistant", layout="wide")

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Hugging Face login
if huggingface_token:
    login(token=huggingface_token)

# Cache heavy resources
@st.cache_resource
def load_text_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./database/chroma_db")

text_embedding_model = load_text_embedding_model()
client = get_chroma_client()
# Use API wrapper to get structured docs
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, load_max_docs=3, load_all_available_meta=True)

# Utility: hash text
def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# Extract text from uploaded PDFs
def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for f in uploaded_files:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text

# Process and store text in ChromaDB
def process_text_and_store(all_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(all_text)
    print(chunks)
    collection = client.get_or_create_collection(name="knowledge_base")

    # Clear existing docs
    existing = collection.get()
    if existing.get("ids"):
        collection.delete(ids=existing["ids"])

    # Add new chunks
    for i, chunk in enumerate(chunks):
        emb = text_embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[emb],
            metadatas=[{"source": "text", "chunk_id": i}],
            documents=[chunk]
        )
    st.session_state.collection = collection
    return collection

# Semantic search
def semantic_search(query: str, collection, top_k: int = 3):
    q_emb = text_embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return results

# Generate response via LLM
def generate_response(query: str, context: str) -> str:
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"  
    resp = completion(
        model="gemini/gemini-1.5-flash",
        messages=[{"role": "user", "content": prompt}],
        api_key=gemini_api_key
    )
    return resp['choices'][0]['message']['content']

# Streamlit UI
st.title("🧠 RAG-powered Research Paper Assistant")

# Sidebar for inputs
with st.sidebar:
    st.header("Options")
    mode = st.radio("Mode", ["Upload PDFs", "Search arXiv"] )
    st.markdown("---")

if mode == "Upload PDFs":
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        if st.button("Process PDFs"):
            with st.spinner("Extracting and indexing PDF content..."):
                text = extract_text_from_pdfs(uploaded)
                # avoid reprocessing same text
                text_hash = get_text_hash(text)
                if st.session_state.get("last_hash") != text_hash:
                    collection = process_text_and_store(text)
                    st.session_state.last_hash = text_hash
                    st.success("PDF content indexed.")
                else:
                    collection = st.session_state.collection
                    st.info("Content unchanged. Using cached index.")
        if st.session_state.get("collection"):
            query = st.text_input("Enter your query:")
            if st.button("Get Answer") and query:
                with st.spinner("Running semantic search and generating answer..."):
                    results = semantic_search(query, st.session_state.collection)
                    docs = results['documents'][0]
                    # Show contexts in expander
                    for idx, doc in enumerate(docs):
                        with st.expander(f"Context chunk {idx}"):
                            st.write(doc)
                    answer = generate_response(query, "\n".join(docs))
                    st.subheader("Answer")
                    st.markdown(answer)

else:
    search_q = st.text_input("Search arXiv for papers:")
    if st.button("Search"): 
        with st.spinner("Querying arXiv..."):
            docs = arxiv_wrapper.load(search_q)
            # Display results in table
            rows = []
            for doc in docs:
                meta = doc.metadata
                rows.append({
                    "Title": meta.get("title"),
                    "Authors": ", ".join(meta.get("authors", [])),
                    "Published": meta.get("published"),
                    "ID": meta.get("entry_id") or meta.get("id")
                })
            st.dataframe(rows)
            # index concatenated content
            full_text = "\n".join([d.page_content for d in docs])
            process_text_and_store(full_text)
            st.success("Paper abstracts indexed.")
    if st.session_state.get("collection"):
        q2 = st.text_input("Ask a question about these papers:")
        if st.button("Get Paper Answer") and q2:
            with st.spinner("Generating answer from paper context..."):
                res = semantic_search(q2, st.session_state.collection)
                docs = res['documents'][0]
                for idx, doc in enumerate(docs):
                    with st.expander(f"Context chunk {idx}"):
                        st.write(doc)
                ans = generate_response(q2, "\n".join(docs))
                st.subheader("Paper Answer")
                st.markdown(ans)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, ChromaDB, SentenceTransformers, and Gemini LLM.")