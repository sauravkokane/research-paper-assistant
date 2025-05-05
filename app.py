# Import all dependencies
import os
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from litellm import completion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import ArxivQueryRun
from dotenv import load_dotenv
from huggingface_hub import login

#loading of environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")


# huggingface login
if huggingface_token:
    login(token=huggingface_token)

# configure and create vector database
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./database/chroma_db")

@st.cache_resource
def load_text_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

text_embedding_model = load_text_embedding_model()
client = get_chroma_client()

arxiv_tool = ArxivQueryRun()

# pdf to text - transcript fetch
def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text


# text processing - splitter + store in vectors
def process_text_and_store(all_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(all_text)

    try:
        collection = client.get_or_create_collection(name="knowledge_base")
        
        # Manually delete all existing documents
        existing = collection.get()
        if "ids" in existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception as e:
        st.error(f"Error accessing collection: {e}")
        return None

    for i, chunk in enumerate(chunks):
        embedding = text_embedding_model.encode(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"source": "pdf", "chunk_id": i}],
            documents=[chunk]
        )

    st.session_state["collection"] = collection
    return collection


# search in vector database - Retriver
def semantic_search(query, collection, top_k=3):
    query_embedding = text_embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k
    )
    return results

# create prompt and generate response - Augmentation + Generation
def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    response = completion(
        model="gemini/gemini-1.5-flash",
        messages=[{"content": prompt, "role": "user"}],
        api_key=gemini_api_key
    )
    return response['choices'][0]['message']['content']

def main():
    st.title("RAG-powered Research Paper Assistant")

    # Option to choose between PDF upload and arXiv search
    option = st.radio("Choose an option:", ("Upload PDFs", "Search arXiv"))

    if option == "Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            st.write("Processing uploaded files...")
            all_text = extract_text_from_pdfs(uploaded_files)
            collection = process_text_and_store(all_text)
            st.success("PDF content processed and stored successfully!")

            query = st.text_input("Enter your query:")
            if st.button("Execute Query") and query:
                results = semantic_search(query, collection)
                context = "\n".join(results['documents'][0])
                response = generate_response(query, context)
                st.subheader("Generated Response:")
                st.write(response)

    elif option == "Search arXiv":
        query = st.text_input("Enter your search query for arXiv:")

        if st.button("Search ArXiv") and query:
            arxiv_results = arxiv_tool.invoke(query)
            st.session_state["arxiv_results"] = arxiv_results  
            st.subheader("Search Results:")
            st.write(arxiv_results)

            collection = process_text_and_store(arxiv_results)
            st.session_state["collection"] = collection  

            st.success("arXiv paper content processed and stored successfully!")

        # Only allow querying if search has been performed
        if "arxiv_results" in st.session_state and "collection" in st.session_state:
            query = st.text_input("Ask a question about the paper:")
            if st.button("Execute Query on Paper") and query:
                results = semantic_search(query, st.session_state["collection"])
                context = "\n".join(results['documents'][0])
                response = generate_response(query, context)
                st.subheader("Generated Response:")
                st.write(response)


if __name__ == '__main__':
	main()