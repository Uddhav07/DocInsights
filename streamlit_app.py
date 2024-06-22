import streamlit as st
import os
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    ImageCaptionLoader,
    UnstructuredXMLLoader,
    CSVLoader
)
import ollama

# Constants
PROMPT_TEMPLATE = """
Answer the question based on the following context and the history:

{context}

---

Answer the question based on the above context: 
{question}
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_selected" not in st.session_state:
    st.session_state.model_selected = None
if "store_selected" not in st.session_state:
    st.session_state.store_selected = None

# Functions
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()

def process_file(file_path):
    if file_path.endswith((".png", ".jpg")):
        loader = ImageCaptionLoader(path_images=[file_path])
    elif file_path.endswith((".pdf", ".docx", ".txt")):
        loader = UnstructuredFileLoader(file_path)
    elif file_path.endswith(".xml"):
        loader = UnstructuredXMLLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        st.warning(f"Unsupported file type: {file_path}")
        return []
    return loader.load()

def create_vector_store(document_chunks, embeddings, store_type):
    if store_type == "FAISS":
        return FAISS.from_documents(document_chunks, embeddings)
    elif store_type == "Chroma":
        return Chroma.from_documents(document_chunks, embeddings)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

def process_uploaded_files():
    if not st.session_state.uploaded_files:
        st.warning("Please upload files first.")
        return

    with st.spinner("Processing files..."):
        documents = []
        for uploaded_file in st.session_state.uploaded_files:
            file_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            documents.extend(process_file(file_path))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)
        
        embeddings = load_embeddings()
        vectorstore = create_vector_store(document_chunks, embeddings, st.session_state.store_selected)
        
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }
    st.success("Files processed and indexed successfully!")

def answer_question(prompt):
    if "processed_data" not in st.session_state:
        st.warning("Please upload and index files first.")
        return

    vectorstore = st.session_state.processed_data["vectorstore"]
    results = vectorstore.similarity_search_with_score(prompt, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompted = prompt_template.format(context=context_text, question=prompt)
    
    return prompted
    


# Streamlit UI
st.title("DocInsight")

with st.sidebar:
    st.session_state.uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    st.button("Index Files", on_click=process_uploaded_files)
    st.selectbox("Select Model", [model["name"] for model in ollama.list()["models"]],key="model_selected")
    st.selectbox("Select Vector Store", ["FAISS", "Chroma"], key="store_selected")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your documents")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        prompted = answer_question(prompt)
        llm = Ollama(model=st.session_state.model_selected)
        response = st.write_stream(llm.stream(prompted))
        st.session_state.messages.append({"role": "assistant", "content": response})