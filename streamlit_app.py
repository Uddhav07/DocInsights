import streamlit as st
import os
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ImageCaptionLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
# from langchain_community.embeddings import OllamaEmbeddings
# embeddings = OllamaEmbeddings()
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import ollama

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

st.title("DocInsight")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

models = [model["name"] for model in ollama.list()["models"]]
vstores = ("FAISS","Chroma")

if "model_selected" not in st.session_state:
    st.session_state.model_selected = models[0]
if "store_selected" not in st.session_state:
    st.session_state.store_selected = vstores[0]

output_parser=StrOutputParser()
model_selected = st.session_state.model_selected
llm=Ollama(model=model_selected)

prompt = st.chat_input("Yooo wassup?")

def vstore():
    if uploaded_files:
        document_chunks = st.session_state.document_chunks 
        store_selected = st.session_state.store_selected
        if store_selected == "FAISS":
            vectorstore = FAISS.from_documents(document_chunks, embeddings)
        if store_selected == "Chroma":
            vectorstore = Chroma.from_documents(document_chunks, embeddings)
        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

def uploaded():
    global document_chunks
    if uploaded_files:
        global vectorstore
        # Print the number of files uploaded or YouTube URL provided to the console
        st.write(f"Number of files uploaded: {len(uploaded_files)}")
            # Load the data and perform preprocessing only if it hasn't been loaded before
        
        # Load the data from uploaded files
        documents = []

        
        for uploaded_file in uploaded_files:
            # Get the full file path of the uploaded file
            file_path = os.path.join(os.getcwd(), uploaded_file.name)

            # Save the uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Check if the file is an image
            if file_path.endswith((".png", ".jpg")):
                # Use ImageCaptionLoader to load the image file
                image_loader = ImageCaptionLoader(path_images=[file_path])

                # Load image captions
                image_documents = image_loader.load()

                # Append the Langchain documents to the documents list
                documents.extend(image_documents)
                
            elif file_path.endswith((".pdf", ".docx", ".txt")):
                # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
                loader = UnstructuredFileLoader(file_path)
                loaded_documents = loader.load()

                # Extend the main documents list with the loaded documents
                documents.extend(loaded_documents)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        st.session_state.document_chunks = document_chunks
        
        vstore()
      

with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    st.button("index files", on_click=uploaded)
    st.selectbox("Select Model", models , key="model_selected")
    st.selectbox("select store", vstores, key="store_selected")


if prompt:
    # user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    if  ("processed_data" in st.session_state):
        vectorstore = st.session_state.processed_data["vectorstore"]
        results = vectorstore.similarity_search_with_score(prompt, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompted = prompt_template.format(context=context_text, question=prompt)

        with st.chat_message("assistant"):
            response=st.write_stream(llm.stream(prompted))
            # chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            response=st.write("Please Upload and Index a file first")
            # chat history
            st.session_state.messages.append({"role": "assistant", "content": "Please Upload and Index a file first"})

    