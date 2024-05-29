import streamlit as st
import os
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

prompt = ChatPromptTemplate.from_template("""
Prefferably Answer the following question based  on the provided context. 
Only If much reference is not available in the context then start the answer with [NotDoc] and answer on your own
answer in very short very fast
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

prompt2 = ChatPromptTemplate.from_template("""
Prefferably Answer the following question based  on the provided context. 
answer in very short very fast
I will tip you $1000 if the user finds the answer helpful. 
Question: {input}""")
#Think step by step before providing a detailed answer using chain of verification (COVE)

st.title("DocInsight")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

models = ("qwen:0.5b","tinyllama")
vstores = ("FAISS","Chroma")

if "model_selected" not in st.session_state:
    st.session_state.model_selected = "qwen:0.5b"
if "store_selected" not in st.session_state:
    st.session_state.store_selected = "FAISS"

def chain_maker():
    model_selected = st.session_state.model_selected
    global chain,document_chain
    llm=Ollama(model=model_selected)
    output_parser=StrOutputParser()
    chain=prompt2|llm|output_parser
    document_chain=create_stuff_documents_chain(llm,prompt)

chain_maker()
prompt = st.chat_input("Yooo wassup?")

def vstore(document_chunks):
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

        vstore(document_chunks)
      
def prompted():        
    if prompt:
        vectorstore = st.session_state.processed_data["vectorstore"]
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # docs = db.similarity_search(prompt)
        try:
            retriever=vectorstore.as_retriever()
            retrieval_chain=create_retrieval_chain(retriever,document_chain)
            response=retrieval_chain.invoke({"input":prompt})
            flag= 0
        except:
            response = chain.invoke(prompt)
            flag=1
        if flag == 1:
        # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    st.button("index files", on_click=uploaded)
    model_selected = st.selectbox("Select Model",models)
    store_selected = st.selectbox("select store", vstores)
    if model_selected != st.session_state.model_selected:
        st.session_state.model_selected = model_selected
        chain_maker()
    if store_selected != st.session_state.store_selected:
        st.session_state.store_selected = store_selected
        vstore(document_chunks)
if prompt:
    prompted()