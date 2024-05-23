import streamlit as st
# import pandas as pd
# from io import StringIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Help the user understand things better"),
        ("user","Question:{question}")
    ]
)
st.title("DocInsight")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

llm=Ollama(model="phi3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

db=0
# React to user input
if prompt := st.chat_input("Yooo wassup?"):
    if db:
        retireved_results=db.similarity_search(prompt)
        print(retireved_results[0].page_content)
    
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chain.invoke({"question":prompt})
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
from langchain.docstore.document import Document
# if uploaded_files is not :
if 0:
    # # Extract text from the PDF
    # pdf_file = BytesIO(uploaded_file.read())
    # pdf_text = extract_text_from_pdf(uploaded_file)
    
    # # Display the extracted text
    # # st.write("PDF Text:")
    # # st.write(pdf_text)
    # broken_pdf=text_splitter.split_documents(pdf_text)
    
    # Create a file-like object from the uploaded file
    pdf_file = BytesIO(uploaded_file.read())

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_file)

    doc = Document(page_content=pdf_text)

    # Split the document
    broken_pdf = text_splitter.split_documents([doc])
    
    # Extract the page_content from each Document object
    texts = [doc.page_content for doc in broken_pdf]

    # Embed the texts
    vec_pdf = embeddings.embed_documents(texts)
    db = FAISS.from_documents(vec_pdf, embeddings)