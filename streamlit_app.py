import streamlit as st
# import pandas as pd
# from io import StringIO

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama


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


llm=Ollama(model="phi3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser


# React to user input
if prompt := st.chat_input("Yooo wassup?"):
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
    
# uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=1)