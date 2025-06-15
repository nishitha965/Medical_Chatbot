# app.py

# Phase 1: Imports
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Phase 2: LangChain & Groq
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3: Vectorstore Loading
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("Hi! I am your RAG chatbot!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Load FAISS vectorstore (cached)
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

# Prompt Input
prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Initialize Groq LLM
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything, you always give the best, 
        the most accurate and most precise answers. Answer the following Question: {user_prompt}.
        Start the answer directly. No small talk please"""
    )

    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {str(e)}")
