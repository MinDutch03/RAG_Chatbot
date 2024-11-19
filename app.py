import time
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set up page
st.set_page_config(page_title="RAG-Based Health Assistant", page_icon="🚑")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Health Assistant 👨‍⚕️")
    st.write("Your AI-powered Assistant")

# Initialize environment
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# API Keys
groq_api_key = st.secrets["GROQ_API_KEY"]
cohere_api_key = st.secrets["CO_API_KEY"]

if not groq_api_key:
    st.error("GROQ_API_KEY not found! Please set it in the .env file.")
if not cohere_api_key:
    st.error("CO_API_KEY not found! Please set it in the .env file.")

# File paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

# Setting up models and retrievers
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15, api_key=groq_api_key)
llm = ChatCohere(temperature=0.15, api_key=cohere_api_key)
embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)
kb_retriever = vectorDB.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# Rephrasing prompt
rephrasing_template = """
    TASK: Convert context-dependent questions into standalone queries.
    INPUT:
    - chat_history: Previous messages
    - question: Current user query
    RULES:
    1. Replace pronouns (it/they/this) with specific referents
    2. Expand contextual phrases ("the above", "previous")
    3. Return original if already standalone
    4. NEVER answer or explain - only reformulate
    OUTPUT: Single reformulated question, preserving original intent and style.
"""
rephrasing_prompt = ChatPromptTemplate.from_messages(
    [("system", rephrasing_template), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
history_aware_retriever = create_history_aware_retriever(
    llm=chatmodel, retriever=kb_retriever, prompt=rephrasing_prompt
)

# QA prompt
system_prompt_template = (
    "As a Health Assistant Chatbot specializing in health queries, "
    "your primary objective is to provide accurate and concise information based on user queries. "
    "You will adhere strictly to the instructions provided, offering relevant "
    "context from the knowledge base while avoiding unnecessary details. "
    "Your responses will be brief, to the point, concise, and in compliance with the established format. "
    "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. Please contact our doctors."
    "\nCONTEXT: {context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template), ("placeholder", "{chat_history}"), ("human", "{input}")]
)
qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Initialize session state for multiple chat sessions
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = []  # Stores histories of all chat sessions
if "current_session" not in st.session_state:
    st.session_state["current_session"] = []  # Tracks the current chat session

# Functions
def reset_conversation():
    st.session_state["current_session"] = []

def new_chat():
    if st.session_state["current_session"]:
        st.session_state["chat_sessions"].append(st.session_state["current_session"])
    st.session_state["current_session"] = []

# Display messages from the current session
for message in st.session_state["current_session"]:
    with st.chat_message(message.type):
        st.write(message.content)

# User input
user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            result = coversational_rag_chain.invoke(
                {"input": user_query, "chat_history": st.session_state["current_session"]}
            )
            message_placeholder = st.empty()
            full_response = (
                "⚠️ **_This information is not intended as a substitute for health advice. \n"
                "_Please consult a healthcare professional for personalized recommendations._** \n\n\n"
            )
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")
    st.session_state["current_session"].extend(
        [HumanMessage(content=user_query), AIMessage(content=result["answer"])]
    )

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    st.button("Reset Conversation 🗑️", on_click=reset_conversation)
with col2:
    st.button("New Chat ➕", on_click=new_chat)
