import time
import streamlit as st
import os
from numpy.core.defchararray import endswith
import sys
__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initializing the UI
st.set_page_config(page_title="RAG-Based Health Assistant", page_icon="🚑")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Health Assistant 👨‍⚕️")
    st.write("Your AI-powered Assistant")

# Setting up env
groq_api_key = st.secrets["GROQ_API_KEY"]
cohere_api_key = st.secrets["CO_API_KEY"]

# Check if API keys are loaded
if not groq_api_key:
    st.error("GROQ_API_KEY not found! Please set it in the .env file.")
if not cohere_api_key:
    st.error("CO_API_KEY not found! Please set it in the .env file.")

# LangChain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Setting up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

# Setting up the LLM
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15, api_key=groq_api_key)
llm = ChatCohere(temperature=0.15, api_key=cohere_api_key)

# Setting up -> Streamlit session state
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None

# Function to start a new chat
def start_new_chat():
    new_chat_id = len(st.session_state["chats"]) + 1
    st.session_state["chats"][new_chat_id] = {
        "messages": [],
        "id": new_chat_id
    }
    st.session_state["current_chat"] = new_chat_id
    return new_chat_id

# Function to reset current chat
def reset_current_chat():
    if st.session_state["current_chat"] is not None:
        st.session_state["chats"][st.session_state["current_chat"]]["messages"] = []

# Function to delete a chat session
def delete_chat_session(chat_id):
    if chat_id in st.session_state["chats"]:
        del st.session_state["chats"][chat_id]
        if st.session_state["current_chat"] == chat_id:
            st.session_state["current_chat"] = None
        st.success(f"Chat session {chat_id} deleted successfully.")
        if len(st.session_state["chats"]) > 0:
            st.session_state["current_chat"] = list(st.session_state["chats"].keys())[0]
        else:
            st.session_state["current_chat"] = None

# Open-source embedding model from HuggingFace
embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Loading the vector database from local
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)

# Setting up the retriever
kb_retriever = vectorDB.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# Setting up the history aware retriever
rephrasing_template = (
    """
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
)

rephrasing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrasing_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm=chatmodel,
    retriever=kb_retriever,
    prompt=rephrasing_prompt
)

# Setting up the document chain
system_prompt_template = (
    "As a Health Assistant Chatbot specializing in health queries, "
    "your primary objective is to provide accurate and concise information based on user queries. "
    "You will adhere strictly to the instructions provided, offering relevant "
    "context from the knowledge base while avoiding unnecessary details. "
    "Your responses will be brief, to the point, concise and in compliance with the established format. "
    "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. Please contact our doctors."
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", "{input}"),
        ("chat_history", "{chat_history}")
    ]
)

qa_chain = create_stuff_documents_chain(chatmodel, qa_prompt)
# Final RAG chain
coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Sidebar for chat management
st.sidebar.title("Chat Sessions")

# Add New Chat button
if st.sidebar.button("New Chat 🆕"):
    start_new_chat()

# Display and select existing chats
chat_options = list(st.session_state["chats"].keys())
if chat_options:
    selected_chat = st.sidebar.selectbox(
        "Select a Chat",
        options=chat_options,
        format_func=lambda x: f"Chat {x}"
    )
    st.session_state["current_chat"] = selected_chat

    # Add Delete button
    if st.sidebar.button("Delete Selected Chat 🗑️"):
        delete_chat_session(selected_chat)

# If no chats exist, create a new chat
if not chat_options:
    start_new_chat()

# Check if a chat is selected
if st.session_state["current_chat"] is None:
    start_new_chat()

# Get current chat messages
current_chat_messages = st.session_state["chats"][st.session_state["current_chat"]]["messages"]

# Get all previous chat messages from all sessions (collecting all history)
all_chat_history = []
for chat_id, chat in st.session_state["chats"].items():
    all_chat_history.extend(chat["messages"])

# Print all messages in the current chat
for message in current_chat_messages:
    with st.chat_message(message["type"]):
        st.write(message["content"])

user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            # Invoke the chain to fetch the result
            result = coversational_rag_chain.invoke({
                "input": user_query,
                "chat_history": all_chat_history  # Using chat history from all sessions
            })

            message_placeholder = st.empty()

            full_response = (
                "⚠️ **_This information is not intended as a substitute for health advice. \n"
                "_Please consult a healthcare professional for personalized recommendations._** \n\n\n"
            )

            # Displaying the output on the dashboard
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)  # Simulate the output feeling of ChatGPT

                message_placeholder.markdown(full_response + " ▌")

    # Appending conversation turns to the current chat
    current_chat_messages.extend(
        [
            {"type": "user", "content": user_query},
            {"type": "assistant", "content": result['answer']}
        ]
    )

# Add Reset Current Chat button
if st.button('Reset Current Chat 🗑️'):
    reset_current_chat()

# Save chat sessions to a file (for persistence across refreshes)
def save_chats():
    with open("chats.json", "w") as f:
        json.dump(st.session_state["chats"], f)

# Load chat sessions from file
def load_chats():
    if os.path.exists("chats.json"):
        with open("chats.json", "r") as f:
            st.session_state["chats"] = json.load(f)
