import time
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
import os

## initializing the UI
st.set_page_config(page_title="RAG-Based Health Assistant", page_icon="🚑")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Health Assistant 👨‍⚕️")
    st.write("Your AI-powered Assistant")

## setting up env
from numpy.core.defchararray import endswith

# Get the API keys
groq_api_key = st.secrets["GROQ_API_KEY"]
cohere_api_key = st.secrets["CO_API_KEY"]

# Check if API keys are loaded
if not groq_api_key:
    st.error("GROQ_API_KEY not found! Please set it in the .env file.")
if not cohere_api_key:
    st.error("CO_API_KEY not found! Please set it in the .env file.")

## LangChain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
## implementation of LangChain ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

## setting up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")
chats_directory = os.path.join(current_dir, "chat_history")

# Create chat history directory if it doesn't exist
os.makedirs(chats_directory, exist_ok=True)

## setting up the LLM
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15, api_key=groq_api_key)
llm = ChatCohere(temperature=0.15, api_key=cohere_api_key)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None

# Function to save chat history
def save_chat_history(chat_id, messages):
    """Save chat history to a JSON file"""
    filename = os.path.join(chats_directory, f"chat_{chat_id}.json")
    with open(filename, 'w') as f:
        # Convert messages to a serializable format
        serializable_messages = [
            {"type": msg.type, "content": msg.content}
            for msg in messages
        ]
        json.dump(serializable_messages, f)

# Function to load chat history
def load_chat_history(chat_id):
    """Load chat history from a JSON file"""
    filename = os.path.join(chats_directory, f"chat_{chat_id}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            message_data = json.load(f)
            # Reconstruct message objects
            messages = []
            for msg in message_data:
                if msg['type'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['type'] == 'ai':
                    messages.append(AIMessage(content=msg['content']))
            return messages
    return []

# Function to list existing chat files
def list_existing_chats():
    """List all existing chat files"""
    chat_files = [f for f in os.listdir(chats_directory) if f.startswith('chat_') and f.endswith('.json')]
    return [int(f.split('_')[1].split('.')[0]) for f in chat_files]

# Function to start a new chat
def start_new_chat():
    # Find the next available chat ID
    existing_chats = list_existing_chats()
    new_chat_id = max(existing_chats + [0]) + 1

    # Create a new chat session
    st.session_state["chats"][new_chat_id] = {
        "messages": [],
        "id": new_chat_id
    }

    # Set the current chat to the new chat
    st.session_state["current_chat"] = new_chat_id

    return new_chat_id

# Function to reset current chat
def reset_current_chat():
    if st.session_state["current_chat"] is not None:
        # Clear messages in session state
        st.session_state["chats"][st.session_state["current_chat"]]["messages"] = []

        # Remove the chat history file
        chat_id = st.session_state["current_chat"]
        chat_file = os.path.join(chats_directory, f"chat_{chat_id}.json")
        if os.path.exists(chat_file):
            os.remove(chat_file)

## remaining setup code (embeddings, retriever, prompts, etc.) stays the same...

# Sidebar for chat management
st.sidebar.title("Chat Sessions")

# Add New Chat button
if st.sidebar.button("New Chat 🆕"):
    start_new_chat()

# Display and select existing chats
chat_options = list_existing_chats()
if chat_options:
    selected_chat = st.sidebar.selectbox(
        "Select a Chat",
        options=chat_options,
        format_func=lambda x: f"Chat {x}"
    )

    # Load the selected chat's messages
    if selected_chat not in st.session_state["chats"]:
        st.session_state["chats"][selected_chat] = {
            "messages": load_chat_history(selected_chat),
            "id": selected_chat
        }

    st.session_state["current_chat"] = selected_chat

# Check if a chat is selected
if st.session_state["current_chat"] is None:
    start_new_chat()

# Get current chat messages
current_chat_messages = st.session_state["chats"][st.session_state["current_chat"]]["messages"]

## printing all messages in the current chat
for message in current_chat_messages:
    with st.chat_message(message.type):
        st.write(message.content)

user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            ## invoking the chain to fetch the result
            result = coversational_rag_chain.invoke({
                "input": user_query,
                "chat_history": current_chat_messages
            })

            message_placeholder = st.empty()

            full_response = (
                "⚠️ **_This information is not intended as a substitute for health advice. \n"
                "_Please consult a healthcare professional for personalized recommendations._** \n\n\n"
            )

        ## displaying the output on the dashboard
        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02) ## <- simulate the output feeling of ChatGPT

            message_placeholder.markdown(full_response + " ▌")

    ## appending conversation turns to the current chat
    current_chat_messages.extend(
        [
            HumanMessage(content=user_query),
            AIMessage(content=result['answer'])
        ]
    )

    # Update session state
    st.session_state["chats"][st.session_state["current_chat"]]["messages"] = current_chat_messages

    # Save the updated chat history
    save_chat_history(st.session_state["current_chat"], current_chat_messages)

# Add Reset Current Chat button
if st.button('Reset Current Chat 🗑️'):
    reset_current_chat()
