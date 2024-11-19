import os
import json
import time
import streamlit as st

# Constants for storage
CHAT_STORAGE_FILE = "chat_sessions.json"

# Function to load saved chats
def load_chats():
    if os.path.exists(CHAT_STORAGE_FILE):
        with open(CHAT_STORAGE_FILE, "r") as file:
            return json.load(file)
    return {}

# Function to save chats
def save_chats():
    with open(CHAT_STORAGE_FILE, "w") as file:
        json.dump(st.session_state["chats"], file)

# Initialize Streamlit app
st.set_page_config(page_title="RAG-Based Health Assistant", page_icon="🚑")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Health Assistant 👨‍⚕️")
    st.write("Your AI-powered Assistant")

# Load chats into session state
if "chats" not in st.session_state:
    st.session_state["chats"] = load_chats()

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None

# Function to start a new chat
def start_new_chat():
    new_chat_id = len(st.session_state["chats"]) + 1
    st.session_state["chats"][new_chat_id] = {"messages": [], "id": new_chat_id}
    st.session_state["current_chat"] = new_chat_id
    save_chats()  # Save chats after starting a new one
    return new_chat_id

# Function to reset current chat
def reset_current_chat():
    if st.session_state["current_chat"] is not None:
        st.session_state["chats"][st.session_state["current_chat"]]["messages"] = []
        save_chats()  # Save chats after resetting

# Function to delete a chat session
def delete_chat_session(chat_id):
    if chat_id in st.session_state["chats"]:
        del st.session_state["chats"][chat_id]
        if st.session_state["current_chat"] == chat_id:
            st.session_state["current_chat"] = None
        save_chats()  # Save chats after deletion

# Sidebar for chat management
st.sidebar.title("Chat Sessions")

if st.sidebar.button("New Chat 🆕"):
    start_new_chat()

chat_options = list(st.session_state["chats"].keys())
if chat_options:
    selected_chat = st.sidebar.selectbox(
        "Select a Chat", options=chat_options, format_func=lambda x: f"Chat {x}"
    )
    st.session_state["current_chat"] = selected_chat

    if st.sidebar.button("Delete Selected Chat 🗑️"):
        delete_chat_session(selected_chat)

if not chat_options:
    start_new_chat()

if st.session_state["current_chat"] is None:
    start_new_chat()

# Current chat messages
current_chat_messages = st.session_state["chats"][st.session_state["current_chat"]]["messages"]

for message in current_chat_messages:
    with st.chat_message(message["type"]):
        st.write(message["content"])

user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            # Example assistant response
            response = f"Here's a response to: {user_query}"
            time.sleep(0.02)

        st.write(response)
        current_chat_messages.append({"type": "user", "content": user_query})
        current_chat_messages.append({"type": "assistant", "content": response})
        save_chats()  # Save chats after each interaction

# Reset Current Chat button
if st.button("Reset Current Chat 🗑️"):
    reset_current_chat()
