import time
import streamlit as st
import json
import os

# Initializing the UI
st.set_page_config(page_title="RAG-Based Health Assistant", page_icon="🚑")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Health Assistant 👨‍⚕️")
    st.write("Your AI-powered Assistant")

# Check if chat data exists
chat_data_file = "chat_sessions.json"
if os.path.exists(chat_data_file):
    with open(chat_data_file, "r") as f:
        st.session_state["chats"] = json.load(f)
else:
    st.session_state["chats"] = {}

# Setting up the LLM
groq_api_key = st.secrets["GROQ_API_KEY"]
cohere_api_key = st.secrets["CO_API_KEY"]

# Setting up -> Streamlit session state
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
    save_chat_data()
    return new_chat_id

# Function to reset current chat
def reset_current_chat():
    if st.session_state["current_chat"] is not None:
        st.session_state["chats"][st.session_state["current_chat"]]["messages"] = []
        save_chat_data()

# Function to delete a chat session
def delete_chat_session(chat_id):
    if chat_id in st.session_state["chats"]:
        del st.session_state["chats"][chat_id]
        # Reset current chat to None if the deleted session is the current one
        if st.session_state["current_chat"] == chat_id:
            st.session_state["current_chat"] = None
        st.success(f"Chat session {chat_id} deleted successfully.")
        save_chat_data()

# Function to save chat data to a file
def save_chat_data():
    with open(chat_data_file, "w") as f:
        json.dump(st.session_state["chats"], f)

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
    with st.chat_message(message.type):
        st.write(message.content)

user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            # Invoke the chain to fetch the result, now including all chat history from all sessions
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
            HumanMessage(content=user_query),
            AIMessage(content=result['answer'])
        ]
    )
    save_chat_data()

# Add Reset Current Chat button
if st.button('Reset Current Chat 🗑️'):
    reset_current_chat()
