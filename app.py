import time
import streamlit as st
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = [{"id": 0, "messages": []}]
if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = 0

# UI Layout
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Health Assistant 👨‍⚕️")
    st.write("Your AI-powered Assistant")

# Sidebar for chat management
with st.sidebar:
    st.title("Chat Sessions")

    # New Chat button
    if st.button("New Chat 📝"):
        new_chat_id = len(st.session_state["chat_sessions"])
        st.session_state["chat_sessions"].append({"id": new_chat_id, "messages": []})
        st.session_state["current_chat_id"] = new_chat_id
        st.rerun()

    # Display chat sessions
    for session in st.session_state["chat_sessions"]:
        if st.button(f"Chat {session['id']}", key=f"chat_{session['id']}"):
            st.session_state["current_chat_id"] = session['id']
            st.rerun()

        # Add button to reset individual chat session
        if st.button(f"Reset Chat {session['id']}", key=f"reset_chat_{session['id']}"):
            session["messages"] = []
            st.rerun()

# Set up APIs and LLMs
groq_api_key = st.secrets["GROQ_API_KEY"]
cohere_api_key = st.secrets["CO_API_KEY"]
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15, api_key=groq_api_key)
llm = ChatCohere(temperature=0.15, api_key=cohere_api_key)

# Get current chat session's messages
current_chat = next(
    (chat for chat in st.session_state["chat_sessions"]
     if chat["id"] == st.session_state["current_chat_id"]),
    {"messages": []}
)

# Print current chat messages
for message in current_chat["messages"]:
    with st.chat_message(message.type):
        st.write(message.content)

# Handle user input
user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            result = coversational_rag_chain.invoke({
                "input": user_query,
                "chat_history": current_chat["messages"]
            })

            message_placeholder = st.empty()
            full_response = "⚠️ This information is not intended as a substitute for health advice. Please consult a healthcare professional for personalized recommendations.\n\n"

        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02)
            message_placeholder.markdown(full_response + " ▌")

    # Update the current chat session's messages
    new_messages = [
        HumanMessage(content=user_query),
        AIMessage(content=result['answer'])
    ]

    for chat in st.session_state["chat_sessions"]:
        if chat["id"] == st.session_state["current_chat_id"]:
            chat["messages"].extend(new_messages)
            break
