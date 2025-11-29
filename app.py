import streamlit as st
from agent_setup import agent_executor

# Streamlit Page Config
st.set_page_config(page_title="Synapse AI Support", page_icon="🤖", layout="centered")
st.title("🤖 Synapse AI – Support Assistant")
st.write("Ask any support query and I'll help you!")

# Initialize chat history in session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Run the agent and append response
    try:
        response = agent_executor.invoke({"input": user_input})
        final_output = response.get("output", "⚠ Agent failed to process your request.")
    except Exception as e:
        final_output = f"⚠ Agent failed: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": final_output})

# Display full chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])