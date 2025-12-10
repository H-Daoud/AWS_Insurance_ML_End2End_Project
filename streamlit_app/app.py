import streamlit as st
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000/rag/query"

st.set_page_config(page_title="HUK-COBURG Expert", page_icon="🛡️")

# UI Header
st.title("🛡️ HUK-COBURG Insurance Assistant")
st.markdown("---")
st.markdown("**Status:** System Ready | **Model:** GPT-4o")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handler
if prompt := st.chat_input("How can I help you with your insurance?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get Bot Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Analyzing policy documents...*")
        
        try:
            payload = {"query": prompt}
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No response received.")
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = f"Error {response.status_code}: Unable to reach API."
                message_placeholder.error(error_msg)
        except Exception as e:
            message_placeholder.error(f"Connection Failed: {str(e)}")
            st.warning("Make sure the FastAPI server is running on port 8000.")