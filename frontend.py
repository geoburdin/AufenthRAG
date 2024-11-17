# frontend.py

import streamlit as st
import requests
from io import BytesIO
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("German Law Q&A (AufenthG) with Voice Input")


# Function to display message history with chat bubbles
def display_message_history():
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='background-color: #DCF8C6; padding: 10px; border-radius: 10px; text-align: left; max-width: 80%; margin-bottom: 10px;'>
                    <strong>You:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            st.markdown(
                f"""
                <div style='background-color: #FFFFFF; padding: 10px; border-radius: 10px; text-align: left; max-width: 80%; margin-bottom: 10px;'>
                    <strong>Assistant:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<hr>", unsafe_allow_html=True)  # Separator between messages
    st.markdown("<a id='end'></a>", unsafe_allow_html=True)


# Display the existing message history
st.header("Conversation History")
display_message_history()

# Scroll to the latest message
if st.session_state["messages"]:
    st.markdown(
        "<script>document.getElementById('end').scrollIntoView();</script>",
        unsafe_allow_html=True,
    )

st.header("New Query")

# Mode: Text or Voice
mode = st.radio("Input Mode", ["Text", "Voice"])

if mode == "Text":
    user_question = st.text_input(
        "Ask a question about the Residence Act:", key="user_input"
    )
    if st.button("Submit"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:

            with st.spinner("Assistant is typing..."):
                try:
                    # Prepare the payload with history
                    payload = {"question": user_question, "context": [], "history": st.session_state["messages"]}

                    response = requests.post(
                        "http://127.0.0.1:8000/query", json=payload
                    )
                    # Remove or comment out the debugging line in production
                    st.write("Backend Response:", response.text)  # Debugging line

                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer available.")
                        updated_history = data.get(
                            "history", st.session_state["messages"]
                        )

                        # Append assistant message to history
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": answer}
                        )
                        st.success("Answer added to conversation history.")

                        # Update the history in session state
                        st.session_state["messages"] = updated_history

                    else:
                        error_message = response.json().get("error", "Unknown error.")
                        st.error(f"Error: {error_message}")
                except Exception as e:
                    st.error(f"Error: {e}")


# Option to clear conversation history
if st.button("Clear Conversation History"):
    st.session_state["messages"] = []
    st.success("Conversation history cleared.")
