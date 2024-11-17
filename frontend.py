import streamlit as st
from audiorecorder import audiorecorder
import requests
from io import BytesIO

st.title("German Law Q&A with Voice and Text Input")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display message history
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

st.header("Conversation History")
display_message_history()

# Input Mode Selection
st.header("New Query")
mode = st.radio("Choose Input Mode", ["Text", "Voice"])

# Text Mode
if mode == "Text":
    user_question = st.text_input("Enter your question about the Residence Act:")
    if st.button("Submit Text Query"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                payload = {"question": user_question, "context": [], "history": st.session_state["messages"]}
                response = requests.post("http://127.0.0.1:8000/query", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer available.")
                    updated_history = data.get("history", st.session_state["messages"])
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["messages"] = updated_history
                else:
                    st.error("Error querying the assistant.")
            except Exception as e:
                st.error(f"Error: {e}")

# Voice Mode
elif mode == "Voice":
    st.write("Record your question below:")

    audio = audiorecorder("Start Recording", "Stop Recording")

    if len(audio) > 0:
        # Export AudioSegment to raw WAV bytes
        audio_bytes = BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)

        # Play recorded audio
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Submit Voice Query"):
            try:
                # Send audio for transcription
                # Send audio for transcription
                files = {"audio": ("temp_audio.wav", audio_bytes, "audio/wav")}
                response = requests.post("http://127.0.0.1:8000/transcribe_audio", files=files)

                if response.status_code == 200:
                    transcription = response.json().get("transcription", "")
                    st.success(f"Transcription: {transcription}")

                    # Query the backend with the transcription
                    payload = {"question": transcription, "context": [], "history": st.session_state["messages"]}
                    query_response = requests.post("http://127.0.0.1:8000/query", json=payload)

                    if query_response.status_code == 200:
                        data = query_response.json()
                        answer = data.get("answer", "No answer available.")
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                        st.session_state["messages"] = data.get("history", st.session_state["messages"])
                    else:
                        st.error("Error querying the assistant.")
                else:
                    st.error("Error in transcription.")
            except Exception as e:
                st.error(f"Error during transcription: {e}")

# Option to clear conversation history
if st.button("Clear Conversation History"):
    st.session_state["messages"] = []
    st.success("Conversation history cleared.")
