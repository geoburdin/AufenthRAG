# Aufenthaltsgesetz Q&A with Voice and Text Input

This project is a Q&A application that allows users to ask questions about [German migration law](https://www.gesetze-im-internet.de/aufenthg_2004/BJNR195010004.html) using either text or voice input. The application leverages Streamlit for the frontend, FastAPI for the backend, and integrates with OpenAI's API for generating responses and transcribing audio. Additionally, it uses FAISS for vector search.

## Features

- **Text Input**: Users can type their questions and receive answers.
- **Voice Input**: Users can record their questions and receive transcriptions and answers.
- **Conversation History**: The application maintains a history of the conversation.
- **Hallucination Check**: Ensures the generated answers are accurate and refines queries if necessary.
- **Contextual Search**: Uses FAISS to perform similarity searches for better context in responses.



1. **Frontend (Streamlit)**
- Sends text or voice queries to the FastAPI backend.

2. **Backend (FastAPI Endpoints)**
- `/query`: Handles text queries.
- `/transcribe_audio`: Handles voice queries. Transcribes the audio and sends the transcribed text as a text `/query`.
  
Processes text and voice requests.  
  Interacts with OpenAI for generating responses and transcribing audio.  
  Uses FAISS for vector store search.

**OpenAI API**  
  Generates answers to questions.  
  Performs audio transcription.

**FAISS Vector Store**  
Stores and searches through vector representations of the document.


3. **Internal Functions and Components**
- `orchestrator`: Determines if vector search is needed and calls the SMART or SIMPLE model.
- `vectorstore.similarity_search`: Performs similarity search if needed.
- `generate_answer`: Generates an answer with OPENAIs SMART or SIMPLE model.
- `hallucination_check`: Checks for hallucinations in the generated answer. If hallucinations are detected, refines the query and calls `/query` again.
- `refine_query`: Refines the query if hallucinations are detected and calls `/query` again.
- `summarize_history`: Summarizes the conversation history if it is too long.
- `generate_corrected_transcript`: Corrects the transcription of the audio.

### Interaction Flow

1. **Text Query Flow**
- **Frontend** sends a text query to `/query`.
- **FastAPI** receives the query and calls `orchestrator` to check if vector search is needed.
- If vector search is needed, **FastAPI** calls `vectorstore.similarity_search` to retrieve context.
- **FastAPI** calls `generate_answer` to generate an answer using the SIMPLE or SMART_LLM.
- **FastAPI** calls `hallucination_check` to check for hallucinations.
- If hallucinations are detected, **FastAPI** calls `refine_query` and recursively processes the refined query.
- **FastAPI** calls `summarize_history` if the conversation history is too long.
- **FastAPI** returns the answer and updated history to the **Frontend**.

2. **Voice Query Flow**
- **Frontend** sends an audio file to `/transcribe_audio`.
- **FastAPI** receives the audio file and calls `generate_corrected_transcript` to transcribe with whisper and correct the audio with SIMPLE_LLM.
- **FastAPI** processes the transcribed text as a text query (following the text query flow).


### Diagram
[Detailed Interaction Diagram](https://drive.google.com/file/d/1LT_duH10zKQlx9Aovb6NI_iwRWx6z0gS/view?usp=sharing)


![Detailed Interaction Diagram](lawrag.drawio.svg)