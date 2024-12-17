# backend.py

from fastapi import FastAPI, UploadFile, HTTPException, File
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import warnings
import os
import logging
from langchain_community.vectorstores import FAISS
from utils import split_into_paragraphs
import dotenv
from openai import OpenAI
from models import Query
from langsmith import traceable
from utils import (
    orchestrator,
    generate_answer,
    hallucination_check,
    refine_query,
    summarize_history,
    generate_corrected_transcript,
)

dotenv.load_dotenv()

# Suppress the FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Logs will be saved in app.log
        logging.StreamHandler(),  # Continue showing logs in the console
    ],
)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup for embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS vector store
vectorstore = None


def initialize_vectorstore(vectorstore):
    logger = logging.getLogger(__name__)
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS vectorstore loaded from local index.")
        return vectorstore
    else:
        try:
            with open("aufenthg.txt", "r", encoding="utf-8") as file:
                law_text = file.read()

            paragraphs = split_into_paragraphs(law_text)
            vectorstore = FAISS.from_texts(paragraphs, embeddings)
            vectorstore.save_local("faiss_index")
            logger.info("FAISS vectorstore created and saved locally.")
            return vectorstore
        except FileNotFoundError:
            logger.error("Law text file 'aufenthg.txt' not found.")
            raise
        except Exception as e:
            logger.exception("Failed to initialize FAISS vectorstore.")
            raise


vectorstore = initialize_vectorstore(vectorstore)



@app.post("/query")
async def query_start(query: Query):
    """
    FastAPI endpoint that calls the traceable function.
    """
    try:
        response = await query_law_docs(query)
        return response
    except Exception as e:
        logger.exception("An error occurred while processing the query.")
        raise HTTPException(status_code=500, detail=str(e))

@traceable(name="question_law_docs")
async def query_law_docs(query: Query):
    if vectorstore is None:
        logger.error("Vectorstore not initialized")
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")

    try:
        logger.info(f"Received query: {query.question}")

        # Step 1: Determine if vector search is needed
        use_vector, model_use = orchestrator(query)
        if use_vector:
            context = vector_search(query, vectorstore)
        else:
            logger.info("Vector search not required. Proceeding without context.")
            context = ""
        query.context = query.context + [context]
        # Step 2: Use LLM to generate an answer based on retrieved documents
        answer = generate_answer(query.question, context, query.history, model_use)

        # Step 3: Check for hallucinations
        if hallucination_check(query.question, answer, context):
            logger.warning("Hallucination detected in the answer. Refining the query.")
            query.question = refine_query(query.question, answer)
            return await query_law_docs(query)  # Recursive call with refined query

        # Step 4: Prepare response with conversation history
        updated_history = query.history.copy()
        updated_history.append({"role": "user", "content": query.question})
        updated_history.append({"role": "assistant", "content": answer})

        # Summarize history if too large
        if len(updated_history) > 2000:
            logger.info("Conversation history too long. Summarizing history.")
            summary = summarize_history(updated_history)
            response_history = summary
        else:
            response_history = updated_history

        return {"answer": answer, "history": response_history}

    except Exception as e:
        logger.exception("An error occurred while processing the query.")
        raise HTTPException(status_code=500, detail=str(e))


@traceable(name="vector_search")
def vector_search(query: Query, vectorstore):
    logger.info("Vector search required. Performing similarity search.")
    retrieved_docs = vectorstore.similarity_search(query.question, k=5)
    context = "\n\n ".join([doc.page_content for doc in retrieved_docs])
    logger.info(
        f"Retrieved context from vector search: {context[:100]}..."
    )  # Log first 100 chars
    return context


@traceable(name="transcribe_audio")
@app.post("/transcribe_audio")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_path = f"/tmp/{audio.filename}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(await audio.read())

        # Transcription with Whisper
        transcription = client.audio.transcriptions.create(
            file=open(temp_path, "rb"), model="whisper-1"
        ).text

        logger.info(f"Original transcription: {transcription}")

        corrected_text = generate_corrected_transcript(transcription)
        logger.info(f"Corrected transcription: {corrected_text}")

        # Optionally delete the temporary file
        os.remove(temp_path)

        return {"transcription": corrected_text}

    except Exception as e:
        logger.exception("An error occurred during transcription.")
        raise HTTPException(status_code=500, detail=str(e))


# CORS configuration
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8000",
    "http://192.168.0.160:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
