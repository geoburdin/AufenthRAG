# backend.py

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import whisper
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import logging
import warnings

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

# Initialize Whisper model
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load Whisper model.")
    raise e

# Setup for embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS vector store
vectorstore = None


def split_into_paragraphs(text: str) -> list:
    """
    Splits the given text into paragraphs based on double line breaks.
    """
    paragraphs = [
        paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
    ]
    return paragraphs


# Example Usage
def initialize_vectorstore():
    global vectorstore

    if os.path.exists("faiss_index"):
        # Load FAISS index
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        logger.info("FAISS vectorstore loaded from local index.")
    else:
        # Load and split text
        try:
            with open("aufenthg.txt", "r", encoding="utf-8") as file:
                law_text = file.read()

            # Split text into paragraphs
            paragraphs = split_into_paragraphs(law_text)

            # Build FAISS index from paragraphs
            vectorstore = FAISS.from_texts(paragraphs, embeddings)
            vectorstore.save_local("faiss_index")
            logger.info("FAISS vectorstore created and saved locally.")
        except FileNotFoundError:
            logger.error("Law text file 'aufenthg.txt' not found.")
            raise
        except Exception as e:
            logger.exception("Failed to initialize FAISS vectorstore.")
            raise


initialize_vectorstore()


# Pydantic model for queries
class Query(BaseModel):
    question: str
    context: list  # List of retrieved documents
    history: list  # List of previous messages


@app.post("/query")
async def query_law_docs(query: Query):
    if vectorstore is None:
        logger.error("Vectorstore not initialized")
        raise HTTPException(status_code=500, detail="Vectorstore not initialized")

    try:
        logger.info(f"Received query: {query.question}")

        # Step 1: Determine if vector search is needed
        if is_vector_search_needed(query):
            logger.info("Vector search required. Performing similarity search.")
            retrieved_docs = vectorstore.similarity_search(query.question, k=5)
            context = "\n\n ".join([doc.page_content for doc in retrieved_docs])
            logger.info(
                f"Retrieved context from vector search: {context[:100]}..."
            )  # Log first 100 chars
        else:
            logger.info("Vector search not required. Proceeding without context.")
            context = ""
        query.context = query.context + [context]
        # Step 2: Use LLM to generate an answer based on retrieved documents
        answer = generate_answer(query.question, context, query.history)

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


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    audio_path = f"temp_audio_{file.filename}"
    try:
        with open(audio_path, "wb") as audio_file:
            audio_file.write(await file.read())
        logger.info(f"Audio file saved to {audio_path}")

        # Transcription with Whisper
        transcription = whisper_model.transcribe(audio_path)["text"]
        logger.info(f"Original transcription: {transcription}")

        corrected_text = generate_corrected_transcript(transcription)
        logger.info(f"Corrected transcription: {corrected_text}")

        os.remove(audio_path)
        logger.info(f"Temporary audio file {audio_path} removed.")

        return {"transcription": corrected_text}

    except Exception as e:
        logger.exception("An error occurred during transcription.")
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequired(BaseModel):
    required: bool


# Load LLM model names from environment variables
SMART_LLM = os.getenv("SMART_LLM", "gpt-4o-mini")  # Default to gpt-4o-mini if not set
SIMPLE_LLM = os.getenv(
    "SIMPLE_LLM", "gpt-3.5-turbo"
)  # Default to gpt-3.5-turbo if not set


def is_vector_search_needed(query: Query) -> bool:
    messages = [
        {
            "role": "system",
            "content": "You are an assistant deciding whether to perform vector search to send a response on the user's message. Vector search is over german law documents Gesetz über den Aufenthalt, die Erwerbstätigkeit und die Integration von Ausländern im Bundesgebiet1) (Aufenthaltsgesetz - AufenthG)"
            "Respond with 'True' if vector search for more information about the german law is required and 'False' if no additional information is needed.",
        },
        {"role": "user", "content": f"{query.question}"},
        {"role": "system", "content": "Conversation history:"},
        {"role": "system", "content": str(query.history)},
    ]
    completion = client.beta.chat.completions.parse(
        model=SMART_LLM, messages=messages, response_format=SearchRequired
    )
    result = completion.choices[0].message.parsed.required
    logger.info(f"Search required: {result}")
    return result


def generate_answer(query: str, context: str, history: list) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant providing answers based on provided context.",
        },
        {"role": "system", "content": "The history of the conversation is as follows:"},
        {"role": "system", "content": str(history)},
    ]
    if context:
        messages.append(
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:",
            }
        )
    else:
        messages.append({"role": "user", "content": f"Question: {query}\nAnswer:"})

    completion = client.chat.completions.create(model=SMART_LLM, messages=messages)
    logger.info("Generated answer using OpenAI.")
    return completion.choices[0].message.content.strip()


from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()


class Hallucination(BaseModel):
    reasoning: str
    isHallucination: bool


def hallucination_check(query: str, answer: str, context: str) -> bool:
    """
    Check if the answer aligns with the retrieved context.
    """
    validation_prompt = (
        f"Validate if the following answer is accurate based on the provided context. "
        f"Respond with 'True' if the answer is not factual and 'False' if it is a good answer. Answer True only if the answer is really completely unrelated to the question.\n\n"
        f"Context: {context}\n\nQuestion: {query}\nAnswer: {answer}"
    )
    messages = [
        {
            "role": "system",
            "content": "You are a validator ensuring factual alignment of answers.",
        },
        {"role": "user", "content": validation_prompt},
    ]
    completion = client.beta.chat.completions.parse(
        model=SMART_LLM, messages=messages, response_format=Hallucination
    )
    result = completion.choices[0].message.parsed.isHallucination
    logger.info(f"Hallucination check result: {result}")
    return result


def refine_query(query: str, answer: str) -> str:
    refinement_prompt = (
        "The given answer appears to be inaccurate. Suggest a refined version of the query "
        "to get a more reliable answer.\n\n"
        f"Original Query: {query}\nInaccurate Answer: {answer}\n"
    )
    messages = [
        {"role": "system", "content": "You are an assistant refining user queries."},
        {"role": "user", "content": refinement_prompt},
    ]
    completion = client.chat.completions.create(model=SIMPLE_LLM, messages=messages)
    refined_query = completion.choices[0].message.content.strip()
    logger.info(f"Refined query: {refined_query}")
    return refined_query


def summarize_history(history: list) -> list:
    conversation = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"

    summary_prompt = (
        "Summarize the following conversation while retaining key information:\n\n"
        f"{conversation}"
    )

    completion = client.chat.completions.create(
        model=SIMPLE_LLM,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversations.",
            },
            {"role": "user", "content": summary_prompt},
        ],
    )

    summary = completion.choices[0].message.content.strip()
    logger.info("Conversation history summarized.")
    return [{"role": "system", "content": summary}]


def generate_corrected_transcript(transcription: str) -> str:
    from prompts import correct_prompt

    messages = [
        {"role": "system", "content": correct_prompt},
        {"role": "user", "content": transcription},
    ]
    completion = client.chat.completions.create(model=SIMPLE_LLM, messages=messages)
    corrected_transcription = completion.choices[0].message.content.strip()
    logger.info("Generated corrected transcription.")
    return corrected_transcription


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
