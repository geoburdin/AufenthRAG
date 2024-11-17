import os
import logging
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from models import Query, SearchRequired, Hallucination

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)
# Load LLM model names from environment variables
SMART_LLM = os.getenv("SMART_LLM", "gpt-4o-mini")  # Default to gpt-4o-mini if not set
SIMPLE_LLM = os.getenv(
    "SIMPLE_LLM", "gpt-3.5-turbo"
)  # Default to gpt-3.5-turbo if not set


def orchestrator(query: Query) -> tuple[bool, str]:
    messages = [
        {
            "role": "system",
            "content": "You are an assistant deciding whether to perform vector search to send a response on the user's message. Vector search is over german law documents Gesetz über den Aufenthalt, die Erwerbstätigkeit und die Integration von Ausländern im Bundesgebiet1) (Aufenthaltsgesetz - AufenthG)"
            "Respond with 'True' if vector search for more information about the german law is required and 'False' if no additional information is needed."
            "Considering how difficult the question is, the you shall decide whether to use the SMART LLM model or the SIMPLE LLM model to generate the answer.",
        },
        {"role": "user", "content": f"{query.question}"},
        {"role": "system", "content": "Conversation history:"},
        {"role": "system", "content": str(query.history)},
        {"role": "system", "content": "Context:"},
        {"role": "system", "content": str(query.context)},
    ]
    completion = client.beta.chat.completions.parse(
        model=SMART_LLM, messages=messages, response_format=SearchRequired
    )
    required = completion.choices[0].message.parsed.required
    model_use = completion.choices[0].message.parsed.useModel
    logger.info(f"Search required: {required}, Model to use: {model_use}")
    return required, model_use


def generate_answer(query: str, context: str, history: list, model_use: str) -> str:
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

    completion = client.chat.completions.create(model=model_use, messages=messages)
    logger.info("Generated answer using OpenAI.")
    return completion.choices[0].message.content.strip()


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


def split_into_paragraphs(text: str) -> list:
    paragraphs = [
        paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
    ]
    return paragraphs
