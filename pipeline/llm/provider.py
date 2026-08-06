"""Groq/Llama LLM provider and the answer-generation chain."""

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from platform_layer.config.settings import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    MAX_ANSWER_TOKENS,
)
from pipeline.llm.prompts import get_analyst_prompt, get_summary_prompt


def get_llm() -> ChatGroq:
    """Create the Groq-hosted Llama model, validating the API key first."""
    if not GROQ_API_KEY or GROQ_API_KEY.strip() in ("", "your_groq_api_key_here"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys before starting the app."
        )
    return ChatGroq(
        temperature=LLM_TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        max_tokens=MAX_ANSWER_TOKENS,
    )


def build_answer_chain(llm):
    """Build the source-grounded answer chain (prompt -> llm -> text).

    Takes a pre-built, source-labeled context string (see
    retrieval.context_builder) plus conversation-memory fields and returns the
    model's formatted answer as a string.
    """
    return get_analyst_prompt() | llm | StrOutputParser()


def summarize_conversation(llm, history: str) -> str:
    """Produce a short factual summary of a conversation transcript."""
    prompt = get_summary_prompt()
    chain = prompt | llm
    result = chain.invoke({"history": history})
    # ChatGroq returns a message object; fall back to str() for safety.
    return getattr(result, "content", str(result)).strip()
