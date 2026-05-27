from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from playlist_rag.config import settings
from playlist_rag.schemas import QueryIntent

_SYSTEM = """You parse natural-language playlist requests into structured search intent.

Rules:
- semantic_query MUST be in English, 1-3 sentences, evocative and specific (for embedding search).
- target_duration_minutes: extract from phrases like "2 hours" (120), "45 min" (45). Null if unspecified.
- moods: pick from closed set only when clearly implied. Can be empty.
- energy_levels: pick from closed set when energy level is implied. Can be empty.
- exclude_artists / exclude_genres: extract explicit bans ("sin Kanye", "no reggaeton", "without Drake").
- include_genres: ONLY playlist genre tags from this list when explicitly requested:
  rock, pop, jazz, metal, electronic, hip-hop, r&b, soul, country, folk, blues,
  classical, ambient, indie, latin, punk, k-pop, j-pop, gospel, wellness, gaming.
  Do NOT put "instrumental" here — use min_instrumentalness=0.5 instead.
- min_instrumentalness: set to 0.5 when user wants instrumental / no-vocals focus.
- min/max tempo and energy: only when user gives numeric or clear qualitative bounds.
- prefer_obscure vs prefer_popular: only one should be true; default both false.
- languages: ONLY when the user explicitly wants songs in that language
  (e.g. "solo en español", "english songs only"). Do NOT infer language from
  the language the user wrote their request in.

Mood closed set: joyful, melancholic, angry, calm, energetic, romantic, introspective, nostalgic, rebellious, uplifting

Energy closed set: very_low, low, medium, high, very_high

{format_instructions}
"""

_USER = """User request:
{user_query}

Parse into structured intent."""

_parser = PydanticOutputParser(pydantic_object=QueryIntent)
_format_instructions = _parser.get_format_instructions()

_llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.1,
    timeout=settings.llm_timeout_seconds,
    max_retries=0,
    api_key=settings.openai_api_key or None,
)

_prompt = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("user", _USER)]
).partial(format_instructions=_format_instructions)

_chain = _prompt | _llm | _parser


@retry(
    retry=retry_if_exception_type((OutputParserException, ValidationError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def parse_query(user_query: str) -> QueryIntent:
    if not user_query or not user_query.strip():
        raise ValueError("parse_query requires a non-empty user query")
    return _chain.invoke({"user_query": user_query.strip()})
