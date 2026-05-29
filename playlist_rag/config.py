from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    database_url: str = (
        "postgresql://playlist:password@localhost:5435/playlist_rag"
    )

    openai_api_key: str = ""
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    indexer_version: str = "0.1.0"
    lyrics_excerpt_chars: int = 800
    batch_commit_size: int = 10

    retrieval_top_k: int = 80
    default_duration_minutes: float = 60.0
    max_playlist_tracks: int = 50
    artist_spacing: int = 2

    eval_judge_batch_size: int = 10
    eval_max_retrieved_to_judge: int = 40
    eval_judge_temperature: float = 0.0
    eval_answer_relevance_num_questions: int = 3


settings = Settings()
