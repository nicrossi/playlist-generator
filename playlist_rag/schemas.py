from typing import Literal, Optional

from pydantic import BaseModel, Field

Mood = Literal[
    "joyful",
    "melancholic",
    "angry",
    "calm",
    "energetic",
    "romantic",
    "introspective",
    "nostalgic",
    "rebellious",
    "uplifting",
]

EnergyQualitative = Literal["very_low", "low", "medium", "high", "very_high"]


class TrackSemantics(BaseModel):
    description: str = Field(
        min_length=30,
        max_length=600,
        description=(
            "English prose description of the track's character, mood, and "
            "sonic identity. Specific, evocative, ~2-4 sentences. Avoid "
            "generic phrasing like 'this song explores feelings'."
        ),
    )
    themes: list[str] = Field(
        min_length=1,
        max_length=5,
        description=(
            "Short lowercase noun phrases capturing what the track is about "
            "(e.g., 'heartbreak', 'late-night driving', 'youthful rebellion')."
        ),
    )
    mood: Mood = Field(description="Primary mood. Pick one from the closed set.")
    inferred_subgenre: str = Field(
        max_length=50,
        description=(
            "Specific subgenre label (e.g., 'shoegaze', 'trap-soul', "
            "'cumbia villera'). Free text; may extend beyond playlist tags."
        ),
    )
    energy_qualitative: EnergyQualitative = Field(
        description="Qualitative energy level on a five-point scale."
    )


class NormalizedTrack(BaseModel):
    spotify_track_id: str
    track_name: str
    track_artist: str
    track_album_name: Optional[str] = None
    track_album_release_date: Optional[str] = None

    track_popularity: Optional[int] = None
    popularity_tier: str = "unknown"
    playlist_genres: list[str] = Field(default_factory=list)
    playlist_subgenres: list[str] = Field(default_factory=list)
    playlist_names: list[str] = Field(default_factory=list)

    danceability: Optional[float] = None
    energy: Optional[float] = None
    key: Optional[int] = None
    loudness: Optional[float] = None
    mode: Optional[int] = None
    speechiness: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None
    liveness: Optional[float] = None
    valence: Optional[float] = None
    tempo: Optional[float] = None
    duration_ms: Optional[int] = None
    time_signature: Optional[int] = None

    lyrics_clean: Optional[str] = None
    language: Optional[str] = None
    genius_tag: Optional[str] = None

    @property
    def has_lyrics(self) -> bool:
        return bool(self.lyrics_clean and self.lyrics_clean.strip())


class QueryIntent(BaseModel):

    semantic_query: str = Field(
        description=(
            "English search phrase capturing mood, activity, and sonic intent. "
            "Used for embedding-based retrieval."
        ),
    )
    target_duration_minutes: float | None = Field(
        default=None,
        description="Desired playlist length in minutes.",
    )
    moods: list[Mood] = Field(default_factory=list)
    energy_levels: list[EnergyQualitative] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    include_genres: list[str] = Field(default_factory=list)
    exclude_genres: list[str] = Field(default_factory=list)
    exclude_artists: list[str] = Field(default_factory=list)
    min_tempo: float | None = None
    max_tempo: float | None = None
    min_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    min_instrumentalness: float | None = Field(default=None, ge=0.0, le=1.0)
    prefer_popular: bool = False
    prefer_obscure: bool = False


class RetrievedTrack(BaseModel):
    track_id: int
    spotify_track_id: str
    track_name: str
    track_artist: str
    description: str
    mood: str | None = None
    energy_qualitative: str | None = None
    inferred_subgenre: str | None = None
    tempo: float | None = None
    energy: float | None = None
    valence: float | None = None
    instrumentalness: float | None = None
    duration_ms: int | None = None
    popularity_tier: str | None = None
    vector_score: float = 0.0
    final_score: float = 0.0


class PlaylistTrack(RetrievedTrack):
    position: int
    reason: str = ""


class PlaylistResult(BaseModel):
    query: str
    intent: QueryIntent
    tracks: list[PlaylistTrack]
    total_duration_minutes: float
    explanation: str


class GenerationTrace(BaseModel):

    retrieved_candidates: list[RetrievedTrack] = Field(default_factory=list)
    retrieval_relaxed: bool = False


class GenerationRun(BaseModel):

    result: PlaylistResult
    trace: GenerationTrace
