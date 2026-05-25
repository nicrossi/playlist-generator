from contextlib import contextmanager
from typing import Iterator

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from playlist_rag.config import settings


class Base(DeclarativeBase):
    pass


class Track(Base):
    __tablename__ = "tracks"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    spotify_track_id = Column(String, nullable=False, unique=True)
    track_name = Column(Text, nullable=False)
    track_artist = Column(Text, nullable=False)
    track_album_name = Column(Text)
    track_album_release_date = Column(String)

    danceability = Column(Float)
    energy = Column(Float)
    key = Column(Integer)
    loudness = Column(Float)
    mode = Column(Integer)
    speechiness = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    valence = Column(Float)
    tempo = Column(Float)
    duration_ms = Column(Integer)
    time_signature = Column(Integer)

    track_popularity = Column(Integer)
    popularity_tier = Column(String, nullable=False)
    playlist_genres = Column(ARRAY(String))
    playlist_subgenres = Column(ARRAY(String))
    playlist_names = Column(ARRAY(String))

    lyrics = Column(Text)
    language = Column(String)
    genius_tag = Column(String)

    description = Column(Text)
    mood = Column(String)
    inferred_subgenre = Column(String)
    energy_qualitative = Column(String)

    indexed_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    indexer_version = Column(String, nullable=False)


class TrackEmbedding(Base):
    __tablename__ = "track_embeddings"

    track_id = Column(
        BigInteger,
        ForeignKey("tracks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding = Column(Vector(1536), nullable=False)
    model_name = Column(String, nullable=False)
    embedded_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TrackTheme(Base):
    __tablename__ = "track_themes"

    track_id = Column(
        BigInteger,
        ForeignKey("tracks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    theme = Column(String, primary_key=True)


class TrackIndexStatus(Base):
    __tablename__ = "track_index_status"

    spotify_track_id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    last_error = Column(Text)
    last_attempt_at = Column(DateTime(timezone=True))
    attempt_count = Column(Integer, nullable=False, server_default="0")


_engine = create_engine(
    settings.database_url, pool_pre_ping=True, future=True
)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_session() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
