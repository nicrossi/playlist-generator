"""initial schema: tracks, track_embeddings, track_themes, track_index_status

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-24

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tracks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("spotify_track_id", sa.String, nullable=False, unique=True),
        sa.Column("track_name", sa.Text, nullable=False),
        sa.Column("track_artist", sa.Text, nullable=False),
        sa.Column("track_album_name", sa.Text),
        sa.Column("track_album_release_date", sa.String),
        sa.Column("danceability", sa.Float),
        sa.Column("energy", sa.Float),
        sa.Column("key", sa.Integer),
        sa.Column("loudness", sa.Float),
        sa.Column("mode", sa.Integer),
        sa.Column("speechiness", sa.Float),
        sa.Column("acousticness", sa.Float),
        sa.Column("instrumentalness", sa.Float),
        sa.Column("liveness", sa.Float),
        sa.Column("valence", sa.Float),
        sa.Column("tempo", sa.Float),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("time_signature", sa.Integer),
        sa.Column("track_popularity", sa.Integer),
        sa.Column("popularity_tier", sa.String, nullable=False),
        sa.Column("playlist_genres", sa.ARRAY(sa.String)),
        sa.Column("playlist_subgenres", sa.ARRAY(sa.String)),
        sa.Column("playlist_names", sa.ARRAY(sa.String)),
        sa.Column("lyrics", sa.Text),
        sa.Column("language", sa.String),
        sa.Column("genius_tag", sa.String),
        sa.Column("description", sa.Text),
        sa.Column("mood", sa.String),
        sa.Column("inferred_subgenre", sa.String),
        sa.Column("energy_qualitative", sa.String),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("indexer_version", sa.String, nullable=False),
    )
    op.create_index("idx_tracks_energy", "tracks", ["energy"])
    op.create_index("idx_tracks_tempo", "tracks", ["tempo"])
    op.create_index("idx_tracks_valence", "tracks", ["valence"])
    op.create_index("idx_tracks_mood", "tracks", ["mood"])
    op.create_index("idx_tracks_language", "tracks", ["language"])
    op.create_index("idx_tracks_popularity_tier", "tracks", ["popularity_tier"])
    op.execute(
        "CREATE INDEX idx_tracks_playlist_genres ON tracks USING GIN (playlist_genres)"
    )

    op.create_table(
        "track_embeddings",
        sa.Column(
            "track_id",
            sa.BigInteger,
            sa.ForeignKey("tracks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("model_name", sa.String, nullable=False),
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "track_themes",
        sa.Column(
            "track_id",
            sa.BigInteger,
            sa.ForeignKey("tracks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("theme", sa.String, primary_key=True),
    )
    op.create_index("idx_track_themes_theme", "track_themes", ["theme"])

    op.create_table(
        "track_index_status",
        sa.Column("spotify_track_id", sa.String, primary_key=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("last_error", sa.Text),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True)),
        sa.Column(
            "attempt_count", sa.Integer, nullable=False, server_default="0"
        ),
    )
    op.create_index(
        "idx_track_index_status_status", "track_index_status", ["status"]
    )


def downgrade() -> None:
    op.drop_table("track_index_status")
    op.drop_table("track_themes")
    op.drop_table("track_embeddings")
    op.drop_table("tracks")
