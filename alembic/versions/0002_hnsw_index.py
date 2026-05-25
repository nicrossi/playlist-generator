"""HNSW vector index on track_embeddings.embedding (cosine).

Revision ID: 0002_hnsw_index
Revises: 0001_initial
Create Date: 2026-05-24

Apply this migration AFTER `python -m playlist_rag.cli.index` has completed
on the full dataset. Building HNSW on a populated table is 5-10x faster than
maintaining the index incrementally during inserts.

Workflow:
    alembic upgrade 0001_initial      # create tables, NO HNSW
    python -m playlist_rag.cli.index --parquet data/unified_tracks.parquet
    alembic upgrade head              # build HNSW on populated table
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0002_hnsw_index"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX idx_track_embeddings_hnsw
        ON track_embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_track_embeddings_hnsw")
