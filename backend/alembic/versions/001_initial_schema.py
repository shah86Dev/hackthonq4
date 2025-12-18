"""Initial database schema for Book-Embedded RAG Chatbot

Revision ID: 001
Revises:
Create Date: 2025-12-17 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create books table
    op.create_table('books',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create chunks table
    op.create_table('chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('book_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chapter', sa.String(length=200), nullable=True),
        sa.Column('section', sa.String(length=200), nullable=True),
        sa.Column('page_range', sa.String(length=50), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('chunk_id', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['book_id'], ['books.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create queries table
    op.create_table('queries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('book_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('mode', sa.String(length=20), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('selected_text', sa.Text(), nullable=True),
        sa.Column('retrieved_chunk_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('latency', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['book_id'], ['books.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create answers table
    op.create_table('answers',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('query_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('answer_text', sa.Text(), nullable=False),
        sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['query_id'], ['queries.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create feedback table
    op.create_table('feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('answer_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_rating', sa.String(length=10), nullable=True),
        sa.Column('correction', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['answer_id'], ['answers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("user_rating IN ('positive', 'negative', 'neutral')", name='feedback_user_rating_check')
    )


def downgrade() -> None:
    # Drop feedback table first (due to foreign key constraints)
    op.drop_table('feedback')

    # Drop answers table
    op.drop_table('answers')

    # Drop queries table
    op.drop_table('queries')

    # Drop chunks table
    op.drop_table('chunks')

    # Drop books table
    op.drop_table('books')