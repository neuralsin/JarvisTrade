"""Inverted signal weaponization columns

This migration adds columns to the models table to track inverted signal state.

Revision ID: add_inversion_columns
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


def upgrade():
    """Add inversion tracking columns to models table"""
    op.add_column('models', sa.Column('is_inverted', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('models', sa.Column('model_state', sa.Text(), nullable=True, server_default='NORMAL'))
    op.add_column('models', sa.Column('inversion_metadata', JSONB(), nullable=True))


def downgrade():
    """Remove inversion tracking columns"""
    op.drop_column('models', 'inversion_metadata')
    op.drop_column('models', 'model_state')
    op.drop_column('models', 'is_inverted')
