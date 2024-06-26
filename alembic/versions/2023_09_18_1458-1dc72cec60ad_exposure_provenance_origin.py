"""exposure_provenance_origin

Revision ID: 1dc72cec60ad
Revises: 93d7c3c93a06
Create Date: 2023-09-18 14:58:14.964366

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1dc72cec60ad'
down_revision = '93d7c3c93a06'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('exposures', sa.Column('provenance_id', sa.String(), nullable=False))
    op.add_column('exposures', sa.Column('origin_identifier', sa.Text(), nullable=True))
    op.drop_index('ix_exposures_telescope', table_name='exposures')
    op.create_index(op.f('ix_exposures_origin_identifier'), 'exposures', ['origin_identifier'], unique=False)
    op.create_index(op.f('ix_exposures_provenance_id'), 'exposures', ['provenance_id'], unique=False)
    op.create_foreign_key(None, 'exposures', 'provenances', ['provenance_id'], ['id'], ondelete='CASCADE')
    op.drop_column('exposures', 'telescope')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('exposures', sa.Column('telescope', sa.TEXT(), autoincrement=False, nullable=False))
    op.drop_constraint(None, 'exposures', type_='foreignkey')
    op.drop_index(op.f('ix_exposures_provenance_id'), table_name='exposures')
    op.drop_index(op.f('ix_exposures_origin_identifier'), table_name='exposures')
    op.create_index('ix_exposures_telescope', 'exposures', ['telescope'], unique=False)
    op.drop_column('exposures', 'origin_identifier')
    op.drop_column('exposures', 'provenance_id')
    # ### end Alembic commands ###
