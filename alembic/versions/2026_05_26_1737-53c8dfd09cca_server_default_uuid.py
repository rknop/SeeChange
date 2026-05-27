"""server_default_uuid

Revision ID: 53c8dfd09cca
Revises: a5d1ea8761c0
Create Date: 2026-05-26 17:37:36.602065

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '53c8dfd09cca'
down_revision = 'a5d1ea8761c0'
branch_labels = None
depends_on = None


tabs = [ 'backgrounds', 'archive_locks', 'calibrator_files', 'calibfile_downloadlock',
         'catalog_excerpts', 'gaiadr3_downloadlock', 'cutouts', 'data_files',
         'deepscore_sets', 'deepscores', 'exposures', 'fake_sets', 'fake_analysis',
         'images', 'sensor_sections', 'knownexposures', 'pipelineworkers',
         'measurement_sets', 'measurements', 'objects', 'object_positions',
         'code_versions', 'provenance_tags', 'psfs', 'refs', 'refsets', 'reports',
         'source_lists', 'world_coordinates', 'zero_points' ]
                 
def upgrade() -> None:
    conn = op.get_bind()
    for tab in tabs:
        conn.execute(
            sa.sql.text( f"ALTER TABLE {tab} ALTER COLUMN _id SET DEFAULT gen_random_uuid()" )
        )


def downgrade() -> None:
    conn = op.get_bind()
    for tab in tabs:
        conn.execute(
            sa.sql.text( f"ALTER TABLE {tab} ALTER COLUMN _id DROP DEFAULT" )
        )
