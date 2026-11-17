import pytest
from psycopg import sql

from models.base import PGDB
from models.provenance import Provenance
from pipeline.pipeline import Pipeline


@pytest.fixture
def pipeline_for_tests( loaded_ls4_singleamp_exposure ):
    try:
        pip = Pipeline.get( 'image_reduction', exposure=loaded_ls4_singleamp_exposure, provtag='imred_pip_test' )
        yield pip
    finally:
        # TODO DELETE ALL PRODUCTS
        with PGDB() as pgdb:
            rows, _cols = pgdb.execute( "SELECT t.provenance_id, p.process "
                                        "FROM provenance_tags t "
                                        "INNER JOIN provenances p ON t.provenance_id=p._id "
                                        "WHERE tag='imred_pip_test'" )
            provs = [ r[0] for r in rows if r[1] != 'acquire_exposure' ]
            if len(provs) > 0:
                pgdb.execute( "DELETE FROM provenance_tags WHERE tag='imred_pip_test'" )
                pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE _id=ANY(ARRAY[{provs}])" )
                              .format( provs=sql.SQL(",").join(provs) ) )
                pgdb.commit()


def test_image_reduction_pipeline( pipeline_for_tests ):
    # Make sure the provenances all got created
    provs = Provenance.get_for_tag( 'imred_pip_test' )
    assert set( provs.keys() ) == { 'acquire_exposure', 'split_exposure', 'secure_calibrators',
                                    'preprocessing', 'extraction', 'astrocal', 'photocal' }
