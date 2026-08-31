import pytest

from psycopg import sql

from models.base import PGDB
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.coaddition import Coadder
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.scoring import Scorer
from pipeline.fakeinjection import FakeInjector
from pipeline.top_level import Pipeline


@pytest.fixture
def preprocessor( test_config ):
    p = Preprocessor( **test_config.value('preprocessing') )
    return p


@pytest.fixture
def extractor( test_config ):
    extr = Detector( **test_config.value('extraction') )
    return extr


@pytest.fixture
def astrometor( test_config ):
    a = AstroCalibrator( **test_config.value('astrocal') )
    return a


@pytest.fixture
def coadder( test_config ):
    c = Coadder( **test_config.value('coaddition.coaddition') )
    return c


@pytest.fixture
def subtractor( test_config ):
    s = Subtractor( **test_config.value('subtraction') )
    return s


@pytest.fixture
def detector( test_config ):
    d = Detector( **test_config.value('detection') )
    return d


@pytest.fixture
def cutter( test_config ):
    c = Cutter( **test_config.value('cutting') )
    return c


@pytest.fixture
def measurer( test_config ):
    m = Measurer( **test_config.value('measuring') )
    return m


@pytest.fixture
def scorer( test_config ):
    s = Scorer( **test_config.value('scoring') )
    return s


@pytest.fixture
def fakeinjector( test_config ):
    f = FakeInjector( **test_config.value('fakeinjection') )
    return f


@pytest.fixture
def pipeline_for_tests():
    kwargs = { 'pipeline': { 'provenance_tag': 'pipeline_for_tests' } }
    # Unlike most of the component objects, top_level.Pipeline actually does
    #   itself read the config in its __init__, and uses the passed kwargs
    #   to override the config.
    p = Pipeline( **kwargs )
    yield p

    # Clean up the provenance tag potentially created by the pipeline
    with PGDB() as pgdb:
        pgdb.execute_nofetch( sql.SQL( "DELETE FROM provenance_tags WHERE tag={tag}" )
                              .format( tag='pipeline_for_tests' ) )
        pgdb.commit()
