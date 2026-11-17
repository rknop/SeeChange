import pytest
import contextlib

from psycopg import sql

from models.base import PGDB
from models.provenance import Provenance, ProvenanceTag
from models.calibrator_set import CalibratorSet
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from pipeline.pipeline import Pipeline
from pipeline.parameters import Parameters
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from util.util import NotDefined


@pytest.fixture( scope="module" )
def pipeline_test_rawprov():
    try:
        prov = Provenance( process='test_image' )
        prov.insert_if_needed()
        yield prov
    finally:
        with PGDB() as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE _id={provid}" ).format( provid=prov.id ) )
            pgdb.commit()


@pytest.fixture( scope="module" )
def pipeline_test_calprov():
    try:
        prov = Provenance( process='secure_calibrators' )
        prov.insert_if_needed()
        yield prov
    finally:
        with PGDB() as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE _id={provid}" ).format( provid=prov.id ) )
            pgdb.commit()


@pytest.fixture( scope="module" )
def pipeline_test_provs( pipeline_test_rawprov, pipeline_test_calprov ):
    rawprov = pipeline_test_rawprov
    calprov = pipeline_test_calprov

    @contextlib.contextmanager
    def pipeline_test_prov_factory( pipeline=None, provtag=None, usecal=True, pars={} ):
        pars = { 'preprocessing': {}, 'extraction': {}, 'astrocal': {}, 'photocal': {} } | pars
        provs = { rawprov.process: rawprov }
        if usecal:
            provs[ calprov.process ] = calprov
        try:
            with PGDB() as pgdb:
                keys = [ 'preprocessing', 'extraction', 'astrocal', 'photocal' ]
                clses = [ Preprocessor, Detector, AstroCalibrator, PhotCalibrator ]
                for i, (key, cls) in enumerate( zip(keys, clses) ):
                    procob = cls( pipeline=pipeline, **(pars[key]) )
                    procpars = procob.pars.get_critical_pars()
                    if i == 0:
                        upstrs = [ rawprov, calprov ] if usecal else [ rawprov ]
                    else:
                        upstrs = [ provs[ keys[ i-1 ] ] ]
                    provs[ key ] = Provenance( process=key, parameters=procpars, upstreams=upstrs, pgdb=pgdb )

                if provtag is not None:
                    ProvenanceTag.addtag( provtag, list( provs.values() ), pgdb=pgdb )

            yield provs

        finally:
            with PGDB() as pgdb:
                if provtag is not None:
                    for prov in provs.values():
                        pgdb.execute( sql.SQL( "DELETE FROM provenance_tags "
                                               "WHERE provenance_id={provid} "
                                               "AND tag={tag}"
                                              ).format( provid=prov.id, tag=provtag ) )
                for prov in provs.values():
                    if prov.process in ( rawprov.process, calprov.process ):
                        continue
                    pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE _id={provid}"
                                          ).format( provid=prov.id ) )
                pgdb.commit()

    return pipeline_test_prov_factory


class ParsPipelineForTests( Parameters ):
    def __init__( self, **kwargs ):
        super().__init__()

        self.add_par(
            name = 'extraction',
            default = {},
            par_types = dict,
            docstring = "Modifications to base extraction config for this pipeline",
            critical = True
        )

        self._enforce_new_new_attrs = True
        self.override( kwargs )


class PipelineForTests( Pipeline ):
    _name = "pipeline_for_tests"
    _process_name = "pipeline_for_tests"
    _pars_class = ParsPipelineForTests

    _all_steps = [ 'preprocessing', 'extraction', 'astrocal', 'photocal' ]
    _starting_products = { 'raw_image': Image,
                           'calibrator_set': CalibratorSet
                           }
    _process_products = { 'preprocessing': [ (Image, 'image') ],
                          'extraction': [ (SourceList, 'sources'),
                                          (Background, 'bg'),
                                          (PSF, 'psf') ],
                          'astrocal': [ (WorldCoordinates, 'wcs') ],
                          'photocal': [ (ZeroPoint, 'zp') ]
                         }
    _product_provenance_processes = { 'bg': None,
                                      'psf': None
                                      }
    _product_upstreams = { 'image': [ 'raw_image', 'calibrator_set' ],
                           'sources': [ 'image' ],
                           'bg': [ 'sources' ],
                           'psf': [ 'sources' ],
                           'wcs': [ 'sources' ],
                           'zp': [ 'wcs' ]
                          }
    _process_classes = { 'preprocessing': Preprocessor,
                         'extraction': Detector,
                         'astrocal': AstroCalibrator,
                         'photocal': PhotCalibrator
                        }


class NoCalPipelineForTests( PipelineForTests ):
    _name = 'no_cal_pipeline_for_tests'
    _process_name = 'no_cal_pipeline_for_tests'
    _starting_products = [ 'raw_image' ],
    _product_upstreams = PipelineForTests._product_upstreams | { 'image': [ 'raw_image' ] }


def test_pipeline( pipeline_test_provs, pipeline_test_rawprov,  pipeline_test_calprov ):

    pip = Pipeline.get( 'pipeline_for_tests',
                        starting_prov={ 'raw_image': pipeline_test_rawprov,
                                        'calibrator_set': pipeline_test_calprov } )

    with pipeline_test_provs( pip ) as expectedprovs:
        for process, prov in expectedprovs.items():
            assert prov.id == pip.provenances[ process ].id
        assert pip.starting_prov == { p: expectedprovs[p] for p in [ 'raw_image', 'calibrator_set' ] }
        assert pip.provtag is None
        assert pip.product_upstreams == { 'raw_image': NotDefined,
                                          'calibrator_set': NotDefined,
                                          'image': [ 'raw_image', 'calibrator_set' ],
                                          'sources': [ 'image' ],
                                          'bg': [ 'sources' ],
                                          'psf': [ 'sources' ],
                                          'wcs': [ 'sources' ],
                                          'zp': [ 'wcs' ]
                                         }
        assert pip.product_processes == { 'raw_image': NotDefined,
                                          'calibrator_set': NotDefined,
                                          'image': 'preprocessing',
                                          'sources': 'extraction',
                                          'bg': 'extraction',
                                          'psf': 'extraction',
                                          'wcs': 'astrocal',
                                          'zp': 'photocal'
                                         }


    # TODO : test passing modified upstreams / classes to the constructor



    import pdb; pdb.set_trace()
