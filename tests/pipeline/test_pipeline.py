import pytest
import contextlib

from psycopg import sql

from models.base import PGDB
from models.provenance import Provenance, ProvenanceTag
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from pipeline.pipeline import Pipeline
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator


@pytest.fixture( scope="module" )
def pipeline_test_rawprov():
    try:
        prov = Provenance( process='test_image' )
        prov.insert_if_needed()
        yield prov
    finally:
        with PGDB() as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE id={provid}" ).format( provid=prov.id ) )
            pgdb.commit()


@pytest.fixture( scope="module" )
def pipeline_test_calprov():
    try:
        prov = Provenance( process='secure_calibrators' )
        prov.insert_if_needed()
        yield prov
    finally:
        with PGDB() as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE id={provid}" ).format( provid=prov.id ) )
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
                procob = Preprocessor( pipeline=pipeline, **(pars['preprocessing']) )
                pars = procob.get_critical_parameters()
                upstrs = [ rawprov, calprov ] if usecal else [ rawprov ]
                provs[ 'preprocessing' ] = Provenance( process='preprocessing', parameters=pars,
                                                       upstreams=upstrs, pgdb=pgdb )
                provs[ 'preprocessing' ].insert_if_needed( pgdb=pgdb )
                procob = Detector( pipeline=pipeline, **(pars['extraction']) )
                pars = procob.get_critical_parameters()
                provs[ 'extraction' ] = Provenance( process='extraction', parameters=pars,
                                                    upstreams=[ provs['preprocessing'] ], pgdb=pgdb )
                provs[ 'extraction' ].insert_if_needed( pgdb=pgdb )
                procob = AstroCalibrator( pipeline=pipeline, **(pars['astrocal']) )
                pars = procob.get_critical_parameters()
                provs[ 'astrocal' ] = Provenance( process='astrocal', parameters=pars,
                                                  upstreams=[ provs['extraction'] ], pgdb=pgdb )
                provs[ 'astrocal' ].insert_if_needed()
                procob = PhotCalibrator( pipeline=pipeline, **(pars['photocal']) )
                pars = procob.get_critical_parameters()
                provs[ 'photocal' ] = Provenance( process='photocal', parameters=pars,
                                                  upstreams=[ provs['astrocal'] ], pgdb=pgdb )
                provs['photocal'].insert_if_needed()

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
                    pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE provenance_id={provid}"
                                          ).format( provid=prov.id ) )
                pgdb.commit()

    return pipeline_test_prov_factory



class PipelineForTests( Pipeline ):
    _name = "pipeline_for_tests"
    _all_steps = [ 'preprocessing', 'extraction', 'astrocal', 'photocal' ]
    _starting_products = [ 'raw_image', 'calibrator_set' ]
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
    _starting_products = [ 'raw_image' ],
    _product_upstreams = PipelineForTests._product_upstreams | { 'image': [ 'raw_image' ] }


def test_pipeline( pipeline_test_provs, pipeline_test_rawprov,  pipeline_test_calprov ):

    pip = Pipeline.get( 'pipeline_for_tests',
                        starting_prov={ 'split_exposure': pipeline_test_rawprov,
                                        'secure_calibrators': pipeline_test_calprov } )
    import pdb; pdb.set_trace()
    pass
