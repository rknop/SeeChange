import copy

from models.provenance import Provenance
from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.reference import Reference
from models.cutouts import Cutouts
from models.measurement import MeasurementSet, Measurements
from models.deepscore import DeepScoreSet, DeepScores

from pipeline.pipeline import Pipeline
from pipeline.exposure_splitter import ExposureSplitter
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astro_cal import AstroCalibrator
from pipeline.photo_cal import PhotCalibrator
from improc.alignment import Aligner
from pipeline.subtraction import Subtractor
from pipeline.cutting import Cutter
from pipeline.measuring import Measurer
from pipeline.scoring import Scorer
from pipeline.alerting import Alerter

from util.logger import SCLogger
from util.util import NotDefined


class DifferenceImagingTransientDiscoveryPipeline( Pipeline ):
    _name = "dia_transientsearch"

    _all_steps = [ 'split_exposure', 'secure_calibrators', 'preprocessing', 'extraction', 'astrocal', 'photocal',
                   'secure_reference', 'alignment', 'subtraction', 'detection', 'cutting', 'measuring', 'scoring',
                   'alerting' ]

    _starting_products = { 'exposure': Exposure }

    _process_products = { 'split_exposure': [ (Image, 'raw_image') ],
                          'secure_calibrators': [],
                          'preprocessing': [ (Image, 'image') ],
                          'extraction': [ (SourceList, 'sources'),
                                          (Background, 'bg'),
                                          (PSF, 'psf') ],
                          'astrocal': [ (WorldCoordinates, 'wcs') ],
                          'photocal': [ (ZeroPoint, 'zp') ],
                          'alignment': [ (Image, 'aligned_ref'),
                                         (SourceList, 'aligned_ref_sources'),
                                         (Background, 'aligned_ref_bg' ),
                                         (PSF, 'aligned_ref_psf' ),
                                         (WorldCoordinates, 'aligned_ref_wcs'),
                                         (ZeroPoint, 'aligned_ref_zp') ],
                          'secure_reference': [ (Reference, 'ref') ],
                          'subtraction': [ (Image, 'subimage') ],
                          'detection': [ (SourceList, 'detections') ],
                          'cutting': [ (Cutouts, 'cutouts') ],
                          'measuring': [ (MeasurementSet, 'measurementset'),
                                         (list, 'measurements', Measurements) ],
                          'scoring': [ (DeepScoreSet, 'deepscoreset'),
                                       (list, 'deepscores', DeepScores) ],
                          'alerting': []
                         }

    # For processes that don't show up in the list, Pipeline will use
    #   the key from _process_products.
    _product_provenance_processes = { 'ref': 'referencing',
                                      'aligned_ref': 'Image.trim',
                                      'aligned_ref_sources': 'Image.trim.sources',
                                      'aligned_ref_wcs': 'Image.trim.wcs',
                                      'aligned_ref_zp': 'Image.trim.zp',
                                     }


    _product_upstreams = { 'exposure': NotDefined,
                           'raw_image': [ 'exposure' ],
                           'image': [ 'raw_image' ],
                           'sources': [ 'image' ],
                           'bg': [ 'sources' ],
                           'psf': [ 'sources' ],
                           'wcs': [ 'sources' ],
                           'zp': [ 'wcs' ],
                           'ref': NotDefined,
                           'aligned_ref': [ 'ref', 'wcs' ],
                           'aligned_ref_sources': [ 'aligned_ref' ],
                           'aligned_ref_bg': [ 'aligned_ref_sources' ],
                           'aligned_ref_psf': [ 'aligned_ref_sources' ],
                           'aligned_ref_wcs': [ 'aligned_ref_sources' ],
                           'aligned_ref_zp': [ 'aligned_ref_wcs' ],
                           'subimage': [ 'aligned_ref_zp', 'zp' ],
                           'detections': [ 'subimage' ],
                           'cutouts': [ 'detections' ],
                           'measurementset': [ 'cutouts' ],
                           'measurements': [ 'measurementset' ],
                           'deepscoreset': [ 'measurementset' ],
                           'deepscores': [ 'deepscoreset' ]
                          }

    _process_classes = { 'split_exposure': ExposureSplitter,
                         'secure_calibrators': "OMG ROB FIGURE THIS OUT",
                         'preprocessing': Preprocessor,
                         'extraction': Detector,
                         'astrocal': AstroCalibrator,
                         'photocal': PhotCalibrator,
                         'secure_reference': "OMG ROB FIGURE THIS OUT",
                         'alignment': Aligner,
                         'subtraction': Subtractor,
                         'detection': Detector,
                         'cutting': Cutter,
                         'measuring': Measurer,
                         'scoring': Scorer,
                         'alerting': Alerter
                        }


    def make_prov_tree( self, starting_prov, starting_point=Exposure,
                        save_raw_image=False, image_already_preprocessed=False,
                        **kwargs ):
        """See Pipeline::make_prov_tree.

        starting_point must be either Exposure or Image

        If starting_point is Exposure, then the pipeline will start at
        split_exposure.  Otherwise, starting_point must be Image.  In
        that case, if the process of starting_prov is 'preprocessing',
        then the pipeline will start at 'secure_calibrators' followed by
        'preprocessing'.  Otherwise, the pipeline will start at
        'extraction'.

        """

        if starting_point not in ( Exposure, Image, 'Exposure', 'Image' ):
            raise TypeError( f"starting_point must be Exposure or Image, not {starting_point}" )

        starting_prov = Provenance.get( starting_prov )
        if starting_prov is None:
            raise ValueError( f"Failed to find provenance {starting_prov}" )

        if image_already_preprocessed and save_raw_image:
            SCLogger.warning( "save_raw_image ignored,image_already_preprocessed is set" )

        upstreams = copy.deepcopy( self._tree_upstreams )
        if starting_point in ( Image, 'Image' ):
            del upstreams[ 'split_exposure' ]
            if image_already_preprocessed:
                del upstreams[ 'preprocessing' ]
                # TODO THINK : is secure_calibrators only for preprocessing?
                del upstreams[ 'secure_calibrators' ]
                upstreams['extraction'] = [ 'starting_point' ]
            else:
                upstreams['preprocessing'] = [ 'starting_point', 'secure_calibrators' ]

        else:
            if image_already_preprocessed:
                del upstreams['preprocessing']
                # TODO THINK : is secure_calibrators only for preprocessing?
                del upstreams[ 'secure_calibrators' ]
                upstreams['extraction'] = [ 'split_exposure' ]
            elif not save_raw_image:
                del upstreams['split_exposure']
                upstreams['preprocessing'] = [ 'starting_point', 'secure_calibrators' ]

        return super().make_prov_tree( starting_prov, _tree_upstreams=upstreams, **kwargs )
