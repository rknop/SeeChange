from models.exposure import Exposure
from models.image import Image
from models.source_list import SourceList
from models.background import Background
from models.psf import PSF
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.pipeline import Pipeline
from pipeline.exposure_splitter import ExposureSplitter
from pipeline.secure_calibrator import CalibratorSecurer
from pipeline.preprocessing import Preprocessor
from pipeline.detection import Detector
from pipeline.astrocal import AstroCalibrator
from pipeline.photocal import PhotCalibrator

from util.util import NotDefined


class ImageReductionPipeline( Pipeline ):
    _name = "image_reduction"
    _process_name = "image_reduction"

    _all_steps = [ 'split_exposure', 'secure_calibrators', 'preprocessing', 'extraction', 'astrocal', 'photocal' ]

    _external_pipeline_processes = { 'secure_calibrators', 'referencing' }

    _starting_products = { 'exposure': Exposure }

    _process_products = { 'split_exposure': [ (Image, 'raw_image') ],
                          'secure_calibrators': [],
                          'preprocessing': [ (Image, 'image') ],
                          'extraction': [ (SourceList, 'sources'),
                                          (Background, 'bg'),
                                          (PSF, 'psf') ],
                          'astrocal': [ (WorldCoordinates, 'wcs') ],
                          'photocal': [ (ZeroPoint, 'zp') ],
                         }

    # For processes that don't show up in the list, Pipeline will use
    #   the key from _process_products.
    _product_provenance_processes = { 'exposure': NotDefined,
                                      'bg': None,
                                      'psf': None,
                                     }

    _product_upstreams = { 'raw_image': [ 'exposure' ],
                           'image': [ 'raw_image' ],
                           'sources': [ 'image' ],
                           'bg': [ 'sources' ],
                           'psf': [ 'sources' ],
                           'wcs': [ 'sources' ],
                           'zp': [ 'wcs' ],
                          }

    _process_classes = { 'split_exposure': ExposureSplitter,
                         'secure_calibrators': CalibratorSecurer,
                         'preprocessing': Preprocessor,
                         'extraction': Detector,
                         'astrocal': AstroCalibrator,
                         'photocal': PhotCalibrator,
                        }
