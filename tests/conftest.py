import os
import io
import warnings
import pytest
import uuid
import shutil
import pathlib
import math
import random
import subprocess

import numpy as np
from scipy.integrate import dblquad

import sqlalchemy as sa
import sqlalchemy.orm

from astropy.io import fits

import selenium.webdriver

from util.config import Config
from models.base import (
    FileOnDiskMixin,
    SmartSession,
    Psycopg2Connection,
    CODE_ROOT,
    get_all_database_objects,
    setup_warning_filters,
    get_archive_object
)
from models.provenance import Provenance
from models.catalog_excerpt import CatalogExcerpt
from models.user import AuthUser, AuthGroup
from models.image import Image
from models.source_list import SourceList
from models.psf import PSF, PSFExPSF
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.fakeset import FakeSet

from util.archive import Archive
from util.util import remove_empty_folders, env_as_bool
from util.retrydownload import retry_download
from util.logger import SCLogger

from pipeline.data_store import DataStore, ProvenanceTree

# Set this to False to avoid errors about things left over in the database and archive
#   at the end of tests.  In general, we want this to be True, so we can make sure
#   that our tests are properly cleaning up after themselves.  However, the errors
#   from this can hide other errors and failures, so when debugging, set it to False.
verify_archive_database_empty = True
# verify_archive_database_empty = False


pytest_plugins = [
    'tests.fixtures.simulated',
    'tests.fixtures.decam',
    'tests.fixtures.ztf',
    'tests.fixtures.ptf',
    'tests.fixtures.pipeline_objects',
    'tests.fixtures.datastore_factory',
    'tests.fixtures.conductor',
    'tests.fixtures.webap',
]

ARCHIVE_PATH = None

SKIP_WARNING_TESTS = False

# We may want to turn this on only for tests, as it may add a lot of runtime/memory overhead
# ref: https://www.mail-archive.com/python-list@python.org/msg443129.html
# os.environ["SEECHANGE_TRACEMALLOC"] = "1"


# this fixture should be the first thing loaded by the test suite
# (session is the pytest session, not an SQLAlchemy session)
def pytest_sessionstart(session):
    # Will be executed before the first test
    global SKIP_WARNING_TESTS

    if False:  # this is only to make the warnings into errors, so it is easier to track them down...
        warnings.filterwarnings('error', append=True)  # comment this out in regular usage
        SKIP_WARNING_TESTS = True

    setup_warning_filters()  # load the list of warnings that are to be ignored (not just in tests)
    # below are additional warnings that are ignored only during tests:

    # ignore warnings from photometry code that occur for cutouts with mostly zero values
    warnings.filterwarnings('ignore', message=r'.*Background mean=.*, std=.*, normalization skipped!.*')

    # make sure to load the test config
    test_config_file = os.getenv( "SEECHANGE_CONFIG", None )
    if test_config_file is None:
        test_config_file = str((pathlib.Path(__file__).parent.parent
                                / 'tests' / 'seechange_config_test.yaml').resolve())
    Config.get(configfile=test_config_file, setdefault=True)
    FileOnDiskMixin.configure_paths()
    # SCLogger.setLevel( logging.INFO )

    # get rid of any catalog excerpts from previous runs:
    with SmartSession() as session:
        catexps = session.scalars(sa.select(CatalogExcerpt)).all()
        for catexp in catexps:
            if os.path.isfile(catexp.get_fullpath()):
                os.remove(catexp.get_fullpath())
            session.delete(catexp)
        session.commit()


def any_objects_in_database( dbsession ):
    """Look in the database, print errors and return False if things are left behind.

    The "allowed" tables (CodeVersion, SensorSection,
    CatalogExcerpt, Provenance, Object, PasswordLink) will not cause
    False to be returned, but will just print a debug message.

    Parameters
    ----------
      dbsession: Session

    Returns
    -------
      True if there are only database rows in allowed tables.
      False if there are any databse rows in non-allowed tables.

    """

    objects = get_all_database_objects( session=dbsession )
    any_objects = False
    for Class, ids in objects.items():
        # TODO: check that surviving provenances have test_parameter
        # ...I don't think this should be a TODO.  Check this, but I
        #    think the pipeline will automatically add provenances if
        #    they don't exist.  As such, the tests may implicitly
        #    add provenances they don't explicitly track.
        if Class.__name__ in ['CodeVersion', 'SensorSection', 'CatalogExcerpt',
                              'Provenance', 'Object', 'ObjectLegacySurveyMatch', 'PasswordLink']:
            if len(ids) > 0:
                SCLogger.debug(f'There are {len(ids)} {Class.__name__} objects in the database. These are OK to stay.')
            continue

        # Special case handling for the 'current' Provenance Tag, which may have
        #   been added automatically by top_level.py
        if Class.__name__ == "ProvenanceTag":
            currents = []
            notcurrents = []
            for id in ids:
                obj = Class.get_by_id( id, session=dbsession )
                if obj.tag == 'current':
                    currents.append( obj )
                else:
                    notcurrents.append( obj )
            if len(currents) > 0:
                SCLogger.debug( f'There are {len(currents)} {Class.__name__} "current" objects in the database. '
                                F'These are OK to stay.' )
            objs = notcurrents
        else:
            objs = [ Class.get_by_id( i, session=dbsession) for i in ids ]

        if len(objs) > 0:
            any_objects = True
            strio = io.StringIO()
            strio.write( f'There are {len(objs)} {Class.__name__} objects in the database. '
                         f'Please make sure to cleanup!')
            for obj in objs:
                strio.write( f'\n    {obj}' )
            SCLogger.error( strio.getvalue() )

    return any_objects

# Uncomment this fixture to run the "empty database" check after each
# test.  This can be useful in figuring out which test is leaving stuff
# behind.  Because of session scope fixtures, it will cause nearly every
# (or every) test to fail, but at least you'll have enough debug output
# to (hopefully) find the tests that are leaving behind extra stuff.
#
# NOTE -- for this to work, ironically, you have to set
# verify_archive_database_empty to False at the top of this file.
# Otherwise, at the end of all the tests, the things left over in the
# databse you are looking for will cause everything to fail, and you
# *only* get that message instead of all the error messages from here
# that you wanted to get!  (Oh, pytest.)
#
# (This is probably not practical, becasuse there is *so much* module
# and session scope stuff that lots of things are left behind by tests.
# You will have to sift through a lot of output to find what you're
# looking for.  We need a better way.)
# @pytest.fixture(autouse=True)
# def check_empty_database_at_end_of_each_test():
#     yield True
#     with SmartSession() as dbsession:
#         assert not any_objects_in_database( dbsession )


# This will be executed after the last test (session is the pytest session, not the SQLAlchemy session)
# It will completely wipe the database (ideally)
def pytest_sessionfinish(session, exitstatus):
    global verify_archive_database_empty

    # SCLogger.debug('Final teardown fixture executing! ')
    with SmartSession() as dbsession:
        # ISSUE 479 this will find and DEBUG report the codeversions that are about to get killed in the next line.
        any_objects = any_objects_in_database( dbsession )

        # We'll need the catalog excerpts later after we delete the table
        catexps = dbsession.scalars(sa.select(CatalogExcerpt)).all()

    # ...SQLAlchemy sometimes seems determined to have dangling sessions
    # even when I tell it to close them.  See long rant in comments in
    # models/base.py::SmartSession.  To try to not make that hang when
    # cleaning up tests, try to totally shut down sqlalchemy altogether
    # and just use postgres directly.  (Honestly, we should never have
    # used sqlalchemy in the first place, its benefits have come nowhere
    # close to offsetting its headcaches.)
    #
    # Issue #516
    sqlalchemy.orm.session.close_all_sessions()

    with Psycopg2Connection() as conn:
        cursor = conn.cursor()

        # delete the CodeVersion objects (this should remove all provenances as well,
        # and that should cascade to *almost* everything else)
        cursor.execute( "TRUNCATE TABLE code_versions CASCADE" )

        # remove any Object objects, as these are not automatically cleaned up
        # Will cascade to object legacy survey matches
        cursor.execute( "TRUNCATE TABLE objects CASCADE" )

        # make sure there aren't any CalibratorFileDownloadLock rows
        # left over from tests that failed or errored out
        cursor.execute( "DELETE FROM calibfile_downloadlock" )

        # remove SensorSections, though see Issue #487
        cursor.execute( "DELETE FROM sensor_sections" )

        # remove RefSets, because those won't have been deleted by the code version / provenance cascade
        cursor.execute( "DELETE FROM refsets" )

        # remove any residual KnownExposures and PipelineWorkers
        cursor.execute( "DELETE FROM knownexposures" )
        cursor.execute( "DELETE FROM pipelineworkers" )

        # remove database records for any catalog excerpts.  (We'll remove the files below.)
        cursor.execute( "DELETE FROM catalog_excerpts" )

        conn.commit()

    # remove empty folders from the archive
    if ARCHIVE_PATH is not None:
        # remove catalog excerpts manually
        for catexp in catexps:
            if os.path.isfile(catexp.get_fullpath()):
                os.remove(catexp.get_fullpath())
            archive_file = os.path.join(ARCHIVE_PATH, catexp.filepath)
            if os.path.isfile(archive_file):
                os.remove(archive_file)

        remove_empty_folders( ARCHIVE_PATH, remove_root=False )

        # check that there's nothing left in the archive after tests cleanup
        if os.path.isdir(ARCHIVE_PATH):
            files = list(pathlib.Path(ARCHIVE_PATH).rglob('*'))

            if len(files) > 0:
                if verify_archive_database_empty:
                    strio = io.StringIO()
                    strio.write( f'There are files left in the archive after tests cleanup: {files}' )
                    if any_objects:
                        strio.write('\nThere are objects left in the database.  '
                                    'Some tests are not properly cleaning up')
                    raise RuntimeError( strio.getvalue() )
                else:
                    warnings.warn( f'There are files left in the archive after tests cleanup: {files}' )

    if any_objects and verify_archive_database_empty:
        raise RuntimeError('There are objects in the database.  Some tests are not properly cleaning up!')


@pytest.fixture(scope='session')
def download_url():
    return 'https://portal.nersc.gov/cfs/m4616/SeeChange_testing_data'


# data that is included in the repo and should be available for tests
@pytest.fixture(scope="session")
def persistent_dir():
    return os.path.join(CODE_ROOT, 'data')


# this is a cache folder that should survive between test runs
@pytest.fixture(scope="session", autouse=True)
def cache_dir():
    path = os.path.join(CODE_ROOT, 'data/cache')
    os.makedirs(path, exist_ok=True)
    return path


# this will be configured to FileOnDiskMixin.local_path, and used as temporary data location
@pytest.fixture(scope="session")
def data_dir():
    temp_data_folder = FileOnDiskMixin.local_path
    tdf = pathlib.Path( temp_data_folder )
    tdf.mkdir( exist_ok=True, parents=True )
    with open( tdf / 'placeholder', 'w' ):
        pass  # make an empty file inside this folder to make sure it doesn't get deleted on "remove_data_from_disk"

    # SCLogger.debug(f'temp_data_folder: {temp_data_folder}')

    yield temp_data_folder

    ( tdf / 'placeholder' ).unlink( missing_ok=True )

    # remove all the files created during tests
    # make sure the test config is pointing the data_dir
    # to a different location than the rest of the data
    # shutil.rmtree(temp_data_folder)


@pytest.fixture(scope="session")
def blocking_plots():
    """Control how and when plots will be generated.

    If the env var MAKE_PLOTS is True, then properly-written tests that
    do nothing but make plots will be run, and plots will be saved to
    tests/plots.

    If the env var MAKE_PLOTS is True, and the env var INTERACTIVE is
    True, then properly written tests will both save plots to
    tests/plots, and will show the plot on the screen, waiting for it to
    be closed before continuing with the tests.

    For test writers:

    If a test only makes plots, it should be marked with
    @pytest.mark.skipif( not env_as_bool('MAKE_PLOTS'), reason='Set MAKE_PLOTS to run this test' )

    If a test does stuff you want run and *also* makes plots, it should
    wrap the plot-building in an if block that tests
    env_as_bool('MAKE_PLOTS').

    Any tests that make plots should include this fixture.  They can
    optionally wrap any matplotlib show commands around an if on the
    value of this fixture, as it will be True only if INTERACTIVE is
    set.

    This fixture will also set the matplotlib backend to 'Agg' if
    INTERACTIVE is false, so that matplotlib will never block on a show
    call.  (If that really works, then it's not actually necessary to
    wrap show statements in the if block described in the previous
    paragraph.

    """
    import matplotlib
    backend = matplotlib.get_backend()

    # make sure there's a folder to put the plots in
    if not os.path.isdir(os.path.join(CODE_ROOT, 'tests/plots')):
        os.makedirs(os.path.join(CODE_ROOT, 'tests/plots'))

    inter = env_as_bool('INTERACTIVE')
    if not inter:  # for non-interactive plots, use headless plots that just save to disk
        # ref: https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython
        matplotlib.use("Agg")

    yield inter

    matplotlib.use(backend)


def rnd_str(n):
    rng = np.random.default_rng()
    return ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), n))


@pytest.fixture(scope='session', autouse=True)
def test_config():
    return Config.get()


@pytest.fixture
def provenance_base():
    p = Provenance(
        process="test_process",
        parameters={"test_parameter": uuid.uuid4().hex},
        upstreams=[],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


@pytest.fixture
def provenance_extra( provenance_base ):
    p = Provenance(
        process="test_extra_process",
        code_version_id=provenance_base.code_version_id,
        parameters={"test_parameter": uuid.uuid4().hex},
        upstreams=[provenance_base],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


@pytest.fixture
def provenance_tags_loaded( provenance_base, provenance_extra ):
    try:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES (%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': 'xyzzy', 'provid': provenance_base.id } )
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES (%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': 'plugh', 'provid': provenance_base.id } )
            cursor.execute( "INSERT INTO provenance_tags(_id,tag,provenance_id) "
                            "VALUES (%(id)s,%(tag)s,%(provid)s)",
                            { 'id': uuid.uuid4(), 'tag': 'plugh', 'provid': provenance_extra.id } )
            conn.commit()
        yield True
    finally:
        with Psycopg2Connection() as conn:
            cursor = conn.cursor()
            cursor.execute( "DELETE FROM provenance_tags WHERE tag IN ('xyzzy', 'plugh')" )
            conn.commit()


# use this to make all the pre-committed Image fixtures
@pytest.fixture(scope="session")
def provenance_preprocessing():
    p = Provenance(
        process="preprocessing",
        parameters={"test_parameter": "test_value"},
        upstreams=[],
        is_testing=True,
    )
    p.insert_if_needed()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


@pytest.fixture(scope="session")
def provenance_extraction():
    p = Provenance(
        process="extraction",
        parameters={"test_parameter": "test_value"},
        upstreams=[],
        is_testing=True,
    )
    p.insert()

    yield p

    with SmartSession() as session:
        session.execute( sa.delete( Provenance ).where( Provenance._id==p.id ) )
        session.commit()


@pytest.fixture(scope="session", autouse=True)
def archive_path(test_config):
    if test_config.value('archive.local_read_dir', None) is not None:
        archivebase = test_config.value('archive.local_read_dir')
    elif os.getenv('SEECHANGE_TEST_ARCHIVE_DIR') is not None:
        archivebase = os.getenv('SEECHANGE_TEST_ARCHIVE_DIR')
    else:
        raise ValueError('No archive.local_read_dir in config, and no SEECHANGE_TEST_ARCHIVE_DIR env variable set')

    # archive.path_base is usually /test
    archivebase = pathlib.Path(archivebase) / pathlib.Path(test_config.value('archive.path_base'))
    global ARCHIVE_PATH
    ARCHIVE_PATH = archivebase
    return archivebase


@pytest.fixture(scope="session")
def archive(test_config, archive_path):
    archive_specs = test_config.value('archive')
    if archive_specs is None:
        raise ValueError( "archive in config is None" )
    archive_specs[ 'logger' ] = SCLogger
    archive = Archive( **archive_specs )

    archive.test_folder_path = archive_path  # track the place where these files actually go in the test suite
    yield archive

    # try:
    #     # To tear down, we need to blow away the archive server's directory.
    #     # For the test suite, we've also mounted that directory locally, so
    #     # we can do that
    #     try:
    #         shutil.rmtree( archivebase )
    #     except FileNotFoundError:
    #         pass
    #
    # except Exception as e:
    #     warnings.warn(str(e))


@pytest.fixture( scope="module" )
def catexp(data_dir, cache_dir, download_url):
    filename = "Gaia_DR3_151.0926_1.8312_17.0_19.0.fits"
    cachepath = os.path.join(cache_dir, filename)
    filepath = os.path.join(data_dir, filename)

    if not os.path.isfile(cachepath):
        retry_download(os.path.join(download_url, filename), cachepath)

    if not os.path.isfile(filepath):
        shutil.copy2(cachepath, filepath)

    yield CatalogExcerpt.create_from_file( filepath, 'gaia_dr3' )

    if os.path.isfile(filepath):
        os.remove(filepath)


# ======================================================================
# PSF Palette fixtures
#
# This makes an image with a regularly space grid of elliptical gaussians,
#   for purposes of testing PSF extraction and the like

class PSFPaletteMaker:
    def __init__( self, round=False ):
        tempname = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
        self.imagename = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.fits'
        self.weightname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.weight.fits'
        self.flagsname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.flags.fits'
        self.catname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.cat'
        self.psfname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.psf'
        self.psfxmlname = pathlib.Path( FileOnDiskMixin.temp_path ) / f'{tempname}.psf.xml'

        self.nx = 1024
        self.ny = 1024

        self.clipwid = 17

        self.flux = 200000.
        self.noiselevel = 5.

        self.x0 = self.nx/2.
        self.sigx0 = 1.25
        if round:
            self.sigxx = 0.
            self.sigxy = 0.
        else:
            self.sigxx = 0.5 / self.nx
            self.sigxy = 0.

        self.y0 = self.ny/2.
        if round:
            self.sigy0 = 1.25
            self.sigyx = 0.
            self.sigyy = 0.
        else:
            self.sigy0 = 1.75
            self.sigyx = -0.5 / self.nx
            self.sigyy = 0.

        self.theta0 = 0.
        if round:
            self.thetax = 0.
            self.thetay = 0.
        else:
            self.thetax = 0.
            self.thetay = math.pi / 2. / self.ny

        # Positions where we're going to put the PSFs.  Want to have
        # about 100 of them, but also don't want them all to fall right
        # at the center of the pixel (hence the nonintegral spacing)
        self.xpos = np.arange( 25., 1000., 102.327 )
        self.ypos = np.arange( 25., 1000., 102.327 )

    @staticmethod
    def psffunc ( yr, xr, sigx, sigy, theta ):
        xrot =  xr * math.cos(theta) + yr * math.sin(theta)
        yrot = -xr * math.sin(theta) + yr * math.cos(theta)
        return 1/(2*math.pi*sigx*sigy) * math.exp( -( xrot**2/(2.*sigx**2) + yrot**2/(2.*sigy**2) ) )

    def psfpixel( self, x, y, xi, yi ):
        sigx = self.sigx0 + (x - self.x0) * self.sigxx + (y - self.y0) * self.sigxy
        sigy = self.sigy0 + (x - self.x0) * self.sigyx + (y - self.y0) * self.sigyy
        theta = self.theta0 + (x - self.x0) * self.thetax + (y - self.y0) * self.thetay

        res = dblquad( PSFPaletteMaker.psffunc, xi-x-0.5, xi-x+0.5, yi-y-0.5, yi-y+0.5, args=( sigx, sigy, theta ) )
        return res[0]

    def make_psf_palette( self ):
        self.img = np.zeros( ( self.nx, self.ny ) )

        for i, xc in enumerate( self.xpos ):
            xi0 = int( math.floor(xc)+0.5 )
            SCLogger.info( f"Making psf palette, on x {i} of {len(self.xpos)}" )
            for yc in self.ypos:
                yi0 = int( math.floor(yc)+0.5 )
                for xi in range(xi0 - (self.clipwid//2), xi0 + self.clipwid//2 + 1):
                    for yi in range(yi0 - (self.clipwid//2), yi0 + self.clipwid//2 + 1):
                        self.img[yi, xi] = self.flux * self.psfpixel( xc, yc, xi, yi )

        # Have to have some noise in there, or sextractor will choke on the image
        # Seed it, so we don't have to make our tests flaky.
        rng = np.random.default_rng( seed=42 )
        self.img += rng.normal( 0., self.noiselevel, self.img.shape )

        hdu = fits.PrimaryHDU( data=self.img )
        hdu.writeto( self.imagename, overwrite=True )
        hdu = fits.PrimaryHDU( data=np.zeros_like( self.img, dtype=np.uint8 ) )
        hdu.writeto( self.flagsname, overwrite=True )
        hdu = fits.PrimaryHDU( data=np.full( self.img.shape, 1. / ( self.noiselevel**2 ) ) )
        hdu.writeto( self.weightname, overwrite=True )

    def extract_and_psfex( self ):
        astromatic_dir = None
        cfg = Config.get()
        if cfg.value( 'astromatic.config_dir' ) is not None:
            astromatic_dir = pathlib.Path( cfg.value( 'astromatic.config_dir' ) )
        elif cfg.value( 'astromatic.config_subdir' ) is not None:
            astromatic_dir = pathlib.Path( CODE_ROOT ) / cfg.value( 'astromatic.config_subdir' )
        if astromatic_dir is None:
            raise FileNotFoundError( "Can't figure out where astromatic config directory is" )
        if not astromatic_dir.is_dir():
            raise FileNotFoundError( f"Astromatic config dir {str(astromatic_dir)} doesn't exist "
                                     f"or isn't a directory." )

        conv = astromatic_dir / "default.conv"
        nnw = astromatic_dir / "default.nnw"
        paramfile = astromatic_dir / "sourcelist_sextractor.param"

        SCLogger.info( "Running sextractor..." )
        # Run sextractor to give psfex something to do
        command = [ 'source-extractor',
                    '-CATALOG_NAME', self.catname,
                    '-CATALOG_TYPE', 'FITS_LDAC',
                    '-PARAMETERS_NAME', paramfile,
                    '-FILTER', 'Y',
                    '-FILTER_NAME', str(conv),
                    '-WEIGHT_TYPE', 'MAP_WEIGHT',
                    '-RESCALE_WEIGHTS', 'N',
                    '-WEIGHT_IMAGE', self.weightname,
                    '-FLAG_IMAGE', self.flagsname,
                    '-FLAG_TYPE', 'OR',
                    '-PHOT_APERTURES', '4.7',
                    '-SATUR_LEVEL', '1000000',
                    '-STARNNW_NAME', nnw,
                    '-BACK_TYPE', 'MANUAL',
                    '-BACK_VALUE', '0.0',
                    self.imagename
                   ]
        res = subprocess.run( command, capture_output=True, timeout=60 )
        assert res.returncode == 0

        SCLogger.info( "Runing psfex..." )
        # Run psfex to get the psf and psfxml files
        command = [ 'psfex',
                    '-PSF_SIZE', '31',
                    '-SAMPLE_FWHMRANGE', '1.0,10.0',
                    '-SAMPLE_VARIABILITY', '0.5',
                    '-CHECKPLOT_DEV', 'NULL',
                    '-CHECKPLOT_TYPE', 'NONE',
                    '-CHECKIMAGE_TYPE', 'NONE',
                    '-WRITE_XML', 'Y',
                    '-XML_NAME', self.psfxmlname,
                    '-XML_URL', 'file:///usr/share/psfex/psfex.xsl',
                    self.catname
                   ]
        res = subprocess.run( command, capture_output=True, timeout=60 )
        assert res.returncode == 0

        self.psf = PSFExPSF()
        self.psf.load( psfpath=self.psfname, psfxmlpath=self.psfxmlname )
        self.psf.fwhm_pixels = float( self.psf.header['PSF_FWHM'] )

    def cleanup( self ):
        self.imagename.unlink( missing_ok=True )
        self.weightname.unlink( missing_ok=True )
        self.flagsname.unlink( missing_ok=True )
        self.catname.unlink( missing_ok=True )
        self.psfname.unlink( missing_ok=True )
        self.psfxmlname.unlink( missing_ok=True )


@pytest.fixture(scope="module")
def round_psf_palette():
    palette = PSFPaletteMaker( round=True )
    palette.make_psf_palette()
    palette.extract_and_psfex()

    yield palette

    palette.cleanup()


@pytest.fixture(scope="module")
def psf_palette():
    palette = PSFPaletteMaker( round=False )
    palette.make_psf_palette()
    palette.extract_and_psfex()

    yield palette

    palette.cleanup()



# ======================================================================
# Fixtures for conductor and web ap tests

@pytest.fixture
def user():
    # username test, password test_password
    with SmartSession() as session:
        user = AuthUser( id='fdc718c3-2880-4dc5-b4af-59c19757b62d',
                         username='test',
                         displayname='Test User',
                         email='testuser@mailhog'
                        )
        user.pubkey = '''-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEArBn0QI7Z2utOz9VFCoAL
+lWSeuxOprDba7O/7EBxbPev/MsayA+MB+ILGo2UycGHs9TPBWihC9ACWPLG0tJt
q5FrqWaHPmvXMT5rb7ktsAfpZSZEWdrPfLCvBdrFROUwMvIaw580mNVm4PPb5diG
pM2b8ZtAr5gHWlBH4gcni/+Jv1ZKYh0b3sUOru9+IStvFs6ijySbHFz1e/ejP0kC
LQavMj1avBGfaEil/+NyJb0Ufdy8+IdgGJMCwFIZ15HPiIUFDRYWPsilX8ik+oYU
QBZlFpESizjEzwlHtdnLrlisQR++4dNtaILPqefw7BYMRDaf1ggYiy5dl0+ZpxYO
puvcLQlPqt8iO1v3IEuPCdMqhmmyNno0AQZq+Fyc21xRFdwXvFReuOXcgvZgZupI
XtYQTStR9t7+HL5G/3yIa1utb3KRQbFkOXRXHyppUEIr8suK++pUORrAablj/Smj
9TCCe8w5eESmQ+7E/h6M84nh3t8kSBibOlcLaNywKm3BEedQXmtu4KzLQbibZf8h
Ll/jFHv5FKYjMBbVw3ouvMZmMU+aEcaSdB5GzQWhpHtGmp+fF0bPztgTdQZrArja
Y94liagnjIra+NgHOzdRd09sN9QGZSHDanANm24lZHVWvTdMU+OTAFckY560IImB
nRVct/brmHSH0KXam2bLZFECAwEAAQ==
-----END PUBLIC KEY-----
'''
        user.privkey = {"iv": "pXz7x5YA79o+Qg4w",
                        "salt": "aBtXrLT7ds9an38nW7EgbQ==",
                        "privkey": "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"  # noqa: E501
                        }
        session.add( user )
        session.commit()

    yield True

    with SmartSession() as session:
        user = session.query( AuthUser ).filter( AuthUser.username=='test' ).all()
        for u in user:
            session.delete( u )
        session.commit()


@pytest.fixture
def admin_user():
    # In the noble tradition of routers everywhere, username admin, password admin
    with SmartSession() as session:
        user = AuthUser( id='684ece1d-c33a-4f80-b90e-646ae54021b7',
                         username='admin',
                         displayname='Admin User',
                         email='admin@mailhog',
                        )
        user.pubkey = '''-----BEGIN PUBLIC KEY-----
 MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAw+8D5eIkY1BKHkjmiifL
 XTzpitg84xW292YWoLNd5nYCzudzZLb0wCZmIyB9Go7Qy/1vEAHctpD7u74QCevN
 eDhfryTYHwD4Cfj1756htLtT4/M6UVxBV+gpQAQjko+otM6y900Rpkq8sidvNEWO
 yZlgYU3jUDO4Xa6Pxxzfuf8W+hU1beIZosIVjqiJr4aX2ceItHJfEjKh/LtZjFsj
 2HUjPFLuU+AIFUxVdfHqPmX25OhMmCE/TffaUZUfNIE0B+kV5m9nlhdccwlhPrO0
 3mVS5rSOz4TE6TYEVmACaMYhMFI79tBk4/ld1p0ngW6VhAwW1GqQwOCUoXs9alUJ
 Zh7qmSZy3t7XK/IUIAJdPmwfBF5tKwmoahCjn2LxzS3OCXoRwKPL0hvHUpmvfJKk
 XTNX/gJ21mk4tncdbwy7y+TcfP38amLvC9axIm7rmXOL22K1FGsxBFxllzpYiRY2
 dSOtSY3roih06f4wcyOfrWhWCbsR3zLZQ54L+pn4r1jae+RqSBqhslvsJQopnsMK
 YTEaYDKcP9d+SsIh2N71VURYSVBEUko4QTBIxisPwgWewhKMN/vOOpiY1kXp2CQv
 YoIHDVu28bZHOry9/T7nw5iwrqymXr9P2cYPosjmPj2Mee3XhgMlaPxj2B6z4qiT
 lgq/YFNl4MiGzKRlUCURCgMCAwEAAQ==
 -----END PUBLIC KEY-----
'''
        user.privkey = {"iv": "P3iF0nASKzuz/9BR",
                        "salt": "tZXnze39AoYqUsGYQoAIWw==",
                        "privkey": "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"  # noqa: E501
                        }
        session.add( user )

        group = AuthGroup( id='d6de6f07-96e0-46a4-addd-56f6c29b47d3',
                           name='admin',
                           description='Admin Group' )
        session.add( group )
        session.commit()
        session.execute( sa.text( "INSERT INTO auth_user_group(userid,groupid) VALUES(:user,:group)" ),
                         { 'user': '684ece1d-c33a-4f80-b90e-646ae54021b7',
                           'group': 'd6de6f07-96e0-46a4-addd-56f6c29b47d3' } )

        session.commit()

    yield True

    with SmartSession() as session:
        user = session.query( AuthUser ).filter( AuthUser.username=='admin' ).all()
        for u in user:
            session.delete( u )
        group = session.query( AuthGroup ).filter( AuthGroup.name=='admin' ).all()
        for g in group:
            session.delete( g )
        session.commit()


@pytest.fixture
def browser():
    opts = selenium.webdriver.FirefoxOptions()
    opts.add_argument( "--headless" )
    ff = selenium.webdriver.Firefox( options=opts )
    # This next line lets us use self-signed certs on test servers
    ff.accept_untrusted_certs = True
    yield ff
    ff.close()
    ff.quit()


# ======================================================================
# Fake objects for testing stuff

@pytest.fixture
def bogus_image_factory( provenance_base ):
    def load_bogus_image( _id, filepath ):
        img = Image( _id=_id,
                     format='fits',
                     type='Sci',
                     provenance_id=provenance_base.id,
                     mjd=60000.,
                     end_mjd=60000.00052,
                     exp_time=45.,
                     instrument='DemoInstrument',
                     telescope='DemoTelescope',
                     filter='r',
                     section_id=1,
                     project='test',
                     target='test',
                     filepath=filepath,
                     ra=120.,
                     dec=5.,
                     ra_corner_00=119.9,
                     ra_corner_01=119.9,
                     ra_corner_10=120.1,
                     ra_corner_11=120.1,
                     minra=110.9,
                     maxra=120.1,
                     dec_corner_00=4.9,
                     dec_corner_10=4.9,
                     dec_corner_01=5.1,
                     dec_corner_11=5.1,
                     mindec=4.9,
                     maxdec=5.1,
                     lim_mag_estimate=22.5,
                     bkg_mean_estimate=0.,
                     bkg_rms_estimate=1.,
                     fhwm_estimate=2.35,
                     airmass=1.2,
                     sky_sub_done=True,
                     astro_cal_done=True,
                    )

        # Give the image some numpy arrays for things that will look at the size
        img.header = fits.Header( { 'TESTKW': 'testval' } )
        img.data = np.zeros( ( 1024, 1024 ), dtype=np.float32 )
        img.weight = np.full( ( 1024, 1024 ), 0.01, dtype=np.float32 )
        img.flags = np.zeros( ( 1024, 1024 ), dtype=np.int16 )

        img.save()
        img.insert()

        return img

    return load_bogus_image


@pytest.fixture
def bogus_sources_factory( provenance_base ):
    def load_bogus_sources( _id, filepath, image ):
        improv = Provenance.get( image.provenance_id )
        prov = Provenance( code_version_id=improv.code_version_id,
                           process='extraction',
                           parameters={ 'method': 'sextractor' },
                           upstreams=[ improv ],
                           is_testing=True )
        prov.insert_if_needed()
        src = SourceList( _id=_id,
                          image_id=image.id,
                          format='sextrfits',
                          num_sources=42,
                          provenance_id=prov.id,
                          filepath=filepath,
                          md5sum=uuid.uuid4() )
        src.insert()

        return src

    return load_bogus_sources


@pytest.fixture
def bogus_image( bogus_image_factory ):
    img = bogus_image_factory( uuid.UUID('13de30a0-cb73-40d7-a708-10354005b7e4'), 'fake_bogus_image' )

    yield img

    # Doing this manually rather than calling img.delete_from_disk_and_database
    #   because of the bogus_datastore cleanup below
    with SmartSession() as session:
        session.execute( sa.delete( Image ).where( Image._id==img.id ) )
        session.commit()
    archive = get_archive_object()
    for comp in [ 'image', 'weight', 'flags' ]:
        p = pathlib.Path( FileOnDiskMixin.local_path ) / f'fake_bogus_image.{comp}.fits'
        if p.is_file():
            p.unlink()
        archive.delete( f'fake_bogus_image.{comp}.fits', okifmissing=True )


@pytest.fixture
def bogus_sources_and_psf( bogus_image, bogus_sources_factory ):
    src = bogus_sources_factory( uuid.UUID('717d6591-0630-448a-9748-0097e40c8272'),
                                 'fake_bogus_source_list.fits',
                                 bogus_image )

    psf = PSFExPSF( _id=uuid.UUID('c9e410de-a6d1-4db4-9b95-8ee866997784'),
                    sources_id=src.id,
                    fwhm_pixels=2.5,
                    filepath='fake_bogus_psf.fits',
                    md5sum=uuid.uuid4() )
    psf.insert()

    yield src, psf

    with SmartSession() as session:
        session.execute( sa.delete( PSF ).where( PSF._id==psf.id ) )
        session.execute( sa.delete( SourceList ).where( SourceList._id==src.id ) )
        session.commit()


@pytest.fixture
def bogus_bg( bogus_sources_and_psf ):
    bogus_sources, _ = bogus_sources_and_psf
    bg = Background( format='scalar',
                     method='zero',
                     sources_id=bogus_sources.id,
                     value=0.,
                     noise=1.,
                     image_shape=(256,256),
                     filepath='fake_bogus_bg.h5',
                     md5sum=uuid.uuid4() )
    bg.insert()

    yield bg

    with SmartSession() as session:
        session.execute( sa.delete( Background ).where( Background._id==bg.id ) )
        session.commit()


@pytest.fixture
def bogus_wcs( bogus_sources_and_psf ):
    bogus_sources, _ = bogus_sources_and_psf
    srcprov = Provenance.get( bogus_sources.provenance_id )
    prov = Provenance( code_verson_id=srcprov.code_version_id,
                       process='astrocal',
                       parameters={ 'solution_method': 'scamp' },
                       upstreams=[ srcprov ],
                       is_testing=True )
    prov.insert_if_needed()
    wcs = WorldCoordinates( sources_id=bogus_sources.id,
                            provenance_id=prov.id,
                            filepath='fake_bogus_wcs.txt',

                            md5sum=uuid.uuid4() )
    wcs.insert()

    yield wcs

    with SmartSession() as session:
        session.execute( sa.delete( WorldCoordinates ).where( WorldCoordinates._id==wcs.id ) )
        session.commit()


@pytest.fixture
def bogus_zp( bogus_wcs ):
    wcsprov = Provenance.get( bogus_wcs.provenance_id )
    prov = Provenance( code_version_id=wcsprov.code_version_id,
                       process='photocal',
                       parameters={ 'cross_match_catalog': 'gaia_dr3' },
                       upstreams=[ wcsprov ],
                       is_testing=True )
    prov.insert_if_needed()
    zp = ZeroPoint( wcs_id=bogus_wcs.id,
                    zp=25.,
                    dzp=0.1,
                    provenance_id=prov.id )
    zp.insert()

    yield zp

    with SmartSession() as session:
        session.execute( sa.delete( ZeroPoint ).where( ZeroPoint._id==zp.id ) )
        session.commit()


@pytest.fixture
def bogus_datastore( bogus_image, bogus_sources_and_psf, bogus_bg, bogus_wcs, bogus_zp ):
    ds = DataStore()
    ds.image = bogus_image
    ds.sources = bogus_sources_and_psf[0]
    ds.psf = bogus_sources_and_psf[1]
    ds.bg = bogus_bg
    ds.wcs = bogus_wcs
    ds.zp = bogus_zp
    ds.edit_prov_tree( ProvenanceTree( { 'starting_point': Provenance.get( ds.image.provenance_id ),
                                         'extraction': Provenance.get( ds.sources.provenance_id ),
                                         'backgrounding': Provenance.get( ds.bg.provenance_id ),
                                         'astrocal': Provenance.get( ds.wcs.provenance_id ),
                                         'photocal': Provenance.get( ds.zp.provenance_id ) },
                                       upstream_steps = { 'starting_point': [],
                                                          'extraction': ['starting_point'],
                                                          'backgrounding': ['extraction'],
                                                          'astrocal': ['extraction'],
                                                          'photocal': ['astrocal'] } ) )

    yield ds

    # Do this just in case other stuff got added
    ds.delete_everything()



# ======================================================================a


@pytest.fixture
def bogus_fakeset_saved( bogus_zp ):
    xs = [  1127.68, 1658.71, 1239.56, 1601.83, 1531.19, 921.57 ]
    ys = [  2018.91, 1998.84, 2503.77, 2898.47, 3141.27, 630.95  ]
    mags = [ 20.32,  19.90,   23.00,   22.00,  21.00,    23.30 ]
    hostdexen = [ -1., 0., -1., -1., 5., -1. ]

    zpprov = Provenance.get( bogus_zp.provenance_id )
    prov = Provenance( code_version_id=zpprov.code_version_id,
                       process='fakeinjection',
                       parameters={ 'random_seed': 42 },
                       upstreams=[ zpprov ],
                       is_testing=True )
    prov.insert_if_needed()
    fakeset = FakeSet( zp_id=bogus_zp.id,
                       provenance_id=prov.id )
    fakeset.random_seed = 42
    fakeset.fake_x = np.array( xs )
    fakeset.fake_y = np.array( ys )
    fakeset.fake_mag = np.array( mags )
    fakeset.host_dex = np.array( hostdexen )

    fakeset.filepath = "bogus_fakeset.h5"
    fakeset.save()
    fakeset.insert()

    yield fakeset

    fakeset.delete_from_disk_and_database()
