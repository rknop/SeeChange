import pytest
import pathlib

from models.instrument import Instrument
from models.base import FileOnDiskMixin
from util.retrydownload import retry_download


@pytest.fixture( scope='session' )
def loaded_ls4_singleamp_exposure( download_url, cache_dir ):
    expobj = None
    try:
        relpath = pathlib.Path( "LS4/20260409/20260409005320s_00014.fits.fz" )
        cachepath = pathlib.Path( cache_dir ) / relpath
        if not cachepath.is_file():
            cachepath.parent.mkdir( parents=True, exist_ok=True )
            retry_download( f'{download_url}/{relpath}', cachepath )

        ls4cam = Instrument.get_instrument_instance( 'LS4Cam' )
        expobj = ls4cam.manually_load_exposure( cachepath )

        # Make sure it didn't levae behind a temp file
        tmpfile = pathlib.Path( FileOnDiskMixin.temp_path ) / "20260409005320s_00014.fits"
        assert not tmpfile.exists()
        tmpfile = tmpfile.parent / f"{tmpfile.name}.fz"
        assert not tmpfile.exists()

        yield expobj

    finally:
        if expobj is not None:
            fullpath = pathlib.Path( expobj.get_fullpath() )
            expobj.delete_from_disk_and_database()
            assert not fullpath.exists()
