import pytest
import pathlib

import models.ls4cam  # noqa: F401
from models.instrument import get_instrument_instance
from models.exposure import Exposure
from util.retrydownload import retry_download


@pytest.fixture( scope='module' )
def loaded_dualamp_exposure( download_url, cache_dir ):
    expobj = None
    try:
        relpath = pathlib.Path( "LS4/20251027065735s_00015.fits.fz" )
        cachepath = pathlib.Path( cache_dir ) / relpath
        if not cachepath.is_file():
            cachepath.parent.mkdir( parents=True, exist_ok=True )
            retry_download( f'{download_url}/{relpath}', cachepath )

        ls4cam = get_instrument_instance( 'LS4Cam_dualamp' )
        expobj = ls4cam.manually_load_exposure( cachepath )
        yield expobj

    finally:
        if expobj is not None:
            fullpath = pathlib.Path( expobj.get_fullpath() )
            expobj.delete_from_disk_and_database()
            assert not fullpath.exists()


def test_manual_load_exposure( loaded_dualamp_exposure ):
    exp = loaded_dualamp_exposure

    assert exp.origin_identifier == '20251027065735s_00015.fits.fz'
    assert exp.instrument == 'LS4Cam_dualamp'
    assert exp.instrument_object.__class__.__name__ == 'LS4Cam_dualamp'
    assert exp.filter is None
    assert exp.filter_array == [ 'i', 'z', 'g', 'i' ]
    assert exp.ra == pytest.approx( 49.5120, abs=1e-4 )
    assert exp.dec == pytest.approx( -20.5736, abs=1e-4 )
    assert exp.filepath == 'ls4_20251027_065741_None_IYQ2Z6.fits.fz'
    assert exp.type == 'Sci'
    assert exp.format == 'fitsfz'
    assert exp.mjd == pytest.approx( 60975.2901, abs=1e-4 )
    assert exp.exp_time == pytest.approx( 15.0, abs=0.1 )
    assert exp.airmass == pytest.approx( 1.985, abs=0.001 )

    # Make sure the file is there and it's really in the database
    assert pathlib.Path( exp.get_fullpath() ).is_file()
    dbexp = Exposure.get_by_id( exp.id )
    for prop in [ 'id', 'filepath', 'instrument', 'filter', 'type', 'format' ]:
        assert getattr( exp, prop ) == getattr( dbexp, prop )
