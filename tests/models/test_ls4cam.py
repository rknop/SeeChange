import pytest
import pathlib

import models.ls4cam  # noqa: F401
from models.base import FileOnDiskMixin
from models.exposure import Exposure
from models.instrument import get_instrument_instance, SensorSection
from pipeline.data_store import DataStore
from pipeline.top_level import Pipeline
from util.retrydownload import retry_download


# @pytest.fixture( scope='module' )
# def loaded_dualamp_exposure( download_url, cache_dir ):
#     expobj = None
#     try:
#         relpath = pathlib.Path( "LS4/20251027065735s_00015.fits.fz" )
#         cachepath = pathlib.Path( cache_dir ) / relpath
#         if not cachepath.is_file():
#             cachepath.parent.mkdir( parents=True, exist_ok=True )
#             retry_download( f'{download_url}/{relpath}', cachepath )

#         ls4cam = get_instrument_instance( 'LS4Cam_dualamp' )
#         expobj = ls4cam.manually_load_exposure( cachepath )
#         yield expobj

#     finally:
#         if expobj is not None:
#             fullpath = pathlib.Path( expobj.get_fullpath() )
#             expobj.delete_from_disk_and_database()
#             assert not fullpath.exists()

@pytest.fixture( scope='module' )
def loaded_singleamp_multifile_exposure( download_url, cache_dir ):
    expobj = None
    try:
        for ctrlr in range(4):
            for chip in range(8):
                relpath = pathlib.Path( f"LS4/20260410/20260410004924sC{ctrlr}_00025_{chip:02d}.fits" )
                cachepath = pathlib.Path( cache_dir ) / relpath
                if not cachepath.is_file():
                    cachepath.parent.mkdir( parents=True, exist_ok=True )
                    retry_download( f'{download_url}/{relpath}', cachepath )

        ls4cam = get_instrument_instance( 'LS4Cam' )
        # ...just use the last cachepath from the for loop, any of them *should* work
        # (...probably we ought to test more than one, huh.)
        expobj = ls4cam.manually_load_exposure( cachepath )

        # Make sure it didn't leave behind a temp file
        tmpfile = pathlib.Path( FileOnDiskMixin.temp_path ) / "20260410004924s_0025.fits"
        assert not tmpfile.exists()
        tmpfile = tmpfile.parent / f"{tmpfile.name}.fz"
        assert not tmpfile.exists()

        yield expobj

    finally:
        if expobj is not None:
            fullpath = pathlib.Path( expobj.get_fullpath() )
            expobj.delete_from_disk_and_database()
            assert not fullpath.exists()



def test_section_stuff():
    ls4cam = get_instrument_instance( 'LS4Cam' )

    expectedsecs = []
    for quadrant in [ 'NE', 'NW', 'SE', 'SW' ]:
        for chipinquad in [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' ]:
            expectedsecs.append( f"{quadrant}_{chipinquad}" )

    assert ls4cam.get_section_ids() == expectedsecs
    for sec in expectedsecs:
        ls4cam.check_section_id( sec )
    with pytest.raises( ValueError, match=r'section_id must start with one of .*, not AB' ):
        ls4cam.check_section_id( 'AB_C' )
    with pytest.raises( ValueError, match=r'section_id\[2\] must be _, not -' ):
        ls4cam.check_section_id( 'NE-C' )
    with pytest.raises( ValueError, match=r'section_id\[3\] must be in the range A..H, not X' ):
        ls4cam.check_section_id( 'NE_X' )
    with pytest.raises( ValueError, match=r'All LS4 section_ids are length 4; got 3' ):
        ls4cam.check_section_id( 'foo' )
    with pytest.raises( ValueError, match=r'The section_id must be a string.  Got' ):
        ls4cam.check_section_id( 1 )

    sec = ls4cam._make_new_section( 'NE_A' )
    assert isinstance( sec, SensorSection )
    assert sec.instrument == 'LS4Cam'
    assert sec.identifier == 'NE_A'
    assert sec.size_x == 2048
    assert sec.size_y == 4096
    assert sec.filter_array_index == 0
    # TODO : other properties once we know them better

    # TODO : get section offsets

    for sec in expectedsecs:
        dex = ls4cam.get_section_filter_array_index( sec )
        if sec[:2] == 'NE':
            assert dex == 0
        elif sec[:2] == 'NW':
            assert dex == 1
        elif sec[:2] == 'SE':
            assert dex == 2
        elif sec[:2] == 'SW':
            assert dex == 3


def test_manual_load_exposure( loaded_singleamp_multifile_exposure ):
    expobj = loaded_singleamp_multifile_exposure

    assert expobj.origin_identifier == '20260410004924sC3_00025.fits'
    assert expobj.instrument == 'LS4Cam'
    assert expobj.instrument_object.__class__.__name__ == 'LS4Cam'
    assert expobj.telescope == 'ESO 1.0-m Schmidt'
    assert expobj.project == 'unknown'
    assert expobj.target== 'unknown'

    assert expobj.filter is None
    assert expobj.filter_array == [ 'i', 'z', 'g', 'i' ]
    assert expobj.ra == pytest.approx( 159.018, abs=1e-4 )
    assert expobj.dec == pytest.approx( -25.8087, abs=1e-4 )
    assert expobj.filepath == 'ls4_20260410_004930_None_3XSWYA.fits.fz'
    assert expobj.type == 'Sci'
    assert expobj.format == 'fitsfz'
    assert expobj.mjd == pytest.approx( 61140.034375, abs=1e-5 )
    assert expobj.exp_time == pytest.approx( 60., abs=0.01 )
    assert expobj.airmass == pytest.approx( 1.049, abs=0.001 )

    # Make sure we have all the sections we expect
    chips = set( f'{quadrant}_{letter}' for quadrant in ['NW', 'NE', 'SW', 'SE'] for letter in 'ABCDEFGH' )
    for chip in chips:
        sechdr = expobj.section_headers[chip]
        assert sechdr['CCD_LOC'] == chip

    # TODO : look at data?

    # Make sure the file is there and it's really in the database
    assert pathlib.Path( expobj.get_fullpath() ).is_file()
    dbexp = Exposure.get_by_id( expobj.id )
    for prop in [ 'id', 'filepath', 'instrument', 'filter', 'type', 'format' ]:
        assert getattr( expobj, prop ) == getattr( dbexp, prop )


def test_overscan( loaded_singleamp_multifile_exposure ):
    expobj = loaded_singleamp_multifile_exposure

    # SE_F and SE_E are the ones that are half-bad
    chipstodo = [ 'NE_G', 'SE_F', 'SE_E', 'NW_B' ]

    for chip in chipstodo:
        pip = Pipeline( pipeline={ 'through_step': 'preprocessing' },
                        preprocessing={ 'steps_required': ['overscan'] } )
        ds = DataStore( expobj, chip )
        ds = pip( ds )
        import pdb; pdb.set_trace()
        pass


# def test_dualamp_manual_load__exposure( loaded_dualamp_exposure ):
#     exp = loaded_dualamp_exposure

#     assert exp.origin_identifier == '20251027065735s_00015.fits.fz'
#     assert exp.instrument == 'LS4Cam_dualamp'
#     assert exp.instrument_object.__class__.__name__ == 'LS4Cam_dualamp'
#     assert exp.filter is None
#     assert exp.filter_array == [ 'i', 'z', 'g', 'i' ]
#     assert exp.ra == pytest.approx( 49.5120, abs=1e-4 )
#     assert exp.dec == pytest.approx( -20.5736, abs=1e-4 )
#     assert exp.filepath == 'ls4_20251027_065741_None_IYQ2Z6.fits.fz'
#     assert exp.type == 'Sci'
#     assert exp.format == 'fitsfz'
#     assert exp.mjd == pytest.approx( 60975.2901, abs=1e-4 )
#     assert exp.exp_time == pytest.approx( 15.0, abs=0.1 )
#     assert exp.airmass == pytest.approx( 1.985, abs=0.001 )

#     # Make sure the file is there and it's really in the database
#     assert pathlib.Path( exp.get_fullpath() ).is_file()
#     dbexp = Exposure.get_by_id( exp.id )
#     for prop in [ 'id', 'filepath', 'instrument', 'filter', 'type', 'format' ]:
#         assert getattr( exp, prop ) == getattr( dbexp, prop )


# def test_dualamp_load_section_image( loaded_dualamp_exposure ):
#     ls4cam = get_instrument_instance( 'LS4Cam_dualamp' )
#     data = ls4cam.load_section_image( loaded_dualamp_exposure.get_fullpath(), 'NW_C' )
#     assert data.shape == ( 4120, 2100 )
#     assert np.median( data ) == pytest.approx( 4338.0, abs=0.1 )
#     assert np.mean( data ) == pytest.approx( 4374.0, abs=0.1 )
#     assert np.std( data ) == pytest.approx( 315.0, abs=0.1 )


# def test_dualamp_read_header( loaded_dualamp_exposure ):
#     ls4cam = get_instrument_instance( 'LS4Cam_dualamp' )
#     hdr = ls4cam.read_header( loaded_dualamp_exposure.get_fullpath(), 'NW_C' )
#     import pdb; pdb.set_trace()
#     pass
