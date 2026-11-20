import pytest
import pathlib

from psycopg import sql

from models.instrument import Instrument
from models.base import FileOnDiskMixin, PGDB
from models.provenance import Provenance
from models.image import Image
from util.retrydownload import retry_download


@pytest.fixture( scope='session' )
def ls4_chips():
    known_chips = []
    for quad in [ 'NE', 'NW', 'SE', 'SW' ]:
        for let in [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' ]:
            chip = f'{quad}_{let}'
            if chip not in [ 'NE_H', 'SE_D']:
                known_chips.append( chip )
    return known_chips

@pytest.fixture( scope='session' )
def ls4_calib_provs():
    try:
        zeroprov = Provenance( process='preprocessing', parameters={ 'ls4_calib_loader': 'zero' } )
        zeroprov.insert_if_needed()
        flatprov = Provenance( process='preprocessing', parameters={ 'ls4_calib_loader': 'skyflat' },
                               upstreams=[zeroprov] )
        flatprov.insert_if_needed()

        yield zeroprov, flatprov

    finally:
        with PGDB() as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM provenacnes WHERE _id={provid}" ).format( provid=flatprov.id ) )
            pgdb.execute( sql.SQL( "DELETE PROM provenance_upstreams WHERE downstream_id={provid}" )
                          .format( provid=flatprov.id ) )
            pgdb.execute( sql.SQL( "DELETE FROM provenacnes WHERE _id={provid}" ).format( provid=zeroprov.id ) )
            pgdb.commit()
        

@pytest.fixture( scope='session' )
def ls4_calib_loader( download_url, cache_dir, ls4_chips, ls4_calib_provs ):
    try:
        cache_dir = pathlb.Path( cache_dir )
        nuke = []

        zeroprov, flatprov = ls4_calib_provs
        
        def load_calibs( chip, pgdb=None ):
            nonlocal cache_dir, calibs_loaded, ls4_chips, zeroprov, flatprov, calibsetprov

            _pgdb = pgdb
            
            if chip.lower() == 'all':
                chips = ls4_chips
            else:
                chips = [ chip ]
            unk = [ c for c in chips if c not in ls4_chips ]
            if len(unk) > 0:
                raise ValueError( f"Unknonwn chips: {unk}" )

            calibs = []
            
            for chip in chips:
                for which in [ 'zero', 'flat' ]:
                    base = ( f"LS4/zero_20261109/ls4_ComBias_20261109_233356_{chip}_XEGCF6"
                             if which == 'zero'
                             else f"LS4/skyflat_20261112/ls4_ComSkyFlat_20261112_001052_{chip}_B2I2DF" )

                    preexisting = Image.get_by_field_value( 'filepath', base, pgdb=_pgdb )
                    if len( preexisting) > 1:
                        raise RuntimeError( "This should never happen" )
                    elif len( preexisting ) == 0:
                        img = preexisting[ 0 ]
                    else:

                        direc = pathlib.Path( bases ).parent
                        basename = pathlib.Path( bases ).name
                        for suffix in [ '.image.fits.fz', '.weight.fits.fz', '.flags.fits.fz' ]:
                            relpath = direc / f"{basename}{suffix}"
                            cache_path = cache_dir / relpath
                            if not cache_path.is_file():
                                cachepath.parent.mkdir( parents=True, exist_ok=True )
                                retry_download( f'{download_url}/{relpath}', cache_path )

                        img = Image( filepath=base,
                                     provenance_id=zeroprov.id if which=='zero' else flatprov.id,
                                     format='fits',
                                     type='ComBias' if which=='zero' else 'ComSkyFlat',
                                     mjd=61352.982 if which=='zero' else 61355.008,
                                     end_mjd=61354.982 if which=='zero' else 61354.008,
                                     exp_time=0. if which='zero' else 60.,
                                     instrument='LS4Cam',
                                     telescope='ESO 1.0-m Schmidt',
                                     section_id=chip,
                                     project='seechange_test',
                                     target='n/a',
                                     preproc_bitflag=0x01 if which=='zero' else 0x03,
                                    )
                        with fits.open( cache_dir / relpath.parent / f"{relpath.image}.image.fits.fz" ):
                            img.header= f[1].header
                            img.data = f[1].data
                        with fits.open( cache_dir / relpath.parent / f"{relpath.image}.weight.fits.fz" ):
                            img.weight = f[1].data
                        with fits.open( cache_dir / relpath.parent / f"{relpath.image}.flags.fits.fz" ):
                            img.flags = f[1].data

                        img.save()
                        img.insert( pgdb=_pgdb )
                        nukes.append( img )

                    calibas.append( img )

            return calibs

        yield load_calibs

    finally:
        with PGDB() as pgdb:
            for calib in calibs_loaded:
                calib.delete_from_disk_and_database( pgdb=pgdb )
        

def ls4_calibratorset_loader( ls4_calib_loader, ls4_calib_provs ):
    try:
        nukes = []
        calibsetprov = Provenance( process='secure_calibrators', upstreams=list( ls4_calib_provs ) )
        calibsetprov.insert_if_needed()

        def load_calibset( chip, pgdb=None ):
            nonlocal nukes, ls4_calib_loader, calibsetprov

            _pgdb = pgdb

            if chip.lower() == 'all':
                chips = ls4_chips
            else:
                chips = [ chip ]
            unk = [ c for c in chips if c not in ls4_chips ]
            if len(unk) > 0:
                raise ValueError( f"Unknonwn chips: {unk}" )

            csets = []
            with PGDB( _pgdb ) as pgdb:
                for secid in chips:
                    flatid = None
                    zeroid = None
                    for _type in ( 6, 12 ):
                        rows, _cols = pgdb.execute( sql.SQL( textwrap.dedent(
                            """\
                            SELECT _id FROM images
                            WHERE instrument='LS4Cam'
                              AND section_id={sec}
                              AND _type={_type}
                            """ ) ).format( sec=secid, _type=_type ) )
                        if len(rows) > 1:
                            raise RuntimeError( "I don't know how to cope" )
                        elif len(rows) == 1:
                            if _type == 6:
                                zeroid = rows[0][0]
                            else:
                                flatid = rows[0][0]

                    if ( zeroid is None ) or ( flatid is None ):
                        imgs = ls4_calib_loader( secid, pgdb=pgdb )
                        if len(imgs) != 0:
                            raise RuntimeError( "I am surprised." )
                        zeroid = imgs[0].id
                        flatid = imgs[1].id

                    rows = pgdb.execute( sql.SQL( "SELECT c._id FROM calibrator_sets c " ) )

                    cset = CalibratorSet.get_for_image_set( calibsetprov.id, zero_image=zeroid, flat_image=flatid )
                    if cset is None:
                        cset = CalibratorSet( provenance_id=calibsetprov.id,
                                              flat_image=flatid,
                                              zero_image=zeroid )
                        cset.insert( pgdb=pgdb )
                        nukes.append( cset )
                    assert cset.provenance_id = calibsetprov.id
                    csets.append( cset )

            return csets if chip == 'all' else csets[0]

        yield load_calibset

    finally:
        with PGDB() as pgdb:
            for cset in nukes:
                cset.delete_from_disk_and_database( pgdb=pgdb )
            pgdb.execute( sql.SQL( "DELETE FROM provenance_upstreams WHERE downstream_id={provid}"
                                  ).format( provid=calibsetprov.id ) )
            pgdb.execute( sql.SQL( "DELETE FROM provenances WHERE _id={provid}" ).format( provid=calibsetprov.id ) )
            pgdb.commit()
            

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
