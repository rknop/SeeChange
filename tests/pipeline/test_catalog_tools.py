import os
import pathlib
import types
import pytest

from util.exceptions import CatalogNotFoundError
from util.config import Config
from util import ldac
from pipeline.catalog_tools import download_gaia_dr3, fetch_gaia_dr3_excerpt
from models.base import FourCorners
from models.image import Image


def do_basic_download_dr3( temp_dir, data_dir ):
    firstfilepath = None
    secondfilepath = None
    try:
        catexp, firstfilepath, dbfile = download_gaia_dr3( 150.9427, 151.2425, 1.75582, 1.90649,
                                                                     padding=0.1, minmag=18., maxmag=22. )
        assert firstfilepath == os.path.join(temp_dir, 'gaia_dr3_excerpt/94/Gaia_DR3_151.0926_1.8312_18.0_22.0.fits')
        assert dbfile == os.path.join(data_dir, 'gaia_dr3_excerpt/94/Gaia_DR3_151.0926_1.8312_18.0_22.0.fits')
        assert catexp.num_items == 178
        assert catexp.format == 'fitsldac'
        assert catexp.origin == 'gaia_dr3'
        assert catexp.minmag == 18.
        assert catexp.maxmag == 22.
        assert ( catexp.dec_corner_11 - catexp.dec_corner_00 ) == pytest.approx( 1.2 * (1.90649-1.75582), abs=1e-4 )
        catexp, secondfilepath, dbfile = download_gaia_dr3( 150.9427, 151.2425, 1.75582, 1.90649,
                                                                      padding=0.1, minmag=17., maxmag=19. )
        assert secondfilepath == os.path.join(temp_dir, 'gaia_dr3_excerpt/94/Gaia_DR3_151.0926_1.8312_17.0_19.0.fits')
        assert dbfile == os.path.join(data_dir, 'gaia_dr3_excerpt/94/Gaia_DR3_151.0926_1.8312_17.0_19.0.fits')
        assert catexp.num_items == 59
        assert catexp.minmag == 17.
        assert catexp.maxmag == 19.

        _, tbl = ldac.get_table_from_ldac( secondfilepath, imghdr_as_header=True )
        for col in [ 'X_WORLD', 'Y_WORLD', 'ERRA_WORLD', 'ERRB_WORLD', 'PM', 'PMRA', 'PMDEC',
                     'MAG_G', 'MAGERR_G', 'MAG_BP', 'MAGERR_BP', 'MAG_RP', 'MAGERR_RP', 'STARPROB',
                     'OBSDATE', 'FLAGS' ]:
            assert col in tbl.columns
        assert ( tbl['STARPROB'] > 0.95 ).sum() == 59

    finally:
        if firstfilepath is not None:
            pathlib.Path( firstfilepath ).unlink( missing_ok=True )
        if secondfilepath is not None:
            pathlib.Path( secondfilepath ).unlink( missing_ok=True )


def test_download_gaia_dr3( temp_dir, data_dir ):
    do_basic_download_dr3( temp_dir, data_dir )


def test_download_gaia_dr3_noirlab(temp_dir, data_dir):
    # NEVER DO THIS.  If you modify the _static field of a Config object, you're doing it wrong.
    # But.... for this test to work we have to do it wrong.
    cfg = Config.get()
    cfg._static = False
    orig_use_server = cfg.value( 'catalog_gaiadr3.use_server' )
    orig_fallback_datalab = cfg.value( 'catalog_gaiadr3.fallback_datalab' )
    try:
        cfg.set_value( 'catalog_gaiadr3.use_server', False )
        cfg.set_value( 'catalog_gaiadr3.fallback_datalab', True )
        do_basic_download_dr3( temp_dir, data_dir )
    finally:
        cfg.set_value( 'catalog_gaiadr3.use_server', orig_use_server )
        cfg.set_value( 'catalog_gaiadr3.fallback_datalab', orig_fallback_datalab )
        cfg._static = True


def test_fetch_gaia_dr3_excerpt( test_config ) :
    fakeimage = types.SimpleNamespace()
    fakeimage.ra = 180.25
    fakeimage.dec = -30.25
    fakeimage.ra_corner_00 = 180.0
    fakeimage.ra_corner_01 = 180.0
    fakeimage.ra_corner_10 = 180.5
    fakeimage.ra_corner_11 = 180.5
    fakeimage.dec_corner_00 = -30.5
    fakeimage.dec_corner_10 = -30.5
    fakeimage.dec_corner_01 = -30.
    fakeimage.dec_corner_11 = -30.
    fakeimage.minra = 180.0
    fakeimage.maxra = 180.5
    fakeimage.mindec = -30.5
    fakeimage.maxdec = -30.

    catexp_list = {}
    try:
        firstcatexp = fetch_gaia_dr3_excerpt( fakeimage )
        catexp_list[ firstcatexp.id ] = firstcatexp
        # ... this changed from 3139 to 3155 at some point before 2026-08-28,
        #   which alarms me.  Is gaia DR3 not always the same thing?
        assert len( firstcatexp.data ) == 3155

        catexp = fetch_gaia_dr3_excerpt( fakeimage, maxmags=21, magrange=2 )
        catexp_list[ catexp.id ] = catexp
        assert len( catexp.data ) < len( firstcatexp.data )
        assert all( catexp.data['MAG_G'] >= 19 )
        assert all( catexp.data['MAG_G'] <= 21 )
        assert catexp.id != firstcatexp.id

        # WARNING.  Playing with config here
        # in ways you never should, so don't use this test as an example
        # of things you should do.  We're screwing up the gaia server to
        # that we'll get an error if the gaiaserver is contacted.
        orig_gaia_url = test_config.value( 'catalog_gaiadr3.server_url' )
        orig_static = test_config._static
        try:
            test_config._static = False
            test_config.set_value( 'catalog_gaiadr3.server.url', 'https://localhost:666' )

            # Make sure if we ask for something we've already received, we
            # get the same thing back.
            recatexp = fetch_gaia_dr3_excerpt( fakeimage, maxmags=21, magrange=2 )
            catexp_list[ recatexp.id ] = recatexp
            assert recatexp.id == catexp.id

            # One weird thing is that if we ask for something that's a subset of an
            # already-saved excerpt, we should get the bigger excerpt back
            fakeimage2 = types.SimpleNamespace()
            fakeimage.ra, fakeimage.dec = 180.1, -30.1
            fakeimage2.ra_corner_00, fakeimage2.ra_corner_01, fakeimage2.minra = 180.05, 180.05, 180.05
            fakeimage2.ra_corner_10, fakeimage2.ra_corner_11, fakeimage2.maxra = 180.15, 180.15, 180.15
            fakeimage2.dec_corner_00, fakeimage2.dec_corner_10, fakeimage2.mindec = -30.15, -30.15, -30.15
            fakeimage2.dec_corner_01, fakeimage2.dec_corner_11, fakeimage2.maxdec = -30.05, -30.05, -30.05
            catexp = fetch_gaia_dr3_excerpt( fakeimage2 )
            catexp_list[ catexp.id ] = catexp
            assert catexp.id == firstcatexp.id
        finally:
            test_config.set_value( 'catalog_gaiadr3.server_url', orig_gaia_url )
            test_config._static = orig_static


    finally:
        for catexp in catexp_list.values():
            catexp.delete_from_disk_and_database()



def test_gaia_dr3_excerpt_failures( ztf_datastore_uncommitted, ztf_gaia_dr3_excerpt ):
    ds = ztf_datastore_uncommitted
    try:
        # Make sure it fails if we give it a ridiculous max mag
        with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
            catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=5.0, magrange=4, minstars=50 )

        # ...but make sure it succeeds if we also give it a reasonable max mag
        catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[5.0, 20.0], magrange=4.0, minstars=50 )
        assert catexp.id == ztf_gaia_dr3_excerpt.id

        # Make sure it fails if we ask for too many stars
        with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
            catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.0], magrange=4.0, minstars=50000 )

        # Make sure it fails if mag range is too small
        with pytest.raises( CatalogNotFoundError, match="Failed to fetch Gaia DR3 stars at" ):
            catexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.0], magrange=0.01, minstars=50 )

    finally:
        catexp.delete_from_disk_and_database()


def test_gaia_dr3_excerpt( ztf_datastore_uncommitted, ztf_gaia_dr3_excerpt ):
    catexp = ztf_gaia_dr3_excerpt
    ds = ztf_datastore_uncommitted

    assert catexp.num_items == 172
    assert catexp.num_items == len( catexp.data )
    assert catexp.filepath == 'gaia_dr3_excerpt/30/Gaia_DR3_153.6459_39.0937_16.0_20.0.fits'
    assert pathlib.Path( catexp.get_fullpath() ).is_file()
    assert catexp.object_ras.min() == pytest.approx( 153.413563, abs=0.1/3600. )
    assert catexp.object_ras.max() == pytest.approx( 153.877110, abs=0.1/3600. )
    assert catexp.object_decs.min() == pytest.approx( 38.914110, abs=0.1/3600. )
    assert catexp.object_decs.max() == pytest.approx( 39.274596, abs=0.1/3600. )
    assert ( catexp.data['X_WORLD'] == catexp.object_ras ).all()
    assert ( catexp.data['Y_WORLD'] == catexp.object_decs ).all()
    assert catexp.data['MAG_G'].min() == pytest.approx( 16.076, abs=0.001 )
    assert catexp.data['MAG_G'].max() == pytest.approx( 19.994, abs=0.001 )
    assert catexp.data['MAGERR_G'].min() == pytest.approx( 0.0004, abs=0.0001 )
    assert catexp.data['MAGERR_G'].max() == pytest.approx( 0.018, abs=0.001 )

    # Test reading of cache
    newcatexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.0], magrange=4.0, minstars=50, onlycached=True )
    assert newcatexp.id == catexp.id

    # Make sure we can't read the cache for something that doesn't exist
    with pytest.raises( CatalogNotFoundError, match='Failed to fetch Gaia DR3 stars' ):
        newcatexp = fetch_gaia_dr3_excerpt( ds.image, maxmags=[20.5], magrange=4.0, minstars=50, onlycached=True )


def do_download_gaia_dr3_excerpt_ra_span_zero():
    stars = None
    firstcat = None
    try:
        ra = 0.
        dec = 0.
        ras = [ 359.79, 359.81, 0.19, 0.21 ]
        decs = [ -0.19, 0.21, -0.21, 0.19 ]
        ras, decs, minra, maxra, mindec, maxdec = FourCorners.sort_radec( ras, decs )
        img = Image( ra=ra, dec=dec,
                     ra_corner_00=ras[0],
                     ra_corner_01=ras[1],
                     ra_corner_10=ras[2],
                     ra_corner_11=ras[3],
                     minra=minra,
                     maxra=maxra,
                     dec_corner_00=decs[0],
                     dec_corner_01=decs[1],
                     dec_corner_10=decs[2],
                     dec_corner_11=decs[3],
                     mindec=mindec,
                     maxdec=maxdec )
        stars = fetch_gaia_dr3_excerpt( img )
        assert stars.num_items == 776
        lt0ras = stars.object_ras[ stars.object_ras > 180. ]
        gt0ras = stars.object_ras[ stars.object_ras < 180. ]
        assert len(lt0ras) == 399
        assert len(gt0ras) == 377
        # The numbers below take into account the 5% padding on each side that fetch_gaia_dr3_excerpt uses
        assert min(lt0ras) == pytest.approx( 359.748, abs=0.003 )
        assert max(lt0ras) == pytest.approx( 360., abs=0.001 )
        assert min(gt0ras) == pytest.approx( 0., abs=0.001 )
        assert max(gt0ras) == pytest.approx( 0.252, abs=0.003 )
        assert min(stars.object_decs) == pytest.approx( -0.252, abs=0.003 )
        assert max(stars.object_decs) == pytest.approx( 0.252, abs=0.003 )

        # Do it again, make sure we get the same thing
        firstcat = stars
        stars = fetch_gaia_dr3_excerpt( img )
        assert stars.id == firstcat.id

    finally:
        # Clean up
        if stars is not None:
            stars.delete_from_disk_and_database()
        if firstcat is not None:
            firstcat.delete_from_disk_and_database()


def test_gaia_dr3_excerpt_ra_span_zero():
    do_download_gaia_dr3_excerpt_ra_span_zero()


def test_gaia_dr3_excerpt_ra_span_zero_noirlab():
    # NEVER DO THIS.  If you modify the _static field of a Config object, you're doing it wrong.
    # But.... for this test to work we have to do it wrong.
    cfg = Config.get()
    cfg._static = False
    orig_use_server = cfg.value( 'catalog_gaiadr3.use_server' )
    orig_fallback_datalab = cfg.value( 'catalog_gaiadr3.fallback_datalab' )
    try:
        cfg.set_value( 'catalog_gaiadr3.use_server', False )
        cfg.set_value( 'catalog_gaiadr3.fallback_datalab', True )
        do_download_gaia_dr3_excerpt_ra_span_zero()
    finally:
        cfg.set_value( 'catalog_gaiadr3.use_server', orig_use_server )
        cfg.set_value( 'catalog_gaiadr3.fallback_datalab', orig_fallback_datalab )
        cfg._static = True
