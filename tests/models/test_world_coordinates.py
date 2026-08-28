import pytest
import hashlib
import os
import pathlib
import uuid

import psycopg.errors
import numpy as np
from astropy.wcs import WCS

from models.image import Image
from models.source_list import SourceList
from models.world_coordinates import WorldCoordinates


def test_world_coordinates( ztf_datastore_uncommitted, provenance_base, provenance_extra ):
    ds = ztf_datastore_uncommitted
    ds.image.instrument = 'DECam'    # hack - otherwise invent_filepath will not work as 'ZTF' is not an Instrument

    origwcs = WCS( ds.image.header )
    origscs = origwcs.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )

    # Make sure we can construct a WorldCoordinates object from a WCS object

    wcobj = WorldCoordinates()
    wcobj.wcs = origwcs
    wcobj.set_corners_from_wcs( ds.image, setradec=True )
    header_excerpt = wcobj.wcs.to_header( relax=True ).tostring( sep='\n', padding=False)
    md5 = hashlib.md5( header_excerpt.encode('ascii') )
    assert md5.hexdigest() == 'a13d6bdd520c5a0314dc751025a62619'

    # Make sure that we can construct a WCS from a WorldCoordinates

    old_wcs = wcobj.wcs
    wcobj = WorldCoordinates()
    wcobj.wcs = old_wcs
    wcobj.set_corners_from_wcs( ds.image, setradec=True )
    scs = wcobj.wcs.pixel_to_world( [ 0, 0, 1024, 1024 ], [ 0, 1024, 0, 1024 ] )
    for sc, origsc in zip( scs, origscs ):
        assert sc.ra.value == pytest.approx( origsc.ra.value, abs=0.01/3600. )
        assert sc.dec.value == pytest.approx( origsc.dec.value, abs=0.01/3600. )

    # save the WCS to file and DB
    try:
        ds.image.provenance_id = provenance_base.id
        ds.image.save()
        ds.image.insert()
        ds.sources.provenance_id = provenance_extra.id
        ds.sources.save()
        ds.sources.insert()
        ds.psf.save( image=ds.image, sources=ds.sources )

        wcobj.sources_id = ds.sources.id
        wcobj.provenance_id = provenance_base.id
        wcobj.save( image=ds.image )
        wcobj.insert()

        # add a second WCS object and make sure we cannot accidentally commit it, too
        wcobj2 = WorldCoordinates()
        wcobj2.wcs = old_wcs
        wcobj2.set_corners_from_wcs( ds.image, setradec=True )
        wcobj2.sources_id = ds.sources.id
        wcobj2.provenance_id = provenance_base.id
        wcobj2.save( image=ds.image ) # overwrite the save of wcobj

        with pytest.raises( psycopg.errors.UniqueViolation,
                            match='duplicate key value violates unique constraint "_wcs_source_list_provenance_uc"' ):
            wcobj2.insert()

        # also test the filename uniqueness
        sl = SourceList( image_id=ds.image.id, format='sepnpy', num_sources=1, provenance_id=provenance_base.id,
                         filepath="foo", md5sum=uuid.uuid4() )
        sl.insert()
        wcobj2.sources_id = sl.id
        with pytest.raises( psycopg.errors.UniqueViolation,
                            match='duplicate key value violates unique constraint "ix_world_coordinates_filepath"' ):
            wcobj2.insert()
        sl.delete_from_disk_and_database()

        # ensure you cannot overwrite when explicitly setting overwrite=False
        wcobj2.sources_id = ds.sources.id
        wcobj2.provenance_id = provenance_base.id
        with pytest.raises( OSError, match=".txt already exists" ):
            wcobj2.save(overwrite=False)

    finally:
        if 'wcobj' in locals():
            wcobj.delete_from_disk_and_database()

        if 'wcobj2' in locals():
            wcobj2.delete_from_disk_and_database()

        ds.sources.delete_from_disk_and_database()
        ds.image.delete_from_disk_and_database()

    # Do lots of tests of set_corners_from_image

    origra = ds.image.ra
    origdec = ds.image.dec

    def reset_nulls( image, wcs ):
        image.ra = origra
        image.dec = origdec
        wcs.ra = None
        wcs.dec = None
        for radec in [ 'ra', 'dec' ]:
            for good in [ False, True ]:
                for minmax in [ 'min', 'max'] :
                    setattr( wcs, f'{"good_" if good else ""}{minmax}{radec}', None )
                    if not good:
                        setattr( image, f'{minmax}{radec}', None )
                for corner in [ '00', '01', '10', '11' ]:
                    setattr( wcs, f'{radec}_{"good" if good else "corner"}_{corner}', None )
                    if not good:
                        setattr( image, f'{radec}_corner_{corner}', None )

    def check_corners( obj, setradec=False ):
        expecteds = {
            'minra': pytest.approx( 153.45158, abs=1e-4 ),
            'maxra': pytest.approx( 153.84018, abs=1e-4 ),
            'mindec': pytest.approx( 38.94299, abs=1e-4 ),
            'maxdec': pytest.approx( 39.24451, abs=1e-4 ),
            'ra_corner_00': pytest.approx( 153.47066, abs=1e-4 ),
            'ra_corner_01': pytest.approx( 153.45158, abs=1e-4 ),
            'ra_corner_10': pytest.approx( 153.84018, abs=1e-4 ),
            'ra_corner_11': pytest.approx( 153.82262, abs=1e-4 ),
            'dec_corner_00': pytest.approx( 38.94299, abs=1e-4 ),
            'dec_corner_01': pytest.approx( 39.23028, abs=1e-4 ),
            'dec_corner_10': pytest.approx( 38.95711, abs=1e-4 ),
            'dec_corner_11': pytest.approx( 39.24451, abs=1e-4 )
        }
        for radec in [ 'ra', 'dec' ]:
            for minmax in [ 'min', 'max' ]:
                assert getattr( obj, f'{minmax}{radec}' ) == expecteds[ f'{minmax}{radec}' ]
                if isinstance( obj, WorldCoordinates ):
                    assert getattr( obj, f'good_{minmax}{radec}' ) == expecteds[ f'{minmax}{radec}' ]
            for corner in [ '00', '01', '10', '11' ]:
                assert getattr( obj, f'{radec}_corner_{corner}' ) == expecteds[ f'{radec}_corner_{corner}' ]
                if isinstance( obj, WorldCoordinates ):
                    assert getattr( obj, f'{radec}_good_{corner}' ) == expecteds[ f'{radec}_corner_{corner}' ]

        if setradec:
            assert obj.ra == pytest.approx( 153.64608, abs=1e-4 )
            assert obj.dec == pytest.approx( 39.09371, abs=1e-4 )
        elif isinstance( obj, Image ):
            assert obj.ra == pytest.approx( 153.64626, abs=1e-4 )
            assert obj.dec == pytest.approx( 39.090189, abs=1e-4 )


    reset_nulls( ds.image, wcobj )
    ds.image.set_corners_from_wcs( wcobj )
    check_corners( ds.image )
    wcobj.set_corners_from_wcs( ds.image )
    check_corners( wcobj )

    reset_nulls( ds.image, wcobj )
    ds.image.set_corners_from_wcs( wcobj, setradec=True )
    check_corners( ds.image, setradec=True )
    wcobj.set_corners_from_wcs( ds.image, setradec=True )
    check_corners( wcobj, setradec=True )

    reset_nulls( ds.image, wcobj )
    ds.image.set_corners_from_wcs( wcobj, ds.image.width, ds.image.height )
    check_corners( ds.image )
    wcobj.set_corners_from_wcs( width=ds.image.width, height=ds.image.height )
    check_corners( wcobj )

    reset_nulls( ds.image, wcobj )
    ds.image.set_corners_from_wcs( wcobj, ds.image.width, ds.image.height, setradec=True )
    check_corners( ds.image, setradec=True )
    wcobj.set_corners_from_wcs( width=ds.image.width, height=ds.image.height, setradec=True )
    check_corners( wcobj, setradec=True )

    mask = np.zeros( ( ds.image.height, ds.image.width ), dtype=np.int16 )
    reset_nulls( ds.image, wcobj )
    wcobj.set_corners_from_wcs( mask=mask )
    check_corners( wcobj )
    reset_nulls( ds.image, wcobj )
    wcobj.set_corners_from_wcs( mask=mask, setradec=True )
    check_corners( wcobj, setradec=True )

    # TODO test exceptions from bad arguments to set_corners_from_wcs


def test_save_and_load_wcs(ztf_datastore_uncommitted, provenance_base, provenance_extra):
    ds = ztf_datastore_uncommitted
    ds.image.instrument = 'DECam' # otherwise invent_filepath will not work as 'ZTF' is not an Instrument
    ds.image.provenance_id = provenance_base.id
    ds.sources.provenance_id = provenance_extra.id

    origwcs = WCS( ds.image.header )
    wcobj = WorldCoordinates()
    wcobj.wcs = origwcs
    wcobj.set_corners_from_wc( ds.image )
    wcobj.sources_id = ds.sources.id
    wcobj.provenance_id = provenance_extra.id

    try:
        wcobj.save( image=ds.image )
        txtpath = pathlib.Path( wcobj.local_path ) / f'{wcobj.filepath}'

        # check for an error if the file is not found when loading
        os.remove(txtpath)
        with pytest.raises( OSError, match="File .* not found!" ):
            wcobj.load( download=False )

        # ensure you can create an identical wcs from a saved one
        wcobj.save( image=ds.image )
        wcobj2 = WorldCoordinates()
        wcobj2.load( txtpath=txtpath )

        assert wcobj2.wcs.to_header( relax=True ) == wcobj.wcs.to_header( relax=True )

    finally:
        if "wcobj" in locals():
            wcobj.delete_from_disk_and_database()
        if "wcobj2" in locals():
            wcobj2.delete_from_disk_and_database()
