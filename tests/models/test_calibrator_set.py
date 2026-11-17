import pytest
import uuid
import random

from psycopg import sql

from models.base import PGDB
from models.calibrator_set import CalibratorSet


def test_calibrator_set_create( bogus_image_factory, bogus_datafile_factory ):
    nukes = []
    barf = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    cset = None
    try:
        flat = bogus_image_factory( uuid.uuid4(), f'{barf}_flat' )
        nukes.append( flat )
        zero = bogus_image_factory( uuid.uuid4(), f'{barf}_zero' )
        nukes.append( zero )
        linearity = bogus_datafile_factory( uuid.uuid4(), f'{barf}_linearity.dat' )
        nukes.append( linearity )

        with pytest.raises( RuntimeError, match=( "Don't instantiate Calibratorset directly, "
                                                  "call Calibratorset.create" ) ):
            _ = CalibratorSet( flat_image=flat, zero_image=zero, linearity_datafile=linearity )

        cset = CalibratorSet.create( flat_image=flat, zero_image=zero, linearity_datafile=linearity )

        cset_prov = cset.generate_provenance()
        nukes.append( cset_prov )
        assert cset.provenance_id == cset_prov.id

        for attr in CalibratorSet.calibitems:
            assert getattr( cset, f'{attr}_id' ) is None

        assert set( cset._file_cache.keys() ) == { 'flat_image_id', 'zero_image_id', 'linearity_datafile_id' }
        assert cset._file_cache[ 'flat_image_id' ] == flat.id
        assert cset._file_cache[ 'zero_image_id' ] == zero.id
        assert cset._file_cache[ 'linearity_datafile_id' ] == linearity.id

        assert cset.flat_image == flat
        assert cset.flat_image_id == flat.id
        assert cset.zero_image == zero
        assert cset.zero_image_id == zero.id
        assert cset.linearity_datafile == linearity
        assert cset.linearity_datafile_id == linearity.id
        assert cset.flat_datafile is None
        assert cset.zero_datafile is None
        assert cset.linearity_image is None
        for suffix in ( 'image', 'image_id', 'datafile', 'datafile_id' ):
            assert all( getattr( cset, f'{i}_{suffix}' ) is None
                        for i in CalibratorSet.calibitems
                        if i not in ( 'flat', 'zero', 'linearity' ) )

        cset.insert()

        assert all( getattr( cset, f'{attr}_id' ) is not None for attr in ( 'flat', 'zero', 'linearity' ) )
        assert all( getattr( cset, f'{attr}_id' ) is None
                    for attr in CalibratorSet.calibitems
                    if attr not in ( 'flat', 'zero', 'linearity' ) )

        # Verify cache reloading works

        cset._file_cache = {}
        assert cset.flat_image == flat
        assert cset.flat_image_id == flat.id
        assert cset.zero_image == zero
        assert cset.zero_image_id == zero.id
        assert cset.linearity_datafile == linearity
        assert cset.linearity_datafile_id == linearity.id
        assert cset.flat_datafile is None
        assert cset.zero_datafile is None
        assert cset.linearity_image is None
        for suffix in ( 'image', 'image_id', 'datafile', 'datafile_id' ):
            assert all( getattr( cset, f'{i}_{suffix}' ) is None
                        for i in CalibratorSet.calibitems
                        if i not in ( 'flat', 'zero', 'linearity' ) )

        oldid = cset.id
        cset._id = None
        with pytest.raises( RuntimeError, match=r"CalibSet already exists." ):
            cset.insert()
        cset.insert( exist_ok=True )
        assert cset.id == oldid

    finally:
        with PGDB() as pgdb:
            if cset is not None:
                cset.delete_from_disk_and_database( pgdb=pgdb )
                pgdb.execute( sql.SQL( "DELETE FROM datafile_or_image_id WHERE _id=ANY(ARRAY[{ids}])"
                                      ).format( ids=sql.SQL(",").join( [ cset.flat_id, cset.zero_id,
                                                                         cset.linearity_id ] ) ) )
                pgdb.commit()
            for nuke in nukes:
                nuke.delete_from_disk_and_database( pgdb=pgdb )
