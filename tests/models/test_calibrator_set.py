import uuid
import random

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

        cset = CalibratorSet( flat_image=flat, zero_image=zero, linearity_datafile=linearity )

        cset_prov = cset.generate_provenance()
        nukes.append( cset_prov )
        assert cset.provenance_id == cset_prov.id

        assert set( cset.images.keys() ) == { 'flat', 'zero' }
        assert set( cset.datafiles.keys() ) == { 'linearity' }
        assert cset.image( 'flat' ) == flat
        assert cset.image( 'zero' ) == zero
        assert cset.image( 'linearity' ) is None
        assert cset.image( 'foo' ) is None
        assert cset.datafile( 'linearity' ) == linearity
        assert cset.datafile( 'flat' ) is None
        assert cset.datafile( 'zero' ) is None
        assert cset.datafile( 'foo' ) is None

        assert set( cset.image_ids.keys() ) == { 'flat', 'zero' }
        assert cset.image_ids[ 'flat' ] == flat.id
        assert cset.image_ids[ 'zero' ] == zero.id
        assert set( cset.datafile_ids.keys() ) == { 'linearity' }
        assert cset.datafile_ids[ 'linearity' ] == linearity.id

        # Make sure it can lazy load
        cset._images = {}
        cset._datafiles = {}
        assert cset.image( 'flat' ).id == flat.id
        assert cset.image( 'zero' ).id == zero.id
        assert cset.image( 'linearity' ) is None
        assert cset.image( 'foo' ) is None
        assert cset.datafile( 'linearity' ).id == linearity.id
        assert cset.datafile( 'flat' ) is None
        assert cset.datafile( 'zero' ) is None
        assert cset.datafile( 'foo' ) is None

        cset.insert()
        nukes.append( cset )

        newcset = CalibratorSet.get_by_id( cset.id )
        assert newcset.id == cset.id
        assert newcset.provenance_id == cset.provenance_id
        assert newcset._images == {}
        assert newcset._datafiles == {}
        assert set( newcset.image_ids.keys() ) == { 'flat', 'zero' }
        assert set( newcset.datafile_ids.keys() ) == { 'linearity' }
        assert newcset.image_ids[ 'flat' ] == cset.image_ids[ 'flat']
        assert newcset.image_ids[ 'zero' ] == cset.image_ids[ 'zero']
        assert newcset.datafile_ids[ 'linearity' ] == cset.datafile_ids[ 'linearity']

        assert newcset.image( 'flat' ).id == cset.image( 'flat' ).id
        assert newcset.image( 'zero' ).id == cset.image( 'zero' ).id
        assert newcset.datafile( 'linearity' ).id == cset.datafile( 'linearity' ).id
        assert newcset.image( 'linearity' ) is None
        assert newcset.image( 'foo' ) is None
        assert newcset.datafile( 'flat' ) is None
        assert newcset.datafile( 'zero' ) is None
        assert newcset.datafile( 'foo' ) is None

    finally:
        with PGDB() as pgdb:
            nukes.reverse()
            for nuke in nukes:
                nuke.delete_from_disk_and_database( pgdb=pgdb )
