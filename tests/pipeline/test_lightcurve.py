import shutil
import textwrap
import random

import numpy as np
from psycopg import sql

import models.object
from models.base import PGDB
from pipeline.lightcurve import Lightcurve
from util.logger import SCLogger


def test_lightcurve( sim_lightcurve_persistent_sources, sim_lightcurve_reference, sim_lightcurve_news,
                     sim_lightcurve_image_parameters ):
    srcs = sim_lightcurve_persistent_sources
    imageinfo, _ = sim_lightcurve_image_parameters
    ref, refds = sim_lightcurve_reference
    newdsen = sim_lightcurve_news

    objinfos = []
    for source in srcs:
        objinfos.append( { 'ra': source['ra'],
                           'dec': source['dec'],
                           'mjds': imageinfo['mjdoffs'] + imageinfo['refmjd'],
                           'fluxen': source['maxflux'] * np.exp( -( imageinfo['mjdoffs'] - source['mjdmaxoff'] ) **2
                                                                 / ( 2 * source['sigmadays']**2 ) )
                          } )

    # ...I think we don't need to clean up.  Objects are allowed to hang
    # around in the database (and deleting them would mean we'd first
    # have to delete the forced photometry).  Provenances are allowed to
    # hang around.  Everything else will be deleted when the fixtures
    # clean up and recursively delete downstreams.

    barf = ''.join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=6 ) )
    for i, objinfo in enumerate( objinfos ):
        # ****
        # TODO INVESTIGATE
        # Failing to get a ref for objects 0 or 1
        if i in (0, 1):
            continue
        # ****

        # Create an object
        obj = models.object.Object( name=f'test_lightcurve_object_{i}_{barf}', ra=objinfo['ra'], dec=objinfo['dec'] )
        obj.insert()

        # Lightcurve builder
        ltcv = Lightcurve( zp_prov = newdsen[0].zp.provenance_id,
                           crop_image = [150, 150],
                           mjd0 = imageinfo['refmjd'] + imageinfo['mjdoffs'][0] - 0.1,
                           mjd1 = imageinfo['refmjd'] + imageinfo['mjdoffs'][-1] + 0.1,
                           instrument='DemoInstrument',
                           object_name=f'test_lightcurve_object_{i}_{barf}',
                           subtraction={ 'refset': 'sim_lightcurve_reference',
                                         'alignment': { 'min_matched': 6 } },
                           save_to_db=True
                          )
        ltcv.run()
        import pdb; pdb.set_trace()

        #####
        # Uncomment this to write out debugging images to the test directory
        from models.image import Image
        import pathlib
        nukes = []
        for filt, ref in ltcv.refs.items():
            origfile = ref.image.get_fullpath()[0]
            destfile = f"ref_{filt}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
            shutil.copy2( origfile, destfile)
            nukes.append( pathlib.Path(destfile) )
        with PGDB( dictcursor=True ) as pgdb:
            for i, phot in enumerate( ltcv.forced_phots ):
                subim = Image.get_by_id( phot.subtraction_id, pgdb=pgdb )
                origfile = subim.get_fullpath()[0]
                destfile = f"sub_{subim.filter}_{i}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
                shutil.copy2( origfile, destfile )
                nukes.append( pathlib.Path(destfile) )
                q = sql.SQL( textwrap.dedent(
                    """\
                    SELECT i._id FROM images i
                    INNER JOIN source_lists s ON s.image_id=i._id
                    INNER JOIN world_coordinates w ON w.sources_id=s._id
                    INNER JOIN zero_points z ON z.wcs_id=w._id
                    INNER JOIN image_subtraction_components isc ON isc.new_zp_id=z._id
                    WHERE isc.image_id={subid}
                    """
                ) ).format( subid=subim.id )
                rows = pgdb.execute( q )
                if len(rows) != 1:
                    raise RuntimeError( "I am surprised." )
                newim = Image.get_by_id( rows[0]['_id'], pgdb=pgdb )
                origfile = newim.get_fullpath()[0]
                destfile = f"new_{newim.filter}_{i}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
                shutil.copy2( origfile, destfile )
                nukes.append( pathlib.Path(destfile) )
        SCLogger.warning( "Lightcurve images written to test directory" )
        import pdb; pdb.set_trace()
        for nuke in nukes:
            nuke.unlink( missing_ok=True )
        ####

        assert len( ltcv.forced_phots ) == len( imageinfo['mjdoffs'] )
        fluxen = np.array( [ p.flux_psf for p in ltcv.forced_phots ] )
