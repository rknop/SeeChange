import shutil
import textwrap
import random

import numpy as np
from astropy.io import fits
from psycopg import sql

import models.object
from models.base import PGDB
from pipeline.lightcurve import Lightcurve
from util.logger import SCLogger


def test_lightcurve( sim_lightcurve_persistent_sources, sim_lightcurve_news, sim_lightcurve_forcedphot_references,
                     sim_lightcurve_image_parameters ):
    srcs = sim_lightcurve_persistent_sources
    imageinfo, _ = sim_lightcurve_image_parameters
    newdsen = sim_lightcurve_news

    objinfos = []
    for source in srcs:
        objinfos.append( { 'ra': source['ra'],
                           'dec': source['dec'],
                           'mjds': imageinfo['mjdoffs'] + imageinfo['refmjd'],
                           'fluxen': source['maxflux'] * np.exp( -( imageinfo['mjdoffs'] - source['mjdmaxoff'] ) **2
                                                                 / ( 2 * source['sigmadays']**2 ) )
                          } )

    # Ideally, everything is cleaned up when upstreams in the fixtures
    # are deleted.  However, I don't think we can count on them doing it
    # in the right order.  So, try to clean up everything we make here.
    nukes = { 'loose_files': [],
              'forcedphot': [],
              'subimids': [],
              'objects': [] }
    try:
        for obji, objinfo in enumerate( objinfos ):
            # ****
            # TODO INVESTIGATE
            # Failing to get a ref for objects 0 or 1
            # if i in (0, 1):
            #     continue
            # ****

            # Create an object
            obj = models.object.Object( name=f'test_lightcurve_object_{obji}',
                                        ra=objinfo['ra'], dec=objinfo['dec'] )
            obj.insert()
            nukes['objects'].append( obj )

            # Lightcurve builder
            ltcv = Lightcurve( zp_prov = newdsen[0].zp.provenance_id,
                               crop_image = [150, 150],
                               mjd0 = imageinfo['refmjd'] + imageinfo['mjdoffs'][0] - 0.1,
                               mjd1 = imageinfo['refmjd'] + imageinfo['mjdoffs'][-1] + 0.1,
                               instrument='DemoInstrument',
                               object_name=f'test_lightcurve_object_{obji}',
                               # NOTE : currently using the search reference, not centered lightcurve
                               #   references, so filtering on max_dist will throw out two of the
                               #   three candidates.  TODO, actual lightcurve refs.
                               subtraction={ 'refset': 'sim_lightcurve_forcedphot_reference',
                                             'alignment': { 'min_matched': 6 },
                                             'reference': { 'max_dist': None }
                                            },
                               save_to_db=True
                              )
            import pdb; pdb.set_trace()
            ltcv.run( cache_aligned_images=True )
            nukes['forcedphot'].extend( ltcv.forced_phots )
            nukes['subimids'].extend( p.subtraction_id for p in ltcv.forced_phots )

            #####
            # Uncomment this to write out debugging images to the test directory
            from models.image import Image
            from models.source_list import SourceList
            import pathlib
            comps = ['image', 'weight', 'flags']
            atts = ['data', 'weight', 'flags']
            for filt, ref in ltcv.refs.items():
                origfiles = ref.image.get_fullpath( components=comps )
                for comp, origfile in zip( comps, origfiles ):
                    destfile = f"ref_{comp}_{filt}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
                    shutil.copy2( origfile, destfile)
                    nukes['loose_files'].append( pathlib.Path(destfile) )
                destfile = f"ref_{comp}_{filt}.reg"
                ref.sources.ds9_regfile( destfile )
                nukes['loose_files'].append( pathlib.Path( destfile ) )
            with PGDB( dictcursor=True ) as pgdb:
                for phot in ltcv.forced_phots:
                    subim = Image.get_by_id( phot.subtraction_id, pgdb=pgdb )
                    origfiles = subim.get_fullpath( comps )
                    for comp, origfile in zip( comps, origfiles ):
                        destfile = f"sub_{comp}_{subim.filter}_{obji}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
                        shutil.copy2( origfile, destfile )
                        nukes['loose_files'].append( pathlib.Path(destfile) )
                    q = sql.SQL( textwrap.dedent(
                        """\
                        SELECT s.* FROM source_lists s
                        INNER JOIN world_coordinates w ON w.sources_id=s._id
                        INNER JOIN zero_points z ON z.wcs_id=w._id
                        INNER JOIN image_subtraction_components isc ON isc.new_zp_id=z._id
                        WHERE isc.image_id={subid}
                        """
                    ) ).format( subid=subim.id )
                    rows = pgdb.execute( q )
                    if len(rows) != 1:
                        raise RuntimeError( "I am surprised." )
                    newsrcs = SourceList.create( **(rows[0]) )
                    newim = Image.get_by_id( newsrcs.image_id, pgdb=pgdb )
                    origfiles = newim.get_fullpath( components=comps )
                    for comp, origfile in zip( atts, origfiles ):
                        destfile = f"new_{comp}_{newim.filter}_{obji}.fits{'.fz' if origfile[-3:]=='.fz' else ''}"
                        shutil.copy2( origfile, destfile )
                        nukes['loose_files'].append( pathlib.Path(destfile) )
                    destfile = f"new_{comp}_{newim.filter}_{obji}.reg"
                    newsrcs.ds9_regfile( destfile )
                    nukes['loose_files'].append( pathlib.Path( destfile ) )
            for aligned in ltcv.aligned_cache:
                hdr = fits.Header( aligned['wcs'].wcs.to_header( relax=True ) )
                for comp, att in zip( comps, atts ):
                    destfile = f"alignedref_{comp}_{newim.filter}_{obji}.fits"
                    fits.writeto( destfile, data=getattr( aligned['ref_image'], att ), header=hdr )
                    nukes['loose_files'].append( pathlib.Path( destfile ) )
                    destfile = f"alignedref_{comp}_{newim.filter}_{obji}.reg"
                    aligned['ref_sources'].ds9_regfile( destfile )
                    nukes['loose_files'].append( pathlib.Path( destfile ) )
            SCLogger.warning( "Lightcurve images written to test directory" )
            import pdb; pdb.set_trace()
            ####

            assert len( ltcv.forced_phots ) == len( imageinfo['mjdoffs'] )
            fluxen = np.array( [ p.flux_psf for p in ltcv.forced_phots ] )

    finally:
        # Delete test files if any
        for f in nukes['loose_files']:
            f.unlink( missing_ok=True )

        # Delete forced phot first, because it's furthest downstream
        for p in nukes['forcedphot']:
            p.delete_from_disk_and_database()

        # Should be safe to delete objects now:
        for o in nukes['objects']:
            o.delete_from_disk_and_database()

        # For subtractions, trace back to the parent trimmed image so
        # that we can delete that, trusting on its downstream deleting
        # to get down to the subtraction.  (Again, these trimmed images
        # are supposed to be deleted as downstreams of the fixture-made
        # images, but the fixtures aren't currently deleting things in
        # the right order to avoid all RESTRICT foreign keys.  Besides,
        # it's nice to clean up after ourselves, yes?)
        import pdb; pdb.set_trace()
        if len( nukes['subimids'] ) > 0:
            with PGDB( dictcursor=True ) as pgdb:
                q = sql.SQL( textwrap.dedent(
                    """
                    SELECT i.* FROM images i
                    INNER JOIN source_lists s ON s.image_id=i._id
                    INNER JOIN world_coordinates w ON w.sources_id=i._id
                    INNER JOIN zero_points z ON z.wcs_id=w._id
                    INNER JOIN image_subtraction_components isc ON isc.new_zp_id=z._id
                    WHERE isc.image_id=ANY(ARRAY[{subids}])
                    """
                ) ).format( subids=sql.SQL(",").join( nukes['subimids'] ) )
                rows = pgdb.execute( q )
            for row in rows:
                img = Image.create( **(row) )
                img.delete_from_disk_and_database()
