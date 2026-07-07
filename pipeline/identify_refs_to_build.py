import io
import textwrap
import argparse

import numpy as np
from psycopg import sql

from util.config import Config
from util.logger import SCLogger
from util.radec import ra_cross_zero_avgra
from models.base import PGDB
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.image import Image
from models.exposure import Exposure
from models.reference import Reference
from pipeline.ref_maker import RefMaker


def identify_refs_to_build( mjd0, mjd1, provtag=None, overlapfrac=None, refset=None, no_make_refset=None ):
    cfg = Config.get()
    provtag = provtag if provtag is not None else cfg.value('pipeline.provenance_tag')
    overlapfrac  = overlapfrac if overlapfrac is not None else cfg.value('subtraction.reference.minovfrac')
    refset = refset if refset is not None else cfg.value('subtraction.refset')

    kwargs = { 'maker': { 'name': refset } }
    if no_make_refset is not None:
        kwargs['maker'].update( { 'ignore_config_use_config_from_refset': no_make_refset,
                                  'refset_must_already_exist': no_make_refset } )
    refmaker = RefMaker( **kwargs )
    refmaker.make_refset()
    SCLogger.info( f"Using refset {refmaker.pars.name} with reference provenance id {refmaker.ref_prov.id}" )

    SCLogger.info( f"Searching for images with provenance tag {provtag} between "
                   f"mjd {mjd0:.2f} and {mjd1:.2f}" )

    with PGDB( dictcursor=True ) as pgdb:
        q = sql.SQL( textwrap.dedent(
            """
            SELECT i._id AS imgid, w._id AS wcsid, z._id as zpid
            FROM images i
            INNER JOIN source_lists s ON s.image_id=i._id
            INNER JOIN world_coordinates w ON w.sources_id=s._id
            INNER JOIN zero_points z ON z.wcs_id=w._id
            INNER JOIN provenance_tags t ON t.provenance_id=z.provenance_id
            WHERE i.mjd>={mjd0} AND i.mjd<={mjd1}
              AND t.tag={provtag}
            """
        ) ).format( mjd0=mjd0, mjd1=mjd1, provtag=provtag )
        rows = pgdb.execute( q )

        SCLogger.info( f"Grabbing {len(rows)} images (etc.) from provenance tag {provtag}..." )

        imgs = [ Image.get_by_id( row['imgid'], pgdb=pgdb ) for row in rows ]
        wcses = [ WorldCoordinates.get_by_id( row['wcsid'], pgdb=pgdb ) for row in rows ]
        zps = [ ZeroPoint.get_by_id( row['zpid'], pgdb=pgdb ) for row in rows ]

        # Make sure we don't spend a long time "idle in transaction"
        pgdb.rollback()

        SCLogger.info( "Checking what mooching will be possible... prepare for n²..." )

        mooch = {}
        for dex in range( len(imgs) ):
            for subdex in range( dex ):
                # Plausibility cut so we don't do the full mechanism of get_overlap_frac most of the time
                dec0 = ( wcses[dex].good_mindec + wcses[dex].good_maxdec ) / 2.
                dec1 = ( wcses[subdex].good_mindec + wcses[subdex].good_maxdec ) / 2.
                # TODO, make this cutoff configurable!  1. is chosen because it's 3600 pixels on LS4
                if np.fabs( dec0 - dec1 ) > 1.:
                    continue
                avgdec = ( dec0 + dec1 )
                one = 1. / np.cos( avgdec * np.pi / 180. )
                ra0 = ra_cross_zero_avgra( wcses[dex].good_minra, wcses[dex].good_maxra )
                ra1 = ra_cross_zero_avgra( wcses[subdex].good_minra, wcses[subdex].good_maxra )
                if not ( ( ( ra0 < one or ra0 > 360-one ) and ( ra1 < one or ra1 > 360-one ) )
                         or
                         ( np.fabs( ra0 - ra1 ) <= one )
                        ):
                    continue

                # OK, be more careful
                if WorldCoordinates.get_overlap_frac( wcses[dex], wcses[subdex] ) > overlapfrac:
                    mooch[imgs[dex].id] = imgs[subdex].id
                    break

    SCLogger.info( f"...{len(mooch)} of {len(imgs)} images can mooch" )

    SCLogger.info( "Looking for existing references and buildable references..." )

    haverefs = []
    somebodyelseswillbegood = []
    canbuildrefs = []
    canbuildrefswcses = []
    nope = []
    zpprovid = None

    for ndone, (img, wcs, zp) in enumerate( zip( imgs, wcses, zps ) ):
        if ndone % 10 == 0:
            SCLogger.info( f"...did {ndone} of {len(imgs)}" )

        if zpprovid is None:
            zpprovid = zp.provenance_id
        else:
            if zpprovid != zp.provenance_id:
                raise RuntimeError( "This should never happen." )

        refs, refimgs = Reference.get_references( image=img, filter=img.filter, instrument=img.instrument,
                                                  refset=refset, overlapfrac=overlapfrac )
        if len(refs) > 0:
            haverefs.append( img )

        else:
            if img.id in mooch.keys():
                somebodyelseswillbegood.append( img )
            else:
                ( images, match_pos, match_count
                  ) = refmaker.choose_reference_images_to_coadd( img.id,
                                                                 image_zp_prov_id=zp.provenance_id,
                                                                 filter=img.filter,
                                                                 log_to_info=False )
                if images is not None:
                    canbuildrefs.append( img )
                    canbuildrefswcses.append( wcs )
                else:
                    nope.append( img )

    SCLogger.info( "Grouping by exposure..." )

    # Group by exposure
    exposures = {}
    for which, collection in zip( ['haverefs', 'other', 'canbuild', 'nope'],
                                  [haverefs, somebodyelseswillbegood, canbuildrefs, nope] ):
        for img in collection:
            if img.exposure_id not in exposures:
                exp = Exposure.get_by_id( img.exposure_id )
                exposures[img.exposure_id] = { 'identifier': exp.origin_identifier,
                                               'filepath': exp.filepath,
                                               'haverefs': [],
                                               'other': [],
                                               'canbuild': [],
                                               'nope': [] }
            exposures[img.exposure_id][which].append( img )

    return exposures, refmaker, zpprovid



def main():
    cfg = Config.get()

    parser = argparse.ArgumentParser( 'identify_refs_to_build.py' )
    parser.add_argument( 'mjd0', type=float, help="Look at images starting on this mjd" )
    parser.add_argument( 'mjd1', type=float, help="Look at images ending on this mjd" )
    parser.add_argument( '-p', '--prov', default=cfg.value( 'pipeline.provenance_tag' ),
                         help=( "Provenance tag of images/wcses/zps to look for; defaults to config value of "
                                "pipeline.provenance_tag" ) )
    parser.add_argument( '-o', '--overlapfrac', default=cfg.value( 'subtraction.reference.minovfrac' ),
                         help=( "Fraction of area that an existing ref must overlap an image for it to be "
                                "considered a good reference for that image.  *Also* the overlap fraction "
                                "for two images to decide if they can share the same newly-built reference."
                                "Defaults to config value of subtraction.reference.minovfrac" ) )
    parser.add_argument( '-r', '--refset', default=cfg.value( "subtraction.refset" ),
                         help=( "The reset to search for existing references in.  Defaults to config value of "
                                "subtraction.refset" ) )
    parser.add_argument( '-n', '--no-make-refset',
                         default=cfg.value( "referencing.maker.ignore_config_use_config_from_refset"),
                         help=( "Don't make a new refset, only use a pre-existing one, and use the config from "
                                "that pre-existing refset even if the rest of refmaker config conflicts with it.  "
                                "Defaults to what is set "
                                "in config referencing.maker.ignore_config_use_config_from_refset" ) )
    parser.add_argument( '-w', '--write-builder-script', default=None,
                         help=( "If given, write a bash script that will run the ref maker for exposures that have "
                                "refs that can be built." ) )
    parser.add_argument( '-m', '--max-no-hope', type=int, default=99999,
                         help=( "Ignored if --write-builder-script is not given.  If not all chips can have "
                                "references made, don't includ exposures that have more than this many chips "
                                "that can't get a reference.  The default is a big enough number that it should "
                                "always build references where possible." ) )
    parser.add_argument( '-v', '--verbose', action='store_true', default=False,
                         help="Show DEBUG log messages (default: just show INFO)" )
    args = parser.parse_args()

    SCLogger.setLevel( "DEBUG" if args.verbose else "INFO" )

    exposures, refmaker, zpprovid = identify_refs_to_build( args.mjd0, args.mjd1, provtag=args.prov,
                                                            overlapfrac=args.overlapfrac, refset=args.refset,
                                                            no_make_refset=args.no_make_refset )

    exporder = list( exposures.keys() )
    exporder.sort( key=lambda x: exposures[x]['filepath'] )
    nimgs = 0
    nhaveref = 0
    ncanbuild = 0
    nother = 0
    nnope = 0

    strio = io.StringIO()
    for expdex in exporder:
        exp = exposures[expdex]
        strio.write( f"{exp['filepath']:48s} : {len(exp['haverefs']):2d} existing, "
                     f"{len(exp['canbuild']):2d} can be built, {len(exp['other']):2d} can mooch, "
                     f"{len(exp['nope']):2d} no hope.\n" )
        nimgs += len(exp['haverefs']) + len(exp['canbuild']) + len(exp['other']) + len(exp['nope'])
        nhaveref += len(exp['haverefs'])
        ncanbuild += len(exp['canbuild'])
        nother += len(exp['other'])
        nnope += len(exp['nope'])
    SCLogger.info( f"Status of exposures:\n{strio.getvalue()}" )

    SCLogger.info( textwrap.dedent(
        f"""\
        Summary: refset is {refmaker.pars.name} ({refmaker.pars.description}), prov. id {refmaker.ref_prov.id}
                 Found {nimgs} new images in {len(exposures)} exposures.
                 {nhaveref} images have pre-existing refs.
                 {ncanbuild} references identified to be built, which will serve for {ncanbuild+nother} images.
                 {nnope} can't get a ref.
        """
    ) )

    if args.write_builder_script is not None:
        with open( args.write_builder_script, 'w' ) as ofp:
            ofp.write( "#/bin/bash\n\n" )
            for expdex in exporder:
                expinfo = exposures[expdex]
                if len(expinfo['nope']) <= args.max_no_hope:
                    for img in expinfo['canbuild']:
                        ofp.write( f"python /seechange/pipeline/ref_maker.py -i {img.filepath} -f {img.filter} "
                                   f"-z {zpprovid}\n" )


# ======================================================================
if __name__ == "__main__":
    main()
