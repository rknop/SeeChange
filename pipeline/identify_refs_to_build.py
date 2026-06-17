import textwrap
import argparse

from psycopg import sql

from util.config import Config
from util.logger import SCLogger
from models.base import PGDB
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint
from models.image import Image
from models.reference import Reference
from pipeline.ref_maker import RefMaker


def main():
    cfg = Config.get()

    parser = argparse.ArgumentParser( 'identify_refs_to_build.py' )
    parser.add_argument( 'mjd0', type=float, help="Look at images starting on this mjd" )
    parser.add_argument( 'mjd1', type=float, help="Look at images ending on this mjd" )
    parser.add_argument( '-p', '--prov', default=cfg.value( 'pipeline.provenance_tag' ),
                         help=( "Provenance tag of images/wcses/zps to look for; defaults to config value of "
                                "pipeline.provenance_tag" ) )
    parser.add_argument( '-o', '--overlapfrac', default=cfg.value( 'referencing.maker.overlap_fraction' ),
                         help=( "Fraction of area that an existing ref must overlap an image for it to be "
                                "considered a good reference for that image.  *Also* the overlap fraction "
                                "for two images to decide if they can share the same newly-built reference."
                                "Defaults to config value of referencing.maker.overlap_fraction" ) )
    parser.add_argument( '-r', '--refset', default=cfg.value( "subtraction.refset" ),
                         help=( "The reset to search for existing references in.  Defaults to config value of "
                                "subtraction.refset" ) )
    args = parser.parse_args()

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
        ) ).format( mjd0=args.mjd0, mjd1=args.mjd1, provtag=args.prov )
        rows = pgdb.execute( q )

        SCLogger.info( f"Considering {len(rows)} images from provenance tag {args.prov}..." )

        imgs = [ Image.get_by_id( row['imgid'], pgdb=pgdb ) for row in rows ]
        wcses = [ WorldCoordinates.get_by_id( row['wcsid'], pgdb=pgdb ) for row in rows ]
        zps = [ ZeroPoint.get_by_id( row['zpid'], pgdb=pgdb ) for row in rows ]

        for row in rows:
            wcs = WorldCoordinates.get_by_id( row['wcsid'], pgdb=pgdb )
            if not any( WorldCoordinates.get_overlap_frac( wcs, w ) > args.overlapfrac for w in wcses ):
                wcses.append( wcs )
                imgs.append( Image.get_by_id( row['imgid'], pgdb=pgdb ) )
                zps.append( ZeroPoint.get_by_id( row['zpid'], pgdb=pgdb ) )

    haverefs = []
    somebodyelseswillbegood = []
    canbuildrefs = []
    canbuildrefswcses = []
    nope = []

    for ndone, (img, wcs, zp) in enumerate( zip( imgs, wcses, zps ) ):
        if ndone % 10 == 0:
            SCLogger.debug( f"...did {ndone} of {len(imgs)}" )

        refs, refimgs = Reference.get_references( image=img, filter=img.filter, instrument=img.instrument,
                                                  refset=args.refset, overlapfrac=args.overlapfrac )
        if len(refs) > 0:
            haverefs.append( img )

        else:
            if any( WorldCoordinates.get_overlap_frac( wcs, w ) > args.overlapfrac for w in canbuildrefswcses ):
                somebodyelseswillbegood.append( img )
            else:
                refmaker = RefMaker()
                ( images, match_pos, match_count
                  ) = refmaker.choose_reference_images_to_coadd( img.id, image_zp_prov_id=zp.provenance_id,
                                                                 filter=img.filter )
                if images is not None:
                    canbuildrefs.append( img )
                    canbuildrefswcses.append( wcs )
                else:
                    nope.append( img )


    print( f"Found {len(imgs)} new images." )
    print( f"{len(haverefs)} images have pre-existing refs." )
    print( f"{len(canbuildrefs)} references identified to be built, which will serve for "
           f"{len(canbuildrefs)+len(somebodyelseswillbegood)} images." )
    print( f"{len(nope)} can't get a ref." )


# ======================================================================
if __name__ == "__main__":
    main()
