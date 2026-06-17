import textwrap
import argparse

from psycopg import sql

from models.base import PGDB
from models.image import Image
from models.world_coordinates import WorldCoordinates


def main():
    parser = argparse.ArgumentParser( 'export_new_ref_sub',
                                      description="Export WCS-laden images from a subtraction" )
    parser.add_argument( 'exposure', help='filepath, origin_ientifier, or id of exposure' )
    parser.add_argument( 'section', help='section_id of image' )
    parser.add_argument( '-o', '--outbase', default="",
                         help="base name of output; will write {outbase}new_image.fits etc.")
    parser.add_argument( '-p', '--provtag', default='20260612e_exgal', help="Provenance tag of subtraction" )
    parser.add_argument( '-w', '--weight', action='store_true', default=False, help="Write weights?" )
    parser.add_argument( '-f', '--flags', action='store_true', default=False, help="Write flags?" )
    args = parser.parse_args()

    q = sql.SQL( textwrap.dedent(
        """\
        SELECT e.origin_identifier,
               i._id AS imgid,
               w._id as imgwcsid,
               sub.subid AS subimgid,
               sub.refimid AS refimid,
               sub.refwcsid AS refwcsid
        FROM exposures e
        LEFT JOIN (
          SELECT i._id, i.section_id, i.exposure_id, i.filepath
          FROM images i
          INNER JOIN provenance_tags t ON i.provenance_id=t.provenance_id
                                      AND t.tag='20260612e_exgal'
        ) i ON i.exposure_id=e._id
        LEFT JOIN (
          SELECT s._id, s.image_id
          FROM source_lists s
          INNER JOIN provenance_tags t ON s.provenance_id=t.provenance_id
                                      AND t.tag='20260612e_exgal'
        ) s ON s.image_id=i._id
        LEFT JOIN (
          SELECT w._id, w.sources_id
          FROM world_coordinates w
          INNER JOIN provenance_tags t ON w.provenance_id=t.provenance_id
                                      AND t.tag='20260612e_exgal'
        ) w ON w.sources_id=s._id
        LEFT JOIN (
          SELECT z._id, z.wcs_id
          FROM zero_points z
          INNER JOIN provenance_tags t ON z.provenance_id=t.provenance_id
                                      AND t.tag='20260612e_exgal'
        ) z ON z.wcs_id=w._id
        LEFT JOIN (
          SELECT isc.new_zp_id AS new_zp_id, i._id AS subid, i.filepath AS subfilepath,
                 r._id AS refid, refwcs._id AS refwcsid, refim._id AS refimid, refim.filepath AS reffilepath
          FROM image_subtraction_components isc
          INNER JOIN images i ON isc.image_id=i._id
          INNER JOIN provenance_tags it ON i.provenance_id=it.provenance_id
                                       AND it.tag='20260612e_exgal'
          INNER JOIN refs r ON r._id=isc.ref_id
          INNER JOIN zero_points refzp ON r.zp_id=refzp._id
          INNER JOIN world_coordinates refwcs ON refzp.wcs_id=refwcs._id
          INNER JOIN source_lists refsrc ON refwcs.sources_id=refsrc._id
          INNER JOIN images refim ON refsrc.image_id=refim._id
        ) sub ON sub.new_zp_id=z._id
        WHERE ( e.filepath LIKE {exppath} OR e.origin_identifier LIKE {exppath} OR e._id::text={expid} )
          AND i.section_id={secid}
        """
    ) ).format( exppath=f'%%{args.exposure}%%', expid=args.exposure, secid=args.section )

    with PGDB( dictcursor=True ) as pgdb:
        rows = pgdb.execute( q )

        if len(rows) == 0:
            raise RuntimeError( "Not found." )

        if len(rows) > 1:
            raise RuntimeError( f"{len(rows)} found" )

        row = rows[0]

        new = Image.get_by_id( row['imgid'], pgdb=pgdb )
        newwcs = WorldCoordinates.get_by_id( row['imgwcsid'], pgdb=pgdb )
        ref = Image.get_by_id( row['refimid'], pgdb=pgdb )
        refwcs = WorldCoordinates.get_by_id( row['refwcsid'], pgdb=pgdb )
        sub = Image.get_by_id( row['subimgid'], pgdb=pgdb )

        newwcs.export_image( f"{args.outbase}new_image.fits", image=new, overwrite=True )
        newwcs.export_image( f"{args.outbase}sub_image.fits", image=sub, overwrite=True )
        refwcs.export_image( f"{args.outbase}ref_image.fits", image=ref, overwrite=True )
        if args.weight:
            newwcs.export_image( f"{args.outbase}new_weight.fits", image=new, which='weight', overwrite=True )
            newwcs.export_image( f"{args.outbase}sub_weight.fits", image=sub, which='weight', overwrite=True )
            refwcs.export_image( f"{args.outbase}ref_weight.fits", image=ref, which='weight', overwrite=True )
        if args.flags:
            newwcs.export_image( f"{args.outbase}new_flags.fits", image=new, which='flags', overwrite=True )
            newwcs.export_image( f"{args.outbase}sub_flags.fits", image=sub, which='flags', overwrite=True )
            refwcs.export_image( f"{args.outbase}ref_flags.fits", image=ref, which='flags', overwrite=True )


# ======================================================================
if __name__ == "__main__":
    main()
