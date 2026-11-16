import time
import pathlib
import textwrap
import random
import shutil
import subprocess
import argparse

import numpy as np
from psycopg import sql

from astropy.io import fits
import astropy.wcs

from models.base import FileOnDiskMixin, PGDB
from models.exposure import Exposure
from models.image import Image
from models.background import Background
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

import improc.tools
import improc.sextrsky

import util.util
from util.logger import SCLogger


def secure_images( exposure, provtag=None, dataprod_provtag=None, include_defective_chips=False,
                   imageprocess="preprocessing", sourcesprocess="extraction",
                   wcsprocess="astrocal", zpprocess="photocal",
                   error_if_missing_sections=False, pgdb=None ):
    _pgdb = pgdb
    dataprod_provtag = provtag if dataprod_provtag is None else dataprod_provtag

    if not isinstance( exposure, Exposure ):
        expid = util.util.asUUID( exposure )
        exposure = Exposure.get_by_id( expid )
    if not isinstance( exposure, Exposure ):
        raise ValueError( "Failed to secure exposure for finding images." )

    instrument = exposure.instrument_object
    sections = set( instrument.get_section_ids( includebad=include_defective_chips ) )

    with PGDB( _pgdb, dictcursor=True ) as pgdb:
        q = sql.SQL( textwrap.dedent(
            """\
            SELECT i.* FROM images i
            INNER JOIN provenances p ON i.provenance_id=p._id
            INNER JOIN provenance_tags t ON t.provenance_id=p._id
            WHERE i.exposure_id={expid}
              AND t.tag={provtag}
              AND p.process={proc}
            ORDER BY i.section_id
            """
        ) )
        rows = pgdb.execute( q.format( expid=exposure.id, provtag=provtag, proc=imageprocess ) )
        images = { row['section_id']:  Image.create( **(row) ) for row in rows }
        if len(images) == 0:
            raise RuntimeError( "No images found" )
        elif len(images) != len(rows):
            raise RuntimeError( f"Some sections multiply defined; this should not happen! "
                                f"sections found: {[row['section_id'] for row in rows]}" )

        secsfound = set( images.keys() )
        if not sections.issubset( secsfound ):
            if error_if_missing_sections:
                raise RuntimeError( f"Failed to find images for sections: {sections-secsfound}" )
        if not secsfound.issubset( sections ):
            raise RuntimeError( f"Found images with unknown sections: {secsfound-sections}" )

        if ( dataprod_provtag == provtag ):
            otherprod_images = images
        else:
            rows = pgdb.execute( q.format( expid=exposure.id, provtag=dataprod_provtag, proc=imageprocess ) )
            otherprod_images = { row['section_id']: Image.create( **(row) ) for row in rows }

        otherproducts = { secid: img.get_sources_wcs_zp( provtag=dataprod_provtag, pgdb=pgdb,
                                                         processes={ 'sources': sourcesprocess,
                                                                     'wcs': wcsprocess,
                                                                     'zp': zpprocess } )
                          for secid, img in otherprod_images.items() }
        otherproducts.update( { secid: { 'wcs': None, 'zp': None, 'bg': None }
                                for secid in secsfound if secid not in otherprod_images.keys() } )

        missings = {}
        for secid, prods in otherproducts.items():
            if any( prods[x] is None for x in ( 'bg', 'wcs', 'zp' ) ):
                missings[secid] = [ x for x in ( 'bg', 'wcs', 'zp' ) if prods[x] is None ]

        if len(missings) > 0:
            if include_defective_chips:
                missinggoods = { k: v for k, v in missings.items()
                                 if not instrument.get_section( images[k].section_id ).defective }
                if len( missinggoods) > 0:
                    nl = "\n  "
                    raise RuntimeError( f"Failed to find downstream products for some non-defective sections:{nl}"
                                        f"{nl.join(f'{k}: {v}' for k, v in missinggoods.items())}" )

                SCLogger.info( f"Going to try to guess a bg, wcs, and zp for defective chips "
                               f"{list(missings.keys())}" )
                for secid in missings.keys():
                    badimg = images[ secid ]
                    others = otherproducts[ secid ]

                    # For zeropoint, just take the median of all the good images with the same filter
                    # (Whateves.)
                    usezps = np.array( [ otherproducts[i]['zp'].zp for i in secsfound
                                         if ( ( images[i].filter == badimg.filter ) and
                                              ( otherproducts[i]['zp'] is not None ) ) ] )
                    others['zp'] = ZeroPoint( zp=np.median( usezps ) )

                    # For WCS, find the closest chip, steal its WCS, offseting CRVAL accordingly.
                    # This is of course just a guess, but whatevs.
                    myoffra, myoffdec = instrument.get_section_radec_offset( secid )
                    closesec = None
                    dra = None
                    ddec = None
                    mindist = 1e32
                    for othersecid in images.keys():
                        if othersecid in missings.keys():
                            continue
                        offra, offdec = instrument.get_section_radec_offset( othersecid )
                        wcs = otherproducts[ othersecid ]['wcs'].wcs
                        _ra, dec = wcs.pixel_to_world_values( images[othersecid].width / 2.,
                                                              images[othersecid].height / 2. )
                        thisdra = myoffra - offra
                        thisddec = myoffdec - offdec
                        dist = np.sqrt( thisdra**2 + thisddec**2 )
                        if dist < mindist:
                            closesec = othersecid
                            mindist = dist
                            dra = thisdra
                            ddec = thisddec

                    wcshdr = otherproducts[closesec]['wcs'].wcs.to_header( relax=True )
                    wcshdr['CRVAL1'] += dra / np.cos( dec * np.pi/180. )
                    wcshdr['CRVAL2'] += ddec
                    others['wcs'] = WorldCoordinates( wcs = astropy.wcs.WCS( header=wcshdr ), setradec=True )
                    others['wcs'].set_corners_from_wcs( width=badimg.width, height=badimg.height )

                    # For background, just run sextrsky and say hell with it
                    sky, skysig = improc.sextrsky.sextrsky( images[secid].data )
                    others['bg'] = Background( format='map', image_shape=images[secid].data.shape )
                    others['bg'].counts = sky

                    # Fill in the weight if we don't have it
                    if badimg.weight is None:
                        badimg.weight = np.full_like( 1 / skysig**2, badimg.data, dtype=np.float32 )

            else:
                nl = "\n  "
                raise RuntimeError( f"Failed to find downstream products for some sections:{nl}"
                                    f"{nl.join(f'{k}: {v}' for k, v in missings.items())}" )

        return images, otherproducts


def mosaic_images( images, otherproducts, pixelscale=4., mosaiczp=26., constweight=False,
                   zero_masked=False, zero_weightless=False, zero_value=0.,
                   outpath=None, outweightpath=None, clobber=False ):
    barf = "".join( random.choices( 'abcdefghijklmnopqrstuvwxyz', k=10 ) )
    coaddpath = ( pathlib.Path( outpath ) if outpath is not None
                  else FileOnDiskMixin.temp_path / f'{barf}_coadd.image.fits' )
    coaddweightpath = ( pathlib.Path( outweightpath ) if outweightpath is not None
                        else FileOnDiskMixin.temp_ppath / f'{barf}_coadd.weight.fits' )
    workdir = FileOnDiskMixin.temp_path / barf
    workdir.mkdir( exist_ok=True, parents=True )

    for nuke in ( coaddpath, coaddweightpath ):
        if nuke.exists():
            if not clobber:
                raise FileExistsError( f"Output file {nuke} exists and clobber is false." )
            elif not nuke.is_file():
                raise FileExistsError( f"Output file {nuke} exists and is not a regular file; aborting." )
            else:
                SCLogger.warning( f"Deleting existing {nuke}" )
                nuke.unlink()

    try:
        # Write out suitably modified images and weights
        for i, secid in enumerate( images.keys() ):
            img = images[ secid ]
            other = otherproducts[ secid ]
            imagedata = other['bg'].subtract_me( img.data )
            imagedata *= 10 ** ( (mosaiczp - other['zp'].zp ) / 2.5 )
            if constweight:
                weightdata = np.full_like( imagedata, 1.0, dtype=np.float32 )
            else:
                # THINK, did I do this right?  (The -2 because it's 1/variance.)
                weightdata = img.weight * ( 10 ** ( -2. * (mosaiczp - other['zp'].zp ) ) )
            if zero_masked:
                # Make sure masked pixels have 0 weight
                # If constweight is false, this is probably true anyway from the pipeline
                weightdata[ img.flags != 0 ] = 0.
            hdr = img.header.copy()
            improc.tools.strip_wcs_keywords( hdr )
            hdr.update( other['wcs'].wcs.to_header( relax=True ) )
            fits.writeto( workdir /  f'image_{i:03d}.fits', imagedata, header=hdr )
            fits.writeto( workdir / f'weight_{i:03d}.fits', weightdata, header=hdr )
            # Try to clean up memory usage
            img.free()
            del imagedata
            del weightdata

        # Figure out the target WCS and write it out
        minra = min( [ o['wcs'].minra for o in otherproducts.values() ] )
        maxra = max( [ o['wcs'].maxra for o in otherproducts.values() ] )
        mindec = min( [ o['wcs'].mindec for o in otherproducts.values() ] )
        maxdec = max( [ o['wcs'].maxdec for o in otherproducts.values() ] )
        absmindec = min( np.fabs( mindec ), np.fabs( maxdec ) )
        dra = ( maxra - minra ) * np.cos( absmindec * np.pi / 180. )
        height = int( np.ceil( ( maxdec - mindec ) * 3600. / pixelscale ) )
        width = int( np.ceil( dra * 3600. / pixelscale ) )

        targhdr = astropy.wcs.WCS(
            { 'CRPIX1': width / 2.,
              'CRPIX2': height / 2.,
              'CRVAL1': ( minra + maxra ) / 2.,
              'CRVAL2': ( mindec + maxdec ) / 2.,
              'CDELT1': -pixelscale / 3600.,
              'CDELT2': pixelscale / 3600.,
              'PC1_1': 1.,
              'PC1_2': 0.,
              'PC2_1': 0.,
              'PC2_2': 1.,
              # 'CDELT1': 1.0,
              # 'CDELT2': 1.0,
              # 'PC1_1': -pixelscale / 3600.,
              # 'PC1_2': 0.,
              # 'PC2_1': 0.,
              # 'PC2_2': pixelscale / 3600.,
              'CTYPE1': "RA---TAN",
              'CTYPE2': "DEC--TAN",
              'CUNIT1': 'deg',
              'CUNIT2': 'deg' }
        ).to_header( relax=True )
        targhdr.insert( 0, ( 'NAXIS2', height ) )
        targhdr.insert( 0, ( 'NAXIS1', width ) )
        # This is that mysterious not-well-documented swarp behavior
        targhdr.tofile( workdir / 'coadd.head' )

        # Run swarp
        ( workdir / 'resample' ).mkdir()
        ( workdir / 'vmem' ).mkdir()
        command = [ 'swarp' ]
        command.extend( [ str(workdir / f'image_{i:03d}.fits') for i in range( len(images) ) ] )
        command.extend( [ '-IMAGEOUT_NAME', str( workdir / 'coadd.fits' ),
                          '-WEIGHTOUT_NAME', str( workdir / 'coadd.weight.fits' ),
                          '-RESCALE_WEIGHTS', 'N',
                          '-SUBTRACT_BACK', 'N',
                          '-RESAMPLE_DIR', str( workdir / 'resample' ),
                          '-VMEM_DIR', str( workdir / 'vmem' ),
                          '-VMEM_MAX', '1024',
                          '-MEM_MAX', '1024',
                          '-WRITE_XML', 'N',
                          '-INTERPOLATE', 'Y',
                          '-FSCALE_KEYWORD', 'THIS_KEYWORD_WILL_NEVER_EXIST',
                          '-FSCALE_DEAFAULT', '1.0',
                          '-GAIN_KEYWORD', 'THIS_KEYWORD_WILL_NEVER_EXIST',
                          '-GAIN_DEFAULT', '0.0',
                          '-SATLEV_KEYWORD', 'THIS_KEYWORD_WILL_NEVER_EXIST',
                          '-SATLEV_DEFAULT', '1e10',
                          '-COMBINE', 'Y',
                          '-COMBINE_TYPE', 'CLIPPED',
                          '-WEIGHT_TYPE', 'MAP_WEIGHT',
                          '-WEIGHT_IMAGE', ','.join( str( workdir / f'weight_{i:03d}.fits' )
                                                     for i in range( len(images) ) )
                         ]
                       )
        SCLogger.info( f"Running swarp to mosaic together {len(images)} images..." )
        t0 = time.perf_counter()
        res = subprocess.run( command, capture_output=True, timeout=600. )
        t1 = time.perf_counter()
        SCLogger.info( f"...swarp to mosaic {len(images)} images took {t1-t0:.2f} seconds." )
        if res.returncode != 0:
            SCLogger.error( f"Swarp failed!  Returned {res.returncode}\n"
                            f"   arguments: {res.args}\n"
                            f"   stdout:\n{res.stdout.decode('utf-8')}\n"
                            f"   stderr:\n{res.stderr.decode('utf-8')}\n" )
            raise RuntimeError( "swarp failed" )

        if zero_weightless:
            with fits.open( workdir / 'coadd.fits' ) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data
            with fits.open( workdir / 'coadd.weight.fits' ) as hdul:
                weight = hdul[0].data

            data[ weight <= 0. ] = zero_value

            fits.writeto( coaddpath, data, header=hdr )
            fits.writeto( coaddweightpath, weight, header=hdr )
        else:
            shutil.copy2( workdir / 'coadd.fits', coaddpath )
            shutil.copy2( workdir / 'coadd.weight.fits', coaddweightpath )

        return coaddpath, coaddweightpath

    finally:
        # Clean up
        util.util.scorch_the_earth( workdir )


# ======================================================================

def main():
    parser = argparse.ArgumentParser( "mosaic_exposure_redimages.py" )
    parser.add_argument( "exposureid", help="UUID of exposure" )
    parser.add_argument( "-p", "--provtag", "--provenance-tag", required=True,
                         help="Provenance tag for find images" )
    parser.add_argument( "-d", "--dataprod-provtag", "--data-product-provenance-tag", default=None,
                         help="Provenance tag for data products other than images; will use PROVTAG if not given." )
    parser.add_argument( "--include-defective-chips", action='store_true', help="Include defective chips?" )
    parser.add_argument( "-i", "--image", default=None, help="Output image file" )
    parser.add_argument( "-w", "--weight", default=None, help="Ouptut weight file" )
    parser.add_argument( "-s", "--pixelscale", type=float, default=4., help="Pixel Scale in arcsec/pix (default 4)" )
    parser.add_argument( "-z", "--zp", type=float, default=26., help="Output image zeropoint (default 26.)" )
    parser.add_argument( "--const-weight", action='store_true', default=False,
                         help="Don't use image weights.  Needed if you want to show masked pixels." )
    parser.add_argument( "--zero-weightless", action='store_true', default=False,
                         help="Set all pixels with no weight in the final sum to ZERO_VALUE" )
    parser.add_argument( "--zero-value", type=float, default=0.,
                         help=( "If --zero-weightless is given, set all pixels with no weight in the final sum "
                                "to this value." ) )
    parser.add_argument( "--clobber", action='store_true', default=False,
                         help="Overwrite output image and weight files if they exist." )
    args = parser.parse_args()

    # TODO, allow passing processes, error_if_missing_sections
    images, otherproducts = secure_images( args.exposureid,
                                           provtag=args.provtag,
                                           dataprod_provtag=args.dataprod_provtag,
                                           include_defective_chips=args.include_defective_chips,
                                           error_if_missing_sections=True )
    coaddimage, coaddweight = mosaic_images( images, otherproducts, pixelscale=args.pixelscale, mosaiczp=args.zp,
                                             constweight=args.const_weight,
                                             zero_weightless=args.zero_weightless, zero_value=args.zero_value,
                                             outpath=args.image, outweightpath=args.weight, clobber=args.clobber )


# ======================================================================
if __name__ == "__main__":
    main()
