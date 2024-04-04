import pathlib
import random
import time
import io
import subprocess

import astropy.io
from astropy.wcs import WCS
from astropy import units


from models.base import FileOnDiskMixin, _logger
from util import ldac
from util.exceptions import SubprocessFailure, BadMatchException
from util import config


def solve_wcs_scamp( sources, catalog, crossid_radius=2.,
                     max_sources_to_use=2000, min_frac_matched=0.1,
                     min_matched=10, max_arcsec_residual=0.15,
                     magkey='MAG', magerrkey='MAGERR',
                     sources_regfile=None, sources_regxy=True,
                     catalog_regfile=None, catalog_regxy=False ):
    """Solve for the WCS of image with sourcelist sources, based on catalog.

    If scamp does not succeed, will raise a SubprocessFailure
    exception (see utils/exceptions.py).

    Parameters
    ----------
      sources: Path or str
        A file with a FITS LDAC table of sources from the image whose
        WCS we want.

      catalog: Path or str
        A file with a FITS LDAC table of the catalog.  Must include
        'X_WORLD', 'Y_WORLD', 'MAG', and 'MAGERR' (where the latter two
        may be overridden with the magkey and magerrkey keywords).

      crossid_radius: float, default 2.0
        The radius in arcseconds for the initial scamp match (not the final solution).

      max_sources_to_use: int, default 2000
        If specified, if the number of objects in sources is larger
        than this number, tell scamp to only use this many sources.
        for the initial match.  (This makes the initial match go faster.)

      min_frac_matched: float, default 0.1
        scamp must be able to match at least this fraction of
        min(number of sources, number of catalog objects) for the
        match to be considered good.

      min_matched: int, default 10
        scamp must be able to match at least this many objects
        for the match to be considered good.

      max_arcsec_residual: float, default 0.15
        maximum residual in arcseconds, along both RA and Dec
        (i.e. not a radial residual), for the WCS solution to be
        considered successful.

      magkey: str, default MAG
        The keyword to use for magnitudes in the catalog file.

      magerrkey: str, default MAGERR
        The keyword to use for the magnitude errors in the catalog file.

      sources_regfile: str or Path, default None
        If non-None, will write out a ds9 region file for debugging purposes.
        This has the detections from sources.  WARNING: this file will
        not be deleted automatically.

      sources_regxy: bool, default True
        If sources_regfile is non-none, write out the ds9 region file in
        image coordinates; otherwise, world coordinates.

      catalog_regfile: str or Path, default None
        If non-None, will write out a ds9 region file for debugging purposes.
        This has detections from catalog.  WARNING: this file will
        not be deleted automatically.

      catalog_refxy: bool, default False
        If catalog_regfile is non-none, write out the ds9 region file in
        image coordinates; otherwise, world coordinates.

    Returns
    -------
      astropy.wcs.WCS
        The WCS for image

    """

    sourcefile = pathlib.Path( sources )
    catfile = pathlib.Path( catalog )
    barf = ''.join( random.choices( 'abcdefghijlkmnopqrstuvwxyz', k=10 ) )
    xmlfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f'scamp_{barf}.xml'
    fulloutcatfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"scamp_full_{barf}.fits"
    mergedoutcatfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"scamp_merged_{barf}.fits"
    # Scamp will have written a file stripping .fits from sourcefile and adding .head
    # I'm sad that there's not an option to scamp to explicitly specify this output filename.
    headfile = sourcefile.parent / f"{sourcefile.name[:-5]}.head"

    sourceshdr, sources = ldac.get_table_from_ldac( sourcefile, imghdr_as_header=True )
    cathdr, cat = ldac.get_table_from_ldac( catfile, imghdr_as_header=True )

    max_nmatch = 0
    if ( max_sources_to_use is not None ) and ( len(sources) > max_sources_to_use ):
        max_nmatch = max_sources_to_use

    try:
        # Something I don't know : does scamp only use MATCH_NMAX
        # stars for the whole astrometric solution?  Back in the day
        # (we're talking ca. 1999 or 2000) when I wrote my own
        # transformation software, I had a parameter that allowed it
        # to use a subset of the list to get an initial match
        # (because that's an N² process), but then once I had that
        # initial match I used it to match all of the stars on the
        # list, and then used all of those stars on the list in the
        # solution.  I don't know if scamp works that way, or if it
        # just does the entire astrometric solution on MATCH_NMAX
        # stars.  The documentation is silent on this....
        command = [ 'scamp', sourcefile,
                    '-ASTREF_CATALOG', 'FILE',
                    '-ASTREFCAT_NAME', catfile,
                    '-MATCH', 'Y',
                    '-MATCH_NMAX', str( max_nmatch ),
                    '-SOLVE_PHOTOM', 'N',
                    '-CHECKPLOT_DEV', 'NULL',
                    '-CHECKPLOT_TYPE', 'NONE',
                    '-CHECKIMAGE_TYPE', 'NONE',
                    '-SOLVE_ASTROM', 'Y',
                    '-PROJECTION_TYPE', 'TPV',
                    '-WRITE_XML', 'Y',
                    '-XML_NAME', xmlfile,
                    '-CROSSID_RADIUS', str( crossid_radius ),
                    '-ASTREFMAG_KEY', magkey,
                    '-ASTREFMAGERR_KEY', magerrkey,
                   ]

        if ( sources_regfile is not None ) or ( catalog_regfile is not None ):
            command.extend( [ '-MERGEDOUTCAT_TYPE', 'FITS_LDAC',
                              '-MERGEDOUTCAT_NAME', mergedoutcatfile,
                              '-FULLOUTCAT_TYPE', 'FITS_LDAC',
                              '-FULLOUTCAT_NAME', fulloutcatfile ] )

        t0 = time.perf_counter()
        res = subprocess.run( command, capture_output=True )
        t1 = time.perf_counter()
        _logger.debug( f"Scamp with {len(sources)} sources and {len(cat)} catalog stars "
                       f"(with match_nmax={max_nmatch}) took {t1-t0:.2f} seconds" )
        if res.returncode != 0:
            raise SubprocessFailure( res )

        scampstat = astropy.io.votable.parse( xmlfile ).get_table_by_index( 1 )
        nmatch = scampstat.array["AstromNDets_Reference"][0]
        sig0 = scampstat.array["AstromSigma_Reference"][0][0]
        sig1 = scampstat.array["AstromSigma_Reference"][0][1]
        infostr = ( f"Scamp on {pathlib.Path(catfile).name} yielded {nmatch} matches out of "
                    f"{len(sources)} sources and {len(cat)} catalog objects, "
                    f"with position sigmas of ({sig0:.2f}\", {sig1:.2f}\")" )
        if not ( ( nmatch > min_frac_matched * min( len(sources), len(cat) ) )
                 and ( nmatch > min_matched )
                 and ( ( sig0 + sig1 ) / 2. <= max_arcsec_residual )
                ):
            infostr += ( f", which isn't good enough.\n"
                         f"Scamp command: {res.args}\n"
                         f"-------------\nScamp stderr:\n{res.stderr.decode('utf-8')}\n"
                         f"-------------\nScamp stdout:\n{res.stdout.decode('utf-8')}\n" )
            # A warning not an error in case something outside is iterating
            _logger.warning( infostr )
            raise BadMatchException( infostr )

        _logger.info( infostr )

        # Create a WCS object based on the information
        # written in the ".head" file scamp created.
        with open( headfile ) as ifp:
            hdrtext = ifp.read()
        # The FITS spec says ASCII, but Emmanuel put a non-ASCII latin-1
        # character in his comments... and astropy.io.fits.Header is
        # anal about only ASCII.  Sigh.
        hdrtext = hdrtext.replace( 'é', 'e' )
        strio = io.StringIO( hdrtext )
        hdr = astropy.io.fits.Header.fromfile( strio, sep='\n', padding=False, endcard=False )
        wcs = WCS( hdr )

        if ( sources_regfile is not None ) or ( catalog_regfile is not None ):
            # scamp didn't actually write the file we told it to, but something else (sigh)
            fulloutcatfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"scamp_full_{barf}_1.fits"
            mergedoutcatfile = pathlib.Path( FileOnDiskMixin.temp_path ) / f"scamp_merged_{barf}_1.fits"
            fullouthdr, fulloutcat = ldac.get_table_from_ldac( fulloutcatfile )
            sourcescat = fulloutcat[ fulloutcat['CATALOG_NUMBER'] == 1 ]
            catalogcat = fulloutcat[ fulloutcat['CATALOG_NUMBER'] == 0 ]

            joincat = astropy.table.join( sourcescat, catalogcat, keys=['SOURCE_NUMBER'] )
            joincat['X_IMAGE'] = joincat['X_IMAGE_1']
            joincat['Y_IMAGE'] = joincat['Y_IMAGE_1']
            joincat['ALPHA_J2000'] = joincat['ALPHA_J2000_2']
            joincat['DELTA_J2000'] = joincat['DELTA_J2000_2']

            def _matchnamer( p ):
                p = pathlib.Path( p )
                base = p.name[:-4] if p.name[-4:] == '.reg' else p.name
                return p.parent / f'{base}_matched.reg'
            matchcat_regfile = _matchnamer( catalog_regfile )
            matchsrc_regfile = _matchnamer( sources_regfile )

            for regfile, srclist, isxy, rad, color in zip(
                    [ sources_regfile, catalog_regfile, matchsrc_regfile, matchcat_regfile ],
                    [ sourcescat, catalogcat, joincat, joincat ],
                    [ sources_regxy, catalog_regxy, True, False ],
                    [ 5, 5, 6, 6 ], [ 'green', 'green', 'yellow', 'yellow' ] ):
                if regfile is not None:
                    with open( regfile, "w" ) as ofp:
                        if isxy:
                            rad /= wcs.proj_plane_pixel_scales()[0].to( units.arcsec ).value
                            for row in srclist:
                                ofp.write( f'image;circle({row["X_IMAGE"]:.2f},{row["Y_IMAGE"]:.2f},{rad}) '
                                           f'# color={color} width=2\n' )
                        else:
                            for row in srclist:
                                ofp.write( f'icrs;circle({row["ALPHA_J2000"]:.6f},{row["DELTA_J2000"]:.6f},{rad}" '
                                           f'# color={color} width=2\n' )

            import pdb; pdb.set_trace()
            pass


        return wcs

    finally:
        xmlfile.unlink( missing_ok=True )
        headfile.unlink( missing_ok=True )
        mergedoutcatfile.unlink( missing_ok=True )
        fulloutcatfile.unlink( missing_ok=True )
