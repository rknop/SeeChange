import io
import time
import pathlib
import shutil
import random
import subprocess

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import improc.scamp

from util.exceptions import CatalogNotFoundError, SubprocessFailure, BadMatchException
from util.logger import SCLogger

from models.world_coordinates import WorldCoordinates
from models.base import FileOnDiskMixin

from pipeline.parameters import Parameters
from pipeline.data_store import DataStore
from pipeline.catalog_tools import fetch_gaia_dr3_excerpt


class ParsAstroCalibrator(Parameters):
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_match_catalog = self.add_par(
            'cross_match_catalog',
            'gaia_dr3',
            str,
            ( 'Which catalog should be used for cross matching for astrometry.  '
              'For method "astrometry.net", this must also be "astrometry.net".' ),
            critical=True
        )
        self.add_alias('catalog', 'cross_match_catalog')

        self.solution_method = self.add_par(
            'solution_method',
            'scamp',
            str,
            ( 'Method/algorithm to use to match the catalog to the image source list.  '
              'Currently suported: scamp, astrometry.net' ),
            critical=True
        )
        self.add_alias( 'method', 'solution_method' )

        self.max_catalog_mag = self.add_par(
            'max_catalog_mag',
            [22.],
            list,
            ( 'Maximum (dimmest) magnitudes to try requesting for the matching catalog (list of float).  It will '
              'try these in order until it gets a catalog excerpt with at least catalog_min_stars, '
              'and until it gets a succesful WCS solution.  (Cached catalog excerpts will be considered a match '
              'if their max mag is within 0.1 mag of the one specified here.)  Not used by all methods.' ),
            critical=True
        )
        self.add_alias( 'max_mag', 'max_catalog_mag' )

        self.mag_range_catalog = self.add_par(
            'mag_range_catalog',
            4.,
            ( float, None ),
            ( 'Range between maximum and minimum magnitudes to request for the catalog. '
              'Make this None to have no lower (bright) limit.  Not used by all methods.' ),
            critical=True
        )
        self.add_alias( 'mag_range', 'mag_range_catalog' )

        self.min_catalog_stars = self.add_par(
            'min_catalog_stars',
            50,
            int,
            'Minimum number of stars the catalog must have.  Not used by all methods.',
            critical=True
        )
        self.add_alias( 'min_stars', 'min_catalog_stars' )

        self.max_arcsec_residual = self.add_par(
            'max_arcsec_residual',
            0.15,
            float,
            ( 'Maximum residual in arcseconds for a WCS solution to be considered succesful.  The exact '
              'meaning of this depends on the method, but it should be something reasonable.' ),
            critical=True
        )
        self.add_alias( 'max_resid', 'max_arcsec_residual' )

        self.crossid_radii = self.add_par(
            'crossid_radii',
            [2.0],
            list,
            'List of initial radius in arcsec for cross-identifications to match; this is a scamp-specific parameter, '
            'passed to scamp via -CROSSID_RADIUS.  Pass the ones to try in order; the algorithm will try '
            'these (inside the mag_range_catalog loop) until it gets a successful WCS solution.  '
            'Not used by all methods.',
            critical=True
        )

        self.min_frac_matched = self.add_par(
            'min_frac_matched',
            0.1,
            float,
            ( 'At least this fraction of the smaller of (image source list length, catalog excerpt length) '
              'must have been matched between the two for a WCS solution to be considered successful.' ),
            critical=True
        )
        self.add_alias( 'min_frac', 'min_frac_matched' )

        self.min_matched_stars = self.add_par(
            'min_matched_stars',
            10,
            int,
            ( 'At least this many stars must be matched between the source list and the catalog excerpt. '
              'Set this to 0 to not use this criterion.  (Both this and min_frac_matched are checked.)' ),
            critical=True
        )
        self.add_alias( 'min_matches', 'min_matched_stars' )

        self.max_sources_to_use = self.add_par(
            'max_sources_to_use',
            2000,
            ( int, list ),
            ( 'If there are more than this many sources on the source list, crop it down this many, '
              'keeping the brightest sources.  Not used by all methods.' ),
            critical=True
        )

        self.subproc_timeout = self.add_par(
            'subproc_timeout',
            300,
            int,
            'Timeout in seconds for a subprocess to run.',
            critical=False
        )

        self.astrometry_net_bindir = self.add_par(
            'astrometry_net_bindir',
            '/astrometry.net/bin',
            str,
            'Directory where the astrometry.net binaries are.',
            critical=False
        )

        self.astrometry_net_indexdir = self.add_par(
            'astrometry_net_indexdir',
            None,
            ( str, type(None) ),
            ( 'Directory where astrometry.net should find index files.  If not given, use the '
              'astrometry.net default (which, I think, is ../data relative to astrometry_net_bindir' ),
            critical=False
        )

        # TODO : these could also be used in the scamp method!
        self.astrometry_net_exposure_radec = self.add_par(
            'astrometry_net_exposure_radec',
            False,
            bool,
            "Tell astrometry.net to look within astrometry_net_radius times the "
            "nominal instrument field radius of the supposed exposure's ra and dec.  "
            "(Normally, it starts without preconceptions (I think).)",
            critical=True
        )

        self.astrometry_net_image_radec = self.add_par(
            'astrometry_net_image_radec',
            False,
            bool,
            "Tell astrometry.net to look within astrometry_net_raidus times the "
            "nominal instrument field radius of the supposed image's ra and dec.  "
            "This takes precenece over astrometry_net_exposure_radec if both are set.  "
            "Normally, astrometry.net starts without preconception... I think.",
            critical=True
        )

        self.astrometry_net_radius = self.add_par(
            'astrometry_net_radius',
            2.0,                            # a generous default
            float,
            "Fraction of the nominal field (if astrometry_net_radec is True) or image "
            "(if astrometry_net_image_radec is True) radius to pull cata logs stars from.",
            critical=True
        )

        self._enforce_no_new_attrs = True

        self.override(kwargs)

    def get_process_name(self):
        return 'astrocal'


class AstroCalibrator:
    def __init__(self, **kwargs):
        self.pars = ParsAstroCalibrator(**kwargs)

        # this is useful for tests, where we can know if
        # the object did any work or just loaded from DB or datastore
        self.has_recalculated = False

    # ----------------------------------------------------------------------

    def _solve_wcs_scamp( self, image, sources, catexp, crossid_radius=2. ):
        """Solve for the WCS of image, updating image.header.

        If scamp does not succeed, will raise a SubprocessFailure
        exception (see utils/exceptions.py).

        Parameters
        ----------
          image: Image
            The image to solve the WCS for.  If the WCS solution
            succeeds, then the header field of the image will be
            updated with the keywords that define the new WCS.

          sources: SourceList
            Sources extracted from image

          catexp: CatalogExcerpt
            Astrometric calibration catalog excerpt that overlaps image.

          crossid_radius: float
            The radius in arcseconds for the initial scamp match (not the final solution).

        Returns
        -------
          astropy.wcs.WCS

        """

        if catexp.format != 'fitsldac':
            raise ValueError( f'_solve_wcs_scamp requires a fitsldac catalog excerpt, not {catexp.format}' )
        if sources.format != 'sextrfits':
            raise ValueError( f'_solve_wcs_scamp requires a sextrffits source list, not {sources.format}' )
        if catexp.origin != 'gaia_dr3':
            raise NotImplementedError( f"Don't know what magnitude key to choose for astrometric reference "
                                       f"{catexp.origin}; only gaia_dr3 is implemented." )

        if sources.filepath is None:
            sources.save( image=image )

        sourcefile = pathlib.Path( sources.get_fullpath() )
        catfile = pathlib.Path( catexp.get_fullpath() )

        wcs = improc.scamp.solve_wcs_scamp(
            sourcefile,
            catfile,
            crossid_radius=crossid_radius,
            max_sources_to_use=self.pars.max_sources_to_use,
            min_frac_matched=self.pars.min_frac_matched,
            min_matched=self.pars.min_matched_stars,
            max_arcsec_residual=self.pars.max_arcsec_residual,
            magkey='MAG_G', magerrkey='MAGERR_G',
            timeout=self.pars.subproc_timeout,
        )

        # Update image.header with the new wcs.  Process this
        # through astropy.wcs.WCS to make sure everything is copacetic.
        image.header.extend( wcs.to_header( relax=True ), update=True )

        return wcs

    # ----------------------------------------------------------------------

    def _run_scamp( self, ds, prov, session=None ):
        """Do the work of run for the scamp matching method."""

        image = ds.get_image( session=session )

        sources = ds.get_sources( session=session )
        if sources is None:
            raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.inputs_str}')

        success = False
        exceptions = []
        for maxmag in self.pars.max_catalog_mag:
            try:
                catexp = fetch_gaia_dr3_excerpt(
                    image=image,
                    minstars=self.pars.min_catalog_stars,
                    maxmags=maxmag,
                    magrange=self.pars.mag_range_catalog,
                    session=session,
                )
            except CatalogNotFoundError as ex:
                SCLogger.info( f"Failed to get a catalog excerpt with enough stars with maxmag {maxmag}, "
                               f"trying the next one." )
                exceptions.append(ex)
                continue

            for radius in self.pars.crossid_radii:
                try:
                    wcs = self._solve_wcs_scamp( image, sources, catexp, crossid_radius=radius )
                    success = True
                    break
                except SubprocessFailure as ex:
                    SCLogger.info( f"Scamp failed for maxmag {maxmag} and crossid_rad {radius}, "
                                   f"trying the next crossid_rad" )
                    exceptions.append(ex)
                    continue
                except BadMatchException as ex:
                    SCLogger.info( f"Scamp didn't produce a successful match for maxmag {maxmag} "
                                   f"and crossid_rad {radius}; trying the next crossid_rad" )
                    exceptions.append(ex)
                    continue

            if success:
                break
            else:
                SCLogger.info( f"Failed to solve for WCS with maxmag {maxmag}, trying the next one." )

        if not success:
            raise RuntimeError( f"_run_scamp failed to find a match. Exceptions that were raised: {exceptions}" )

        # Save these in case something outside wants to
        # probe them (e.g. tests)
        self.maxmag = maxmag
        self.crossid_radius = radius
        self.catexp = catexp

        ds.wcs = WorldCoordinates( sources_id=sources.id, provenance_id=prov.id )
        ds.wcs.wcs = wcs

    # ----------------------------------------------------------------------

    def _run_astrometry_net( self, ds, prov, session=None ):
        if self.pars.cross_match_catalog != 'astrometry.net':
            raise ValueError( f'cross_match_catalog is "{self.pars.cross_match_catalog}", but needs to be '
                              f'"astrometry.net" for method astrometry.net' )

        # OK.  With some though, this could almost certainly be made
        # more efficient.  Most importantly, we already have a source
        # catalog, so we could use that instead of feeding the image
        # into astrometry.net and making it extract it's own catalog!
        #
        # If our instrument classes have halfway decent scales, then the
        # 0.8 - 1.2 range we feed to solve-field is probably too generous,
        # but whatevs.
        #
        # I *think* a substantial fraction of the time is in I/O.  We might improve
        # things by parallelizing less, though that's hard to figure out without
        # changing the structure of top_level.py

        barf = ''.join( random.choices( "abcdefghijklmnopqrstuvwxyz", k=10 ) )
        tmpdir = pathlib.Path( FileOnDiskMixin.temp_path ) / barf
        tmpdir.mkdir( parents=True )
        tmptmpdir = tmpdir / "tmp"
        tmptmpdir.mkdir()
        image = ds.get_image( session=session )
        sources = ds.get_sources( session=session )
        if sources is None:
            raise ValueError( f'Cannot find a source list corresponding to the datastore inputs: {ds.inputs_str}' )

        # Write an xyls file for astrometry.net to eat.  (It could eat
        # an image, but then it would be running sextractor itself,
        # which is a waste of time since we've already done it.)
        #
        # astrometry.net coordinates are 1-offset, so we have to convert
        xyls = Table( { 'XIMAGE': sources.x + 1.,
                        'YIMAGE': sources.y + 1.,
                        'FLUX': sources.psffluxadu()[0] } )
        inputpath = tmpdir / 'input_xyls.fits'
        xyls.write( inputpath, format='fits' )

        try:
            SCLogger.debug( f"Starting astrometry.net on {ds.image.filepath}" )
            # If I did this right, it won't write any files anywhere other than into tmpdir.
            # I'm *trying* to only write 'solved' and 'wcs.fits', but it doesn't seem that
            # --axy none stops it from writing hte axy file.  (It's probably useful, anyway....)
            com = [ str( pathlib.Path( self.pars.astrometry_net_bindir ) / "solve-field" ),
                    '--dir', tmpdir,
                    '-m', tmptmpdir,
                    '--x-column', 'XIMAGE',
                    '--y-column', 'YIMAGE',
                    '--sort-column', 'FLUX',
                    '--width', str( image.data.shape[1] ),
                    '--height', str( image.data.shape[0] ),
                    '-p',                               # png images; -p disables them
                    '-S', tmpdir / 'solved',
                    '-W', tmpdir / 'wcs.fits',
                    # '--axy', tmpdir / 'axyls.fits',     # sources extracted
                    '-U', tmpdir / 'index_xyls.fits',   # x/y of sources from index
                    '-R', tmpdir / 'rdls.fits',         # RA/Dec of sources from index
                    '-B', tmpdir / 'corr.fits',         # stars that match between catalog and image
                    '-M', 'none',
                    '-N', 'none',
                    '-O',
                    '-L', str( 0.8 * image.instrument_object.pixel_scale ),
                    '-H', str( 1.2 * image.instrument_object.pixel_scale ),
                    '-u', 'arcsecperpix',
                    '-t', '3',
                    # '-v', '-v', '-v', '-v',             # verbose for debugging
                   ]

            if self.pars.astrometry_net_indexdir is not None:
                com.extend( [ '--index-dir', self.pars.astrometry_net_indexdir ] )

            if self.pars.astrometry_net_image_radec:
                # hmm... this will trigger a read of the data, if it hasn't been read already.
                # Should we just not worry about it, since often it will have been read earlier
                # in the pipeline?  Or should we be keeping image width and height in the database?
                imrad = ( image.instrument_object.pixel_scale / 3600. *
                          np.sqrt( image.data.shape[0]**2 + image.data.shape[1]**2 ) / 2. )
                imrad *= self.pars.astrometry_net_radius
                SCLogger.debug( f"astrometry.net starting within {imrad:.2f}° of "
                                f"({image.ra:.4f}, {image.dec:.4f})" )
                com.extend( [ '--ra', str( image.ra ),
                              '--dec', str( image.dec ),
                              '--radius', str( imrad ) ] )
            elif self.pars.astrometry_net_exposure_radec:
                if ds.exposure_id is None:
                    raise RuntimeError( "Didn't start with an exposure, don't have exposure ra/dec" )
                exprad = image.instrument_object.max_rad_degree * self.pars.astrometry_net_radius
                SCLogger.debug( f"astrometry.net starting within {exprad:.2f}° of "
                                f"({ds.exposure.ra}:.4f, {ds.exposure.dec:.4f})" )
                com.extend( [ '--ra', str( ds.exposure.ra ),
                              '--dec', str( ds.exposure.dec ),
                              '--radius', str( exprad ) ] )

            com.append( str(inputpath) )

            t0 = time.perf_counter()
            try:
                _anetcom = " ".join( f'"{str(i)}"' if ' ' in str(i) else str(i) for i in com )
                SCLogger.debug( f"Sending to subprocess.run: {_anetcom}" )
                res = subprocess.run( com, capture_output=True, timeout=self.pars.subproc_timeout )
            except Exception:
                strstr = io.StringIO()
                strstr.write( "Exception trying to subprocess.run; contents of com are:\n" )
                for val in com:
                    strstr.write( f"    {val} (type {type(val)})\n" )
                SCLogger.exception( strstr.getvalue() )
                raise
            t1 = time.perf_counter()
            SCLogger.debug( f"astrometry.net/solve-field ran in {t1-t0:.2f} seconds" )
            if res.returncode != 0:
                raise SubprocessFailure( res )
            if not ( tmpdir/'solved' ).is_file():
                raise SubprocessFailure( res, premessage="astrometry.net/solve-field worked but solved file missing" )

            # Read the diagnostic files, do some match checks
            # axyls = Table.read( tmpdir / 'axyls.fits' )
            index_xyls = Table.read( tmpdir / 'index_xyls.fits' )
            corr = Table.read( tmpdir / 'corr.fits' )

            ncat = len(index_xyls)
            # nsrc = len(axyls)
            nsrc = len(xyls)
            nmatch = len(corr)

            if ( self.pars.min_matched_stars > 0 ) and ( nmatch < self.pars.min_matched_stars ):
                raise RuntimeError( f"astrometry.net didn't get a good enough match, only matched "
                                    f"{len(corr)} stars (which is less than {self.pars.min_matched_stars})" )
            if ( nmatch < self.pars.min_frac_match * min( ncat, nsrc ) ):
                raise RuntimeError( f"astrometry.net matched {nmatch} stars out of {nsrc} sources "
                                    f"and {ncat} catalog objects, which isn't enough." )

            dx = corr['field_x'] - corr['index_x']
            dy = corr['field_y'] - corr['field_y']
            resid = np.median( np.sqrt( dx*dx + dy*dy ) )
            resid *= image.instrument_object.pixel_scale
            if resid > self.pars.max_arcsec_residual:
                raise RuntimeError( f"Median residual of astrometry.net matches is {resid:.2f}\", which is more "
                                    f"than the configured limit of {self.pars.max_arcsec_residual}\"" )

            # Extract the wcs

            with fits.open( tmpdir / 'wcs.fits' ) as wcshdu:
                astropy_wcs = WCS( wcshdu[0].header )

            # Update the image header... I'm not fully sure if that actually goes anywhere or not,
            #  but scamp does it, so I'm cargo-culting my earlier self.
            image.header.extend( astropy_wcs.to_header( relax=True ), update=True )

            ds.wcs = WorldCoordinates( sources_id=sources.id, provenance_id=prov.id )
            ds.wcs.wcs = astropy_wcs

        # except Exception as ex:
        #     import remote_pdb; remote_pdb.RemotePdb('127.0.0.1', random.randint(1000,60000)).set_trace()
        #     pass

        finally:
            # import remote_pdb; remote_pdb.RemotePdb('127.0.0.1', random.randint(1000,60000)).set_trace()
            shutil.rmtree( tmpdir )


    # ----------------------------------------------------------------------

    def run(self, *args, **kwargs):
        """Extract sources and use their positions to calculate the astrometric solution.

        Arguments are parsed by the DataStore.parse_args() method.
        Returns a DataStore object with the products of the processing.
        """
        self.has_recalculated = False

        try:
            ds = DataStore.from_args(*args, **kwargs)
            t_start = time.perf_counter()
            if ds.update_memory_usages:
                import tracemalloc
                tracemalloc.reset_peak()  # start accounting for the peak memory usage from here

            self.pars.do_warning_exception_hangup_injection_here()

            # get the provenance for this step:
            prov = ds.get_provenance('astrocal', self.pars.get_critical_pars())

            # try to find the world coordinates in memory or in the database:
            wcs = ds.get_wcs( provenance=prov )

            if wcs is None:  # must create a new WorldCoordinate object
                self.has_recalculated = True
                image = ds.get_image()
                if image.astro_cal_done:
                    SCLogger.warning(
                        f"Failed to find a wcs for image {pathlib.Path( image.filepath ).name}, "
                        f"but it has astro_cal_done=True.  (This may just be because you're doing "
                        f"a new provenance, so may not be a big deal.)"
                    )

                if self.pars.solution_method == 'scamp':
                    self._run_scamp( ds, prov )
                elif self.pars.solution_method == 'astrometry.net':
                    self._run_astrometry_net( ds, prov )
                else:
                    raise ValueError( f'Unknown solution method {self.pars.solution_method}' )

                # If an astro cal wasn't previously run on this image,
                # update the image's ra/dec and corners attributes based on this new wcs
                if not image.astro_cal_done:
                    image.set_corners_from_header_wcs(wcs=ds.wcs.wcs, setradec=True)
                    image.astro_cal_done = True

                if ds.update_runtimes:
                    ds.runtimes['astrocal'] = time.perf_counter() - t_start
                if ds.update_memory_usages:
                    import tracemalloc
                    ds.memory_usages['astrocal'] = tracemalloc.get_traced_memory()[1] / 1024 ** 2  # in MB

            # update the bitflag with the upstreams
            sources = ds.get_sources()
            if sources is None:
                raise ValueError(f'Cannot find a source list corresponding to the datastore inputs: {ds.inputs_str}')
            psf = ds.get_psf()
            if psf is None:
                raise ValueError(f'Cannot find a PSF corresponding to the datastore inputs: {ds.inputs_str}')
            bg = ds.get_background()
            if bg is None:
                raise ValueError(f'Cannot find a background corresponding to the datastore inputs: {ds.inputs_str}')

            ds.wcs._upstream_bitflag = 0
            ds.wcs._upstream_bitflag |= sources.bitflag  # includes badness from Image as well
            ds.wcs._upstream_bitflag |= psf.bitflag
            ds.wcs._upstream_bitflag |= bg.bitflag

            return ds

        except Exception as e:
            SCLogger.exception( f"Exception in Astrometor.run: {e}" )
            ds.exceptions.append( e )
            raise
