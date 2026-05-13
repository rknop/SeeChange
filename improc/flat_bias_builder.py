import pathlib
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psycopg.rows

from astropy.io import fits

from models.base import PsycopgConnection
from models.instrument import Instrument
from models.exposure import Exposure
from models.image import Image
from util.logger import SCLogger
import util.util


class FlatBuilder:
    """Despite the name, this can be used for  both flats and biases--  default paramters are for biases."""

    nmad_k = 1.4826

    def __init__( self, instrument=None, exposure_mode=False, use_masks=False, mask_file=None,
                  section_keyword=None, numproc=32, timeout=900, combine_mode="median", normalize_mode=None,
                  bad_threshold=3., nodb=False, i_know_what_im_doing=False ):
        """Make a FlatBuilder.

        The default of numproc=32 is based on the following considerations:

           * A NERSC Perlmutter CPU node has 512GB of RAM
           * A single 2048x4096 image, double precision, takes 64MiB of RAM
           * Let's say we're going to combine 99 images, conservatively
           * That means one chip takes ~6GB (a bit more) of memory to
           *    store the full stack.
           * LS4 has 32 chips
           * That's a total of ~200GB of memory
           * ...so we can do all chips at once!  W00t!

        If you're on a system with less memory, that could well be the
        limiting factor, rather than the number of CPUs, dpeending on
        how many different exposures you're combining together.

        """
        self.instrument = instrument
        self.telescope = None
        self.exposure_mode = exposure_mode
        self.use_masks = use_masks
        self.mask_file = mask_file
        self.section_keyword = section_keyword
        self.numproc = numproc
        self.timeout = None if timeout == 0. else timeout
        self.numproc = numproc
        self.combmode = combine_mode
        self.normmethod = normalize_mode
        self.bad_threshold = bad_threshold
        self.nodb = nodb
        self.i_know_what_im_doing = i_know_what_im_doing

        # Do some basic checks to make sure the stuff we've specified is implemented.
        if self.normmethod not in [ "median" ]:
            raise ValueError( f"Unknown normalization method {self.normmethod}" )
        if self.combmode not in [ "median", "sigmaclipmedian" ]:
            raise ValueError( f"Unknown combinaton mode {self.combmode}" )

        if self.nodb:
            if self.instrument is None:
                raise ValueError( "Must give an instrument when running with nodb" )
            if self.exposure_mode and ( self.section_keyword is None ):
                raise ValueError( "Must give a section_keyword when running with nodb and exposure_mode" )

        self.results = None


    def set_input_files( self, files ):
        if files is None:
            raise ValueError( "Gotta give me files" )
        self.files = util.util.listify( files )


        if len(self.files) < 5:
            if self.i_know_what_im_doing:
                SCLogger.warning( f"Only {self.files} input files, but you say you know what you're doing so..." )
            else:
                raise RuntimeError( f"Really?  You want to build a bias or flat from only {len(self.files)} images?" )

        if self.nodb:
            if self.use_masks and ( self.mask_file is None ):
                raise ValueError( "With --nodb, if use_masks is True, then mask_file is required." )
            missing = set()
            for f in self.files:
                if not pathlib.Path(f).is_file():
                    missing.add( f )
            if len(missing) > 0:
                raise FileNotFoundError( "Couldn't find some input files: {missing}" )

            if self.instrument is not None:
                self.instrument = Instrument.get_instrument_instance( self.instrument )

            if self.exposure_mode:
                self.section_ids = set()
                withoutsecid = set()
                with fits.open( self.files[0] ) as hdul:
                    for i, hdu in enumerate(hdul):
                        if self.section_keyword in hdu:
                            self.section_ids.add( hdul[ self.section_keyword ] )
                        else:
                            withoutsecid.add( i )
                if len( self.section_ids ) == 0:
                    raise RuntimeError( f"Didn't find any HDUs in the header of {self.files[0]} that had "
                                        f"the header keyword {self.section_keyword}" )
                if len( withoutsecid ) > 0:
                    SCLogger.warning( f"In {self.files[0]} (the canary), the following HDUs didn't have header "
                                      f"keyword {self.section_keyword}, so will be ignored: {withoutsecid}" )


        else:
            with PsycopgConnection() as con:
                cursor = con.cursor( row_factory=psycopg.rows.row_factory )
                if self.exposure_mode:
                    q = "SELECT _id, instrument FROM exposures WHERE _id=ANY(%(flist)s) OR filepath=ANY(%(flist)s)"
                else:
                    q = ( "SELECT filepath, instrument, telescope, section_id FROM images "
                          "WHERE _id=ANY(%(flist)s) OR filepath=ANY(%(flist)s)" )
                cursor.execute( q, { 'flist': self.files } )
                rows = cursor.fetchall()
                if len(rows) != len(self.files):
                    raise RuntimeError( f"Specified {len(self.files)} files, but found {len(rows)} "
                                        f"matches in the database." )

                self.telescope = rows[0]['telescope']
                self.instrument = rows[0]['instrument'] if self.instrument is None else self.instrument
                found_instruments = set( r['instrument'] for r in rows )
                if ( len( found_instruments ) > 1 ) or ( self.instrument not in found_instruments ):
                    raise ValueError( f"Instrument mismatch, they weren't all {self.instrument}: "
                                      f"found {found_instruments}" )
                self.instrument = Instrument.get_instrument_instance( self.instrument )

                if self.exposure_mode:
                    self.section_ids = self.instrument.get_secton_ids()
                if not self.exposure_mode:
                    self.section_id = rows[0]['section_id']
                    found_section_ids = set( r['section_id'] for r in rows )
                    if len( found_section_ids ) > 1:
                        raise ValueError( f"Images didn't all have the same section_id; found: {found_section_ids}" )

                self.files = [ r['id'] for r in rows ]

        self.results = None


    def read_files( self, section_id=None ):
        massive_stack = None

        if section_id is not None:
            SCLogger.info( f"Reading {len(self.files)} images for section_id {section_id}" )
        else:
            SCLogger.info( f"Reading {len(self.files)} images" )


        # Going to store the read images as float64 even though that's almost certainly overkill.
        # Whatever.  Memory is cheap now, yes?

        if self.nodb:
            for i, fname in enumerate( self.files ):
                with fits.open( fname ) as hdul:
                    if self.exposure_mode:
                        hdu = [ h for h in hdul if ( ( self.section_keyword in h.header )
                                                     and ( h.header[self.section_keyword] == section_id ) ) ]
                        if len(hdu) > 1:
                            raise RuntimeError( f"Found {len(hdu)} hdus with section {section_id} in {fname}!" )
                        if len(hdu) == 0:
                            raise RuntimeError( f"Could not find section {section_id} in {fname}" )
                        hdu = hdu[0]
                        data = self.instrument.overscan_and_trim( hdu.header, hdu.data )

                    else:
                        # ASSUMPTION.  If len(hdul) == 2, then this is a fpacked FITS file, and we should
                        #   look at hdu 1, otherwise look at hdu 0 and stick our heads in the sand if
                        #   len(hdul) > 1
                        hdu = hdul[1] if len(hdul) == 2 else hdul[0]
                        data = hdu.data

                    if massive_stack is None:
                        massive_stack = np.empty( ( data.shape[0], data.shape[1], len(self.files) ),
                                                       dtype=np.float64 )
                    if data.shape != massive_stack.shape[0:2]:
                        raise RuntimeError( f"First image {self.files[0]} had shape {massive_stack.shape[0:2]}, "
                                            f"but {fname} has shape {data.shape}" )
                    massive_stack[ :, :, i ] = data

        else:
            with PsycopgConnection() as con:
                for i, objid in enumerate( self.files ):

                    if self.exposure_mode:
                        expobj = Exposure.get_by_id( objid, session=con )
                        fname = expobj.filepath
                        data = expobj.data[ section_id ]
                        if data is None:
                            raise RuntimeError( f"Failed to find section data {section_id} in exposure {fname}" )
                        header = expobj.section_headers[ section_id ]
                        if header is None:
                            raise RuntimeError( f"Failed to find section header {section_id} in exposure {fname}" )
                        del expobj
                        data = self.instrument.overscan_and_trim( header, data )
                    else:
                        imgobj = Image.get_by_id( objid, session=con )
                        fname = imgobj.filepath
                        data = imgobj.data
                        del imgobj

                    if massive_stack is None:
                        massive_stack = np.empty( ( data.shape[0], data.shape[1], len(self.files) ),
                                                       dtype=np.float64 )
                    if data.shape != massive_stack.shape[0:2]:
                        raise RuntimeError( f"First image had shape {massive_stack.shape[0:2]}, but "
                                            f"{fname} has shape {data.shape}" )
                    massive_stack[ :, :, i ] = data


        # TODO READ MASKS

        return massive_stack


    def calculate_flat( self, massive_stack, section_id=None ):
        section_id = '' if section_id is None else f' for {section_id}'
        if self.normmethod is not None:
            SCLogger.info( f"Normalizing all images{section_id}..." )
            for i in range(len(self.files)):
                # TODO USE MASK
                if self.normmethod == 'median':
                    massive_stack[ :, :, i ] /= np.median( massive_stack[ :, :, i ] )
                else:
                    raise ValueError( f"Unknown normalization method {self.normmethod}" )

        # Combine all the images
        SCLogger.info( f"Building combined image{section_id}..." )
        if self.combmode == "median":
            # TODO individual masks?  THOUGHT REQUIRED
            self.combined = np.median( massive_stack, axis=2 )
            mad = np.median( np.abs( massive_stack - self.combined[ :, :, np.newaxis ] ), axis=2 )
            self.nmad = ( self.nmad_k * mad ).astype( np.float32, copy=False )
            # TODO, add to standard mask
            threshold = self.bad_threshold * self.nmad_k * np.median( self.nmad )
            self.mask = ( self.nmad > threshold ).astype( np.int16 )

        else:
            raise ValueError( f"Unknown combination mode {self.combmode}" )

        SCLogger.info( f"...done building combined image{section_id}" )


    def __call__( self, input_files=None ):
        if input_files is not None:
            self.set_input_files()

        if self.exposure_mode:

            def do_the_things( self, section_id ):
                massive_stack = self.read_files( section_id )
                self.calculate_flat( massive_stack, section_id )
                del massive_stack
                return section_id, self.combined, self.nmad, self.mask

            self.results = {}

            SCLogger.info( f"Starting work on {len(self.section_ids)} sections in {self.numproc} workers..." )
            pool = ProcessPoolExecutor( max_workers=self.numproc, max_tasks_per_child=1 )
            for res in pool.map( lambda x: do_the_things(*x),
                                 [ [ self, sec ] for sec in self.section_ids ],
                                 timeout=self.timeout
                                ):
                sec_id, comb, nmad, mask = res
                self.results[sec_id]['comb'] = comb
                self.results[sec_id]['nmad'] = nmad
                self.results[sec_id]['mask'] = mask

            SCLogger.info( f"...done work on {len(self.section_ids)} sections." )

        else:
            # Image mode, no need for multiprocessing.  (That is, we could still use it, and it
            #  would be faster, but whatever.)

            massive_stack = self.read_files()
            self.calculate_flat( massive_stack )
            del massive_stack

            self.results = { 'comb': self.combined,
                             'nmad': self.nmad,
                             'mask': self.mask }


    def write_combination( self, outfile="flat", outmask=None, nmad=None ):
        if self.results is None:
            raise RuntimeError( "Nothing calculated to write." )

        def _write_file( results, section_id=None ):
            nonlocal self, outfile, outmask, nmad

            header = fits.Header()
            if self.telescope is not None:
                header['TELESCOP'] = self.telescope
            if self.instrument is not None:
                header['INSTRUME'] = self.instrument.name
            if section_id is not None:
                header['SEC_ID'] = section_id
            header['COMMENT'] = "Image combined with SeeChange flat_bias_builder.py"
            header['COMMENT'] = f"...normmethod={self.normmethod}"
            header['COMMENT'] = f"...combmode={self.combmode}"
            # TODO MORE
            header['COMMENT'] = "Files combined:"
            for f in self.files:
                header['COMMENT'] = f"  {f}"

            fname = f'{outfile}{f"_{section_id}" if section_id is not None else ""}.fits'
            SCLogger.info( f"Writing combined image to to {fname}..." )
            fits.writeto( fname, results['comb'], header )
            SCLogger.info( "...written." )

            if outmask is not None:
                fname = f'{outmask}{f"_{section_id}" if section_id is not None else ""}.fits'
                SCLogger.info( f"Writing mask to {fname}..." )
                fits.writeto( fname, results['mask'], header )
                SCLogger.info( "...written." )

            if nmad is not None:
                fname = f'{nmad}{f"_{section_id}" if section_id is not None else ""}.fits'
                SCLogger.info( f"Writing nmad to {fname}..." )
                fits.writeto( fname, results['nmad'], header )
                SCLogger.info( "...written." )


        if self.exposure_mode:
            for secid, results in self.results.items():
                _write_file( results, secid )

        else:
            _write_file( self.results )


# ======================================================================

def main():
    parser = argparse.ArgumentParser( "flatbuilder", description="try to make a flat",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "images", nargs='*', default=[],
                         help="Images/exposures to combine (paths, database filepaths, or database ids)" )
    parser.add_argument( "-i", "--instrument", default=None,
                         help="Name of the instrument.  Might not be needed, depending on what you do." )
    parser.add_argument( "-l", "--file-list", default=None,
                         help=( "Text file with list of images or exposures, one per line.  Must not specify any "
                                "images or exposures on the command line otherwise if you use --file-list" ) )
    parser.add_argument( "-e", "--exposure-mode", action='store_true', default=False,
                         help=( "Normally, works on images which have already been overscanned and trimmed "
                                "(and maybe more).  Use this to work on raw exposures.  Maybe want to "
                                "set numproc > 1 in that case." ) )
    parser.add_argument( "-u", "--use-masks", action='store_true', default=False,
                         help=( "Use masks to reject pixels before doing things like normalization.  If image "
                                "masks are available, use those, otherwise use instrument-standard masks." ) )
    parser.add_argument( "--mask-file", default=None,
                         help=( "Manually specify a master mask file; this is the path to that file.  In exposure "
                                "mode, this is just a base path, and _<sensor_section_id>.fits will be appended; "
                                "all those files must exist." ) )
    parser.add_argument( "--section-keyword", default=None,
                         help=( "Normally, use the image record in the database to figure out the sensor section. "
                                "If this is set, instead read it from this keyword of the FITS header.  Required "
                                "in exposure mode if --nodb is set." ) )
    parser.add_argument( "-N", "--numproc", type=int, default=32,
                         help="Number of processes to use in exposure mode.  Ignored if --exposure-mode is not set." )
    parser.add_argument( "-t", "--timeout", type=float, default=900.,
                         help=( "In exposure mode, the timeout to wait for all the processes to finish before "
                                "throwing a fit.  Make this 0 for no timeout." ) )
    parser.add_argument( "-c", "--combine-mode", default="median",
                         help="Mode to combine images.  Currently only median is supported" )
    parser.add_argument( "-1", "--normalize-mode", default=None,
                         help=( "Mode to normalize images before combining.  Do not specify this for a bias, "
                                "use \"median\" for a flat." ) )
    parser.add_argument( "-b", "--bad-threshold", default=3.,
                         help=( "If specified then on the resultant image, any pixels whose nmad is more than "
                                "this factor times 1.4826 times the median NMAD will be masked." ) )
    parser.add_argument( "-o", "--outfile", default="flat",
                         help=( "Base name of output file.  In image mode, .fits will be appended.  In exposure "
                                " mode, _<sensor_section_id>.fits will be appended." ) )
    parser.add_argument( "-m", "--outmask", default=None, help="Filename (or base) for mask output" )
    parser.add_argument( "-n", "--nmad", default=None, help="Filename (or base) for nmad output" )
    parser.add_argument( "--nodb", default=False, action='store_true',
                         help=( "By default, input images are all in the database and you must give database ids "
                                "or database filepaths.  With --nodb, instead just give paths to images." ) )
    parser.add_argument( "--i-know-what-im-doing", default=False, action='store_true', help=argparse.SUPPRESS )

    args = parser.parse_args()

    builder = FlatBuilder( instrument=args.instrument,
                           exposure_mode=args.exposure_mode,
                           use_masks=args.use_masks,
                           mask_file=args.mask_file,
                           section_keyword=args.section_keyword,
                           numproc=args.numproc,
                           timeout=args.timeout,
                           combine_mode=args.combine_mode,
                           normalize_mode=args.normalize_node,
                           bad_threshold=args.bad_threshold,
                           nodb=args.nodb,
                           i_know_what_im_doing=args.i_know_what_im_doing )

    if args.file_list is not None:
        if len(args.images):
            raise ValueError( "Specify images on the command line, or a file with a list of images, not both." )
        files = []
        with open( args.file_list ) as ifp:
            for line in ifp:
                line = line.strip()
                if not ( ( line[0] == '#' ) or ( len(line) == 0 ) ):
                    files.append( line )
    elif len(args.images) == 0:
        raise ValueError( "Must give either images on the command line, or a file with a list of images." )
    else:
        files = args.images

    builder.set_input_files( files )
    builder()
    builder.write_combination( args.outfile, args.outmask, args.nmad )


# ======================================================================
if __name__ == "__main__":
    main()
