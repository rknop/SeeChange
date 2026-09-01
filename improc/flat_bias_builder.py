import re
import pathlib
import argparse
import time
import datetime
import dateutil.parser
import pytz
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import functools
import logging
import tracemalloc

import numpy as np
import psycopg.rows
from psycopg import sql

from astropy.io import fits
import astropy.time

from models.base import PsycopgConnection
from models.provenance import Provenance
from models.instrument import Instrument
from models.calibratorfile import CalibratorFile
from models.enums_and_bitflags import ImageTypeConverter
from models.exposure import Exposure
from models.image import Image
from pipeline.parameters import Parameters
from util.logger import SCLogger
from util.config import Config
import util.fits
import util.util


# THOUGHT REQUIRED
#
# Right now, the massive_stack where all images are read together is
# stored as float64.  For bias, this is definitely overkill; you do not
# need lots of extra precision to calculate a median!  For flats, it's a
# bit more subtle, because you're normalizing all the flats.  Still,
# this is almost certainly vast overkill, and we should probably just
# use float32 for everything.  Yeah, you know what, you're right.
_massive_stack_dtype = np.float32


class ParsFlatBuilder(Parameters):

    def __init__( self, **kwargs ):
        super().__init__()

        self.exposure_mode = self.add_par(
            'exposure_mode',
            True,
            bool,
            "Work on raw exposures rather than images.",
            critical=False
        )

        self.is_flat = self.add_par(
            'is_flat',
            False,
            bool,
            "True if building a flat, False if building a bias",
            critical=False
        )

        self.filter = self.add_par(
            'filter',
            None,
            ( str, type(None) ),
            "Build a flat/bias for this filter.  (Doesn't make sense with bias.)",
            critical=False
        )

        self.section_id = self.add_par(
            'section_id',
            None,
            ( str, type(None) ),
            "Build a flat/bias for this sensor section.  Don't use in exposure mode.",
            critical=False
        )

        self.numproc = self.add_par(
            'numproc',
            32,
            int,
            ( "When in exposure mode, launch this many worker processes.  Ignored in image mode.  "
              "Think about memory, that will likely limit you more than CPUs." ),
            critical=False
        )

        self.numwriteproc = self.add_par(
            'numwriteproc',
            8,
            int,
            "When in exposure mode, launch this many worker threads to write data.  Ignored in image mode.",
            critical=False
        )

        self.timeout = self.add_par(
            'timeout',
            None,
            ( float, type(None) ),
            ( "When in exposure mode, the timeout for the parent process to wait for child processes "
              "to finish.  If None, there is no timeout." ),
            critical=False
        )

        self.images = self.add_par(
            'images',
            None,
            ( list, type(None) ),
            "List of images to combine together.  Can either be full filepaths, or database ids.",
            critical=False
        )

        self.image_list_file = self.add_par(
            'image_list_file',
            None,
            ( str, type(None) ),
            "Text file with paths. or database ids, of images to combine, one per line.",
            critical=False
        )

        self.image_list_annotation = self.add_par(
            'image_list_annotation',
            None,
            ( str, type(None) ),
            ( "Annotation about how you chose the images for the list.  This is here so that if you "
              "are doing different manual things to make lists of images, they will end up having "
              "different provenances.  Probably leave this at None if you're using find_images."
             )
        )

        self.instrument = self.add_par(
            'instrument',
            None,
            ( str, type(None) ),
            "write doc",
        )

        self.find_images = self.add_par(
            'find_images',
            False,
            bool,
            ( "If True, will search the dastabase for images.  If False, then you must either give "
              "images or image_list_file" )
        )

        self.provenance_id = self.add_par(
            'find_provenance_id',
            None,
            ( str, type(None) ),
            ( "If find_images is true, find images with this provenance id.  If find_images is true, must "
              "specify exactly one of this or find_provenance_tag" )
        )
        self.add_alias( 'prov_id', 'find_provenance_id' )

        self.find_provenance_tag = self.add_par(
            'find_provenance_tag',
            None,
            ( str, type(None) ),
            ( "If find_images is true, find iamges with this provenance tag.  If find_images is true, must "
              "specify exactly one of this or find_provenance_id" )
        )
        self.add_alias( 'prov_tag', 'find_provenance_tag' )

        self.mjd = self.add_par(
            "mjd",
            None,
            ( float, type(None) ),
            ( "Search for images around this time.  Suggestion: make this about noon for the observatory. "
              "for Chile, that's something that ends in ~0.6.  (DST, whatever.)" ),
            critical=False
        )

        self.timewindow = self.add_par(
            'timewindow',
            1.0,
            float,
            "When finding images, will search at mjd0 plus or minues timewindow/2."
        )

        self.searchtype = self.add_par(
            'searchtype',
            'Bias',
            str,
            "When searching for images to combine, search for this type"
        )

        self.minexptime = self.add_par(
            'minexptime',
            0.,
            ( float, type(None) ),
            "write doc"
        )

        self.maxexptime = self.add_par(
            'maxexptime',
            0.1,
            ( float, type(None) ),
            "write doc"
        )

        self.dup_reject = self.add_par(
            'dup_reject',
            None,
            ( float, type(None) ),
            ( "When building flats, look at the RA and Dec of the image/exposure.  Make sure that there "
              "isn't more than one that are this close (decimal degrees) along both axes." )
        )

        self.calibrator_set = self.add_par(
            'calibrator_set',
            'nightly',
            str,
            ( "The calibrator_set for the CalibratorFile created.  One of unknown, externally_supplied, "
              "general, nightly, roughly_weekly, or roughly_monthly" ),
        )

        self.flat_type = self.add_par(
            'flat_type',
            None,
            ( str, type(None) ),
            ( "The flat_type for the CalibratorFile created.  One of unknown, externally_supplied, "
              "sky, twilight, or dome" )
        )

        self.combine_mode = self.add_par(
            'combine_mode',
            "median",
            str,
            "Currently must be median"
        )

        self.normalize_mode = self.add_par(
            'normalize_mode',
            None,
            ( str, type(None) ),
            "Normalize images before combining.  You want this at None for biases, median for flats"
        )

        self.bad_threshold = self.add_par(
            'bad_threshold',
            5.,
            ( float, type(None) ),
            "write doc"
        )

        self.use_instrument_mask = self.add_par(
            'use_instrument_mask',
            False,
            bool,
            ( "Use an instrument mask?  It will be used to reject pixels for medians, and ORed into "
              "the output mask.  Depending on what the instrument mask holds, you probably don't want "
              "to use this for biases, but you probably do for flats.  At least, in LS4 that's what "
              "we want." ),
            critical=True
        )

        self.nmin = self.add_par(
            'nmin',
            7,
            int,
            "write doc"
        )

        self.nmax = self.add_par(
            'nmax',
            21,
            int,
            "write doc"
        )

        self.save_to_db = self.add_par(
            'save_to_db',
            False,
            bool,
            "Should we save an Image and CalbiratorFile to the database?",
            critical=False
        )

        self.no_save_on_exception = self.add_par(
            'no_save_on_exception',
            True,
            bool,
            ( "Ignored if save_to_db is False.  If True, and in exposure mode, and there are exceptions for "
              "some chips, then don't save anything.  If False, then save the chips that worked." ),
            critical=False
        )

        self.savetype = self.add_par(
            'savetype',
            'ComBias',
            str,
            "When saving an image to the database, this is its type"
        )

        self.name_convention = self.add_par(
            'bias_name_convention',
            "{inst_name}/biases/{date}/{inst_name}_bias_{date}_{time}_{section_id}_{prov_hash:.6s}",
            str,
            "write doc",
            critical=False
        )

        self.validity_start = self.add_par(
            'validity_start',
            None,
            ( datetime.datetime, None ),
            ( "The flat or bias should be valid starting at this time.  Irrelevant if save_to_db is False.  "
              "Make both this and valdity_start_offset None to be valid from the beginning of time." ),
            critical=False
        )

        self.validity_end = self.add_par(
            'validity_end',
            None,
            ( datetime.datetime, None ),
            ( "The flat or bias should be valid until this time.  Irrelevant if save_to_db is False.  "
              "Make both this and valdity_end_offset None to valid until the end of time." ),
            critical=False
        )

        self.validity_start_offset = self.add_par(
            'validity_start_offset',
            1.0,
            ( float, type(None) ),
            ( "The time of the flat or bias is defined as the average of the earliest and latest "
              "mjd of the images combined together to make the flat or bias.  The CalibratorFile "
              "will have a validity_start that is this many days BEFORE that time.  Ignored if "
              "validity_start is not None.  Make both this and validity_start None to have no "
              "validity_start.  irrelevant if save_to_db is False." ),
            critical=False
        )

        self.validity_end_offset = self.add_par(
            'validity_end_offset',
            2.0,
            ( float, type(None) ),
            "Like validity_start_offset, but for validity_end",
            critical=False
        )

        self._enforce_no_new_attrs = True
        self.override( kwargs )


class FlatBuilder:
    """Despite the name, this can be used for  both flats and biases--  default paramters are for biases."""

    nmad_k = 1.4826

    def __init__( self, nodb=False, section_keyword=None, **kwargs ):
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

        self.nodb = nodb
        self.section_keyword = section_keyword

        cfg = Config.get()
        cfgdata = cfg.value( 'flat_bias_builder' )
        del cfgdata['flat']
        del cfgdata['bias']
        if 'is_flat' in kwargs:
            is_flat = kwargs['is_flat']
        elif 'is_flat' in cfgdata:
            is_flat = cfgdata['is_flat']
        else:
            is_flat = False
        if is_flat:
            cfgdata.update( cfg.value( 'flat_bias_builder.flat' ) )
        else:
            cfgdata.update( cfg.value( 'flat_bias_builder.bias' ) )

        self.pars = ParsFlatBuilder( **cfgdata )
        self.pars.override( kwargs )

        # Do some basic checks to make sure the stuff we've specified is implemented.
        if self.pars.normalize_mode not in [ "median", None ]:
            raise ValueError( f"Unknown normalization method {self.pars.normalize_mode}" )
        if self.pars.combine_mode not in [ "median" ]: # , "sigmaclipmedian" ]:
            raise ValueError( f"Unknown combinaton mode {self.pars.combine_mode}" )

        if self.nodb:
            if self.pars.instrument is None:
                raise ValueError( "Must give an instrument when running with nodb" )
            if self.pars.exposure_mode and ( self.section_keyword is None ):
                raise ValueError( "Must give a section_keyword when running with nodb and exposure_mode" )

        self._dumping_memory = False
        self.results = None


    def memdump( self, when ):
        if not self._dumping_memory:
            if SCLogger.getEffectiveLevel() == logging.DEBUG:
                tracemalloc.start()
                self._dumping_memory=True

        if self._dumping_memory:
            curmem, maxmem = tracemalloc.get_traced_memory()
            SCLogger.debug( f"{when}: cur: {curmem/1024/1024:.2f} MB, max: {maxmem/1024/1024:.2f} MB" )


    def find_input_files( self, i_know_what_im_doing=False ):
        if self.nodb:
            raise RuntimeError( "nodb is inconsistent with find" )
        if self.pars.instrument is None:
            raise RuntimeError( "find requires an instrument" )
        self.instrument = Instrument.get_instrument_instance( self.pars.instrument )
        self.telescope = self.instrument.telescope

        if self.pars.nmin < 5:
            if i_know_what_im_doing:
                SCLogger.warning( f"You allow {self.pars.nmin} minimum files, but you say what you're doing, so..." )
            else:
                raise RuntimeError( f"Really?  You want to build a bias or flat with only {self.pars.nmin} images?" )

        if self.pars.exposure_mode:
            if self.pars.section_id is not None:
                raise ValueError( "Can't give a find sec_id in exposure mode" )
            self.section_ids = self.instrument.get_section_ids()
            table = 'exposures'
        else:
            if self.pars.section_id is None:
                raise ValueError( "Must give a section_id when not in exposure mode" )
            self.section_id = self.pars.section_id
            table = 'images'

        with PsycopgConnection() as con:
            cursor = con.cursor( row_factory=psycopg.rows.dict_row )
            q = ( sql.SQL( "SELECT m._id, m.filepath, m.ra, m.dec FROM {table} m\n" )
                  .format( table=sql.Identifier( table ) ) )
            if self.pars.find_provenance_id is not None:
                if self.pars.find_provenance_tag is not None:
                    raise ValueError( "Only specify one of find_provenance_id or find_provenance_tag" )
                q += sql.SQL( "WHERE provenance_id={provid}\n" ).format( provid=self.pars.find_provenance_id )
                _where = "  AND"
            else:
                if self.pars.find_provenance_tag is None:
                    raise ValueError( "Must specify exactly one of find_provenance_id or find_provenance_tag" )
                q += ( sql.SQL( "INNER JOIN provenance_tags t ON m.provenance_id=t.provenance_id\n"
                                "                            AND t.tag={tag}\n" )
                       .format( tag=self.pars.find_provenance_tag ) )
                _where = "WHERE"
            if self.pars.searchtype is not None:
                imtype = ImageTypeConverter.to_int( self.pars.searchtype )
                q += sql.SQL( "{where} m._type={type}\n" ).format( where=sql.SQL(_where), type=imtype )
                _where = "  AND"
            if self.pars.section_id is not None:
                q += sql.SQL( "{where} m.section_id={secid}\n" ).format( where=sql.SQL(_where),
                                                                         secid=self.pars.section_id )
                _where = "  AND"
            if self.pars.filter is not None:
                q += sql.SQL( "{where} m.filter={filter}\n" ).format( where=sql.SQL(_where),
                                                                      filter=self.pars.filter )
                _where = "  AND"
            if self.pars.mjd is not None:
                mjd0 = float( self.pars.mjd - self.pars.timewindow/2. )
                mjd1 = float( self.pars.mjd + self.pars.timewindow/2. )
                q += sql.SQL( "{where} m.mjd>={mjd0} AND m.mjd<={mjd1}\n" ).format( where=sql.SQL(_where),
                                                                                    mjd0=mjd0, mjd1=mjd1 )
                _where = "  AND"
            if self.pars.minexptime is not None:
                q += sql.SQL( "{where} m.exp_time>={exptime}\n" ).format( where=sql.SQL(_where),
                                                                          exptime=float(self.pars.minexptime) )
                _where = "  AND"
            if self.pars.maxexptime is not None:
                q += sql.SQL( "{where} m.exp_time<={exptime}\n" ).format( where=sql.SQL(_where),
                                                                          exptime=float(self.pars.maxexptime) )
                _where = "  AND"

            if _where == "WHERE":
                raise RuntimeError( "You probably wanted to specify more search criteria." )

            if ( self.pars.nmax is not None ) and ( not self.pars.dup_reject ):
                q += sql.SQL( "ORDER BY m.mjd DESC LIMIT {nmax}\n" ).format( nmax=int(self.pars.nmax) )
            else:
                q += sql.SQL( "ORDER BY m.mjd DESC\n" )

            SCLogger.debug( f"Searching for input files with:\n{q.as_string()}" )

            cursor.execute( q )
            rows = cursor.fetchall()
            if len(rows) < self.pars.nmin:
                raise RuntimeError( f"Only found {len(rows)} {table} that matched, which is < {self.pars.nmin}" )

            if self.pars.dup_reject is not None:
                oldrows = rows
                rows = []
                # There is probably a cleverer way to do this with numpy
                for row in oldrows:
                    if any( ( np.fabs(row['ra'] - r['ra']) * np.cos(row['dec']*np.pi/180.) < self.pars.dup_reject )
                            and ( np.fabs(row['dec'] - r['dec']) < self.pars.dup_reject )
                            for r in rows
                           ):
                        continue
                    rows.append( row )
                if len(rows) < self.pars.min:
                    raise RuntimeError( f"Found {len(oldrows)} {table}, but after duplicate rejection, "
                                        f"only {len(rows)} were left, which is < {self.pars.nmin}" )
                SCLogger.info( f"Found {len(oldrows)} {table}, {len(rows)} left after ra/dec duplicate rejection." )
                if ( self.pars.nmax is not None ) and ( len(rows) > self.pars.nmax ):
                    SCLogger.info( f"Reducing to {self.pars.nmax} images as specified" )
                    rows = rows[:self.pars.nmax]

            self.files = [ r['_id'] for r in rows ]

            _nl = "\n"
            names = [ f'        {pathlib.Path(r["filepath"]).name}' for r in rows ]
            SCLogger.debug( f'Going to combine the following {len(rows)} {table}:\n{_nl.join(names)}' )

        self.results = None


    def set_input_files( self, i_know_what_im_doing=False ):
        if self.pars.images is None:
            if self.pars.image_list_file is None:
                raise ValueError( "Gotta give me files" )
            with open( self.pars.image_list_file ) as ifp:
                self.files = [ line.strip() for line in ifp ]
                spaces = re.compile( r'^\s*$' )
                self.files = [ f for f in self.files if ( not spaces.search(f) ) and ( f[0] != '#' ) ]
        else:
            self.files = util.util.listify( self.pars.images )

        if len(self.files) < self.pars.nmin:
            if i_know_what_im_doing:
                SCLogger.warning( f"Only {self.files} input files, but you say you know what you're doing so..." )
            else:
                raise RuntimeError( f"Really?  You want to build a bias or flat from only {len(self.files)} images?" )

        if self.nodb:
            missing = set()
            for f in self.files:
                if not pathlib.Path(f).is_file():
                    missing.add( f )
            if len(missing) > 0:
                raise FileNotFoundError( "Couldn't find some input files: {missing}" )

            self.instrument = None
            if self.pars.instrument is not None:
                self.instrument = Instrument.get_instrument_instance( self.pars.instrument )

            if self.pars.exposure_mode:
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
                if self.pars.exposure_mode:
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
                self.instrument = Instrument.get_instrument_instance(
                    rows[0]['instrument'] if self.pars.instrument is None else self.pars.instrument
                )
                found_instruments = set( r['instrument'] for r in rows )
                if ( len( found_instruments ) > 1 ) or ( self.instrument.name not in found_instruments ):
                    raise ValueError( f"Instrument mismatch, they weren't all {self.instrument.name}: "
                                      f"found {found_instruments}" )
                self.instrument = Instrument.get_instrument_instance( self.instrument )

                if self.pars.exposure_mode:
                    self.section_ids = self.instrument.get_section_ids()
                else:
                    self.section_id = rows[0]['section_id']
                    found_section_ids = set( r['section_id'] for r in rows )
                    if len( found_section_ids ) > 1:
                        raise ValueError( f"Images didn't all have the same section_id; found: {found_section_ids}" )

                self.files = [ r['_id'] for r in rows ]

        self.results = None


    def read_files( self, section_id=None ):
        massive_stack = None
        min_mjd = None
        max_mjd = None

        if section_id is not None:
            SCLogger.info( f"Reading {len(self.files)} images for section_id {section_id}" )
        else:
            SCLogger.info( f"Reading {len(self.files)} images" )

        if self.nodb:
            # At the very least, need to generate an all-True mask
            raise RuntimeError( "nodb is broken at the moment" )
            for i, fname in enumerate( self.files ):
                with fits.open( fname ) as hdul:
                    if self.pars.exposure_mode:
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
                        # Make the file index the first one, so that the images stay contiguous in memory
                        massive_stack = np.empty( ( len(self.files), data.shape[0], data.shape[1] ),
                                                       dtype=_massive_stack_dtype )
                    if data.shape != massive_stack.shape[1:3]:
                        raise RuntimeError( f"First image {self.files[0]} had shape {massive_stack.shape[1:3]}, "
                                            f"but {fname} has shape {data.shape}" )
                    massive_stack[ i, :, : ] = data
                    mjd = self.instrument.extract_header_info( hdu.header, [ 'mjd' ] )[0]
                    if min_mjd is None:
                        min_mjd = mjd
                        max_mjd = mjd
                    else:
                        min_mjd = min( mjd, min_mjd )
                        max_mjd = max( mjd, max_mjd )

                if ( i % 10 == 0 ) and ( i > 0 ):
                    SCLogger.debug( f"Read {i} of {len(self.files)} images" )

            # TODO : flags/mask images when not in db mode?

        else:
            timings = { 'get': 0.,
                        'fname': 0.,
                        'mjd': 0.,
                        'data': 0.,
                        'flags': 0.,
                        'del': 0.,
                        'tot': 0.
                       }
            with PsycopgConnection() as con:
                for i, objid in enumerate( self.files ):

                    if self.pars.exposure_mode:
                        expobj = Exposure.get_by_id( objid, session=con )
                        fname = expobj.filepath
                        mjd = expobj.mjd
                        data = expobj.data[ section_id ]
                        if data is None:
                            raise RuntimeError( f"Failed to find section data {section_id} in exposure {fname}" )
                        header = expobj.section_headers[ section_id ]
                        if header is None:
                            raise RuntimeError( f"Failed to find section header {section_id} in exposure {fname}" )
                        del expobj
                        data = self.instrument.overscan_and_trim( header, data )
                        flags = np.zeros( data.shape, dtype=np.int16 )
                    else:
                        t0 = time.perf_counter()
                        imgobj = Image.get_by_id( objid, session=con )
                        t1 = time.perf_counter()
                        fname = imgobj.filepath
                        t2 = time.perf_counter()
                        mjd = imgobj.mjd
                        t3 = time.perf_counter()
                        data = imgobj.data
                        t4 = time.perf_counter()
                        flags = imgobj.flags
                        t5 = time.perf_counter()
                        del imgobj
                        t6 = time.perf_counter()

                        timings['get'] += t1 - t0
                        timings['fname'] += t2 - t1
                        timings['mjd'] += t3 - t2
                        timings['data'] += t4 - t3
                        timings['flags'] += t5 - t4
                        timings['del'] += t6 - t5
                        timings['tot'] += t6 - t0

                    if massive_stack is None:
                        # I'm not using a masked array for two reasons.  First, I think if we can do
                        #  np.nanmedian, memory use will be less than doing a median on a masked array
                        #  (We shall see.)  Second, I want the data to be there even where it is masked.
                        massive_stack = np.empty( ( len(self.files), data.shape[0], data.shape[1] ),
                                                  dtype=_massive_stack_dtype )
                        massive_stack_mask = np.full( ( len(self.files), data.shape[0], data.shape[1] ), False )
                    if data.shape != massive_stack.shape[1:3]:
                        raise RuntimeError( f"First image had shape {massive_stack.shape[1:3]}, but "
                                            f"{fname} has shape {data.shape}" )
                    massive_stack[ i ] = data
                    massive_stack_mask[ i ] = ( flags != 0 )
                    if min_mjd is None:
                        min_mjd = mjd
                        max_mjd = mjd
                    else:
                        min_mjd = min( mjd, min_mjd )
                        max_mjd = max( mjd, max_mjd )

                    if ( i % 10 == 0 ) and ( i > 0 ):
                        SCLogger.debug( f"Read {i} of {len(self.files)} images" )

            if not self.pars.exposure_mode:
                SCLogger.debug( f"Read timings: {', '.join(f'{k}={v:.2f}' for k, v in timings.items())}" )

        return massive_stack, massive_stack_mask, min_mjd, max_mjd


    def calculate_flat( self, massive_stack, massive_stack_mask, section_id=None ):
        """Warning: destroys massive_stack."""

        for_section_id = '' if section_id is None else f' for {section_id}'

        instrument_mask = None
        if self.pars.use_instrument_mask:
            if section_id is None:
                raise ValueError( "Need a section_id to use an instrumnent mask." )
            instrument_mask = self.instrument.get_standard_flags_image( section_id )

        # Normalize all the images if necessary
        self.memdump( "Before normalization" )
        if self.pars.normalize_mode is not None:
            SCLogger.info( f"Normalizing all images{for_section_id}..." )
            for i in range(len(self.files)):
                if self.pars.normalize_mode == 'median':
                    # Make a copy of the image so that we can NaNify the bad pixels
                    #   without nuking the corresponding pixels in massive_stack
                    tmpim = np.array( massive_stack[ i, :, : ] )
                    tmpim[ massive_stack_mask[ i, :, : ] ] = np.nan
                    if instrument_mask is not None:
                        tmpim[ instrument_mask != 0 ] = np.nan
                    massive_stack[ i, :, : ] /= np.nanmedian( tmpim )
                    del tmpim
                else:
                    raise ValueError( f"Unknown normalization method {self.pars.normalize_mode}" )

        self.memdump( "After normalization" )

        # Combine all the images
        SCLogger.info( f"Building combined image{for_section_id}..." )
        if self.pars.combine_mode == "median":
            # OMG MAKING A COPY OF massive_stack, so much memory used!
            # Reason: I previously used masked arrays, and I think that actually used *more* memory.
            #    Maybe.  We shall see.  But, also, I need to do this in two passes: I want to
            #    nanmedian, but if things in the result have a nan, then I know they need to be
            #    masked, but I want the value to be something sane, so that flatfields won't create
            #    inf or nan spikes on images where pixels are masked.
            # NOTE: we don't use the instrument mask here, because it would be the same for every image,
            #    so there's no point.  (We aren't comparing between pixels, just the same pixel in
            #    lots of images.)
            self.memdump( "Before copy" )
            massive_tmp = massive_stack.copy()
            self.memdump( "After copy" )
            massive_tmp[ massive_stack_mask ] = np.nan
            self.combined = np.nanmedian( massive_tmp, axis=0 )
            self.memdump( "After nanmedian" )
            del massive_tmp
            self.memdump( "After del massive_tmp" )
            # Now patch in the median of all-masked pixels so we don't have a nan there
            self.combined_mask = np.isnan( self.combined )
            if np.any( self.combined_mask ):
                self.combined[ self.combined_mask ] = np.nanmedian( massive_stack[:, self.combined_mask], axis=0 )
            self.memdump( "After patch median" )
            # If there were any all-nan pixels, paste in the median of the whole image just to have something sane
            if np.any( np.isnan( self.combined ) ):
                self.combined[ np.isnan( self.combined ) ] = np.nanmedian( self.combined )
            self.memdump( "After patch of patch median" )

            # Build the nmad image ("normalized median absolute deviation")
            massive_stack -= self.combined[ np.newaxis, :, : ]
            self.memdump( "After subtracting median" )
            # ...documentation doesn't say we can do this (i.e. make out= the same as the thing
            # we're working on), but it seems like it ought to work...
            massive_stack = np.abs( massive_stack, out=massive_stack )
            # We can freely mangle massive_stack now
            massive_stack[ massive_stack_mask ] = np.nan
            self.memdump( "After abs" )
            del massive_stack_mask
            mad = np.nanmedian( massive_stack, axis=0, overwrite_input=True )
            self.memdump( "After mad" )
            self.nmad = ( self.nmad_k * mad ).astype( np.float32, copy=False )
            self.nmad_median = np.nanmedian( self.nmad )
            self.memdump( "After nmad" )
            if self.pars.bad_threshold is not None:
                threshold = self.pars.bad_threshold * self.nmad_median
                SCLogger.debug( f"Going to try to do a logcal or of self.combined_mask "
                                f"(dtype {self.combined_mask.dtype}), np.isnan(self.nmad) "
                                f"(dtype {np.isnan(self.nmad).dtype}), and self.nmad>threshold "
                                f"(dtype {(self.nmad>threshold).dtype})" )
                SCLogger.debug( f"{(self.nmad>threshold).sum()} pixels have high nmad" )
                SCLogger.debug( f"Before this OR, {self.combined_mask.sum()} pixels masked." )
                self.combined_mask = ( self.combined_mask | np.isnan( self.nmad ) | ( self.nmad > threshold ) )
                SCLogger.debug( f"After this OR, {self.combined_mask.sum()} pixels masked." )

            # Turn the combined mask into a bitfield.  Anything that has been flagged as bad in the process
            #   of building the zero/bias will have the generic "bad pixel" bit (0x01) set.  Then OR in
            #   the instrument mask
            self.combined_mask = self.combined_mask.astype( np.int16 )
            if instrument_mask is not None:
                self.combined_mask = self.combined_mask | instrument_mask

            self.memdump( "End of calculate_flat" )

        else:
            raise ValueError( f"Unknown combination mode {self.pars.comine_mode}" )

        SCLogger.info( f"...done building combined image{for_section_id}" )


    def _make_header( self, results, section_id=None ) :
        kwdict = self.instrument._get_header_keyword_translations()
        header = fits.Header()
        if self.telescope is not None:
            header[ kwdict['telescope'][0] ] = self.telescope
        if self.instrument is not None:
            header[ kwdict['instrument'][0] ] = self.instrument.name
        if section_id is not None:
            header[ kwdict['sec_id'][0] ] = section_id
        header[ kwdict['mjd'][0] ] = results['min_mjd']
        header['COMMENT'] = "Image combined with SeeChange flat_bias_builder.py"
        header['COMMENT'] = f"...normalize_mode={self.pars.normalize_mode}"
        header['COMMENT'] = f"...combine_mode={self.pars.combine_mode}"
        header['COMMENT'] = f"Median nmad: {self.nmad_median:.3g}"
        if self.pars.bad_threshold is not None:
            header['COMMENT'] = f"Masked nmad >= {self.pars.bad_threshold} x median nmad"
        header['COMMENT'] = "Files combined:"
        for f in self.files:
            header['COMMENT'] = f"  {f}"
        return header


    def _save_and_insert_image( self, imkwargs, results, parentproc=False ):
        try:
            # if not parentproc:
            #     SCLogger.multiprocessing_replace()

            # Not necessdary if multiprocessing, but necessary if multithreading
            imkwargs = imkwargs.copy()

            imkwargs.update( { 'mjd': results['min_mjd'],
                               'end_mjd': results['max_mjd'],
                               'section_id': results['section_id']
                            } )
            imgobj = Image( **imkwargs )
            imgobj.header = self._make_header( results, results['section_id'] )
            imgobj.data = results['comb']
            imgobj.flags = results['comb_mask']
            imgobj.weight = results['nmad']
            imgobj.filepath = imgobj.invent_filepath( name_convention=self.pars.name_convention )
            imgobj.save()
            imgobj.insert()

            # SCARY AND THOUGHT REQUIRED
            #   min_mjd and max_mjd should be the same for all sections in an exposure!
            #   But, there might have been failures.  Hope not.
            # Because of that, consider not using the _offset parameters.

            t = pytz.utc.localize( astropy.time.Time( results['min_mjd'], format='mjd' ).datetime )
            if self.pars.validity_start is not None:
                validity_start = dateutil.parser.parse( self.pars.validity_start )
            elif self.pars.validity_start_offset is not None:
                validity_start = t - datetime.timedelta( days=self.pars.validity_start_offset )
            else:
                validity_start = None

            if ( validity_start is not None ) and ( validity_start.tzinfo is None ):
                validity_start = pytz.utc.localize( validity_start )

            if self.pars.validity_end is not None:
                validity_end = dateutil.parser.parse( self.pars.validity_end )
            elif self.pars.validity_end_offset is not None:
                validity_end = t + datetime.timedelta( days=self.pars.validity_end_offset )
            else:
                validity_end = None

            if ( validity_end is not None ) and ( validity_end.tzinfo is None ):
                validity_end = pytz.utc.localize( validity_end )

            cf = CalibratorFile( type='flat' if self.pars.is_flat else 'zero',
                                 calibrator_set=self.pars.calibrator_set,
                                 flat_type=self.pars.flat_type,
                                 instrument=self.instrument.name,
                                 sensor_section=results['section_id'],
                                 image_id=imgobj.id,
                                 datafile_id=None,
                                 validity_start=validity_start,
                                 validity_end=validity_end
                                )
            cf.insert()
            SCLogger.debug( f"Returning True for save of {results['section_id']}" )
            return results['section_id'], True

        except Exception:
            SCLogger.exception( f"Exception trying to save {results['section_id']}" )
            SCLogger.debug( f"Returning False for save of {results['section_id']}" )
            return results['section_id'], False


    # NOTE.  We're saving the nmad as the weight, because, well, it's there.  But,
    #   this is scary, because nmad is *not* weight.
    # TODO: refactor Image so that it can have a variable number of
    def save_to_db( self ):
        prov = Provenance( process='flat_bias_builder', parameters=self.pars.get_critical_pars() )
        prov.insert_if_needed()


        imkwargs = { 'provenance_id': prov.id,
                     'exp_time': -999.,
                     'instrument': self.instrument.name,
                     'telescope': self.instrument.telescope,
                     'project': 'calibratorfile',
                     'target': 'calibratorfile',
                     'type': self.pars.savetype
                  }
        # ...I'm hoping that a position at the pole isn't going to send q3c into some expensive computation
        imkwargs['ra'] = 0.
        imkwargs['dec'] = 90.
        for mm in [ 'min', 'max' ]:
            imkwargs[ f'{mm}ra' ] = 0.
            imkwargs[ f'{mm}dec' ] = 0.
            for x in [ 0, 1 ]:
                for y in [ 0, 1 ]:
                    imkwargs[f'ra_corner_{x}{y}'] = 0.
                    imkwargs[f'dec_corner_{x}{y}'] = 0.

        if self.pars.exposure_mode:
            if self.pars.numwriteproc == 1:
                SCLogger.info( f"Writing {len(self.section_ids)} sections serially." )
                for sec_id in self.section_ids:
                    if sec_id in self.results:
                        self._save_and_insert_image( imkwargs, self.results[sec_id], parentproc=True )
                    else:
                        SCLogger.warning( f"No results for {sec_id}, not saving it." )
            else:
                SCLogger.info( f"Writing {len(self.section_ids)} sections in {self.pars.numwriteproc} processes." )
                doer = functools.partial( FlatBuilder._save_and_insert_image, self, imkwargs )
                # I think threads, not processes, are sufficient here, because these should
                #   be entirely I/O bound.
                pool = ThreadPoolExecutor( max_workers=self.pars.numwriteproc )
                missing = set( self.section_ids ) - set( self.results.keys() )
                if len(missing) > 0:
                    SCLogger.warning( f"No results for the following sections, not saving them: {missing}" )
                res = []
                for i in pool.map( doer, self.results.values(), timeout=self.pars.timeout ):
                    res.append(i)
                if not all( r[1] for r in res ):
                    failed = [ r[0] for r in res if not r[1] ]
                    SCLogger.error( f"The following sections failed to save: {failed}" )
                pool.shutdown()

        else:
            self._save_and_insert_image( imkwargs, self.results, parentproc=True )


    def write_combination( self, outfile="flat", outmask=None, nmad=None ):
        if self.results is None:
            raise RuntimeError( "Nothing calculated to write." )

        # TODO : Parallelize like save_to_db is

        def _write_file( results, section_id=None ):
            nonlocal self, outfile, outmask, nmad

            header = self._make_header( results, section_id=section_id )

            if outfile is not None:
                fname = f'{outfile}{f"_{section_id}" if section_id is not None else ""}.fits.fz'
                SCLogger.info( f"Writing combined image to to {fname}..." )
                util.fits.write_compressed_fits_fz( fname, results['comb'], header, overwrite=True )
                SCLogger.info( "...written." )

            if outmask is not None:
                fname = f'{outmask}{f"_{section_id}" if section_id is not None else ""}.fits.fz'
                SCLogger.info( f"Writing mask to {fname}..." )
                util.fits.write_compressed_fits_fz( fname, results['comb_mask'], header, overwrite=True )
                SCLogger.info( "...written." )

            if nmad is not None:
                fname = f'{nmad}{f"_{section_id}" if section_id is not None else ""}.fits.fz'
                SCLogger.info( f"Writing nmad to {fname}..." )
                util.fits.write_compressed_fits_fz( fname, results['nmad'], header, overwrite=True )
                SCLogger.info( "...written." )


        if self.pars.exposure_mode:
            for secid, results in self.results.items():
                _write_file( results, secid )

        else:
            _write_file( self.results )


    @classmethod
    def _do_the_things( cls, self, section_id ):
        try:
            SCLogger.multiprocessing_replace()
            # If we were already tracemallocing, we want to start over because we're in a new process.
            # Because we did fork, the self._dumping_memory will have been copied over
            self._dumping_memory = False
            self.memdump( "Starting traced memory" )
            massive_stack, massive_stack_mask, min_mjd, max_mjd = self.read_files( section_id )
            self.memdump( "After read_files" )
            self.calculate_flat( massive_stack, massive_stack_mask, section_id )
            self.memdump( "After calculate_flat" )
            del massive_stack
            del massive_stack_mask
            return section_id, self.combined, self.combined_mask, self.nmad, min_mjd, max_mjd, None
        except Exception as ex:
            SCLogger.exception( f"Exception in process running chip {section_id}" )
            return section_id, None, None, None, None, None, str(ex)


    def __call__( self, i_know_what_im_doing=False ):
        self.memdump( "Starting traced memory" )

        if self.pars.find_images:
            self.find_input_files( i_know_what_im_doing=i_know_what_im_doing )
        else:
            self.set_input_files( i_know_what_im_doing=i_know_what_im_doing )

        self.memdump( "After find/set_input_files" )

        if self.pars.exposure_mode:
            failed = []
            succeeded = []
            collected_results = {}

            if self.pars.numproc == 1:
                SCLogger.info( f"Doing {len(self.section_ids)} sections serially" )
                for sec in self.section_ids:
                    try:
                        massive_stack, massive_stack_mask, min_mjd, max_mjd = self.read_files( sec )
                        self.calculate_flat( massive_stack, massive_stack_mask, sec )
                        del massive_stack
                        collected_results[sec] = {
                            'section_id': sec,
                            'comb': self.combined,
                            'comb_mask': self.combined_mask,
                            'nmad': self.nmad,
                            'min_mjd': min_mjd,
                            'max_mjd': max_mjd
                        }
                        succeeded.append( sec )
                    except Exception:
                        SCLogger.exception( f"Exception working on section {sec}, skipping it." )
                        if sec in collected_results:
                            del collected_results[sec]

                        failed.append( sec )

            else:
                SCLogger.info( f"Starting work on {len(self.section_ids)} sections in {self.pars.numproc} workers..." )
                doer = functools.partial( self.__class__._do_the_things, self )
                pool = ProcessPoolExecutor( max_workers=self.pars.numproc,
                                            mp_context=multiprocessing.get_context('fork') )
                for sec, res in zip( self.section_ids,
                                     pool.map( doer, self.section_ids, timeout=self.pars.timeout ) ):
                    sec_id, comb, comb_mask, nmad, min_mjd, max_mjd, errmsg = res
                    if sec_id != sec:
                        raise RuntimeError( "This should never happen." )
                    if errmsg is not None:
                        SCLogger.error( f"Error working on secton {sec}, skipping it: {errmsg}" )
                        if sec in collected_results:
                            # ... this really shouldn't be the case...
                            del collected_results[ sec ]
                        failed.append( sec )
                    collected_results[sec_id] = {
                        'section_id': sec,
                        'comb': comb,
                        'comb_mask': comb_mask,
                        'nmad': nmad,
                        'min_mjd': min_mjd,
                        'max_mjd': max_mjd
                    }
                    succeeded.append( sec )

                pool.shutdown()

            self.memdump( "After building flat" )

            self.results = collected_results
            SCLogger.info( f"...done work on {len(self.section_ids)} sections.\n"
                           f"   ...succeded (maybe): {succeeded}\n"
                           f"   ...failed (definitely): {failed}\n"
                          )

            if self.pars.save_to_db:
                if self.pars.no_save_on_exception and ( len(failed) > 0 ):
                    SCLogger.error( "Because some chips failed, saving nothing to the database." )
                else:
                    SCLogger.info( "Saving successful chips to database..." )
                    self.save_to_db()
                    SCLogger.info( "...done saving." )

                self.memdump( "After saving to database" )

        else:
            # Image mode, no need for multiprocessing.  (That is, we could still use it, and it
            #  would be faster, but whatever.)

            massive_stack, massive_stack_mask, min_mjd, max_mjd = self.read_files()
            self.memdump( "After reading images" )
            self.calculate_flat( massive_stack, massive_stack_mask, self.section_id )
            self.memdump( "After calculating flat" )
            del massive_stack
            self.memdump( "After del massivestack" )

            self.results = { 'section_id': self.section_id,
                             'comb': self.combined,
                             'comb_mask': self.combined_mask,
                             'nmad': self.nmad,
                             'min_mjd': min_mjd,
                             'max_mjd': max_mjd }

            SCLogger.info( "Done work" )

            if self.pars.save_to_db:
                SCLogger.info( "Saving to database..." )
                self.save_to_db()
                SCLogger.info( "...done saving" )
                self.memdump( "After save_to_db" )

        SCLogger.info( "flat_bias_builder all done" )



# ======================================================================

def main():
    parser = argparse.ArgumentParser( "flatbuilder", description="Try to make a bias or flat.  Defaults in config.",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "images", nargs='*', default=argparse.SUPPRESS,
                         help="Images/exposures to combine (paths, database filepaths, or database ids)" )
    parser.add_argument( "-l", "--image-list-file", default=argparse.SUPPRESS,
                         help=( "Text file with list of images or exposures, one per line.  Must not specify any "
                                "images or exposures on the command line otherwise if you use --file-list" ) )
    parser.add_argument( "-i", "--instrument", default=argparse.SUPPRESS,
                         help="Name of the instrument.  Might not be needed, depending on what you do." )

    parser.add_argument( "--nodb", default=False, action='store_true',
                         help=( "By default, input images are all in the database and you must give database ids "
                                "or database filepaths.  With --nodb, instead just give paths to images." ) )
    parser.add_argument( "--section_keyword", default=None,
                         help=( "If using --nodb and --exposure-mode, this is the header keyword to figure out "
                                "which HDU of an exposure has the data for a given sensor section." ) )

    parser.add_argument( "-f", "--find-images", default=argparse.SUPPRESS, action='store_true',
                         help=( "Instead of specifying files, search the database for images and/or exposures." ) )
    parser.add_argument( "--prov-id", default=None, help="Find images/exposures with this provenance id" )
    parser.add_argument( "--prov-tag", default=None, help="Find images/exposures with this provenance tag" )
    parser.add_argument( "--section-id", default=argparse.SUPPRESS,
                         help=( "When finding images, find images of this secton id.  Do not use when in "
                                "exposure mode, but required when not in exposure mode." ) )
    parser.add_argument( "--searchtype", default=argparse.SUPPRESS,
                         help="Search for images of this type (usu. one of Bias, TwiFlat, DomeFlat, SkyFlat)" )
    parser.add_argument( "--filter", default=argparse.SUPPRESS,
                         help=( "When finding images/exposures, find ones with this filter.  None=any filter." ) )
    parser.add_argument( "--find-sec-id", default=argparse.SUPPRESS,
                         help=( "Section ID to search for.  Required if not in exposure mode, forbidden in in "
                                "exposure mode." ) )
    parser.add_argument( "--mjd", default=argparse.SUPPRESS, type=float,
                         help=( "The MJD of the center of the window to search.  If None, there is no limit, and "
                                "you will get the latest images known." ) )
    parser.add_argument( "--timewindow", default=argparse.SUPPRESS, type=float,
                         help="Search MJD by this much around MJD (± timewindow/2)" )
    parser.add_argument( "--minexptime", default=argparse.SUPPRESS,
                         help="Only find images with at least this exposure time" )
    parser.add_argument( "--maxexptime", default=argparse.SUPPRESS,
                         help="Only find images with at most this exposure time" )
    parser.add_argument( "--dup-reject", default=argparse.SUPPRESS, type=float,
                         help=( "When building flats, look at the RA and DEC of the image/exposure. Make sure that "
                                "there isn't mnore than one that are this close (decimal degrees) along both axes." ) )
    parser.add_argument( "--nmin", default=argparse.SUPPRESS, type=int,
                         help="Must find at least this many images or the process will fail." )
    parser.add_argument( "--nmax", default=argparse.SUPPRESS, type=int,
                         help="If there are more than this many images that match, only take the latest ones" )


    parser.add_argument( "--is-flat", action='store_true', default=argparse.SUPPRESS,
                         help=( "Build a flat (default: bias).  Though, really, this just affects how it's tagged "
                                "when saved to the database.  If you give it a non-None --normalize-mode, but say "
                                "you're building a bias, you're really probably getting something more like a flat." )
                        )
    parser.add_argument( "-e", "--exposure-mode", action='store_true', default=argparse.SUPPRESS,
                         help=( "Normally, works on images which have already been overscanned and trimmed "
                                "(and maybe more).  Use this to work on raw exposures.  Maybe want to "
                                "set numproc > 1 in that case." ) )
    parser.add_argument( "--section-keyword", default=argparse.SUPPRESS,
                         help=( "Normally, use the image record in the database to figure out the sensor section. "
                                "If this is set, instead read it from this keyword of the FITS header.  Required "
                                "in exposure mode if --nodb is set." ) )

    parser.add_argument( "-N", "--numproc", type=int, default=argparse.SUPPRESS,
                         help="Number of processes to use in exposure mode.  Ignored if --exposure-mode is not set." )
    parser.add_argument( "--numwriteproc", type=int, default=argparse.SUPPRESS,
                         help="Number of I/O proceses to write files in exposure mode." )
    parser.add_argument( "-t", "--timeout", type=float, default=argparse.SUPPRESS,
                         help=( "In exposure mode, the timeout to wait for all the processes to finish before "
                                "throwing a fit.  Make this 0 for no timeout." ) )

    parser.add_argument( "-c", "--combine-mode", default=argparse.SUPPRESS,
                         help="Mode to combine images.  Currently only median is supported" )
    parser.add_argument( "-1", "--normalize-mode", default=argparse.SUPPRESS,
                         help=( "Mode to normalize images before combining.  Do not specify this for a bias, "
                                "use \"median\" for a flat." ) )
    parser.add_argument( "-b", "--bad-threshold", default=argparse.SUPPRESS,
                         help=( "If specified then on the resultant image, any pixels whose nmad is more than "
                                "this factor times 1.4826 times the median NMAD will be masked." ) )

    parser.add_argument( "-m", "--use-instrument-mask", action='store_true', default=False,
                         help=( "Use the instrument mask?  You probably don't want "
                                "to set this for biases, but you probably do for flats." ) )

    parser.add_argument( "--save-to-db", default=argparse.SUPPRESS, action='store_true',
                         help=( "Save the image(s) created to the database as Images and create a CalibratorFile "
                                "to go with it." ) )
    parser.add_argument( "--savetype", default=argparse.SUPPRESS,
                         help="Save the combined image as this type (usually ComBias, ComTwiFlat, etc.)" )
    parser.add_argument( "-a", "--image-list-annotation", default=argparse.SUPPRESS,
                         help=( "Added to the 'image_list_annotation' parameter that goes into the provenance. "
                                "Use this to force a new provenance.  Leave it None if you don't know what "
                                "you're doing." ) )
    parser.add_argument( "--validity-start", default=argparse.SUPPRESS,
                         help=( "Flats or biases built should be tagged in the CalibratorFile entry as valid "
                                "starting at this time.  If you set neither this nor --validity-start-offset, "
                                "that entry will be left at None (meaning valid from the beginning of time).  "
                                "(Use an ISO formatted datetime string.)" ) )
    parser.add_argument( "--validity-end", default=argparse.SUPPRESS,
                         help=( "Flats or biases built should be tagged in the CalibratorFile entry as valid "
                                "ending at this time.  If you set neither this nor --validity-end-offset, "
                                "that entry will be left at None (meaning valid until the end of time).  "
                                "(Use an ISO formatted datetime string.)" ) )
    parser.add_argument( "--validity-start-offset", default=argparse.SUPPRESS, type=float,
                         help=( "In the CalibratorFile entry, tag the validity_start of this as the midpoint mjd "
                                "of the combined images MINUS this offset.  Ignored if --validity-start is given. "
                                "If neither this nor --validity-start is given, validity_start will be left unset." ) )
    parser.add_argument( "--validity-end-offset", default=argparse.SUPPRESS, type=float,
                         help="Just like --validity-start-offset, only for validity_end" )
    parser.add_argument( "--calibrator-set", default=argparse.SUPPRESS,
                         help=( "The calibrator set for the CalibratorFile entry for this image.  One of unknown, "
                                "externally_supplied, general, or nightly" ) )
    parser.add_argument( "--flat-type", default=argparse.SUPPRESS,
                         help=( "The flat type for the CalibratorFile entry for this image.  Leave unset if "
                                "this is bias image.  One of unknown, externall_supplied, sky, twilight, or dome." ) )

    parser.add_argument( "-o", "--outfile", default=None,
                         help=( "Base name of output file.  In image mode, .fits will be appended.  In exposure "
                                " mode, _<sensor_section_id>.fits will be appended.  If --save-to-db is given, "
                                "these files are written in addition to the files saved to the databse." ) )
    parser.add_argument( "--outmask", "--om", default=None, help="Filename (or base) for mask output" )
    parser.add_argument( "-n", "--nmad", default=None, help="Filename (or base) for nmad output" )

    parser.add_argument( "-v", "--verbose", default=False, action='store_true',
                         help="Set log level to debug (default info)" )
    parser.add_argument( "--i-know-what-im-doing", default=False, action='store_true', help=argparse.SUPPRESS )

    args = parser.parse_args()
    kwargs = vars( args ).copy()

    mainargs = { 'outfile': None, 'outmask': None, 'nmad': None, 'i_know_what_im_doing': False, 'verbose': False }
    for kw in mainargs.keys():
        if kw in kwargs:
            mainargs[kw] = kwargs[kw]
            del kwargs[kw]

    if mainargs['verbose']:
        SCLogger.setLevel( 'DEBUG' )
    else:
        SCLogger.setLevel( 'INFO' )

    builder = FlatBuilder( **kwargs )
    builder( i_know_what_im_doing=mainargs['i_know_what_im_doing'] )

    if any( mainargs[i] is not None for i in [ 'outfile', 'outmask', 'nmad' ] ):
        SCLogger.info( "Writing manual output files." )
        builder.write_combination( mainargs['outfile'], mainargs['outmask'], mainargs['nmad'] )
        SCLogger.info( "Done writing manual output files." )


# ======================================================================
if __name__ == "__main__":
    main()
