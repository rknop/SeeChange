import re
import pathlib
import subprocess

import numpy as np

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.time
import astropy.units as u
from astropy.io import fits

from models.base import FileOnDiskMixin
from models.instrument import ( Instrument,
                                InstrumentOrientation,
                                InstrumentOriginExposures,
                                SensorSection
                               )
from util.logger import SCLogger
from util.fits import read_fits_image


class LS4Cam(Instrument):
    """Default operation mode for LS4Cam: single amp, sorta.

    It's complicated because of optimizations Kenneth made.

    LS4Cam exposures are assumed to always be raw."""

    # LS4 filenames:
    #
    #
    # Unpacked images:
    #   20260410004924sC0_00025_00.fits
    #
    #     First 14 are YYYYMMDDHHMMSS
    #
    #     s = "sky" (shutter opened) ; could be "d" for dark
    #
    #     C? = which controller; ? is 0 through 3
    #
    #     00025 is just an incrementing number
    #
    #     _00 is the chip within the controller.
    #        will be 0-7 for single-amp, 0-15 for dual-amp
    #
    # Packed images:
    #   20260409005320s_00014.fits.fz
    #
    #     Has 33 HDUs if in single-amp mode, so 1 per chip plus dataless HDU 0 (fpack)

    _file_re = re.compile( r'^(?P<filebase>(?P<datetime>\d{14})(?P<sd>[sd])(?P<C>C(?P<ctrlr>\d))?)'
                           r'_(?P<num>\d+)(?P<chipthing>_(?P<chip>\d\d))?\.fits(?P<fz>\.fz)?$' )


    def __init__( self, _save_to_call=False, **kwargs ):
        self.name = 'LS4Cam'
        self.telescope = 'ESO 1.0-m Schmidt'
        self.apperture = 1.0
        self.focal_ratio = None   # FIGURE THIS OUT
        self.square_degree_fov = 20
        self.pixel_scale = 1.0
        self.read_time = None # FIGURE THIS OUT
        self.orientation_fixed = True
        self.orientation = InstrumentOrientation.NupEleft    # VERIFY THIS
        self.read_noise  = 1.0  # FIGURE THIS OUT
        self.dark_current= 0.1  # FIGURE THIS OUT
        self.gain = 4.0         # FIGURE THIS OUT
        self.saturation_limit = 20000  # FIGURE THIS OUT
        self.non_linearity_limit = 20000   # FIGURE THIS OUT
        self.allowed_filters = [ "0" ]

        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

        # self.preprocessing_steps_available = [ 'overscan', 'bias', 'dark', 'linearity', 'flat' ]
        self.preprocessing_steps_available = [ 'overscan' ]
        self.preprocessing_steps_done = []


    @classmethod
    def get_filename_regex( cls ):
        return [ cls._file_re ]


    def get_section_ids( self ):
        """LS4 chip ids."""

        seclist = []
        for quadrant in [ 'NE', 'NW', 'SE', 'SW' ]:
            for chipinquad in [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' ]:
                seclist.append( f"{quadrant}_{chipinquad}" )
        return seclist

    def check_section_id( self, section_id ):
        """Raise an exception if section_id is not valid."""
        if not isinstance( section_id, str ):
            raise ValueError( f"The section_id must be a string.  Got {type(section_id)}." )
        if len(section_id) != 4:
            raise ValueError( f"All LS4 section_ids are length 4; got {len(section_id)}." )
        if section_id[0:2] not in [ 'NE', 'NW', 'SE', 'SW' ]:
            raise ValueError( f"section_id must start with one of NE, NW, SE, SW, not {section_id[0:2]}." )
        if section_id[2] != "_":
            raise ValueError( f"section_id[2] must be _, not {section_id[2]}." )
        if section_id[3] not in [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' ]:
            raise ValueError( f"section_id[3] must be in the range A..H, not {section_id[3]}." )

    def _make_new_section( self, section_id ):
        """Make a SensorSection for the LS4 instrument."""

        # TODO get dx and dy right
        SCLogger.warning( "_make_new_section doesn't have yet right offsets yet for LS4Cam!  FIX THIS!" )
        dx = 0
        dy = 0
        filter_array_index = self.get_section_filter_array_index( section_id )
        # TODO get defective right
        defective = False
        return SensorSection( section_id, self.name, size_x=2048, size_y=4096,
                              offset_x=dx, offset_y=dy, defective=defective,
                              filter_array_index=filter_array_index )


    def get_section_offsets( self, section_id ):
        """Find the offset for a specific section."""

        raise NotImplementedError( "Need to implement get_section_offsets for LS4." )


    def get_section_filter_array_index( self, section_id ):
        """Get the index in the filter array for this section.

        For LS4, the filters are fixed to : NE: i, NW: z, SE: g, SW: i.

        So, for *all* exposures, the filter_array should be ['i', 'z', 'g', 'i'].

        At least, so far.
        """
        secdex = { 'NE': 0,
                   'NW': 1,
                   'SE': 2,
                   'SW': 3 }
        return secdex[ section_id[0:2] ]


    def load_section_image( self, filepath, section_id ):
        self.check_section_id( section_id )
        with fits.open( filepath ) as hdul:
            for hdu in hdul:
                if ( 'CCD_LOC' in hdu.header ) and ( hdu.header['CCD_LOC'] == section_id ):
                    return hdu.data
        raise RuntimeError( f"Didn't find section {section_id} in exposure file {filepath}" )


    def read_header( self, filepath, section_id=None ):
        if isinstance( filepath, list ):
            if not all( isinstance( f, (str, pathlib.Path) ) for f in filepath ):
                raise TypeError( "If you pass a list to read_header, it must be a list of file paths." )
            filepath = filepath[0]

        if not isinstance( filepath, (str, pathlib.Path) ):
            raise TypeError( f"filepath must be a string or path. Got {type(filepath)}" )

        if section_id is None:
            # Get HDU 1, not HDU 0, becasue the global HDU 0 doesn't
            # have everything we need, but I *think* HDU 1 does.
            # TODO make sure this stays true.
            return read_fits_image( filepath, ext=1, output='header' )
        else:
            self.check_section_id( section_id )
            with fits.open( filepath ) as hdul:
                for hdu in hdul:
                    if ( 'CCD_LOC' in hdu.header ) and ( hdu.header['CCD_LOC'] == section_id ):
                        return hdu.header
            raise ValueError( f"Failed to find section {section_id} in FITS file." )


    def extract_header_info( self, header, names ):
        """Get header information from the raw header into common column names.

        The method doc in instrument.dy says that this method is not
        supposed to be overriden, but the structure of it is not
        fleixble enough; it assumes that there would only ever be a unit
        conversion.  Here, we don't have an airmass column in the header
        at all, so it needs to be calculated from muiltiple header keywords.

        """
        somenames = [ n for n in names if n != 'airmass' ]
        output_values = super().extract_header_info( header, somenames )

        if 'instrument' in output_values:
            if output_values['instrument'] != 'LS4Cam':
                raise ValueError( f"LS4Cam header parser found instrument '{output_values['instrument']}', "
                                  f"but expected 'LS4Cam'." )
            if header['amp_direction'] == 'both':
                output_values['instrument'] = 'LS4Cam_dualamp'

        if 'airmass' in names:
            loc = EarthLocation.of_site( 'La Silla Observatory (ESO)' )
            tim = astropy.time.Time( header['STARTOBS'], scale='utc', format='isot', location=loc )
            # I suppose we could be really anal and try to use the chip ra and dec, but hopefully
            #   this will be good enough.  (Plus, the header doesn't currently have the chip ra
            #   and dec....)
            # NOTE : I'm making the assumption that TELE-RA is decimal degrees, because
            #   that's what it was in a test file I looked at.  I hope this stays right!
            radec = SkyCoord( float(header['TELE-RA'])*15., float(header['TELE-DEC']), unit=u.deg )
            altaz = radec.transform_to( AltAz( obstime=tim, location=loc ) )
            output_values['airmass'] = altaz.secz.value

        # ****
        # HACK WARNING
        # We probably don't want to leave this as is, but this is here so I can proceed
        #   with images that are missing header stuff
        if ( 'project' in names ) and ( 'project' not in output_values ):
            output_values['project'] = 'unknown'
        if ( 'target' in names ) and ( 'target' not in output_values ):
            output_values['target'] = 'unknown'
        # ****

        return output_values


    _chip_offsets = None

    def get_ra_dec_for_section( self, ra, dec, section_id ):
        if self.__class__._chip_offsets is None:
            SCLogger.warning( "ra/dec offsets for LS4 cam are currently approximate, need to be measured!" )
            # Kenneth tells me 13.33 pixels is the size of the chip gap
            secgrid = [ [ 'NE_H', 'NE_G', 'NE_F', 'NE_E', 'NW_D', 'NW_D', 'NW_B', 'NW_A' ],
                        [ 'NE_D', 'NE_C', 'NE_B', 'NW_A', 'NW_H', 'NW_G', 'NW_F', 'NW_E' ],
                        [ 'SE_H', 'SE_G', 'SE_F', 'SE_E', 'SW_D', 'SW_D', 'SW_B', 'SW_A' ],
                        [ 'SE_D', 'SE_C', 'SE_B', 'SW_A', 'SW_H', 'SW_G', 'SW_F', 'SW_E' ] ]
            offsets = {}
            for ix, arr in enumerate( secgrid ):
                # - because E to the left is + in RA
                dx = - ( ( ix - 3.5 ) * ( 2048./3600. + 13.33 ) )
                for iy, chip in enumerate( arr ):
                    # - because in secgrid, I listed chips from top to bottom, not bottom to top
                    dy = - ( ( iy - 1.5 ) * ( 4096/3600. + 13.33 ) )
                    offsets[chip] = ( dx, dy )
            self.__class__._chip_offsets = offsets

        ra += self.__class__._chip_offsets[section_id][0] / np.cos( dec * np.pi / 180. )
        dec += self.__class__._chip_offsets[section_id][1]

        return ra, dec


    def get_ra_dec_corners_for_section( self, ra, dec, section_id ):
        raise NotImplementedError( "LS4Cam needs to implement get_ra_dec_corners_for_section" )


    def get_standard_flags_image( self, section_id ):
        SCLogger.warning( "get_standard_flags_image not yet implemetented for LS4cam, returning all zeros!" )
        return super().get_standard_flags_image( section_id )


    def get_gain_at_pixel( self, image, x, y, section_id=None ):
        SCLogger.warning( "get_gain_at_pixel not yet implemented for LS4Cam, returning something basic." )
        return self.gain


    def _get_header_keyword_translations( self ):
        t = dict(
            ra=['CHIP-RA'],          # TODO, figure out if this is really going to be right!
            dec=['CHIP-DEC'],
            mjd=['STARTOBS'],
            project=['PROJECT'],
            target=['TARGETID'],
            width=['NAXIS1'],
            height=['NAXIS2'],
            exp_time=['EXPTIME'],
            filter=['FILTERID'],
            instrument=['INSTRUME'],
            telescope=['TELESCOP'],
            gain=['GAIN'],
            airmass=[]
        )
        return t


    def _get_header_values_converters( self ):
        c = dict(
            mjd=lambda x: astropy.time.Time( x, scale='utc', format='isot' ).mjd
        )
        return c


    def _get_fits_hdu_index_from_section_id( self, section_id ):
        raise RuntimeError( "LS4Cam doesn't know how to get the FITS HDU index just from the section id. "
                            "You should never see this error; if you do, it means that something in "
                            "LS4Cam or LS4Cam_dualamp isn't implemented that needs to be." )



    def _get_file_index_from_section_id( self, section_id ):
        raise NotImplementedError( "_get_file_index_from_section_id doesn't make sense for LS4Cam." )


    def get_short_instrument_name( self ):
        return 'ls4'

    def get_short_filter_name( self, band ):
        # For an exposure, the filter will be None, or, in the header, '0'
        if ( band is None ) or ( band == '0' ):
            return None
        if band not in ['g', 'r', 'i']:
            raise ValueError( f"I only understand filters g, r, and i, not {band}" )
        return band


    def gaia_dr3_to_instrument_mag( self, filter, catdata ):
        raise NotImplementedError( "Need to implement gaia_dr3_to_instrument_mag for LS4Cam" )


    # def get_filter_bandpasses( self ):
    #     # TODO: verify this!  Right now we're just using the lsst values in the base class.


    def _get_default_calibrator( self, mjd, section, calibtype='dark', filter=None ):
        raise NotImplementedError( "Do." )


    # def preprocessing_calibrator_files( self, calibset, flattype, section, filter, mjd, nofetch=False ):
    #     # TODO: figure out if we have to override preprocessing_calibrator_files.  I hope not.


    def overscan_sections( self, header ):
        if header['amp_direction'].strip() != 'left':
            raise RuntimeError( f"{self.__class__.__name} assumes that readout is on the 'left' amp." )

        # Some people, when confronted with a problem, think "I know,
        #   I'll use regular expressions." Now they have two problems.
        #     --Jamie Zawinski
        extractor = re.compile( r'^\s*\[\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*,'
                                r'\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*\]\s*$' )

        # .... OK.  If you ds9 one of the raw images, the stuff towards
        # the left on the screen is the Right amp, and the stuff towards
        # the right on the screen is the Left amp.  Oh well.
        #
        # All the -1's are to go from FITS coords to numpy coords.
        # However, we omit it on the high side, so we get what
        # we'd index numpy with (where the upper index is one past
        # the last pixel).
        # (Also, remember numpy arrays are indexed [y,x].)
        mat = extractor.search( header['BIASSECL'] )
        biasl_x0 = int(mat.group('x0')) - 1
        biasl_x1 = int(mat.group('x1'))
        # biasl_y0 = int(mat.group('y0')) - 1
        # biasl_y1 = int(mat.group('y1'))
        # mat = extractor.search( header['BIASSECR'] )
        # biasr_x0 = int(mat.group('x0')) - 1
        # biasr_x1 = int(mat.group('x1'))
        # biasr_y0 = int(mat.group('y0')) - 1
        # biasr_y1 = int(mat.group('y1'))
        mat = extractor.search( header['DATASECL'] )
        datal_x0 = int(mat.group('x0')) - 1
        datal_x1 = int(mat.group('x1'))
        datal_y0 = int(mat.group('y0')) - 1
        datal_y1 = int(mat.group('y1'))
        mat = extractor.search( header['DATASECR'] )
        datar_x0 = int(mat.group('x0')) - 1
        datar_x1 = int(mat.group('x1'))
        # datar_y0 = int(mat.group('y0')) - 1
        # datar_y1 = int(mat.group('y1'))
        mat = extractor.search( header['PRESECL'] )
        # prel_x0 = int(mat.group('x0')) - 1
        prel_x1 = int(mat.group('x1'))
        # prel_y0 = int(mat.group('y0')) - 1
        # prel_y1 = int(mat.group('y1'))
        # mat = extractor.search( header['PRESECR'] )
        # prer_x0 = int(mat.group('x0')) - 1
        # prer_x1 = int(mat.group('x1'))
        # prer_y0 = int(mat.group('y0')) - 1
        # prer_y1 = int(mat.group('y1'))

        # OK, these numbers are really confusing.  I think the model is,
        #   the numbers assume dual-amp readout.  The files we get have
        #   the L values pasted to the right of the R values, but there
        #   is no R biassec because there was no R readout.  The prescan
        #   is on the left side and comes from presec L because that was
        #   the samp that was actually read out.  The L biassec needs to
        #   be bumped to the right by the size of the R data because the
        #   numbers assume that the R data is not there.  I think.
        #   ZOMG.
        # (Or maybe the left side really is the left side?)
        # (Or the dark side?  If only you knew the power....)
        biassec_x0 = biasl_x0 + ( datar_x1 - datar_x0 )
        biassec_x1 = biassec_x0 + ( biasl_x1 - biasl_x0 )
        datasec_x0 = prel_x1
        datasec_x1 = datasec_x0 + ( datal_x1 - datal_x0 ) + ( datar_x1 - datar_x0 )
        y0 = datal_y0
        y1 = datal_y1

        return [ { 'secname': "",
                   'biassec': { 'x0': biassec_x0, 'x1': biassec_x1, 'y0': y0, 'y1': y1 },
                   'datasec': { 'x0': datasec_x0, 'x1': datasec_x1, 'y0': y0, 'y1': y1 } } ]


    def overscan_trim_keywords_to_strip( self ):
        return [ 'CCDSEC', 'BIASSEC',
                 'DATASECL', 'BIASSECL', 'PRESECL',
                 'DATASECR', 'BAISSECR', 'PRESECR' ]


    def linearity_correct( self, *args, linearitydata=None ):
        raise NotImplementedError( "LS4Cam doesn't yet know how to linearity correct." )



    def acquire_origin_exposure( self, identifier, params, outdir=None ):
        """Download exposure...somehow..."""
        raise NotImplementedError( "Do." )


    def _load_exposure_from_file_or_files( self, filepath, origin_identifier=None, params=None ):
        # Have this here to avoid circular imports (instrument.py)
        from models.exposure import Exposure

        # OK, strict object-oriented design be damned.  This function
        # handles stuff for both this class (LS4Cam) and LS4Cam_dualamp,
        # and knows about both.
        isdualamp = ( self.name == 'LS4Cam_dualamp' )

        filepath = pathlib.Path( filepath )
        obs_type_map = { 'dark': 'Dark',
                         'pmskyflat': 'TwiFlat',
                         'sky': 'Sci' }

        provenance = self.get_exposure_provenance()

        # Try to identify if it's a whole bunch of files, or if it's a single file.
        # If it's a whole bunch of files, then the convention is to pass any one
        #   of the files, and we have to figure out the rest.
        # TODO : compare file parsed controller, chip, sd to what's in the header?
        filematch = self._file_re.search( filepath.name )
        filesd = None
        # filectrlr = None
        # filechip = None
        if filematch is None:
            # If we can't parse the filename, assume it's all chips packed in one file
            manyfiles = False
            isfz = ( ( len(filepath.name) >= 3 ) and ( filepath.name[-3:] == '.fz' ) )
        else:
            filebase = filematch.group( 'filebase' )
            filedatetime = filematch.group( 'datetime' )
            filesd = filematch.group( 'sd' )
            # filectrlr = filematch.group( 'ctrlr' )
            filenum = filematch.group( 'num' )
            manyfiles = ( filematch.group('C') is not None )
            isfz = ( filematch.group('fz') is not None )
            if manyfiles:
                if filematch.group('chipthing') is None:
                    raise ValueError( f'Error prasing ls4cam exposure filename "{filepath.name}": '
                                      f'filename has a C?, but doesn\'t have a chip.' )
                # filecontroller = int( filematch.group('ctrlr') )
                # filechip = int( filematch.group('chip') )
            else:
                if filematch.group('chipthing') is not None:
                    raise ValueError( f'Error prasing ls4cam exposure filename "{filepath.name}": '
                                      f'filename has a chip, but doesn\'t have C?.' )

        # ****
        # One of the example files I had ended in .fits.fz but was impervious to funpack.
        # Not sure what's wrong.
        if not manyfiles:
            raise NotImplementedError( "LS4 packed exposures are broken right now." )
        # ****

        if manyfiles:
            exposurename = f'{filebase}_{filenum}.fits'
            if origin_identifier is None:
                origin_identifier = exposurename
            exposurepath = pathlib.Path( FileOnDiskMixin.temp_path ) / exposurename
            exposurepathfz = exposurepath.parent / f"{exposurepath.name}.fz"
            try:
                # Make sure we can find all the files
                nsubchip = 16 if isdualamp else 8
                nneeded = 64 if isdualamp else 32
                files = []
                missing = []
                for ctrlr in range(0, 4):
                    for subchip in range(0, nsubchip):
                        fz = ( ".fz" if isfz else "" )
                        p = filepath.parent / f'{filedatetime}{filesd}C{ctrlr}_{filenum}_{subchip:02d}.fits{fz}'
                        if p.is_file():
                            files.append( p )
                        else:
                            missing.append( p.name )
                if len(missing) > 0:
                    raise RuntimeError( f"Tried to the {nneeded} individual files that make up the exposure that "
                                        f"goes with {filepath.name}, but some files were missing: {missing}" )
                if len(files) != nneeded:
                    raise RuntimeError( "This should never happen." )

                # Assemble all of these togther into a single exposure
                hdu0 = None
                ra = None
                dec = None
                obs_type = None
                hdus = []
                exphdrinfo = None
                for fitsfile in files:
                    with fits.open( fitsfile ) as hdul:
                        if ( isfz and ( len(hdul) != 2 ) ) or ( ( not isfz ) and ( len(hdul) != 1 ) ):
                            raise RuntimeError( f"Unexpected number of HDUs in file {fitsfile.name}: "
                                                f"expected {2 if isfz else 1} but got {len(hdul)}" )
                        hdu = hdul[1] if isfz else hdul[0]
                        hdr = hdu.header
                        if ( hdr['amp_direction'] == 'both' ) != isdualamp:
                            raise ValueError( f"Header of {fitsfile.name} has amp_direction={hdr['amp_direction']}, "
                                              f"which is not what {self.__class__.name} expects." )
                        # TODO, verify chip and controller vs. filename!!!

                        if hdu0 is None:
                            # There is code elsewhere that will interpret a string ra as h:m:s
                            # WORRY AND THINK : the header doesn't have comments saying the units
                            #   are hours or degrees.  Because in my test image, I got an airmass
                            #   of -3.5 if I interpreted it as degrees, but 1.05 if I interpreted
                            #   it as hours, I'm guessing it's hours.
                            ra = float( hdr['TELE-RA'] ) * 15.
                            dec = float( hdr['TELE-DEC'] )
                            obs_type = obs_type_map[ hdr['IMAGETYP'] ]
                            exphdrinfo = self.extract_header_info( hdr, [ 'mjd', 'exp_time', 'project',
                                                                          'target', 'airmass' ] )
                            if ( exphdrinfo['exp_time'] == 0 ) and ( obs_type == 'Dark' ):
                                obs_type = 'Bias'
                            allhdrinfo = exphdrinfo.copy()
                            allhdrinfo.update( { 'TELE-RA': ra, 'TELE-DEC': dec, 'IMAGETYP': obs_type } )
                            hdu0 = fits.PrimaryHDU( header=fits.Header(allhdrinfo) )
                            hdus.append( hdu0 )

                        hdus.append( fits.ImageHDU( data=hdu.data, header=hdu.header ) )

                # Write the exposure to temp and fpack it
                exphdul = fits.HDUList( hdus )
                exphdul.writeto( exposurepath )
                exphdul = None
                del exphdul
                _result = subprocess.run( [ "fpack", str(exposurepath) ], capture_output=True )

                # Load it and save the loaded object to the right place

                expobj = Exposure( current_file=exposurepathfz, invent_filepath=True, type=obs_type,
                                   ra=ra, dec=dec, format='fitsfz', instrument=self.name,
                                   filter=None, filter_array=['i', 'z', 'g', 'i'],
                                   provenance=provenance.id, origin_identifier=origin_identifier,
                                   header=hdu0.header, preprocc_bitflag=0, components=None, **exphdrinfo )
                expobj.save( exposurepathfz )
                expobj.insert()

                return expobj

            finally:
                # Clean up the temp files if we made them
                if exposurepath.exists():
                    exposurepath.unlink()
                if exposurepathfz.exists():
                    exposurepathfz.unlink()


    def manually_load_exposure( self, filepath, origin_identifier=None, params=None ):
        return self._load_exposure_from_file_or_files( filepath, origin_identifier=origin_identifier, params=params )


    def acquire_and_commit_origin_exposure( self, identifier, params ):
        raise NotImplementedError( "acquire_and_commit_origin_exposure needs to be implemented for LS4Cam" )


    def find_origin_exposures( self,
                               skip_exposures_in_database=True,
                               skip_known_exposures=True,
                               minmjd=None,
                               maxmjd=None,
                               filters=None,
                               containing_ra=None,
                               containing_dec=None,
                               ctr_ra=None,
                               ctr_dec=None,
                               radius=None,
                               minexptime=None,
                               projects=None
                              ):
        raise NotImplementedError( "find_origin_exposures needs to be implemented for LS4Cam" )



class LS4Cam_dualamp(LS4Cam):
    def __init__( self, **kwargs ):
        super().__init__( **kwargs )
        self.name = 'LS4Cam_dualamp'
        raise NotImplementedError( "LS4Cam_dualamp needs to be tested and finished" )
        # TODO : Gain might be different.  Readout time is definitely different.
        # raise NotImplementedError( "I think I have more to do" )


    def _mangle_header_to_single( self, hdr ):
        """Pass the *left* amp header.  Will edit the *SEC* fields to match a stiched raw image, return a new header."""

        newhdr = hdr.copy()

        arrparse = re.compile( r'^\s*\[\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*,\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*\]\s*$' )
        datamatchr = arrparse.search( newhdr['DATASECL'] )
        biasmatchr = arrparse.search( newhdr['BIASSECL'] )
        prematchr = arrparse.search( newhdr['PRESECL'] )
        if any( i is None for i in [ datamatchr, biasmatchr, prematchr ] ):
            raise RuntimeError( "Failed to parse DATASECR, BAISSECR, and/or PRESECR" )
        # The "LEFT" amp is to the right (i.e. West) of the "RIGHT" amp (images are
        #   oriented north up, east left, it seems).  So, have to edit the *L header
        #   keywords.
        for kw in [ 'DATASECL', 'BIASSECL', 'PRESECL' ]:
            secmatch = arrparse.search( newhdr[kw] )
            if secmatch is None:
                raise ValueError( f"Failed to parse image section from {newhdr[kw]} for {kw}" )
            x0 = int( secmatch.group("x0") ) + newhdr['NAXIS2']
            x1 = int( secmatch.group("x1") ) + newhdr['NAXIS2']
            newhdr[kw] = f'[{x0}:{x1},{secmatch.group("y0")}:{secmatch.group("y1")}]'

        return newhdr


    def load_section_image( self, filepath, section_id ):
        lefthdu = None
        righthdu = None
        with fits.open( filepath ) as hdul:
            for hdu in hdul:
                if ( 'CCD_LOC' in hdu.header ) and ( hdu.header['CCD_LOC'] == section_id ):
                    if hdu.header['AMP_NAME'] == 'LEFT':
                        lefthdu = hdu
                    elif hdu.header['AMP_NAME'] == 'RIGHT':
                        righthdu = hdu
                    if ( lefthdu is not None ) and ( righthdu is not None ):
                        break
            if ( lefthdu is None ) or ( righthdu is None ):
                raise RuntimeError( f"Failed to find the two HDUs for section {section_id} of "
                                    f"exposure file {filepath}" )

            if lefthdu.shape[0] != righthdu.shape[0]:
                raise RuntimeError( "Left and right amp vertical shape doesn't match." )

            # The "LEFT" amp is actually to the right (i.e. West) of the "RIGHT" amp.
            # Both are already oriented north up, east left, I think.

            newimg = np.empty( ( lefthdu.shape[0], lefthdu.shape[1] + righthdu.shape[1] ) )
            newimg[ :, 0:righthdu.shape[1] ] = righthdu.data
            newimg[ :, righthdu.shape[1]: ] = lefthdu.data

        # astropy will read FITS files as big-endian
        # But, the sep library depends on native byte ordering
        # So, swap if necessary
        if not newimg.dtype.isnative:
            newimg = newimg.astype( newimg.dtype.name )

        return newimg


    def read_header( self, filepath, section_id=None ):
        """Returns a header that would be a single header for a given chip.

        Because two chips will get stitched together in load_section_image, we have to
        modify the header here to have the right DATASEC and BAISSEC.

        """

        if isinstance( filepath, list ):
            if not all( isinstance( f, (str, pathlib.Path) ) for f in filepath ):
                raise TypeError( "If you pass a list to read_header, it must be a list of file paths." )
            filepath = filepath[0]

        if not isinstance( filepath, (str, pathlib.Path) ):
            raise TypeError( f"filepath must be a string or path. Got {type(filepath)}" )

        if section_id is None:
            # Get HDU 1, not HDU 0, becasue the global HDU 0 doesn't
            # have everything we need, but I *think* HDU 1 does.
            # TODO make sure this stays true.
            hdr = read_fits_image( filepath, ext=1, output='header' )
        else:
            hdr = None
            with fits.open( filepath ) as hdul:
                for hdu in hdul:
                    if ( ( 'CCD_LOC' in hdu.header ) and
                         ( hdu.header['CCD_LOC'] == section_id ) and
                         ( hdu.header['AMP_NAME'] == 'LEFT' )
                        ):
                        hdr = hdu.header
                        break
            if hdu is None:
                raise RuntimeError( f"Failed to find LEFT header for section {section_id}"
                                    f"in exposure file {filepath}" )

        return self._mangle_header_to_single( hdr )



class LS4CamInstrumentOriginExposures( InstrumentOriginExposures ):
    # DO
    pass
