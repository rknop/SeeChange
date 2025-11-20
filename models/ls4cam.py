import numpy as np

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.time
import astropy.units as u

from models.instrument import Instrument, InstrumentOrientation, SensorSection get_instrument_instance
from pipeline.catalog_tools import Bandpass

class LS4Cam(Instrument):

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

        self.preprocessing_steps_available = [ 'overscan', 'bias', 'dark', 'linearity', 'flat' ]
        self.preprocessing_steps_done = []

    @classmethod
    def get_section_ids( cls ):
        """LS4 chip ids."""

        seclist = []
        for quadrant in [ 'NE', 'NW', 'SE', 'SW' ]:
            for chipinqud in [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' ]:
                seclist.append( f"{quadrant}_{chipinquad}" )
        return seclist

    @classmethod
    def check_section_id( cls, section_id ):
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
            raise ValueError( f"section_id[3] must be in the range A..H not {section_id[3]}." )

    @classmethod
    def _make_new_section( self, section_id ):
        """Make a SensorSection for the LS4 instrument."""

        # TODO get dx and dy right
        dx = 0
        dy = 0
        # TODO get defective right
        defective = False
        return SensorSection( section_id, self.name, size_x=2048, size_y=4096,
                              offset_x=dx, offset_y=dy, defective=defective )
        
        
    @classmethod
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


    @classmethod
    def get_filename_regex( cls ):
        # TODO MAKE SURE THIS IS CURRENT
        return [ r'\d{13}s_\d{5}.fits(.fz)?' ]


    def acquire_origin_exposure( self, identifier, params, outdir=None ):
        """Download exposure...somehow...

        """
        raise NotImplementedError( "Do." )


    def acquire_and_commit_origin_exposure( self, identifier, params ):
        # Have this here to avoid circular imports (instrument.py)
        from models.exposure import Exposure
        
        expfile = pathlib.Path( self.acquire_origin_exposure( identifier, params ) )

        obs_type_map = { 'dark': 'Dark',
                         'pmskyflat': 'TwiFlat',
                         'sky': 'Sci' }

        provenance = self.get_exposure_provenance( proc_type=imagetypmap )

        with fits.open( expfile ) as ifp:
            # Going to look at HDU 1, becasue the global HDU 0 doesn't have everything we need,
            #   but I *think* HDU 1 does.  -oo- think ahead and see if this stays true,
            #   things are still evolving right now.
            hdr = ifp[1].header
            ra = hdr['TELE-RA']
            dec = hdr['TELE-DEC']
            obs_type = obs_type_map[ hdr['IMAGETYP'] ]
            filter = None
            filter_array = [ 'i', 'z', 'g', 'i' ]
            instrument = 'LS4cam_dualamp' if len(ifp) > 40 else 'LS4cam'

            if expfile.name[-8:] == '.fits.fz':
                format = 'fitsfz'
            elif expfile.name[-5:] == '.fits':
                format = 'fits'
            else:
                raise ValueError( f"Can't figure out the format of exposure file {expfile}" )

            exphdrinfo = Instrument.extract_header_info( hdr, [ 'ra', 'dec', 'mjd', 'exp_time', 'project', 'target',
                                                                'instrument', 'telescope', 'airmass' ] )
            if ( exphdrinfo['exp_time'] == 0 ) and obs_type == 'Dark':
                obs_type = 'Bias'

        expobj= Exposure( current_file=expfile, invent_filepath=True, type=obs_type,
                          format=format, provenance_id=provenance.id,
                          origin_identifier=identifier, header=hdr,
                          preprocc_bitflag=0, components=None, **exphdrinfo )
        expboj.save( expfile )
        expobj.insert()

        return expobj
            


    
    @classmethod
    def extract_header_info( cls, header, names ):
        """Get header information from the raw header into common column names.

        The method doc in instrument.dy says that this method is not
        supposed to be overriden, but the structure of it is not
        fleixble enough; it assumes that there would only ever be a unit
        conversion.  Here, we don't have an airmass column in the header
        at all, so it needs to be calculated from muiltiple header keywords.

        """
        somenames = [ n for n in names if n != 'airmass' ]
        output_values = super().extract_header_info( header, somenames )

        if 'airmass' in names:
            loc = EarthLocation.of_site( 'La Silla Observatory (ESO)' )
            tim = astropy.time.Time( header['STARTOBS'], scale='utc', format='isot', location=loc )
            # I suppose we could be really anal and try to use the chip ra and dec, but hopefully
            #   this will be good enough.  (Plus, the header doesn't currently have the chip ra
            #   and dec....)
            radec = SkyCoord( header['TELE-RA'], header['TELE-DEC'], unit=u.deg )
            altaz = radec.transform_to( AltAz( obstime=tim, location=loc ) )
            output_values['airmass'] = altaz.sexz.value


    def get_ra_dec_for_section( self, ra, dec, section_id ):
        raise NotImplementedError( "_get_ra_dec_for_section not implemented for LS4" )


    def get_ra_dec_corners_for_section( self, ra, dec, section_id ):
        raise NotImplementedError( "_get_ra_dec_corners_for_section not implemented for LS4" )


    def get_standard_flags_image( self, section_id ):
        raise NotImplementedError( "get_standard_flags_iamge not implemented for LS4" )


    @classmethod
    def _get_header_keyword_translations( cls ):
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
    

    @classmethod
    def _get_header_values_converters( cls ):
        c = dict(
            mjd=lambda x: astropy.time.Time( x, scale='utc', format='isot' ).mjd )
        }
        return c


    def overscan_sections( self, header ):
        raise NotImplementedError( "Still need to do this for one-amp readout." )
    


    @classmethod
    def _get_fits_hdu_index_from_section_ids( cls, section_id ):
        # This will not get called because we are overriding
        #   Insrument.load_section_image
        raise NotImplementedError( "You should never see this for LS4Cam" )
    
    @classmethod
    def get_short_instrument_name( cls ):
        return 'ls4'

    @classmethod
    def get_short_filter_name( cls, band ):
        if filter not in ['g', 'r', 'i']:
            raise ValueError( f"I only understand filters g, r, and i, not {band}" )
        return filter

    @classmethod
    def gaia_dr3_to_instrument_mag( cls, filter, catdata ):
        raise NotImplementedError( "Do." )


    # @classmethod
    # def get_filter_bandpasses( cls ):
    #     # TODO: verify this!  Right now we're just using the lsst values in the base class.


    def _get_default_calibrator( self, mjd, section, calibtype='dark', filter=None ):
        raise NotImplementedError( "Do." )


    # TODO: figure out if we have to override preprocessing_calibrator_files
    
    
    
    
    
class LS4Cam_dualamp(LS4):
    def __init__( self, **kwargs ):
        super().__init__( **kwargs )
        raise NotImplementedError( "I think I have more to do" )


    @classmethod
    def _mangle_header_to_single( cls, hdr ):
        """Pass the *left* amp header.  Will edit the *SEC* fields to match a stiched raw image, return a new header."""

        newhdr = hdr.copy()
        
        arrparse = re.compile( r'^\s*\[\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*,\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*\]\s*$' )
        datamatchr = arrparse.search( newhdr['DATASECR'] )
        biasmatchr = arrparse.search( newhdr['BIASSECR'] )
        prematchr = arrparse.search( newhdr['PRESECR'] )
        if any( i is None for i in [ datmatchr, biasmatchr, prematchr ] ):
            raise RuntimeError( "Failed to parse DATASECR, BAISSECR, and/or PRESECR" )
        for kw in [ 'DATASECR', 'BIASSECR', 'PRESECR' ]:
            secmatch = arrparse.serch( newhdr[kw] )
            if secmatch is None:
                raise ValueError( f"Failed to parse image section from {newhdr[kw]} for {kw}" )
            x0 = int( secmatch.group("x0") ) + newhdr['NAXIS2']
            x1 = int( secmatch.group("x1") ) + newhdr['NAXIS2']
            newhdr[kw] = '[{x0}:{x1},{secmatch.group["y0"]}:{secmatch.group["y1"]}]'

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

        # Because the rest of the code is going to want to have a single
        #   image, we have to do some pre-stiching.

        newhdr = self._mangle_header_to_single( lefthdu.header.copy() )
        if lefthdu.shape[0] != righthdu.shape[0]:
            raise RuntimeError( "Left and right amp vertical shape doesn't match." )
        newimg = np.empty( ( lefthdu.shape[0], lefthdu.shape[1] + righthdu.shape[1] ) )
        newimg[ :, 0:lefthdu.shape[1] ] = lefthdu.data
        newimg[ :, lefthdu.shape[1]: ] = righthdu.data
        
        return newimg

    def read_header( self, filepath, section_id=None ):
        """Returns a header that would be a single header for a given chip.

        Because two chips will get stitched together in load_section_image, we have to
        modify the header here to have the right DATASEC and BAISSEC.

        """
        
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

ROB YOU WERE IN THE MIDDLE OF THINGS

