import re
import types
import pathlib
import datetime
import subprocess

import psycopg.rows
import numpy as np

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.time
import astropy.units as u
from astropy.io import fits

from models.base import FileOnDiskMixin, PsycopgConnection
from models.provenance import Provenance
from models.image import Image
from models.instrument import ( Instrument,
                                InstrumentOrientation,
                                InstrumentOriginExposures,
                                SensorSection
                               )
from util.logger import SCLogger
from util.config import Config
from util.fits import read_fits_image
from util.retrydownload import retry_download


class LS4Cam(Instrument):
    """Default operation mode for LS4Cam: single amp, sorta.

    It's complicated because of optimizations Kenneth made.  All the
    headers claim that the left amp is what's being used for readout,
    but for SE_E, SE_F, SE_C, and SE_D, we're reading out to the right,
    so that half of the chip's CTE doesn't disaster the other half.

    LS4Cam exposures are assumed to always be raw.

    """

    # LS4 filenames:
    #
    #
    # Unpacked images:
    #   20260410004924sC0_00025_00.fits
    #
    #     First 14 are YYYYMMDDHHMMSS
    #
    #     s = "sky" (shutter opened) ; could be "d" for dark, or "e" for evening twilight flat
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

    _file_re = re.compile( r'^(?P<filebase>(?P<datetime>\d{14})(?P<sd>[sdem])(?P<C>C(?P<ctrlr>\d))?)'
                           r'_(?P<num>\d+)(?P<chipthing>_(?P<chip>\d\d))?\.fits(?P<fz>\.fz)?$' )


    def __init__( self, _save_to_call=False, **kwargs ):
        self.name = 'LS4Cam'
        self.telescope = 'ESO 1.0-m Schmidt'
        self.apperture = 1.0
        self.focal_ratio = None   # FIGURE THIS OUT
        self.square_degree_fov = 20   # CHECK THIS
        self.max_rad_degree = 3.5   # CHECK THIS
        self.pixel_scale = 1.0177
        self.read_time = None # FIGURE THIS OUT
        self.orientation_fixed = True
        self.orientation = InstrumentOrientation.NupEright    # VERIFY THIS
        self.read_noise  = 1.0  # FIGURE THIS OUT
        self.dark_current= 0.1  # FIGURE THIS OUT
        self.gain = 2.2         # FIGURE THIS OUT.  (2.2 is a non-absurd approximation for most chips.)
        self.saturation_limit = 20000  # FIGURE THIS OUT
        self.non_linearity_limit = 20000   # FIGURE THIS OUT
        self.allowed_filters = [ "0" ]

        # will apply kwargs to attributes, and register instrument in the INSTRUMENT_INSTANCE_CACHE
        Instrument.__init__(self, **kwargs)

        # self.preprocessing_steps_available = [ 'overscan', 'bias', 'dark', 'linearity', 'flat' ]
        self.preprocessing_steps_available = [ 'overscan', 'zero', 'flat' ]
        self.preprocessing_steps_by_type = { 'Bias': set( [ 'overscan' ] ),
                                             'Dark': set( [ 'overscan', 'zero' ] ),
                                             'DomeFlat': set( [ 'overscan', 'zero' ] ),
                                             'SkyFlat': set( [ 'overscan', 'zero' ] ),
                                             'TwiFlat': set( [ 'overscan', 'zero' ] ),
                                             'Fringe': set( [ 'overscan', 'zero' ] ) }
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
        defective = ( section_id in [ 'SE_D', 'NE_H' ] )
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

    def get_section_filter( self, section_id ):
        secfilt = { 'NE': 'i',
                    'NW': 'z',
                    'SE': 'g',
                    'SW': 'i' }
        return secfilt[ section_id[0:2] ]


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


    _chip_offsets = {
        'NE_A': (-1.2654 ,  0.8498),
        'NE_B': (-1.8960 ,  0.8549),
        'NE_C': (-2.5270 ,  0.8632),
        'NE_D': (-3.1559 ,  0.8743),
        'NE_E': (-1.2598 ,  2.0508),
        'NE_F': (-1.8835 ,  2.0558),
        'NE_G': (-2.5070 ,  2.0636),
        'NE_H': (-3.1559 ,  2.0636),  # Bad chip, couldn't measure, values from neighbors
        'NW_A': ( 1.2330 ,  2.0690),
        'NW_B': ( 0.6103 ,  2.0588),
        'NW_C': (-0.0128 ,  2.0529),
        'NW_D': (-0.6367 ,  2.0505),
        'NW_E': ( 1.2567 ,  0.8688),
        'NW_F': ( 0.6258 ,  0.8590),
        'NW_G': (-0.0041 ,  0.8517),
        'NW_H': (-0.6346 ,  0.8488),
        'SE_A': (-1.2763 , -1.5541),
        'SE_B': (-1.9214 , -1.5489),
        'SE_C': (-2.5664 , -1.5393),  # Bad chip, couldn't measure, values from neighbors
        'SE_D': (-3.1834 , -1.5393),
        'SE_E': (-1.2704 , -0.3524),
        'SE_F': (-1.9085 , -0.3472),
        'SE_G': (-2.5464 , -0.3387),
        'SE_H': (-3.1834 , -0.3261),
        'SW_A': ( 1.2801 , -0.3323),
        'SW_B': ( 0.6427 , -0.3427),
        'SW_C': ( 0.0049 , -0.3499),
        'SW_D': (-0.6325 , -0.3531),
        'SW_E': ( 1.3038 , -1.5326),
        'SW_F': ( 0.6595 , -1.5439),
        'SW_G': ( 0.0145 , -1.5517),
        'SW_H': (-0.6304 , -1.5546)
    }

    def get_ra_dec_for_section( self, ra, dec, section_id ):
        ra += self._chip_offsets[section_id][0] / np.cos( dec * np.pi / 180. )
        dec += self._chip_offsets[section_id][1]

        return ra, dec


    _chip_corners = {
        'NE_A': {
            (   0,    0): (-1.5571,  0.2724),
            (   0, 4095): (-1.5485,  1.4305),
            (2047,    0): (-0.9797,  0.2696),
            (2047, 4095): (-0.9771,  1.4278),
        },
        'NE_B': {
            (   0,    0): (-2.1911,  0.2792),
            (   0, 4095): (-2.1754,  1.4371),
            (2047,    0): (-1.6137,  0.2729),
            (2047, 4095): (-1.6043,  1.4311),
        },
        'NE_C': {
            (   0,    0): (-2.8244,  0.2888),
            (   0, 4095): (-2.8010,  1.4462),
            (2047,    0): (-2.2463,  0.2785),
            (2047, 4095): (-2.2308,  1.4365),
        },
        'NE_D': {
            (   0,    0): (-3.4574,  0.3022),
            (   0, 4095): (-3.4279,  1.4588),
            (2047,    0): (-2.8801,  0.2889),
            (2047, 4095): (-2.8584,  1.4462),
        },
        'NE_E': {
            (   0,    0): (-1.5484,  1.4739),
            (   0, 4095): (-1.5395,  2.6310),
            (2047,    0): (-0.9772,  1.4712),
            (2047, 4095): (-0.9749,  2.6288),
        },
        'NE_F': {
            (   0,    0): (-2.1755,  1.4797),
            (   0, 4095): (-2.1601,  2.6366),
            (2047,    0): (-1.6044,  1.4733),
            (2047, 4095): (-1.5969,  2.6315),
        },
        'NE_G': {
            (   0,    0): (-2.8020,  1.4901),
            (   0, 4095): (-2.7803,  2.6465),
            (2047,    0): (-2.2313,  1.4806),
            (2047, 4095): (-2.2157,  2.6377),
        },
        # NE_H is a bad chip, couldn't get a measurement off of it,
        #   so these numbers are from neighoring chips
        'NE_H': {
            (   0,    0): (-3.4279,  1.4901),
            (   0, 4095): (-3.4279,  2.6465),
            (2047,    0): (-2.8584,  1.4901),
            (2047, 4096): (-2.8584,  1.4901),
        },
        'NW_A': {
            (   0,    0): ( 0.9589,  1.4859),
            (   0, 4095): ( 0.9410,  2.6433),
            (2047,    0): ( 1.5295,  1.4966),
            (2047, 4095): ( 1.5036,  2.6549),
        },
        'NW_B': {
            (   0,    0): ( 0.3327,  1.4763),
            (   0, 4095): ( 0.3201,  2.6338),
            (2047,    0): ( 0.9034,  1.4845),
            (2047, 4095): ( 0.8845,  2.6408),
        },
        'NW_C': {
            (   0,    0): (-0.2943,  1.4725),
            (   0, 4095): (-0.2993,  2.6296),
            (2047,    0): ( 0.2773,  1.4770),
            (2047, 4095): ( 0.2656,  2.6337),
        },
        'NW_D': {
            (   0,    0): (-0.9211,  1.4715),
            (   0, 4095): (-0.9197,  2.6288),
            (2047,    0): (-0.3511,  1.4723),
            (2047, 4095): (-0.3557,  2.6296),
        },
        'NW_E': {
            (   0,    0): ( 0.9799,  0.2842),
            (   0, 4095): ( 0.9597,  1.4422),
            (2047,    0): ( 1.5558,  0.2961),
            (2047, 4095): ( 1.5307,  1.4529),
        },
        'NW_F': {
            (   0,    0): ( 0.3464,  0.2766),
            (   0, 4095): ( 0.3332,  1.4346),
            (2047,    0): ( 0.9224,  0.2843),
            (2047, 4095): ( 0.9060,  1.4416),
        },
        'NW_G': {
            (   0,    0): (-0.2885,  0.2707),
            (   0, 4095): (-0.2943,  1.4288),
            (2047,    0): ( 0.2890,  0.2753),
            (2047, 4095): ( 0.2769,  1.4332),
        },
        'NW_H': {
            (   0,    0): (-0.9226,  0.2696),
            (   0, 4095): (-0.9210,  1.4279),
            (2047,    0): (-0.3451,  0.2705),
            (2047, 4095): (-0.3499,  1.4286),
        },
        'SE_A': {
            (   0,    0): (-1.5747, -2.1308),
            (   0, 4095): (-1.5659, -0.9736),
            (2047,    0): (-0.9835, -2.1335),
            (2047, 4095): (-0.9815, -0.9763),
        },
        'SE_B': {
            (   0,    0): (-2.2233, -2.1234),
            (   0, 4095): (-2.2071, -0.9666),
            (2047,    0): (-1.6323, -2.1304),
            (2047, 4095): (-1.6229, -0.9729),
        },
        'SE_C': {
            (   0,    0): (-2.8723, -2.1122),
            (   0, 4095): (-2.8484, -0.9554),
            (2047,    0): (-2.2816, -2.1229),
            (2047, 4095): (-2.2641, -0.9661),
        },
        # SE_D is a bad chip, couldn't get a measurement off of it,
        #   so these numbers are from neighoring chips
        'SE_D': {
            (   0,    0): (-3.4882, -2.1234),
            (   0, 4095): (-3.4882, -0.9554),
            (2047,    0): (-2.9052, -2.1234),
            (2047, 4095): (-2.9052, -0.9554),
        },
        'SE_E': {
            (   0,    0): (-1.5652, -0.9299),
            (   0, 4095): (-1.5569,  0.2283),
            (2047,    0): (-0.9809, -0.9325),
            (2047, 4095): (-0.9791,  0.2259),
        },
        'SE_F': {
            (   0,    0): (-2.2065, -0.9225),
            (   0, 4095): (-2.1909,  0.2356),
            (2047,    0): (-1.6230, -0.9291),
            (2047, 4095): (-1.6135,  0.2292),
        },
        'SE_G': {
            (   0,    0): (-2.8475, -0.9126),
            (   0, 4095): (-2.8252,  0.2449),
            (2047,    0): (-2.2641, -0.9230),
            (2047, 4095): (-2.2473,  0.2352),
        },
        'SE_H': {
            (   0,    0): (-3.4882, -0.8974),
            (   0, 4095): (-3.4587,  0.2595),
            (2047,    0): (-2.9052, -0.9111),
            (2047, 4095): (-2.8822,  0.2464),
        },
        'SW_A': {
            (   0,    0): ( 0.9997, -0.9164),
            (   0, 4095): ( 0.9802,  0.2410),
            (2047,    0): ( 1.5829, -0.9047),
            (2047, 4095): ( 1.5572,  0.2522),
        },
        'SW_B': {
            (   0,    0): ( 0.3584, -0.9252),
            (   0, 4095): ( 0.3462,  0.2327),
            (2047,    0): ( 0.9423, -0.9171),
            (2047, 4095): ( 0.9234,  0.2404),
        },
        'SW_C': {
            (   0,    0): (-0.2826, -0.9312),
            (   0, 4095): (-0.2887,  0.2271),
            (2047,    0): ( 0.3015, -0.9264),
            (2047, 4095): ( 0.2890,  0.2316),
        },
        'SW_D': {
            (   0,    0): (-0.9241, -0.9319),
            (   0, 4095): (-0.9220,  0.2266),
            (2047,    0): (-0.3403, -0.9316),
            (2047, 4095): (-0.3437,  0.2269),
        },
        'SW_E': {
            (   0,    0): ( 1.0194, -2.1167),
            (   0, 4095): ( 1.0008, -0.9600),
            (2047,    0): ( 1.6144, -2.1029),
            (2047, 4095): ( 1.5830, -0.9491),
        },
        'SW_F': {
            (   0,    0): ( 0.3726, -2.1262),
            (   0, 4095): ( 0.3604, -0.9691),
            (2047,    0): ( 0.9635, -2.1175),
            (2047, 4095): ( 0.9438, -0.9611),
        },
        'SW_G': {
            (   0,    0): (-0.2764, -2.1322),
            (   0, 4095): (-0.2826, -0.9752),
            (2047,    0): ( 0.3145, -2.1272),
            (2047, 4095): ( 0.3017, -0.9702),
        },
        'SW_H': {
            (   0,    0): (-0.9250, -2.1331),
            (   0, 4095): (-0.9232, -0.9759),
            (2047,    0): (-0.3339, -2.1324),
            (2047, 4095): (-0.3389, -0.9752),
        },
    }


    def get_ra_dec_corners_for_section( self, ra, dec, section_id ):
        # Because of the orientation of the image, the min ra is at
        # pixel 0, max ra is at pixel 2047...  which doesn't sound
        # surprising, but RA increses to the East, which is to the left
        # if you're looking at a non-mirrored north-up image of the sky.
        cors = self._chip_corners[ section_id ]
        ra_corner = np.array( [ [ cors[(   0, 0)][0], cors[(   0, 4095)][0] ],
                                [ cors[(2047, 0)][0], cors[(2047, 4095)][0] ] ] )
        dec_corner = np.array( [ [ cors[(   0, 0)][1], cors[(   0, 4095)][1] ],
                                 [ cors[(2047, 0)][1], cors[(2047, 4095)][1] ] ] )
        ra_corner /= np.cos( dec * np.pi / 180. )
        ra_corner += ra
        dec_corner += dec

        return {
            'ra_corner_00': ra_corner[0, 0],
            'ra_corner_01': ra_corner[0, 1],
            'ra_corner_10': ra_corner[1, 0],
            'ra_corner_11': ra_corner[1, 1],
            'dec_corner_00': dec_corner[0, 0],
            'dec_corner_01': dec_corner[0, 1],
            'dec_corner_10': dec_corner[1, 0],
            'dec_corner_11': dec_corner[1, 1],
            'minra': ra_corner.min(),
            'maxra': ra_corner.max(),
            'mindec': dec_corner.min(),
            'maxdec': dec_corner.max()
        }


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
            airmass=[],
            sec_id=['CCD_LOC'],
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
        if band not in ['g', 'i', 'z']:
            raise ValueError( f"I only understand filters g, i, and z, not {band}" )
        return band


    def gaia_dr3_to_instrument_mag( self, filter, catdata ):
        raise NotImplementedError( "Need to implement gaia_dr3_to_instrument_mag for LS4Cam" )


    # def get_filter_bandpasses( self ):
    #     # TODO: verify this!  Right now we're just using the lsst values in the base class.


    def _get_default_calibrator( self, mjd, section, calibtype='dark', filter=None ):
        if calibtype != 'flat':
            raise NotImplementedError( "I only know how to get default calibrator files for flats." )

        if isinstance( section, SensorSection ):
            section_id = section.id
        elif isinstance( section, str ):
            section_id = section
        else:
            raise TypeError( f"section must be a SensorSection or a str, not a {type(section)}" )
        self.check_section_id( section_id )

        from models.calibratorfile import CalibratorFile, CalibratorFileDownloadLock

        cfg = Config.get()
        cv = Provenance.get_code_version( process='LS4Cam Default Calibrator' )
        prov = Provenance( process='LS4Cam Default Calibrator', code_version_id=cv.id )
        prov.insert_if_needed()

        reldatadir = pathlib.Path( 'LS4Cam_default_calibrators' )
        datadir = pathlib.Path( FileOnDiskMixin.local_path ) / reldatadir

        if calibtype == 'flat':
            rempath = pathlib.Path( f'{cfg.value("LS4Cam.calibfiles.flatbase")}_{section_id}.fits' )
        else:
            return None

        url = f'{cfg.value("LS4Cam.calibfiles.urlbase")}{str(rempath)}'
        filepath = reldatadir / calibtype / rempath.name
        fileabspath = datadir / calibtype / rempath.name

        SCLogger.debug( f"ls4cam._get_default_calibrator: getting calibfile lock for {self.name} {section_id} "
                        f"calibset='externally_supplied' calibtype={calibtype}" )
        with CalibratorFileDownloadLock.acquire_lock( instrument='LS4Cam',
                                                      section=section_id,
                                                      calibset='externally_supplied',
                                                      calibtype=calibtype,
                                                      flattype=( 'externally_supplied' if calibtype == 'flat'
                                                                 else None ) ):
            retry_download( url, fileabspath )

            if calibtype == 'flat':
                dbtype = 'ComSkyFlat'
            mjd = float( cfg.value( 'LS4Cam.calibfiles.mjd' ) )
            image = Image( format='fits', type=dbtype, provenance_id=prov.id, instrument='LS4Cam',
                           telescope='ESO 1.0-m Schmidt', filter=self.get_section_filter(section_id),
                           section_id=section_id, filepath=str(filepath), mjd=mjd, end_mjd=mjd,
                           info={}, exp_time=0, ra=0., dec=0.,
                           ra_corner_00=0., ra_corner_01=0.,ra_corner_10=0., ra_corner_11=0.,
                           dec_corner_00=0., dec_corner_01=0., dec_corner_10=0., dec_corner_11=0.,
                           minra=0, maxra=0, mindec=0, maxdec=0,
                           target="", project="" )
            FileOnDiskMixin.save( image, fileabspath )
            calfile = CalibratorFile( type=calibtype,
                                      calibrator_set='externally_supplied',
                                      flat_type='externally_supplied' if calibtype == 'flat' else None,
                                      instrument='LS4Cam',
                                      sensor_section=section_id,
                                      image_id=image.id )
            image.insert()
            calfile.insert()

        SCLogger.debug( f"ls4cam._get_default_calibrator: releasing calibfile lock for {self.name} {section_id} "
                        f"calibset='externally_supplied' calibtype={calibtype}" )

        return calfile


    # def preprocessing_calibrator_files( self, calibset, flattype, section, filter, mjd, nofetch=False ):
    #     # TODO: figure out if we have to override preprocessing_calibrator_files.  I hope not.


    def overscan_sections( self, header ):
        if header['amp_direction'].strip() != 'left':
            raise RuntimeError( f"{self.__class__.__name} assumes that readout is on the 'left' amp." )

        section_id = header['CCD_LOC'].strip()

        # Some people, when confronted with a problem, think "I know,
        #   I'll use regular expressions." Now they have two problems.
        #     --Jamie Zawinski
        extractor = re.compile( r'^\s*\[\s*(?P<x0>\d+)\s*:\s*(?P<x1>\d+)\s*,'
                                r'\s*(?P<y0>\d+)\s*:\s*(?P<y1>\d+)\s*\]\s*$' )

        # Kenneth's Custom Mode : most chips are read out to the left.
        # SE-C, SE-D, SE-E, and SE-F are read out to the right.

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
        mat = extractor.search( header['BIASSECR'] )
        biasr_x0 = int(mat.group('x0')) - 1
        biasr_x1 = int(mat.group('x1'))
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
        #   the numbers assume dual-amp readout.  The actual data section
        #   has the R side on the left and the L side on the right.
        #
        # IF the left amp is used for readout, going from low x to high
        #    x, there is a left block that is PRESECL, then there is the R
        #    side data, then the L side data, then the BIASSECL.
        #    Because the numbers assume dual-amp readout, we have to add
        #    the size of the right side data to the BIASSECL x numbers
        #    to account for that right data being there.
        #
        # IF the right amp is used for readout, going from low x to high
        #    x, there is a left block that is BIASSECL, then there is R
        #    side data, then L side data.  In this case, I think we can
        #    ignore prescan, since it's on the right side, and as such
        #    we don't need to offset anything.

        if section_id in [ 'SE_C', 'SE_D', 'SE_E', 'SE_F' ]:
            biassec_x0 = biasr_x0
            biassec_x1 = biasr_x1
            datasec_x0 = datar_x0
            datasec_x1 = datasec_x0 + ( datal_x1 - datal_x0 ) + ( datar_x1 - datar_x0 )

        else:
            biassec_x0 = biasl_x0 + ( datar_x1 - datar_x0 )
            biassec_x1 = biassec_x0 + ( biasl_x1 - biasl_x0 )
            datasec_x0 = prel_x1
            datasec_x1 = datasec_x0 + ( datal_x1 - datal_x0 ) + ( datar_x1 - datar_x0 )

        y0 = datal_y0
        y1 = datal_y1

        return [ { 'secname': section_id,
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


    def _figure_out_exposure_many_files_or_single( self, filepath, origin_identifier=None,
                                                   ok_no_pattern_match=False ):

        expinfo = types.SimpleNamespace()

        # Try to identify if it's a whole bunch of files, or if it's a single file.
        # If it's a whole bunch of files, then the convention is to pass any one
        #   of the files, and we have to figure out the rest.
        filematch = self._file_re.search( filepath.name )
        expinfo.filesd = None
        # filectrlr = None
        # filechip = None
        if filematch is None:
            if ok_no_pattern_match:
                # If we can't parse the filename, assume it's all chips packed in one file
                expinfo.manyfiles = False
                expinfo.isfz = ( ( len(filepath.name) >= 3 ) and ( filepath.name[-3:] == '.fz' ) )
            else:
                raise ValueError( f"Can't parse LS4 exposure filename {filepath.name}" )
        else:
            expinfo.filebase = filematch.group( 'filebase' )
            expinfo.filedatetime = filematch.group( 'datetime' )
            expinfo.filesd = filematch.group( 'sd' )
            expinfo.filectrlr = filematch.group( 'ctrlr' )
            expinfo.filenum = filematch.group( 'num' )
            expinfo.manyfiles = ( filematch.group('C') is not None )
            expinfo.isfz = ( filematch.group('fz') is not None )
            if expinfo.manyfiles:
                if filematch.group('chipthing') is None:
                    raise ValueError( f'Error prasing ls4cam exposure filename "{filepath.name}": '
                                      f'filename has a C?, but doesn\'t have a chip.' )
                # filecontroller = int( filematch.group('ctrlr') )
                # filechip = int( filematch.group('chip') )
            else:
                if filematch.group('chipthing') is not None:
                    raise ValueError( f'Error prasing ls4cam exposure filename "{filepath.name}": '
                                      f'filename has a chip, but doesn\'t have C?.' )

        if expinfo.manyfiles:
            expinfo.exposurename = f'{expinfo.filedatetime}{expinfo.filesd}_{expinfo.filenum}.fits'
        else:
            expinfo.exposurename = filepath.name[:-3] if expinfo.isfz else filepath.name

        expinfo.origin_identifier = ( origin_identifier if origin_identifier is not None
                                      else expinfo.exposurename )

        expinfo.exposurepath = pathlib.Path( FileOnDiskMixin.temp_path ) / expinfo.exposurename
        expinfo.exposurepathfz = expinfo.exposurepath.parent / f"{expinfo.exposurepath.name}.fz"

        if expinfo.manyfiles:
            # Make sure we can find all the files
            expinfo.isdualamp = ( self.name == 'LS4Cam_dualamp' )
            expinfo.files = []
            expinfo.missing = []
            nsubchip = 16 if expinfo.isdualamp else 8
            for ctrlr in range(0, 4):
                for subchip in range(0, nsubchip):
                    fz = ( ".fz" if expinfo.isfz else "" )
                    p = filepath.parent / ( f'{expinfo.filedatetime}{expinfo.filesd}'
                                            f'C{ctrlr}_{expinfo.filenum}_{subchip:02d}.fits{fz}' )
                    if p.is_file():
                        expinfo.files.append( p )
                    else:
                        expinfo.missing.append( p.name )
        else:
            with fits.open( filepath ) as hdul:
                expinfo.nhdus = len( hdul )

        return expinfo


    def _load_exposure_from_file_or_files( self, filepath, origin_identifier=None, params=None,
                                           proc_type='raw', method='manual_load', code_version=None,
                                           exists_ok=False ):
        # Have this here to avoid circular imports (instrument.py)
        from models.exposure import Exposure

        # OK, strict object-oriented design be damned.  This function
        # handles stuff for both this class (LS4Cam) and LS4Cam_dualamp,
        # and knows about both.
        isdualamp = ( self.name == 'LS4Cam_dualamp' )

        filepath = pathlib.Path( filepath )
        obs_type_map = { 'dark': 'Dark',
                         'pmskyflat': 'TwiFlat',
                         'sky': 'Sci',
                         # ...this next one is a hack, should probably go away.  We ahd some biases
                         #   that were taken manually at the telescope rather than through the
                         #   process by which we'll really get images.
                         'TEST': 'Bias'
                        }

        expinfo = self._figure_out_exposure_many_files_or_single( filepath, origin_identifier=origin_identifier )
        provenance = self.get_exposure_provenance( proc_type=proc_type, method=method, code_version=code_version )

        with PsycopgConnection() as pgcon:
            cursor = pgcon.cursor( row_factory=psycopg.rows.dict_row )
            cursor.execute( "SELECT * FROM exposures WHERE origin_identifier=%(id)s",
                            { 'id': expinfo.origin_identifier } )
            rows = cursor.fetchall()
            if len(rows) > 0:
                if len(rows) > 1:
                    raise RuntimeError( "This should never happen" )
                if not exists_ok:
                    raise RuntimeError( f"Exposure with origin_identifier=\"{expinfo.origin_identifier}\" "
                                        f"already exists in database." )
                row = rows[0]
                if row['provenance_id'] != provenance.id:
                    raise ValueError( f"Exposure with origin_identifier=\"{expinfo.origin_identifier}\" "
                                      f"already exists in the database with provenance "
                                      f"\"{row['provenance_id']}\", but we're trying to load it with "
                                      f"provenance \"{provenance.id}\"." )
                SCLogger.info( f"Exposure with origin identifier=\"{expinfo.origin_identifier}\" already in the "
                               f"database, not doing anything." )
                return Exposure( **row )

        try:
            hdu0 = None
            ra = None
            dec = None
            obs_type = None
            hdus = []
            exphdrinfo = None
            known_chips = set( self.get_section_ids() )
            found_chips = set()

            def process_hdu( hdu ):
                nonlocal hdu0, ra, dec, obs_type, hdus, exphdrinfo, isdualamp
                hdr = hdu.header
                if hdr['CCD_LOC'] not in known_chips:
                    raise ValueError( f"HDU {hdui} of {filepath.name} has CCD_LOC={hdr['CCD_LOC']}, "
                                      f"which isn't a known chip." )
                if ( hdr['amp_direction'] == 'both' ) != isdualamp:
                    raise ValueError( f"Header of {fitsfile.name} has amp_direction={hdr['amp_direction']}, "
                                      f"which is not what {self.__class__.name} expects." )
                if hdu0 is None:
                    # There is code elsewhere that will interpret a string ra as h:m:s
                    # WORRY AND THINK : the header doesn't have comments saying the units
                    #   are hours or degrees.  Because in my test image, I got an airmass
                    #   of -3.5 if I interpreted it as degrees, but 1.05 if I interpreted
                    #   it as hours, I'm guessing it's hours.  (Also, Kenneth told me
                    #   it was hours, and he should know.)
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
                found_chips.add( hdr['CCD_LOC'] )

            if expinfo.manyfiles:
                nneeded = ( 64 if expinfo.isdualamp else 32 )
                if len(expinfo.missing) > 0:
                    raise RuntimeError( f"Tried to the {nneeded} individual files that make up the exposure that "
                                        f"goes with {filepath.name}, but some files were missing: {expinfo.missing}" )
                if len(expinfo.files) != nneeded:
                    raise RuntimeError( "This should never happen." )

                # Assemble all of these togther into a single exposure
                for fitsfile in expinfo.files:
                    with fits.open( fitsfile ) as hdul:
                        if ( expinfo.isfz and ( len(hdul) != 2 ) ) or ( ( not expinfo.isfz ) and ( len(hdul) != 1 ) ):
                            raise RuntimeError( f"Unexpected number of HDUs in file {fitsfile.name}: "
                                                f"expected {2 if expinfo.isfz else 1} but got {len(hdul)}" )
                        hdu = hdul[1] if expinfo.isfz else hdul[0]
                        # TODO, verify chip and controller vs. filename!!!
                        process_hdu( hdu )

            else:  # not expinfo.manyfiles
                if isdualamp:
                    raise NotImplementedError( "Single file and dual amp not working yet." )
                if expinfo.nhdus != 33:
                    raise ValueError( f"Opened a packed FITS file, saw {len(hdul)} HDUs, expected 33." )
                # Note : the LS4 .fz files weren't produced with fpack, and
                #   are (I think) non-standard.  However, even though
                #   funpack doesn't know what to do with them,
                #   astropy.io.fits seems to read them just fine.  As such,
                #   read them, but then rather than just copy them, make new
                #   versions of them and run fpack on them to make them
                #   standard.  I THINK all the data raw data is
                #   integer-encoded, so this should be lossses compression,
                #   so it's not a problem.
                with fits.open( filepath ) as hdul:
                    for hdui, hdu in enumerate(hdul):
                        if hdui == 0:
                            # Exposure header, not an image
                            continue
                        process_hdu( hdu )

            if found_chips != known_chips:
                raise RuntimeError( f"Didn't find all the expected chips in exposure {filepath.name}; "
                                    f"missing are: {known_chips-found_chips}" )
            if len(hdus) != 33:
                raise RuntimeError( "This should never happen." )

            # Write the exposure to temp and fpack it
            exphdul = fits.HDUList( hdus )
            exphdul.writeto( expinfo.exposurepath )
            exphdul = None
            del exphdul
            _result = subprocess.run( [ "fpack", str(expinfo.exposurepath) ], capture_output=True )

            # Load it and save the loaded object to the right place

            expobj = Exposure( current_file=expinfo.exposurepathfz, invent_filepath=True, type=obs_type,
                               ra=ra, dec=dec, format='fitsfz', instrument=self.name,
                               filter=None, filter_array=['i', 'z', 'g', 'i'],
                               provenance_id=provenance.id, origin_identifier=expinfo.origin_identifier,
                               header=hdu0.header, preprocc_bitflag=0, components=None, **exphdrinfo )
            expobj.save( expinfo.exposurepathfz )
            expobj.insert()

            return expobj

        finally:
            # Clean up the temp files if we made them
            if expinfo.exposurepath.exists():
                expinfo.exposurepath.unlink()
            if expinfo.exposurepathfz.exists():
                expinfo.exposurepathfz.unlink()


    def manually_load_exposure( self, filepath, origin_identifier=None, params=None,
                                proc_type='raw', method='manual_load', code_version=None,
                                exists_ok=False ):
        return self._load_exposure_from_file_or_files( filepath, origin_identifier=origin_identifier,
                                                       params=params, proc_type=proc_type,
                                                       method=method, code_version=code_version,
                                                       exists_ok=exists_ok )


    def acquire_and_commit_origin_exposure( self, identifier, params, outdir=None ):
        if ( 'method' in params ) and ( params['method'] == 'localfile' ):
            # OMG THIS IS A HACK.
            # Things should be made into parameters!

            if outdir is not None:
                raise NotImplementedError( "Don't know how to deal with outdir" )

            mat = self._file_re.search( identifier )
            if mat is None:
                raise ValueError( f"Failed to parse identifier {identifier}" )
            filedatetime = mat.group( 'datetime' )
            filesd = mat.group( 'sd' )
            filenum = mat.group( 'num' )

            # Have to find the file
            filedate = datetime.date( int(identifier[0:4]), int(identifier[4:6]), int(identifier[6:8]) )
            filepath = None
            for delta in [ 0, -1, 1 ]:
                if filepath is not None:
                    break
                searchdate = filedate + datetime.timedelta( days=delta )
                masterdirec = pathlib.Path( "/m4616/tmp" )
                direcs = masterdirec.glob( f'{searchdate.year:04d}{searchdate.month:02d}{searchdate.day:02d}*' )
                direcs = [ d for d in direcs if d.is_dir() ]
                direcs.sort()
                for direc in direcs:
                    for f in [ direc / f'{filedatetime}{filesd}_{filenum}.fits',
                               direc / f'{filedatetime}{filesd}_{filenum}.fits.fz',
                               direc / f'{filedatetime}{filesd}C0_{filenum}_00.fits',
                               direc / f'{filedatetime}{filesd}C0_{filenum}_00.fits.fz' ]:
                        if f.is_file():
                            filepath = f
                            break

            if filepath is None:
                raise FileNotFoundError( f"Failed to find file for identifier {identifier}" )

            expobj = self.manually_load_exposure( filepath, exists_ok=True )
            return expobj

        else:
            if 'method' not in params:
                raise ValueError( "No method given to acquire LS4 exposure" )
            else:
                raise ValueError( f"Unknown method to acquire LS4 exposure: {params['method']}" )


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


# Register the instrument in the Instrument dictionaries
LS4Cam.register_this_instrument()
# Dualamp is currently broken
# LS4Cam_dualamp.register_this_instrument()
