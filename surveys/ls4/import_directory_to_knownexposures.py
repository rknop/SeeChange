import pathlib
import argparse

import psycopg
from astropy.io import fits

from models.base import PsycopgConnection
from models.instrument import Instrument
from models.knownexposure import KnownExposure
from util.logger import SCLogger


def main():
    parser = argparse.ArgumentParser( 'import_directory_to_knownexposures',
                                      description='Import a directory of files to known_exposures' )
    parser.add_argument( 'direc', help="Path to directory" )
    parser.add_argument( '--no-hold', action='store_true', default=False,
                         help='By default, loaded exposures are marked as held.  Set this to make them not held.' )
    args = parser.parse_args()

    ls4cam = Instrument.get_instrument_instance( "LS4Cam" )
    # Using 'manual_load' as the proc_type even though that's probably not right any more....
    # Update someday when we reboot everything.
    provenance = ls4cam.get_exposure_provenance( proc_type='raw', method='manual_load' )

    direc = pathlib.Path( args.direc )
    for fpath in direc.iterdir():
        if fpath.is_file():
            try:
                expinfo = ls4cam._figure_out_exposure_many_files_or_single( fpath )
            except Exception as ex:
                SCLogger.warning( f"Something is wrong with file {fpath.name}, skipping it : {ex}" )
                continue

            with PsycopgConnection() as con:
                cursor = con.cursor()
                cursor.execute( "SELECT _id FROM knownexposures WHERE instrument='LS4Cam' and identifier=%(id)s",
                                { 'id': expinfo.origin_identifier } )
                rows = cursor.fetchall()
            if len(rows) > 0:
                SCLogger.info( f"Known exposure {expinfo.origin_identifier} is already known." )
                continue

            filesdmap = { 's': 'Sci', 'd': 'Dark', 'e': 'TwiFlat', 'm': 'TwiFlat' }
            filetype = filesdmap[ expinfo.filesd ]

            if expinfo.manyfiles:
                nneeded = ( 64 if expinfo.isdualamp else 32 )
                if len( expinfo.missing ) > 0:
                    raise FileNotFoundError( f"Tried to the {nneeded} individual files that make "
                                             f"up the exposure that goes with {fpath.name}, but some "
                                             f"files were missing: {expinfo.missing}" )
                with fits.open( fpath ) as hdul:
                    hdu = hdul[ 1 if expinfo.isfz else 0 ]
                    hdrinfo = ls4cam.extract_header_info( hdu.header, [ 'mjd', 'exp_time',
                                                                        'project', 'target' ] )
                    hdrinfo['ra'] = float( hdu.header['TELE-RA'] ) * 15.
                    hdrinfo['dec'] = float( hdu.header['TELE-DEC'] )
                    # ... what is short enough to be a bias?  I hate to say "=0.0" because floats
                    if ( filetype == 'Dark' ) and ( hdrinfo['exp_time'] < 0.1 ):
                        filetype = 'Bias'
            else:
                with fits.open( fpath ) as hdul:
                    if len(hdul) != 33:
                        raise ValueError( f"Opened a {fpath.name}, saw {len(hdul)} HDUs, expected 33." )

                    hdrinfo = ls4cam.extract_header_info( hdul[1].header, [ 'mjd', 'exp_time',
                                                                            'project', 'target' ] )
                    hdrinfo['ra'] = float( hdul[1].header['TELE-RA'] ) * 15.
                    hdrinfo['dec'] = float( hdul[1].header['TELE-DEC'] )
                    if ( filetype == 'Dark' ) and ( hdrinfo['exp_time'] < 0.1 ):
                        filetype = 'Bias'

            ke = KnownExposure( instrument='LS4Cam',
                                identifier=expinfo.origin_identifier,
                                params={'method': 'localfile'},
                                mjd=hdrinfo['mjd'],
                                exp_time=hdrinfo['exp_time'],
                                ra=hdrinfo['ra'],
                                dec=hdrinfo['dec'],
                                project=hdrinfo['project'],
                                target=hdrinfo['target'],
                                type=filetype,
                                state='ready' if args.no_hold else 'held'
                               )
            ke.calculate_coordinates()

            with PsycopgConnection() as con:
                cursor = con.cursor( row_factory=psycopg.rows.dict_row )
                cursor.execute( "SELECT _id FROM exposures WHERE provenance_id=%(prov)s "
                                "  AND origin_identifier=%(id)s",
                                { 'id': expinfo.origin_identifier, 'prov': provenance.id } )
                rows = cursor.fetchall()
                if len(rows) > 1:
                    raise RuntimeError( "This should never happen" )
                if len(rows) == 1:
                    ke.exposure_id = rows[0]['_id']
                    ke.state = 'done'

                ke.insert( session=con )

            SCLogger.info( f"Added known exposure {expinfo.origin_identifier}" )


# ======================================================================
if __name__ == "__main__":
    main()
