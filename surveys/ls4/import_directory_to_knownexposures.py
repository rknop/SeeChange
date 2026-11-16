import time
import pathlib
import argparse
import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from psycopg import sql
from astropy.io import fits

from models.base import PGDB
from models.instrument import Instrument
from models.knownexposure import KnownExposure
from util.logger import SCLogger


class LS4DirectoryImporter:
    def __init__( self, basedirec, no_hold=False, mark_done_if_exposure=False, numproc=8 ):
        self.basedirec = pathlib.Path( basedirec )
        self.no_hold = bool( no_hold )
        self.mark_done_if_exposure = bool( mark_done_if_exposure )
        self.ls4cam = Instrument.get_instrument_instance( "LS4Cam" )
        self.provenance = self.ls4cam.get_exposure_provenance( proc_type='raw', method='manual_load' )
        self.numproc = 1 if numproc <=1 else int(numproc)


    def process_one_file( self, expinfo ):
        if self.numproc > 1:
            SCLogger.multiprocessing_replace()

        try:
            with PGDB( dictcursor=True ) as pgdb:
                rows = pgdb.execute( sql.SQL( "SELECT _id FROM knownexposures "
                                              "WHERE instrument='LS4Cam' and identifier={orid}"
                                             ).format( orid=expinfo.origin_identifier ) )
            if len(rows) > 0:
                SCLogger.info( f"Known exposure {expinfo.origin_identifier} is already known." )
                return 'known'

            filesdmap = { 's': 'Sci', 'd': 'Dark', 'e': 'TwiFlat', 'm': 'TwiFlat' }
            filetype = filesdmap[ expinfo.filesd ]

            if expinfo.manyfiles:
                nneeded = ( 64 if expinfo.isdualamp else 32 )
                if len( expinfo.missing ) > 0:
                    raise FileNotFoundError( f"Tried to the {nneeded} individual files that make "
                                             f"up the exposure that goes with {expinfo.filepath.name}, but some "
                                             f"files were missing: {expinfo.missing}" )
                with fits.open( expinfo.filepath ) as hdul:
                    hdu = hdul[ 1 if expinfo.isfz else 0 ]
                    hdrinfo = self.ls4cam.extract_header_info( hdu.header, [ 'mjd', 'exp_time',
                                                                             'project', 'target' ] )
                    hdrinfo['ra'] = float( hdu.header['TELE-RA'] ) * 15.
                    hdrinfo['dec'] = float( hdu.header['TELE-DEC'] )
                    # ... what is short enough to be a bias?  I hate to say "=0.0" because floats
                    if ( filetype == 'Dark' ) and ( hdrinfo['exp_time'] < 0.1 ):
                        filetype = 'Bias'
            else:
                with fits.open( expinfo.filepath ) as hdul:
                    if len(hdul) != 33:
                        raise RuntimeError( f"Opened {expinfo.filepath.name}, saw {len(hdul)} HDUs, expected 33." )

                    hdrinfo = self.ls4cam.extract_header_info( hdul[1].header, [ 'mjd', 'exp_time',
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
                                state='ready' if self.no_hold else 'held'
                               )
            ke.calculate_coordinates()

            with PGDB( dictcursor=True ) as pgdb:
                if self.mark_done_if_exposure:
                    # Not going to lock the table because normally only one parent process will be
                    #   running, and normally there won't be duplicate files in a directory.
                    #   I may regret this later.
                    rows = pgdb.execute( sql.SQL( "SELECT _id FROM exposures WHERE provenance_id={prov} "
                                                  "  AND origin_identifier={orid}"
                                                 ).format( orid=expinfo.origin_identifier,
                                                           prov=self.provenance.id ) )
                    if len(rows) > 1:
                        raise RuntimeError( "This should never happen" )
                    if len(rows) == 1:
                        ke.exposure_id = rows[0]['_id']
                        ke.state = 'done'

                try:
                    ke.insert( pgdb=pgdb )
                except Exception as ex:
                    # Maybe catch "already exists" exceptions and deal with that?
                    SCLogger.error( f"Exception inserting KnownExposure into database: {ex}" )
                    raise

            SCLogger.info( f"Added known exposure {expinfo.origin_identifier}" )
            return 'loaded'

        except Exception as ex:
            SCLogger.exception( f"Failed loading {expinfo.filepath}: {ex}" )
            return 'failed'


    def __call__( self ):
        t0 = time.perf_counter()
        SCLogger.info( f"Looking in {self.basedirec} for exposures to import" )

        seen_identifiers = set()

        expinfos = []
        for fpath in self.basedirec.iterdir():
            if fpath.is_file():
                try:
                    expinfo = self.ls4cam._figure_out_exposure_many_files_or_single( fpath )
                except Exception as ex:
                    SCLogger.warning( f"Something is wrong with file {fpath.name}, skipping it : {ex}" )
                    continue

                if expinfo.origin_identifier in seen_identifiers:
                    # Empirically, sometimes both the .fits.fz, and the
                    # individual image-per-chip files are in the same
                    # directory!  So, a bit of deduplication.
                    SCLogger.info( f"{fpath.name} has origin_idetifier {expinfo.origin_identifier} "
                                   f"which has already been seen." )
                    continue

                expinfos.append( expinfo )
                seen_identifiers.add( expinfo.origin_identifier )

        SCLogger.info( f"Going to log {len(expinfos)} exposures in {self.numproc} processes." )

        # ...I think I've had trouble passing a method directly to a multiprocessing thing before,
        #   which is why I do this...
        doer = functools.partial( self.__class__.process_one_file, self )

        if self.numproc > 1:
            ppe = ProcessPoolExecutor( max_workers=self.numproc, mp_context=multiprocessing.get_context("fork") )
            # ...I don't 100% understand what ProcessPoolExecutor.map returns, but it seems that
            #   once you iterate through it once, it's gone.  So, turn it into a list so that
            #   we can look at it more than once.
            result = [ r for r in ppe.map( doer, expinfos ) ]
        else:
            result = [ self.doer( e ) for e in expinfos ]

        nsucc = sum( 1 for r in result if r == 'loaded' )
        knowns = [ f'{e.filepath} — {e.origin_identifier}'
                   for r, e in zip(result, expinfos) if r == 'known' ]
        fails = [ f'{e.filepath} — {e.origin_identifier}'
                  for r, e in zip(result, expinfos) if r not in ('loaded', 'known') ]
        nlss = '\n  '
        nlssss = '\n    '

        SCLogger.info( f"Done loading known exposures."
                       f"{f'{nlss}Previously known:{nlssss}{nlssss.join(knowns)}' if len(knowns) > 0 else ''}"
                       f"{f'{nlss}Failed on:{nlssss}{nlssss.join(fails)}' if len(fails) > 0 else ''}" )
        msg = ( f"Loaded {nsucc} of {len(expinfos)} files in {time.perf_counter()-t0:.1f} s "
                f"using {self.numproc} processes; {len(knowns)} already known, {len(fails)} failed." )
        if len(fails) == 0:
            SCLogger.info( msg )
        else:
            SCLogger.error( msg )

        return len(fails) == 0


# ======================================================================

def main():
    parser = argparse.ArgumentParser( 'import_directory_to_knownexposures',
                                      description='Import a directory of files to known_exposures' )
    parser.add_argument( 'basedirec', help="Path to directory holding the files to load" )
    parser.add_argument( '-n', '--numproc', type=int, default=8,
                         help="Run this many subprocesses (default 8); 1=run it all serially, no multiprocessing." )
    parser.add_argument( '--no-hold', action='store_true', default=False,
                         help='By default, loaded exposures are marked as held.  Set this to make them ready.' )
    parser.add_argument( '-d', '--mark-done-if-exposure', action='store_true', default=False,
                         help=( 'By default, loaded exposures are marked as ready or held based on --no-hold. '
                                'Set this to True to check if an exposure already exists in the databse for '
                                'this knownexposure; if so, mark it as done instead of ready or held.' ) )
    args = parser.parse_args()
    loader = LS4DirectoryImporter( **vars( args ) )
    loader()


# ======================================================================
if __name__ == "__main__":
    main()
