import pytest
import datetime
import time

import psycopg2.extras

from models.base import Psycopg2Connection
from models.enums_and_bitflags import CalibratorTypeConverter, CalibratorSetConverter, FlatTypeConverter
from models.calibratorfile import CalibratorFileDownloadLock


# Really only have test for test_acquire_lock

@pytest.fixture
def kwargs():
    return { 'instrument': 'DECam',
             'section': 'N11',
             'calibset': CalibratorSetConverter.to_int('general'),
             'calibtype': CalibratorTypeConverter.to_int('flat'),
             'flattype': FlatTypeConverter.to_int('sky')
            }


def whereclause( kwargs ):
    kwcolmap = { 'instrument': 'instrument',
                 'section': 'sensor_section',
                 'calibset': '_calibrator_set',
                 'calibtype': '_type',
                 'flattype': '_flat_type' }

    q = "WHERE "
    _and = ""
    for key in kwargs:
        q += f"{_and} {kwcolmap[key]}{' IS NULL' if kwargs[key] is None else f'=%({key})s'} "
        _and = "AND"
    return q


def test_acquire_lock( kwargs ):
    # Make sure the lock is not there to start with
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause(kwargs)}", kwargs )
        assert len( cursor.fetchall() ) == 0

    # Get the lock, then make sure it's there
    with CalibratorFileDownloadLock.acquire_lock( **kwargs ):
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause(kwargs)}", kwargs )
            origrows = cursor.fetchall()
            assert len(origrows) > 0

        # Wait 6 seconds, then check to make sure the heartbeat got updated
        time.sleep( 6 )
        with Psycopg2Connection() as conn:
            cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
            cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause(kwargs)}", kwargs )
            rows = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0]['modified'] - origrows[0]['modified'] > datetime.timedelta( seconds=4 )

    # Make sure the lock is not there now that we're done
    # (I'd love also to make sure that the heartbeat process isn't running, but
    # that would require more digging into finding one's own subprocess and
    # looking at them, and I'm too lazy.)
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause(kwargs)}", kwargs )
        assert len( cursor.fetchall() ) == 0

    # Make sure the thing works if we set some of the fields to null
    for nulls in [ [ 'section' ], [ 'section', 'flattype' ] ]:
        tmpargs = kwargs.copy()
        for null in nulls:
            tmpargs[null] = None
        with CalibratorFileDownloadLock.acquire_lock( **tmpargs ):
            with Psycopg2Connection() as conn:
                cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
                cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause(tmpargs)}", tmpargs )
                rows = cursor.fetchall()
                assert len(rows) > 0


def test_lock_contention( kwargs ):
    # Make sure the lock is not there to start with
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause(kwargs)}", kwargs )
        assert len( cursor.fetchall() ) == 0

    # Make sure it times out if another process has created the table lock
    with CalibratorFileDownloadLock.acquire_lock( **kwargs ):
        # ...now try to get it again, which should time out since the outer width holds the lock
        with pytest.raises( RuntimeError, match="Failed to claim the calibfile_downloadlock row" ):
            with CalibratorFileDownloadLock.acquire_lock( maxsleep=2, **kwargs ):
                pass

    # Make sure it times out trying to lock the table if a table lock is held
    with Psycopg2Connection() as conn:
        cursor = conn.cursor()
        cursor.execute( "LOCK TABLE calibfile_downloadlock" )
        with pytest.raises( RuntimeError, match="Failed to get the lock on calibfile_downloadlock after" ):
            with CalibratorFileDownloadLock.acquire_lock( maxsleep=2, **kwargs ):
                pass

    # Make sure that we steal the lock on a heartbeat timeout
    # (This is perverse, since we're setting the heartbeat timeout
    #  to be less than the heartbeat interval.  Never do that in real life,
    #  as it totally subverts the locking.)
    # A better test would be to make subprocess that gets the lock, then
    #  kill -9 it to make it die without releasing its lock....  That would
    #  mean figuring out more about python multiprocessing than I know
    #  off of the top of my head.
    with CalibratorFileDownloadLock.acquire_lock( heartbeat_interval=20, **kwargs ):
        with CalibratorFileDownloadLock.acquire_lock( heartbeat_timeout=2, **kwargs ):
            pass
