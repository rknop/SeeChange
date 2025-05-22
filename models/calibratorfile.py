import os
import time
import math
import datetime
import contextlib
import random
import uuid
import threading
import multiprocessing

import psycopg2.extras

import sqlalchemy as sa
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.schema import UniqueConstraint

from models.base import Base, UUIDMixin, Psycopg2Connection
from models.enums_and_bitflags import CalibratorTypeConverter, CalibratorSetConverter, FlatTypeConverter

from util.logger import SCLogger


class CalibratorFile(Base, UUIDMixin):
    __tablename__ = 'calibrator_files'

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        server_default=sa.sql.elements.TextClause(str(CalibratorTypeConverter.convert( 'unknown' )) ) ,
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):  # noqa: N805
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=True,
        server_default=sa.sql.elements.TextClause( str(CalibratorTypeConverter.convert('unknown')) ),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._calibrator_set )

    @calibrator_set.expression
    def calibrator_set( cls ):  # noqa: N805
        return sa.case( CalibratorSetConverter.dict, value=cls._calibrator_set )

    @calibrator_set.setter
    def calibrator_set( self, value ):
        self._calibrator_set = CalibratorSetConverter.convert( value )

    _flat_type = sa.Column(
        sa.SMALLINT,
        nullable=True,
        index=True,
        doc="Type of flat (unknown, observatory_supplied, sky, twilight, dome), or None if not a flat"
    )

    @hybrid_property
    def flat_type( self ):
        return FlatTypeConverter.convert( self._flat_type )

    @flat_type.inplace.expression
    @classmethod
    def flat_type( cls ):
        return sa.case( FlatTypeConverter.dict, value=cls._flat_type )

    @flat_type.setter
    def flat_type( self, value ):
        self._flat_type = FlatTypeConverter.convert( value )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Instrument this calibrator image is for"
    )

    sensor_section = sa.Column(
        sa.Text,
        nullable=False,
        index=True,
        doc="Sensor Section of the Instrument this calibrator image is for"
    )

    image_id = sa.Column(
        sa.ForeignKey( 'images._id', ondelete='CASCADE', name='calibrator_files_image_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the image (if any) that is this calibrator'
    )

    datafile_id = sa.Column(
        sa.ForeignKey( 'data_files._id', ondelete='CASCADE', name='calibrator_files_data_file_id_fkey' ),
        nullable=True,
        index=True,
        doc='ID of the miscellaneous data file (if any) that is this calibrator'
    )

    validity_start = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file becomes valid.  If None, this '
              ' calibrator is valid from the beginning of time.' )
    )

    validity_end = sa.Column(
        sa.DateTime,
        nullable=True,
        index=True,
        doc=( 'The time (of exposure acquisition) for exposures for '
              ' which this calibrator file is no longer.  If None, this '
              ' calibrator is valid to the end of time.' )
    )

    def __repr__(self):
        return (
            f'<CalibratorFile('
            f'id={self.id}, '
            f'set={self.calibrator_set}, '
            f'type={self.type}, '
            f'image_id={self.image_id}, '
            f'datafile_id={self.datafile_id}, '
            f'for {self.instrument} section {self.sensor_section}'
            f'>'
        )


# This next table is kind of an ugly hack put in place
#   to deal with race conditions; see Instrument.preprocessing_calibrator_files

class CalibratorFileDownloadLock(Base, UUIDMixin):
    __tablename__ = 'calibfile_downloadlock'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            UniqueConstraint( '_type', '_calibrator_set', '_flat_type', 'instrument', 'sensor_section',
                              name='calibfile_downloadlock_unique',
                              postgresql_nulls_not_distinct=True ),
        )

    _type = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=False,
        server_default=sa.sql.elements.TextClause( str(CalibratorTypeConverter.convert( 'unknown' )) ),
        doc="Type of calibrator (Dark, Flat, Linearity, etc.)"
    )

    @hybrid_property
    def type( self ):
        return CalibratorTypeConverter.convert( self._type )

    @type.expression
    def type( cls ):  # noqa: N805
        return sa.case( CalibratorTypeConverter.dict, value=cls._type )

    @type.setter
    def type( self, value ):
        self._type = CalibratorTypeConverter.convert( value )

    _calibrator_set = sa.Column(
        sa.SMALLINT,
        nullable=False,
        index=False,
        server_default=sa.sql.elements.TextClause( str(CalibratorTypeConverter.convert('unknown')) ),
        doc="Calibrator set for instrument (unknown, externally_supplied, general, nightly)"
    )

    @hybrid_property
    def calibrator_set( self ):
        return CalibratorSetConverter.convert( self._calibrator_set )

    @calibrator_set.expression
    def calibrator_set( cls ):  # noqa: N805
        return sa.case( CalibratorSetConverter.dict, value=cls._calibrator_set )

    @calibrator_set.setter
    def calibrator_set( self, value ):
        self._calibrator_set = CalibratorSetConverter.convert( value )

    _flat_type = sa.Column(
        sa.SMALLINT,
        nullable=True,
        index=False,
        doc="Type of flat (unknown, observatory_supplied, sky, twilight, dome), or None if not a flat"
    )

    @hybrid_property
    def flat_type( self ):
        return FlatTypeConverter.convert( self._flat_type )

    @flat_type.inplace.expression
    @classmethod
    def flat_type( cls ):
        return sa.case( FlatTypeConverter.dict, value=cls._flat_type )

    @flat_type.setter
    def flat_type( self, value ):
        self._flat_type = FlatTypeConverter.convert( value )

    instrument = sa.Column(
        sa.Text,
        nullable=False,
        index=False,
        doc="Instrument this calibrator image is for"
    )

    sensor_section = sa.Column(
        sa.Text,
        nullable=True,
        index=False,
        doc="Sensor Section of the Instrument this calibrator image is for"
    )

    def __repr__( self ) :
        return ( f"CalibratorFileDownloadLock("
                 f"id={self.id}, "
                 f"calibrator_set={self.calibrator_set}, "
                 f"type={self.type}, "
                 f"flat_type={self.flat_type}, "
                 f"for {self.instrument} section {self.sensor_section})" )


    @classmethod
    def update_lock_heartbeat( cls, instrument, section, calibset, calibtype, flattype, pipe ):
        """Update the heartbeat described in acquire_lock below."""

        # Some of the SCLogger.debug comments below are useful when debugging the lock
        #   mechanism itself, but are too spammy for general use (even with debug)
        #   Thus, they are commented out, but not deleted so it's easy to put them back.

        # SCLogger.debug( f"Heartbeat process starting with instrument={instrument}, section={section}, "
        #                 f"calibset={calibset}, calibtype={calibtype}, flattype={flattype}" )

        calibset = CalibratorSetConverter.to_int( calibset )
        calibtype = CalibratorTypeConverter.to_int( calibtype )
        flattype = FlatTypeConverter.to_int( flattype )

        while True:
            if pipe.poll( timeout=5 ):
                msg = pipe.recv()
                if msg == "exit":
                    # This next line is too much even for debug, unless debugging the lock mechanism itself
                    # SCLogger.debug( "Heartbeat process exiting." )
                    return

            # SCLogger.debug( "Updating heartbeat." )
            with Psycopg2Connection() as conn:
                cursor = conn.cursor()
                q = ( f"UPDATE calibfile_downloadlock SET modified=%(now)s "
                      f"WHERE instrument=%(inst)s "
                      f"  AND sensor_section{' IS NULL' if section is None else '=%(sec)s'} "
                      f"  AND _calibrator_set{' IS NULL' if calibset is None else '=%(calibset)s'} "
                      f"  AND _type{' IS NULL' if calibtype is None else '=%(calibtype)s'} "
                      f"  AND _flat_type{' IS NULL' if flattype is None else '=%(flattype)s'} " )
                cursor.execute( q, { 'inst': instrument, 'sec': section, 'calibset': calibset,
                                     'calibtype': calibtype, 'flattype': flattype,
                                     'now': datetime.datetime.now( tz=datetime.UTC ) } )
                conn.commit()


    @classmethod
    @contextlib.contextmanager
    def acquire_lock( cls, instrument, section, calibset, calibtype, flattype=None,
                      heartbeat_interval=5, heartbeat_timeout=15, maxsleep=40 ):
        """Get a lock on updating/adding Calibrators of a given type for a given instrument/section.

        This class method should *only* be called as a context manager
        ("with").  Do NOT call this method when you are holding open a
        database connection, because this method could potentially take
        a long time to return (as it waits for another process to finish
        a big download).

        OK.  This is highly unpleasant.  Here are the constraints:

          * We don't want multiple processes all trying to download the
            same calibrator file at the same time.  Not only will this
            lead to contention, this will slow everything down as a big
            (tens of MB) file is downloaded many times when it only
            needs to be downloaded once.

          * We don't want to hold a database connection open while we're
            doing the download, because downloads take long enough that
            that violates the principle of keeping database connections
            open for only short periods of time in order to avoid
            exhausting server resources.  This scotches database locks
            (even row locks) as a solution.  (It's worse than it sounds,
            because not only will the processes doing downloading be
            holding open connections, but so will all the other
            processes waiting for the table lock.)

          * A cooperative, "I promise to free this when I'm done" lock
            isn't good enough.  Even with all the try/finally blocks in
            the world, you'll hit a situation were the computer crashes,
            or the process is killed, while it's in the middle of
            downloading, and it never actually frees the manual
            cooperative lock.  (This is what we tried before, and, yeah,
            we ended up with a bunch of dead locks sitting around.
            You've probably seen this with things like git, where there
            are dead locks sitting around, and you just manually
            override.  We can't manually override in an automated
            pipeline.)

        So what do we do?  A cooperative "I promise to free this when
        I'm done" lock, only with you having to renew the promise
        constantly.  If you go to get a lock, see that somebody else has
        it, but the promise hasn't been renewed recently enough, you can
        grab it.

        (This DOES assume that every computer running the pipeline, and
        the database server, are all time-synced.  Please use ntp!  If
        the clocks are off by seconds, then the timeouts may be too fast
        or too slow.)

        What a mess.

        Parameters
        ----------
        instrument: str
           The instrument

        section: str
           The sensor section, or None (meaning an instrument-global file)

        calibset: str
           The calibrator set

        calibtype: str
           The calibrator type (e.g. 'linearity', 'flat', etc.)

        flattype: str, default None
           The flat type if calibtype is 'flat'

        heartbeat_interval: int, default 5
           Send a "no, really, I have the lock!" heart beat at intervals
           of this many seconds.

        hearbeat_timeout: int, default 15
           If the heartbeat on a lock hasn't been updated in this many
           seconds, assume it's stale and can be ignored.

        maxsleep: int, default 40
           Keep trying for at most this many seconds before finally
           giving up and failing to get the lock.

        """

        sleepmin = 1.
        sleepsigma = 0.5
        sleepmax = 3.  # ...why, yes, I have sleepmax and maxsleep and they mean different things.  What's your point?
        totsleep = 0.
        # Use os.urandom() to see the rng because if we just use the
        #  random module stuff, it will use the system clock.  A bunch
        #  of processes may well hit this line at the same time and get
        #  the same random seed.
        random.seed( os.urandom(4) )

        whereclause = ( f"WHERE instrument=%(inst)s "
                        f"  AND sensor_section{' IS NULL' if section is None else '=%(sec)s'} "
                        f"  AND _calibrator_set{' IS NULL' if calibset is None else '=%(calibset)s'} "
                        f"  AND _type{' IS NULL' if calibtype is None else '=%(calibtype)s'} "
                        f"  AND _flat_type{' IS NULL' if flattype is None else '=%(flattype)s'}" )
        subdict = { 'inst': instrument,
                    'sec': section,
                    'calibset': CalibratorSetConverter.to_int( calibset ),
                    'calibtype': CalibratorTypeConverter.to_int( calibtype ),
                    'flattype': FlatTypeConverter.to_int( flattype )
                   }

        gotlock = False
        mypipe = None
        process = None
        try:
            SCLogger.debug( f"Trying to get lock for {instrument} {section} calibset={calibset} "
                            f"calibtype={calibtype} flattype={flattype}" )

            # To start: try to get a lock on the calibfiles_downloadlock table itself,
            #   and, if we get it, see if the row we're after exists.  If not,
            #   create it.  Otherwise, sleep, and then try again... until the
            #   modified date on the row is too old, at which point we're just
            #   going to assume control of it.
            done = False
            while not done:
                tsleep = min( sleepmax, sleepmin + math.fabs( random.normalvariate( mu=0., sigma=sleepsigma ) ) )
                with Psycopg2Connection() as conn:
                    cursor = conn.cursor( cursor_factory=psycopg2.extras.RealDictCursor )
                    try:
                        cursor.execute( "LOCK TABLE calibfile_downloadlock NOWAIT" )
                    except psycopg2.errors.LockNotAvailable:
                        if totsleep > maxsleep:
                            raise RuntimeError( f"Failed to get the lock on calibfile_downloadlock after "
                                                f"{totsleep:.1f} seconds." )
                        # Next line is useful debugging the lock mechanism, but too spammy for general use
                        # SCLogger.debug( f"sleeping {tsleep:.2f} seconds waiting for table lock" )
                        time.sleep( tsleep )
                        totsleep += tsleep
                        continue

                    now = datetime.datetime.now( tz=datetime.UTC )
                    cursor.execute( f"SELECT * FROM calibfile_downloadlock {whereclause}", subdict )
                    rows = cursor.fetchall()
                    if len(rows) != 0:
                        # Somebody else has the lock.  Be suspicious.  See if the hearbeat has been updated.
                        if now - rows[0]['modified'] < datetime.timedelta( seconds=heartbeat_timeout ):
                            # OK, OK, the claim has been refreshed recently, we'll be nice.
                            if totsleep > maxsleep:
                                raise RuntimeError( f"Failed to claim the calibfile_downloadlock row for "
                                                    f"{instrument} {section} calibset={calibset} calibtype={calibtype} "
                                                    f"flattype={flattype} after {totsleep:.1f}s" )
                            time.sleep( tsleep )
                            # Next line is useful debugging the lock mechanism, but too spammy for general use
                            # SCLogger.debug( f"sleeping {tsleep:.2f} seconds waiting for row not to be there" )
                            totsleep += tsleep
                            continue
                        else:
                            # The lock is too old.  Piss on it to claim it for ourselves.
                            newdict = subdict.copy()
                            newdict['now'] = now
                            SCLogger.debug( "Existing lock is too old, claiming it." )
                            cursor.execute( f"UPDATE calibfile_downloadlock SET modified=%(now)s {whereclause}",
                                            newdict )
                            conn.commit()
                            done = True
                    else:
                        # The row didn't exist!  Make it!
                        SCLogger.debug( "Creating lock row." )
                        insertdict = subdict.copy()
                        insertdict['id'] = uuid.uuid4()
                        cursor.execute( "INSERT INTO calibfile_downloadlock(_id,instrument,sensor_section,"
                                        "                                   _calibrator_set,_type,_flat_type ) "
                                        "VALUES (%(id)s,%(inst)s,%(sec)s,%(calibset)s,%(calibtype)s,%(flattype)s)",
                                        insertdict )
                        conn.commit()
                        done = True

            # Start a subprocess to regularly update the "modified" field of the
            #   table so nobody else does what... well, what we might just have done.
            SCLogger.debug( "Starting lock heartbeat process." )
            # We have to use a thread, not a process, for this, because we may ourselves
            #   be running in a daemonic process (launched from a multiprocessing Pool
            #   by ExposureLauncher (pipeline/exposure_launcher.py)).  Daemonic processes
            #   can't spawn their own subprocesses.  However, threads should just be fine
            #   here, because the whole reason downloading takes a long time is that it
            #   is subject to long i/o waits, which is when threads work well.
            mypipe, theirpipe = multiprocessing.Pipe()
            process = threading.Thread( target=CalibratorFileDownloadLock.update_lock_heartbeat,
                                        kwargs={ 'instrument': instrument, 'section': section,
                                                 'calibset': calibset, 'calibtype': calibtype,
                                                 'flattype': flattype, 'pipe': theirpipe } ) #,
            process.start()

            gotlock = True

            # Yield and let whoever called us play with their shiny new lock

            yield True

        finally:
            # Tell the heartbeat process to stop doing its thing
            if ( process is not None ) and ( mypipe is not None ):
                mypipe.send( "exit" )
                process.join()

            # If we got the lock, we have to delete it
            if gotlock:
                with Psycopg2Connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute( f"DELETE FROM calibfile_downloadlock {whereclause}", subdict )
                    conn.commit()
