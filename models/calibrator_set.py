import uuid

from psycopg import sql

from models.base import PGDB, Base, UUIDMixin, HasProvenance


datafile_or_image_id = sa.Table(
    'datafile_or_image_id',
    Base.metadata,
    sa.Column( '_id', sqlUUID, primary_key=True ),
    sa.Column( 'datafile_id',
               sqlUUID,
               sa.ForeignKey( 'data_files._id', ondelete='CASCADE', name='datafile_or_image_datafile_fkey' ),
               nullable=True,
               index=True ),
    sa.Column( 'image_id',
               sqlUUID,
               sa.ForeignKey( 'images._id', ondelete='CASCADE', name='datafile_or_image_image_fkey' ),
               nullable=True,
               index=True ),
    sa.UniqueConstraint( 'datafile_id', 'image_id', name='datafile_or_image_unique' ),
    sa.CheckConstraint( 'NOT ( datafile_id IS NULL AND image_id IS NULL )', name='datafile_or_image_not_null' ),
    sa.CheckConstraint( 'datafile_id IS NULL OR image_id IS NULL', name='datafile_or_image_null' )
)


class CalibratorSet( Base, UUIDMixin, HasProvenance ):
    """Represents a collection of datafiles and/or images used in preprocessing.

    Outside this class, only ever set the x_datafile_id, and x_image_id
    fields, where x is one of zero, dark, flat, linarity, fringe, or
    illumination.  Only ever get those fields, and id.  Otherwise,
    things will break, horribly.

    """
    
    __tablename__ = 'calibrator_sets'

    @declared_attr
    def __table_args__( cls ): # noqa: N805
        return (
            sa.UniqueConstraint( 'provenance_id', 'dark_id', 'zero_id', 'flat_id',
                                 'linearity_id', 'fringe_id', 'illumination_id', '_is_image' ),
        )

    calibitmes = ( 'zero', 'dark', 'flat', 'linearity', 'fringe', 'illumination' )

    @property
    def id( self ):
        # Never auto-make ids for this class, to avoid database duplication.
        return self._id

    @id.setter
    def id( self, val ):
        raise RuntimeError( "You cannot set the id of a CalibratorSet" )
    

    zero_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_zero_fkey' )
        nullable=True,
        index=True
    )

    dark_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_dark_fkey' )
        nullable=True,
        index=True
    )

    flat_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_flat_fkey' )
        nullable=True,
        index=True
    )

    linearity_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_linearity_fkey' )
        nullable=True,
        index=True
    )

    fringe_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_fringe_fkey' )
        nullable=True,
        index=True
    )

    illumination_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_illumination_fkey' )
        nullable=True,
        index=True
    )

    _is_image = sa.Column(
        sa.SmallInteger,
        nullable=False,
        server_default='0',
        index=False
    )

    def _get_datafile_or_image_id( self, field ):
        dfid = getattr( self, f"_{field}_id" )
        if dfid is None:
            return None

        isimage = CalibratorSetIsImageBits.to_bitmask( field )  self._is_image

        with PGDB( dictcursor=True ) as pgdb:
            rows = pgdb.execute( sql.SQL( textwrap.dedent(

    """\
                SELECT datafile_id, image_id
                FROM datafile_or_image_id
                WHERE _id={dfid}
                """
            ) ).format( dfid=dfid ) )
            if len(rows) > 1:
                raise RuntimeError( "Database corruption in datafile_or_image_id, this should not happen." )
            elif len(rows) = 0:
                raise RuntimeError( "Failed to get datafile_or_image_id from database" )
            else:
                if ( ( ( rows[0][1] is None ) and ( not isimage ) or
                     ( ( rows[0][0] is None ) and isimage ) )
                    ):
                    raise ValueError( f"Iconsistency error; {field} is{'' if isimage else ' not'} "
                                      f"supposed to be an image but is{' not' if isimage else ''}" )

                return isimage, rows[0][1] if isimage else rows[0][0]


    @property
    def zero_image_id( self ):
        if self._zero_image_id is None:
            isimage, fileid = self._get_datafile_or_image( 'zero' )
            if not isimage:
                return None
            self._zero_image_id = fileid
        return self._zero_image_id

    @zero_image_id.setter
    def zero_image_id( self, val ):
        self.zero_image_id = val
        self._is_image |= CalibratorSetIsImageBits.to_bitmask( 'zero' )

    @property
    def zero_datafile_id( self ):
        if self._zero_datafile_id is None:
            isimage, fileid = self._get_datafile_or_image( 'zero' )
            if isimage:
                return None
            self._zero_datafile_id = fileid
        return self._zero_datafile_id

    @zero_datafile_id.setter
    def zero_datafile_id( self, val ):
        self.zero_datafile_id = val
        self._is_image &= 0xffffffff - CalibratorSetIsImageBits.to_bitmask( 'zero' )

    @property
    def dark_image_id( self ):
        if self._dark_image_id is None:
            isimage, fileid = self._get_datafile_or_image( 'dark' )
            if not isimage:
                return None
            self._dark_image_id = fileid
        return self._dark_image_id

    @dark_image_id.setter
    def dark_image_id( self, val ):
        self.dark_image_id = val
        self._is_image |= CalibratorSetIsImageBits.to_bitmask( 'dark' )

    @property
    def dark_datafile_id( self ):
        if self._dark_datafile_id is None:
            isimage, fileid = self._get_datafile_or_image( 'dark' )
            if isimage:
                return None
            self._dark_datafile_id = fileid
        return self._dark_datafile_id

    @dark_datafile_id.setter
    def dark_datafile_id( self, val ):
        self.dark_datafile_id = val
        self._is_image &= 0xffffffff - CalibratorSetIsImageBits.to_bitmask( 'dark' )

    @property
    def flat_image_id( self ):
        if self._flat_image_id is None:
            isimage, fileid = self._get_datafile_or_image( 'flat' )
            if not isimage:
                return None
            self._flat_image_id = fileid
        return self._flat_image_id

    @flat_image_id.setter
    def flat_image_id( self, val ):
        self.flat_image_id = val
        self._is_image |= CalibratorSetIsImageBits.to_bitmask( 'flat' )

    @property
    def flat_datafile_id( self ):
        if self._flat_datafile_id is None:
            isimage, fileid = self._get_datafile_or_image( 'flat' )
            if isimage:
                return None
            self._flat_datafile_id = fileid
        return self._flat_datafile_id

    @flat_datafile_id.setter
    def flat_datafile_id( self, val ):
        self.flat_datafile_id = val
        self._is_image &= 0xffffffff - CalibratorSetIsImageBits.to_bitmask( 'flat' )

    @property
    def fringe_image_id( self ):
        if self._fringe_image_id is None:
            isimage, fileid = self._get_datafile_or_image( 'fringe' )
            if not isimage:
                return None
            self._fringe_image_id = fileid
        return self._fringe_image_id

    @fringe_image_id.setter
    def fringe_image_id( self, val ):
        self.fringe_image_id = val
        self._is_image |= CalibratorSetIsImageBits.to_bitmask( 'fringe' )

    @property
    def fringe_datafile_id( self ):
        if self._fringe_datafile_id is None:
            isimage, fileid = self._get_datafile_or_image( 'fringe' )
            if isimage:
                return None
            self._fringe_datafile_id = fileid
        return self._fringe_datafile_id

    @fringe_datafile_id.setter
    def fringe_datafile_id( self, val ):
        self.fringe_datafile_id = val
        self._is_image &= 0xffffffff - CalibratorSetIsImageBits.to_bitmask( 'fringe' )

    @property
    def linearity_image_id( self ):
        if self._linearity_image_id is None:
            isimage, fileid = self._get_datafile_or_image( 'linearity' )
            if not isimage:
                return None
            self._linearity_image_id = fileid
        return self._linearity_image_id

    @linearity_image_id.setter
    def linearity_image_id( self, val ):
        self.linearity_image_id = val
        self._is_image |= CalibratorSetIsImageBits.to_bitmask( 'linearity' )

    @property
    def linearity_datafile_id( self ):
        if self._linearity_datafile_id is None:
            isimage, fileid = self._get_datafile_or_image( 'linearity' )
            if isimage:
                return None
            self._linearity_datafile_id = fileid
        return self._linearity_datafile_id

    @linearity_datafile_id.setter
    def linearity_datafile_id( self, val ):
        self.linearity_datafile_id = val
        self._is_image &= 0xffffffff - CalibratorSetIsImageBits.to_bitmask( 'linearity' )

    @property
    def illumination_image_id( self ):
        if self._illumination_image_id is None:
            isimage, fileid = self._get_datafile_or_image( 'illumination' )
            if not isimage:
                return None
            self._illumination_image_id = fileid
        return self._illumination_image_id

    @illumination_image_id.setter
    def illumination_image_id( self, val ):
        self.illumination_image_id = val
        self._is_image |= CalibratorSetIsImageBits.to_bitmask( 'illumination' )

    @property
    def illumination_datafile_id( self ):
        if self._illumination_datafile_id is None:
            isimage, fileid = self._get_datafile_or_image( 'illumination' )
            if isimage:
                return None
            self._illumination_datafile_id = fileid
        return self._illumination_datafile_id

    @illumination_datafile_id.setter
    def illumination_datafile_id( self, val ):
        self.illumination_datafile_id = val
        self._is_image &= 0xffffffff - CalibratorSetIsImageBits.to_bitmask( 'illumination' )

    @hybrid_property
    def is_image( self ):
        return CalibratorSetIsImageBits.bitmask_to_strings( self._is_image )

    @is_image.inplace.setter
    def is_image( self, value ):
        self._is_image = CalibratorSetIsImageBits.to_bitmask( value )

    def is_calibrator_image( self, which ):
        return bool( self._is_image & CalibratorSetIsImageBits.to_bitmask( which ) )


    def __init__( self ):
        SeeChangeBase.init( self )

        for calibitem in self.calibitems:
            setattr( self, f"_{calibitem}_image_id", None )
            setattr( self, f"_{calibitem}_datafile_id", None )


    def get_or_insert( self, pgdb=None ):
        """Either load the object id, or create all the necessary database tables.

        With allof x_image_id and x_datafile_id set (where for any par
        of x, at least one is None), and with self._is_image consistent
        with all that, put query the database to see if this is all
        already defined.  If so, set our _id field to the right thing.
        If not, insert all the necessary rows.

        """
        
        with PGDB( pgdb ) as pgdb:
            try:
                pgdb.execute( "LOCK TABLE datafile_or_image_id" )
                pgdb.execute( "LOCK TABLE calibrator_sets" )

                for calibitem in self.calibitems:
                    if CalibratorSetIsImageBits.to_bitmask( calibitem ) & self._is_image:
                        col = 'image_id'
                        fileid = gettatr( self, "_{calibitem}_image_id" )
                    else:
                        col = 'datafile_id'
                        fileid = gettatr( self, "_{calibitem}_datafile_id" )

                    if fielid is not None:
                        q = sql.SQL( "SELECT _id, datafile_id, image_id FROM datafile_or_image_id "
                                     "WHERE {col}={itemid}"
                                    ).format( col=col, itemid=itemid ) )
                        rows = pgdb.execute( q )
                        if len(rows) > 1:
                            raise RuntimeError( "Database corruption in datafile_or_image_id, this should not happen." )
                        elif len(rows) == 1:
                            setattr( self, f"{calibitem}_id", rows[0][0] )
                        else:
                            fileid = uuid.uuid4()
                            insq = sql.SQL( "INSERT INTO datafile_id(_id,{field}) VALUES({myid},{itemid})"
                                           ).format( field=field,
                                                     myid=fileids[calibitem],
                                                     itemid=itemid )
                            pgdb.execute( insq )
                            setattr( self, f"{calibitem}_id", fileid )

                q = sql.SQL( "SELECT _id FROM calibrator_sets" )
                where = " WHERE"
                for calibitem in self.calibitems:
                    itemid = getattr( self, f"{calibitem}_id" )
                    if itemid  is None:
                        q += sql.SQL( "{where} {col} IS NULL" ).format( where=sql.SQL(where),
                                                                        col=sql.Identifier(f'{calibitem}_id') )
                    else:
                        q += sql.SQL( "{where} {col}={val}"
                                     ).format( where.sql.SQL(where), col=sql.Identifier(f'{calibitem}_id')
                                               val=itemid )
                    where = " AND"

                rows = pgdb.execute( q )
                if len(rows) > 1:
                    # Unique constraint should stop this.
                    raise RuntimeError( "This should never happen" )
                if len(rows) == 0:
                    self._id = uuid.uuid4()
                    self.super().insert( pgdb=pgdb, nocommit=True )
                else:
                    self._id = rows[0][0]

                pgdb.commit()

            finally:
                # Make sure we don't hold the lock
                pgdb.rollback()
