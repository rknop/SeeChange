import uuid
import textwrap

from psycopg import sql

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as sqlUUID
from sqlalchemy.ext.declarative import declared_attr

from models.base import PGDB, Base, UUIDMixin, HasProvenance, SeeChangeBase
from models.provenance import Provenance
from models.image import Image
from models.datafile import DataFile
from util.util import asUUID
from util.logger import SCLogger

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

    You can't set fields of this object after creation.

    Properties available include:
       id
       provenance_id
       *_id
       *_image_id
       *_datafile_id
       *_image
       *_datafile

    where * is any of (zero, dark, flat, linearity, fringe, illumination)

    *_id points to the datafile_or_image_id table, and is probably not
     useful to you.

    The only method of general use is insert(), which puts the
    CalibratorSet into the database.

    """

    __tablename__ = 'calibrator_sets'

    @declared_attr
    def __table_args__( cls ): # noqa: N805
        return (
            sa.UniqueConstraint( 'provenance_id', 'dark_id', 'zero_id', 'flat_id',
                                 'linearity_id', 'fringe_id', 'illumination_id' ),
        )

    calibitems = ( 'zero', 'dark', 'flat', 'linearity', 'fringe', 'illumination' )

    @property
    def id( self ):
        # Never auto-make ids for this class, to avoid database duplication.
        return self._id

    @id.setter
    def id( self, val ):
        raise RuntimeError( "You cannot set the id of a CalibratorSet" )


    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete="CASCADE", name='calibrator_set_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="Provenance id for this object"
    )

    zero_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_zero_fkey' ),
        nullable=True,
        index=True
    )

    dark_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_dark_fkey' ),
        nullable=True,
        index=True
    )

    flat_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_flat_fkey' ),
        nullable=True,
        index=True
    )

    linearity_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_linearity_fkey' ),
        nullable=True,
        index=True
    )

    fringe_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_fringe_fkey' ),
        nullable=True,
        index=True
    )

    illumination_id = sa.Column(
        sqlUUID,
        sa.ForeignKey( 'datafile_or_image_id', ondelete='RESTRICT', name='calibrator_set_illumination_fkey' ),
        nullable=True,
        index=True
    )

    @staticmethod
    def create( create_uudify=True, _id=None, id=None, provenance_id=None, **kwargs ):
        """Make a CalibratorSet.

        calibitems is ( 'zero', 'dark', 'flat', 'linearity', 'fringe', 'illumination' )

        Parameters
        ----------
          create_uudify : bool, default True
            Pass id and/or _id through asUUID?

          id, _id: str, UUID; optional
            Set the id of the new CalibratorSet.  If both id and _id are
            given, id is ignored (with a warning).

          provenance_id: Provenance or str; optional
            If given, set the provenance id of the new CalibratorSet
            from this.

          *_id: str or UUID; optional
            * can be any of calibitems.  Directly set the field.  This
            should probably only be used internally.  If you include
            any of these, you cannot include any *_image or *_datafile.

          *_image: Image, str, or UUID; optional
            * can be any of calibitems.  Sets the Image for this
            calibitem to the passed image, or passed image id.

          *_datafile: Datafile, str, or UUID; optional
            * can be any of calibitems.  Sets the DataFile for this
            calibitem to the passed datafile, or passed datafile id.
            Can't give both *_image and *_datafile for the same
            calibitem.

        Returns
        -------
          CalibratorSet

        """

        createargs = { 'create_uudify': create_uudify }
        createassociates = {}

        if id is not None:
            if _id is not None:
                SCLogger.warning( "Gave both _id and id to CalibratorSet.create; ignoring id" )
            else:
                createargs[ '_id' ] = asUUID( id )

        if _id is not None:
            createargs[ '_id' ] = asUUID( _id )

        if provenance_id is not None:
            if isinstance( provenance_id, Provenance ):
                createargs[ 'provenance_id' ] = provenance_id.id
            else:
                createargs[ 'provenance_id' ] = str( provenance_id )

        for attr in CalibratorSet.calibitems:
            if f'{attr}_id' in kwargs:
                if kwargs[f'{attr}_id'] is not None:
                    createargs[ f'{attr}_id' ] = asUUID( kwargs[ f'{attr}_id' ] )
                del kwargs[ f'{attr}_id' ]

            if f'{attr}_image' in kwargs:
                if kwargs[f'{attr}_image'] is not None:
                    if f'{attr}_id' in createargs:
                        raise RuntimeError( f"Can't give both {attr}_id and {attr}_image to CalibratorSet.create" )
                    if isinstance( kwargs[f'{attr}_image'], Image ):
                        createassociates[ f'{attr}_image_id' ] = kwargs[ f'{attr}_image' ].id
                    else:
                        createassociates[ f'{attr}_image_id' ] = asUUID( kwargs[ f'{attr}_image' ] )
                del kwargs[ f'{attr}_image' ]

            if f'{attr}_datafile' in kwargs:
                if kwargs[f'{attr}_datafile'] is not None:
                    if f'{attr}_id' in createargs:
                        raise RuntimeError( f"Can't give both {attr}_id and {attr}_datafile to CalibratorSet.create" )
                    if f'{attr}_image_id' in createassociates:
                        raise RuntimeError( f"Can't specify both {attr}_image and {attr}_datafile" )
                    if isinstance( kwargs[f'{attr}_datafile'], DataFile ):
                        createassociates[ f'{attr}_datafile_id' ] = kwargs[ f'{attr}_datafile' ].id
                    else:
                        createassociates[ f'{attr}_datafile_id' ] = asUUID( kwargs[ f'{attr}_datafile' ] )
                del kwargs[ f'{attr}_datafile' ]

        for prop in ( 'created_at', 'modified' ):
            if prop in kwargs:
                createargs[ prop ] = kwargs[ prop ]
                del kwargs[ prop ]

        if len(kwargs) > 0:
            raise ValueError( f"Unknown parameters passed to CalibratorSet.create: {list(kwargs.keys())}" )

        return CalibratorSet( ok_to_instantiate=True, _createassociates=createassociates, **createargs )


    def __getattribute__( self, attr ):
        """Handle getting *_image_id, *_image, *_datafile_id, and *_datafile."""

        if not isinstance( attr, str ):
            raise TypeError( "attr must be a str" )

        cache = object.__getattribute__( self, '_file_cache' )

        if attr in cache:
            return cache[ attr ]

        elif ( len(attr) > 3 ) and ( attr[-3:] == "_id" ):
            if ( len(attr) > 9 ) and ( attr[-9:] == "_image_id" ):
                attrbase = attr[:-9]
                if attrbase in CalibratorSet.calibitems:
                    isimage, imageid = self._get_datafile_or_image_id( attrbase )
                    if not isimage:
                        return None
                    else:
                        return imageid
            elif ( len(attr) > 12 ) and ( attr[-12:] == "_datafile_id" ):
                attrbase = attr[:-12]
                if attrbase in CalibratorSet.calibitems:
                    isimage, datafileid = self._get_datafile_or_image_id( attrbase )
                    if ( isimage is None ) or isimage:
                        return None
                    else:
                        return datafileid

            else:
                return object.__getattribute__( self, attr )

        elif ( len(attr) > 6 ) and ( attr[-6:] == "_image" ):
            attrbase = attr[:-6]
            if attrbase in CalibratorSet.calibitems:
                with PGDB() as pgdb:
                    isimage, imageid = self._get_datafile_or_image_id( attrbase, pgdb=pgdb )
                    if not isimage:
                        return None
                    else:
                        img = Image.get_by_id( imageid, pgdb=pgdb )
                        cache[ attr ] = img
                        return img

            else:
                return object.__getattribute__( self, attr )

        elif ( len(attr) > 9 ) and ( attr[-9:] == "_datafile" ):
            attrbase = attr[:-9]
            if attrbase in CalibratorSet.calibitems:
                with PGDB() as pgdb:
                    isimage, datafileid = self._get_datafile_or_image_id( attrbase, pgdb=pgdb )
                    if ( isimage is None ) or isimage:
                        return None
                    else:
                        df = DataFile.get_by_id( datafileid, pgdb=pgdb )
                        cache[ attr ] = df
                        return df

            else:
                return object.__getattribute__( self, attr )

        else:
            return object.__getattribute__( self, attr )

    def __setattr__( self, attr, value ):
        if ( ( ( len(attr) > 9 ) and ( attr[-9:] == "_image_id" ) ) or
             ( ( len(attr) > 12 ) and ( attr[-12:] == "_datafile_id" ) )
             ( ( len(attr) > 6 ) and ( attr[-6:] == "_image" ) ) or
             ( ( len(attr) > 9 ) and ( attr[-9:] == "_datafile" ) )
            ):
            raise RuntimeError( "You can't set a CalibratorSet *_image[_id] or *_datafile[_id] "
                                "outside of the constructor." )
        else:
            object.__setattr__( self, attr, value )


    def _get_datafile_or_image_id( self, field, pgdb=None ):
        cache = object.__getattribute__( self, '_file_cache' )

        # TODO : consistency check?
        if f'{field}_image_id' in cache:
            return True, cache[ f'{field}_image_id' ]
        elif  f'{field}_image' in cache:
            return True, cache[ f'{field}_image' ].id
        elif f'{field}_datafile_id' in cache:
            return False, cache[ f'{field}_datafile_id' ]
        elif f'{field}_datafile' in cache:
            return False, cache[ f'{field}_datafile' ].id

        else:
            dfid = object.__getattribute__( self, f"_{field}_id" )
            if dfid is None:
                return None, None

            with PGDB( dictcursor=True, pgdb=pgdb ) as pgdb:
                rows = pgdb.execute( sql.SQL( textwrap.dedent(
                    """\
                    SELECT datafile_id, image_id, _id
                    FROM datafile_or_image_id
                    WHERE _id={dfid}
                    """
                ) ).format( dfid=dfid ) )
                if len(rows) > 1:
                    raise RuntimeError( "Database corruption in datafile_or_image_id, this should not happen." )
                elif len(rows) == 0:
                    raise RuntimeError( f"Failed to get datafile_or_image_id {dfid} from database" )
                else:
                    if ( rows[0][0] is None ) and ( rows[0][1] is None ):
                        raise RuntimeError( f"There is a datafile_or_image_id row ({rows[0][3]}) with "
                                            f"both image_id and datafile_id NULL.  This is bad." )
                    if rows[0][0] is None:
                        return True, rows[0][1]

                    else:
                        return False, rows[0][0]

    def __init__( self, _ok_to_instantiate=False, _createassociates=None, **kwargs ):
        """Do not call the constructor, call CalibratorSet.create()"""

        if not _ok_to_instantiate:
            raise RuntimeError( "Don't instantiate CalibratorSet directly, call Calibratorset.create" )

        if any( f'{field}_id' in kwargs for field in CalibratorSet.calibitems ):
            if ( _createassociates is not None ) and ( len( _createassociates ) > 0 ):
                raise RuntimeError( "CalibratorSet constructor got both *_id fields and _createassociates" )

            object.__setattr__( self, '_createassociates', None )
            for field in CalibratorSet.calibitems:
                if f'{field}_id' in kwargs:
                    object.__setattr__( self, f'{field}_id', kwargs[f'{field}_id'] )
                    del kwargs[ f'{field}_id' ]

            self._file_cache = {}

        else:
            object.__setattr__( self._file_cache, _createassociates )

        SeeChangeBase.init( self, **kwargs )
        self._file_cache = {}


    def generate_provenance( self, process='secure_calibrators', pgdb=None, nocommit=False, set_provenance_id=True ):
        # This is a weird object, in that the object can generate the provenance,
        #   rather than the process that produces it.  (Almost.)
        upstreams = []
        with PGDB.pgdb( pgdb ) as pgdb:
            for attr in CalibratorSet.calibitems:
                img = getattr( self, f'{attr}_image' )
                df = getattr( self, f'{attr}_datafile' )
                if ( img is not None ) and ( df is not None ):
                    raise RuntimeError( f"Calibrator set has both a {attr}_image and {attr}_datafile; this is bad." )
                if img is not None:
                    if img.provenance_id is None:
                        return None
                    upstreams.append( Provenance.get( img.provenance_id, must_exist=True, pgdb=pgdb ) )
                elif df is not None:
                    if df.provenance_id is None:
                        return None
                    upstreams.append( Provenance.get( df.provenance_id, must_exist=True, pgdb=pgdb ) )

            prov = Provenance( process=process, upstreams=upstreams, pgdb=pgdb )
            if set_provenance_id:
                self.provenance_id = prov.id
            if not nocommit:
                prov.insert_if_neeed( pgdb=pgdb )

        return prov


    def existing_entry( self, pgdb=None ):
        # This is more complicated than you'd think because there may
        #   exist datafile_or_image_id entries that we don't know about.
        #   If we have a non-None, _createassociates attribute, that
        #   means we know the relevant image and datafile ids, but not
        #   the {attr}_id fields that point to datafile_or_image_id.  In
        #   that case, gotta do a bunch of joins to find the
        #   datafile_or_image_id values *if* those already exists.

        # ...of course, if we have no provenance, then we can't exist.
        #   (Take that, Descartes!)
        if object.__getattribute__( self, 'provenance_id' ) is None:
            return None

        # If any of the {attr}_id fields is set, then they should all be set.
        # (They only get set on insert or loading from the database.)
        if any( object.__getattribute__( self, f'{i}_id' ) is not None for i in CalibratorSet.calibitems ):
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT c._id FROM calibrator_sets
                WHERE provenance_id={provid}
                """
            ) ).format( isimage=self._isimage, provid=self.provenance_id )
            for field in CalibratorSet.calibitems:
                val = object.__getattribute__( self, f'{field}_id' )
                if val is None:
                    q += sql.SQL( "  AND {field}_id IS NULL\n" ).format( field=sql.SQL(field) )
                else:
                    q += sql.SQL( "  AND {field}_id=val\n" ).format( field=sql.SQL(field), val=val )

        else:
            # In this case, we don't actually know the {attr}_id fields so we can't
            #   just search the calibrator_set table.  Instead, we have to
            #   do ten million joins.
            cache = object.__getattribute__( self, '_file_cache' )
            selectq = sql.SQL( "SELECT c._id" )
            joinq = sql.SQL( "" )
            where = sql.SQL( "WHERE c.provenance_id={provid}\n" ).format( provid=self.provenance_id )

            for field in CalibratorSet.calibitems:
                if ( ( f'{field}_image_id' in cache ) or
                     ( f'{field}_datafile_id' in cache )
                    ):
                    if ( ( f'{field}_image_id' in cache ) and
                         ( f'{field}_datafile_id' in cache )
                        ):
                        raise ValueError( f"CalibratorSet _cache has both "
                                          f"{field}_image_id and {field}_datafile_id " )

                    joinq += sql.SQL( textwrap.dedent(
                        """\
                        INNER JOIN datafile_or_image_id {field}_join ON c.{field}_id={field}_join._id
                        """
                    ) ).format( field=field )

                    if f'{field}_image_id' in cache:
                        selectq += sql.SQL( ", {c}.{field}_id AS {field}_image_id" ).format( field=field )
                        where += sql.SQL( "  AND {field}_join.image_id={imid}\n"
                                         ).format( field=sql.SQL(field),
                                                   imid=cache[f'{field}_image_id'] )
                    else:
                        selectq += sql.SQL( ", {field}_join.datafile_id AS {field}_datafile_id" ).format( field=field )
                        where += sql.SQL( "  AND {field}_join.datafile_id={dfid}\n"
                                         ).format( field=sql.SQL(field),
                                                   imid=cache[f'{field}_datafile_id'] )

            q = sql.SQL( "{selectq}\nFROM calibrator_sets\n{joinq}\n{where}"
                        ).format( selectq=selectq, joinq=joinq, where=where )


        with PGDB( pgdb, dictcursor=True ) as pgdb:
            rows = pgdb.execute( q )
            if len(rows) > 1:
                raise RuntimeError( f"Got multiple rows for query:\n{q.as_string()}" )
            if len(rows) == 1:
                return rows[0]
            else:
                return None


    def insert( self, exist_ok=False, pgdb=None ):
        """Insert the CalibratorSet into the database.

        If exist_ok is True and the entry already exists, update the
        object's id to the existing one, and also update the *_id fields
        if appropriate.

        Parameters
        ----------
          exist_ok : bool, default False
            If True, and a CalibratorSet already exists that's the same
            as this one, then just update the id (and, if appropriate,
            other fields).  If False, raise an exception.

          pgdb: PGDB or None
            Database connection.  If not given, a new one will be opened
            and closed.

        """

        with PGDB( pgdb ) as pgdb:
            if self.provenance_id is None:
                self.generate_provenance( pgdb=pgdb )

            existing_entry = self.existing_entry( pgdb=pgdb )
            # Rollback to free any shared locks we may have picked up;
            # Otherwise, we'll get deadlocks below.
            pgdb.rollback()
            if existing_entry is not None:
                if not exist_ok:
                    raise RuntimeError( "CalibSet already exists." )
                else:
                    setattr( self, '_id', existing_entry["_id"] )
                    for attr in CalibratorSet.calibitems:
                        field = f'{attr}_id'
                        if field in existing_entry:
                            object.__setattr__( self, field, existing_entry[field] )

            else:
                myid = uuid.uuid4()
                insertfields = sql.SQL( "_id" )
                insertvalues = sql.SQL( "{myid}" ).format( myid=myid )
                setvalues = { '_id': myid }
                searchattrs = []
                searchqs = []
                insertqs = []
                cache = object.__getattribute__( self, '_file_cache' )
                for attr in CalibratorSet.items:
                    if ( f'{attr}_image_id' in cache ) or ( f'{attr}_datafile_id' in cache ):
                        if ( ( f'{attr}_image_id' in cache ) and
                             ( f'{attr}_datafile_id' in cache )
                            ):
                            raise RuntimeError( f"Error, CalibratorSet cache has both "
                                                f"{attr}_image_id and {attr}_datafile_id" )
                        thisid = uuid.uuid4()
                        insertfields += sql.SQL( ",{attr}" ).format( attr=sql.Identifier(f'{attr}_id' ) )
                        setvalues[ f'{attr}_id' ] = thisid
                        searchattrs.append( attr )

                        if f'{attr}_image_id' in cache:
                            searchqs.append( sql.SQL( textwrap.dedent(
                                """\
                                SELECT _id FROM datafile_or_image_id
                                WHERE image_id={imid} AND datafile_id IS NULL
                                """
                            ) ).format( imid=f'{attr}_image_id' ) )
                            insertqs.append( sql.SQL( textwrap.dedent(
                                """\
                                INSERT INTO datafile_or_image_id( _id, datafile_id, image_id )
                                VALUES({uuid}, NULL, {imid})
                                """
                            ) ).format( uuid=thisid, imid=cache[f'{attr}_image_id'] ) )
                        else:
                            searchqs.append( sql.SQL( textwrap.dedent(
                                """\
                                SELECT _id FROM datafile_or_image_id
                                WHE
                                RE image_id IS NULL AND datafile_id={dfid}
                                """
                            ) ).format( dfid=f'{attr}_datafile_id' ) )
                            insertqs.append( sql.SQL( textwrap.dedent(
                                """\
                                INSERT INTO datafile_or_image_id( _id, datafile_id, image_id )
                                VALUES({uuid}, {dfid}, NULL)
                                """
                            ) ).format( uuid=thisid, dfid=cache[f'{attr}_datafile_id'] ) )

                try:
                    # To avoid race conditions, lock the tables table and search again
                    pgdb.execute_nofetch( "LOCK TABLE calibrator_sets" )
                    pgdb.execute_nofetch( "LOCK TABLE datafile_or_image_id" )
                    existing_entry = self.existing_entry( pgdb=pgdb )
                    if existing_entry is not None:
                        pgdb.rollback()
                        if not exist_ok:
                            raise RuntimeError( "CalibSet already exists." )
                        else:
                            setattr( self, '_id', existing_entry["_id"] )
                            for attr in CalibratorSet.calibitems:
                                field = f'{attr}_id'
                                if field in existing_entry:
                                    object.__setattr__( self, field, existing_entry[field] )
                            setattr( self, '_cache', None )

                    else:
                        # Insert while we hold our lock
                        for searchq, insertq, attr in zip( searchqs, insertqs, searchattrs ):
                            rows, _cols = pgdb.execute( searchq )
                            if len(rows) == 0:
                                pgdb.execute_nofetch( insertq )
                            else:
                                setvalues[ f'{attr}_id' ] = rows[0][0]
                        insertvalues += sql.SQL(",").join( setvalues[ f'{attr}_id' ] for attr in searchattrs )
                        q = sql.SQL( "INSERT INTO calibrator_sets({fields}} VALUES({values})"
                                    ).format( fields=insertfields, values=insertvalues )
                        pgdb.execute_nofetch( q )
                        pgdb.commit()
                        for attr, val in setvalues:
                            object.__setattr__( self, attr, val )
                finally:
                    # Really make sure locks are released
                    pgdb.rollback()


    def delete_from_disk_and_database( self, remove_folder=False, remove_downstreams=False, archive=False, pgdb=None ):
        if remove_downstreams:
            raise NotImplementedError( "Removing downstreams of a CalibratorSet not implemented." )

        with PGDB( pgdb ) as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM calibrator_sets WHERE _id={uuid}" ).format( uuid=self.id ) )
            pgdb.commit()
