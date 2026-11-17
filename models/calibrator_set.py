import uuid
import textwrap
import types

from psycopg import sql

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as sqlUUID

from models.base import PGDB, Base, UUIDMixin, HasProvenance, SeeChangeBase
from models.provenance import Provenance
from models.image import Image
from models.datafile import DataFile
from util.util import asUUID
from util.logger import SCLogger

calibset_datafile_or_image_id = sa.Table(
    'calibset_datafile_or_image_id',
    Base.metadata,
    sa.Column( 'calibrator_set_id',
               sqlUUID,
               sa.ForeignKey( 'calibrator_sets._id', ondelete='CASCADE', name='cset_datafile_or_image_calibset_fkey' ),
               nullable=False,
               index=True ),
    sa.Column( 'field',
               sa.Text,
               nullable=False,
               index=False,
               doc="Name of the field (zero, dark, flat, etc.)" ),
    sa.Column( 'datafile_id',
               sqlUUID,
               sa.ForeignKey( 'data_files._id', ondelete='RESTRICT', name='cset_datafile_or_image_datafile_fkey' ),
               nullable=True,
               index=True ),
    sa.Column( 'image_id',
               sqlUUID,
               sa.ForeignKey( 'images._id', ondelete='RESTRICT', name='cset_datafile_or_image_image_fkey' ),
               nullable=True,
               index=True ),
    sa.UniqueConstraint( 'calibrator_set_id', 'datafile_id', 'image_id', name='cset_datafile_or_image_unique' ),
    sa.CheckConstraint( 'NOT ( datafile_id IS NULL AND image_id IS NULL )', name='cset_datafile_or_image_not_null' ),
    sa.CheckConstraint( 'datafile_id IS NULL OR image_id IS NULL', name='cset_datafile_or_image_null' )
)


class CalibratorSet( Base, UUIDMixin, HasProvenance ):
    """Represents a collection of datafiles and/or images used in preprocessing.

    You can't set fields of this object after creation.  (I mean, you
    can, python will let you, but you will cause total chaos if you do.)

    Properties available include:
       id
       provenance_id
       *_id
       *_image_id
       *_datafile_id
       *_image
       *_datafile

    where * is any of (zero, dark, flat, linearity, fringe, illumination)

    *_id points to the calibset_datafile_or_image_id table, and is probably not
     useful to you.

    The only method of general use is insert(), which puts the
    CalibratorSet into the database.

    THIS CLASS MAY NOT BE USED WITH SQLALCHEMY ORM STUFF, it doesn't
    have the right support built-in.  But you shouldn't be using any
    SQLAlchemy anywhere in new SeeChange code anyway, because we're
    trying to get rid of it.

    """

    __tablename__ = 'calibrator_sets'

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

    def __init__( self, id=None, _id=None, provenance_id=None, created_at=None, modified=None, **kwargs ):
        """Make a CalibratorSet.

        Parameters
        ----------
          create_uudify : bool, default True
            ...here for compatibility with UUIDMixin.create, but not used.
            Always treated as True.

          id, _id: str, UUID; optional
            Set the id of the new CalibratorSet.  If both id and _id are
            given, id is ignored.

          provenance_id: Provenance or str; optional
            If given, set the provenance id of the new CalibratorSet
            from this.

          created_at, modified: timezone-aware datetime.datetime or None
            Timestamps... don't set these, the database will.  I think.

          *_image: Image, str, or UUID; optional
            * can be any of calibitems.  Sets the Image for this
            calibitem to the passed image, or passed image id.

          *_datafile: Datafile, str, or UUID; optional
            * can be any of calibitems.  Sets the DataFile for this
            calibitem to the passed datafile, or passed datafile id.
            Can't give both *_image and *_datafile for the same
            calibitem.

            """

        self._images = {}
        self.image_ids = {}
        self._datafiles = {}
        self.datafile_ids = {}

        self._id = asUUID(_id) if _id is not None else asUUID(id) if id is not None else None
        self.provenance_id = provenance_id
        self.created_at = created_at
        self.modified = modified

        nukes = set()
        for key, value in kwargs.items():
            if key[-6:] == '_image':
                base = key[:-6]
                if isinstance( value, Image ):
                    self._images[base] = value
                    self.image_ids[base] = value.id
                else:
                    self.image_ids[base] = asUUID( value )
                nukes.add( key )
            elif key[-9:] == '_datafile':
                base = key[:-9]
                if base in self.image_ids.keys():
                    raise RuntimeError( f"Can't specify both {base}_image and {base}_datafile" )
                if isinstance( value, DataFile ):
                    self._datafiles[base] = value
                    self.datafile_ids[base] = value.id
                else:
                    self.datafile_ids[base] = asUUID( value )
                nukes.add( key )

        for key in nukes:
            del kwargs[ key ]
        if len(kwargs) > 0:
            raise ValueError( f"Unrecognized parameters passed to CalibratorSet: {list(kwargs.keys())}" )


        SeeChangeBase.__init__( self )

    @property
    def images( self ):
        return types.MappingProxyType( self._images )

    @property
    def datafiles( self ):
        return types.MappingProxyType( self._datafiles )

    def image( self, attr, pgdb=None ):
        if attr in self._images:
            return self._images[ attr ]

        elif attr in self.image_ids:
            self._images[ attr ] = Image.get_by_id( self.image_ids[attr], pgdb=pgdb )
            return self._images[ attr ]

        else:
            return None

    def datafile( self, attr, pgdb=None ):
        if attr in self._datafiles:
            return self._datafiles[ attr ]

        elif attr in self.datafile_ids:
            self._datafiles[ attr ] = DataFile.get_by_id( self.datafile_ids[attr], pgdb=pgdb )
            return self._datafiles[ attr ]

        else:
            return None

    def validate_consistency( self ):
        # If hitdb is False, this doesn't fully validate consistency

        boths = set()
        not_image = set()
        mismatch_image = set()
        missing_imid = set()
        not_datafile = set()
        mismatch_datafile = set()
        missing_dfid = set()
        for attr, image in self._images.items():
            if not isinstance( image, Image ):
                not_image.add( attr )
            elif attr not in self.image_ids:
                missing_imid.add( attr )
            elif image.id != self.image_ids[ attr ]:
                mismatch_image.add( attr )

        for attr, datafile in self._datafiles.items():
            if ( attr in self._images ) or ( attr in self.image_ids ):
                boths.add( attr )
            if not isinstance( datafile, DataFile ):
                not_datafile.add( attr )
            elif attr not in self.datafile_ids:
                missing_dfid.add( attr )
            elif datafile.id != self.datafile_ids[ attr ]:
                mismatch_datafile.add( attr )

        for attr, datafile_id in self.datafile_ids.items():
            if ( attr in self._images ) or ( attr in self.image_ids ):
                boths.add( attr )

        if any( len(x) > 0 for x in [ boths, not_image, mismatch_image, missing_imid, not_datafile,
                                         mismatch_datafile, missing_dfid ] ):
            # TODO: descriptive error message
            raise RuntimeError( "CalibratorSet not self-consistent." )


    def generate_provenance( self, process='secure_calibrators', pgdb=None, nocommit=False, set_provenance_id=True ):
        # This is a weird object, in that the object can generate the provenance,
        #   rather than the process that produces it.  (Almost.)
        self.validate_consistency()
        upstreams = set()
        with PGDB( pgdb ) as pgdb:
            for attr in self._images.keys():
                im = self.image( attr, pgdb=pgdb )
                upstreams.add( Provenance.get( im.provenance_id, must_exist=True, pgdb=pgdb ) )
            for attr in self._datafiles.keys():
                im = self.datafile( attr, pgdb=pgdb )
                upstreams.add( Provenance.get( im.provenance_id, must_exist=True, pgdb=pgdb ) )
            upstreams = list( upstreams )
                
            prov = Provenance( process=process, upstreams=upstreams, pgdb=pgdb )
            if set_provenance_id:
                self.provenance_id = prov.id
            if not nocommit:
                prov.insert_if_needed( pgdb=pgdb )

        return prov

    def existing_id( self, pgdb=None ):
        self.validate_consistency()

        if self.provenance_id is None:
            return None

        q = sql.SQL( "SELECT c._id FROM calibrator_sets c\n" )

        dex = 0
        for attr, imid in self.image_ids.items():
            q += sql.SQL( textwrap.dedent(
                """\
                INNER JOIN calibset_datafile_or_image_id {t} ON {t}.calibrator_set_id=c._id
                                                            AND {t}.field={field}
                                                            AND {t}.image_id={imid}
                """
            ) ).format( t=sql.Identifier( f't{dex}' ), field=attr, imid=imid )
            dex += 1
        for attr, dfid in self.datafile_ids.items():
            q += sql.SQL( textwrap.dedent(
                """
                INNER JOIN calibset_datafile_or_image_id {t} ON {t}.calibrator_set_id=c._id
                                                            AND {t}.field={field}
                                                            AND {t}.image_id={imid}
                """
            ) ).format( t=sql.Identifier( f't{dex}' ), field=attr, imid=imid )
            dex += 1

        q += sql.SQL( "WHERE provenance_id={provid}" ).format( provid=self.provenance_id )

        with PGDB( pgdb ) as pgdb:
            rows, _cols = pgdb.execute( q )
            if len( rows ) == 0:
                return None
            elif len( rows ) > 1:
                raise RuntimeError( f"Multiple redundant calibrator_sets in database: "
                                    f"{[row[0] for row in rows]}" )
            else:
                return rows[0][0]


    def insert( self, exist_ok=True, exist_mismatch_ok=False, pgdb=None ):
        """Insert the CalibratorSet into the database.

        If exist_ok is True and the entry already exists, update the
        object's id to the existing one.

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

        self.validate_consistency()

        with PGDB( pgdb ) as pgdb:
            if self.provenance_id is None:
                self.generate_provenance( pgdb=pgdb )

            existing_id = self.existing_id( pgdb=pgdb )
            # Release shared locks as fast as possible
            pgdb.rollback()
            if existing_id is not None:
                if not exist_ok:
                    raise RuntimeError( "CalibSet already exists." )
                elif ( not exist_mismatch_ok ) and ( self._id is not None ) and ( existing_id != self._id ):
                    raise RuntimeError( "CalibSet already exists with different id" )
                else:
                    self._id = existing_id
                    return

            else:
                myid = self._id if self._id is not None else uuid.uuid4()
                try:
                    # Lock tables, and to avoid race conditions, search again.
                    pgdb.execute_nofetch( "LOCK TABLE calibrator_sets" )
                    pgdb.execute_nofetch( "LOCK TABLE calibset_datafile_or_image_id" )
                    existing_id = self.existing_id( pgdb=pgdb )
                    if existing_id is not None:
                        pgdb.rollback()
                        if not exist_ok:
                            raise RuntimeError( "CalibSet already exists." )
                        elif ( not exist_mismatch_ok ) and ( self._id is not None ) and ( existing_id != self._id ):
                            raise RuntimeError( "CalibSet already exists with different id" )
                        else:
                            self._id = existing_id
                            return
                    else:
                        self._id = myid
                        pgdb.execute_nofetch( sql.SQL( "INSERT INTO calibrator_sets(_id, provenance_id) "
                                                       "VALUES({uuid}, {provid})"
                                                      ).format( uuid=myid, provid=self.provenance_id ) )
                        for attr, imid in self.image_ids.items():
                            pgdb.execute_nofetch( sql.SQL( textwrap.dedent(
                                """\
                                INSERT INTO calibset_datafile_or_image_id(calibrator_set_id, field, image_id)
                                VALUES ({setid}, {field}, {imid})
                                """
                            ) ).format( setid=self._id, field=attr, imid=imid ) )
                        for attr, dfid in self.datafile_ids.items():
                            pgdb.execute_nofetch( sql.SQL( textwrap.dedent(
                                """\
                                INSERT INTO calibset_datafile_or_image_id(calibrator_set_id, field, datafile_id)
                                VALUES ({setid}, {field}, {dfid})
                                """
                            ) ).format( setid=self._id, field=attr, dfid=dfid ) )
                        pgdb.commit()
                finally:
                    # Really make sure we release the locks
                    pgdb.rollback()


    @classmethod
    def get_by_id( cls, uuid, pgdb=None ):
        with PGDB( pgdb, dictcursor=True ) as pgdb:
            q = sql.SQL( textwrap.dedent(
                """\
                SELECT c._id, c.provenance_id, c.created_at, c.modified, j.field, j.image_id, j.datafile_id
                FROM calibrator_sets c
                LEFT JOIN calibset_datafile_or_image_id j ON j.calibrator_set_id=c._id
                WHERE c._id={uuid}
                """
            ) ).format( uuid=uuid )
            rows = pgdb.execute( q )

        kwargs = { '_id': rows[0]['_id'],
                   'provenance_id': rows[0]['provenance_id'],
                   'created_at': rows[0]['created_at'],
                   'modified': rows[0]['modified']
                  }
        for row in rows:
            if row['image_id'] is not None:
                kwargs[f'{row["field"]}_image'] = row['image_id']
            elif row['datafile_id'] is not None:
                kwargs[f'{row["field"]}_datafile'] = row['datafile_id']
            else:
                SCLogger.warning( "A row of calibset_datafile_or_iamge_id has both image and datafile NULL" )

        return CalibratorSet( **kwargs )


    def delete_from_disk_and_database( self, remove_folder=False, remove_downstreams=False, archive=False, pgdb=None ):
        if remove_downstreams:
            raise NotImplementedError( "Removing downstreams of a CalibratorSet not implemented." )

        with PGDB( pgdb ) as pgdb:
            pgdb.execute( sql.SQL( "DELETE FROM calibrator_sets WHERE _id={uuid}" ).format( uuid=self.id ) )
            pgdb.execute( sql.SQL( "DELETE FROM calibset_datafile_or_image_id WHERE calibrator_set_id={uuid}"
                                  ).format( uuid=self.id ) )
            pgdb.commit()
