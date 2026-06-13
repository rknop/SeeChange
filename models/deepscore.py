import textwrap

from psycopg import sql

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declared_attr

from models.base import Base, UUIDMixin, SeeChangeBase, PGDB, HasBitFlagBadness
from models.enums_and_bitflags import DeepscoreAlgorithmConverter
from models.measurements import MeasurementSet


class DeepScoreSet( Base, UUIDMixin, HasBitFlagBadness ):
    # A DeepScoreSet is a way of having a single upstream for things that depend on
    #   a whole set of DeepScores (like a FakeAnalysis).

    __tablename__ = 'deepscore_sets'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            UniqueConstraint('measurementset_id', '_algorithm', 'provenance_id', name='_deepscoreset_unique_uc'),
        )

    measurementset_id = sa.Column(
        sa.ForeignKey( 'measurement_sets._id', ondelete="CASCADE", name='score_set_meas_set_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the measurement set that this deepscore set is associated with"
    )

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete="CASCADE", name='score_set_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the provenance of this deepscore set."
    )

    _algorithm = sa.Column(
        sa.SMALLINT,
        nullable=False,
        doc=("Integer which represents which of the ML/DL algorithms was used "
             "for this object. Also specifies necessary parameters for a given "
             "algorithm." )
    )

    @hybrid_property
    def algorithm(self):
        return DeepscoreAlgorithmConverter.convert( self._algorithm )

    @algorithm.expression
    def algorithm(cls):  # noqa: N805
        return sa.case( DeepscoreAlgorithmConverter.dict, value=cls._algorithm )

    @algorithm.setter
    def algorithm( self, value ):
        self._algorithm = DeepscoreAlgorithmConverter.convert( value )

    @property
    def deepscores( self ):
        if self._deepscores is None:
            with PGDB() as pgdb:
                q = sql.SQL( textwrap.dedent(
                    """\
                    SEELCT * FROM deepscore
                    WHERE deepscoreset_id={me}
                    ORDER BY index_in_sources
                    """
                ) ).format( me=self.id )
                rows = pgdb.execute( q )
                self._deepscores = [ DeepScore(**row) for row in rows ]

        return self._deepscores

    @deepscores.setter
    def deepscores( self, val ):
        if ( not isinstance( val, list ) ) or ( not all( isinstance( d, DeepScore ) for d in val ) ):
            raise TypeError( "deepscores must be a list of DeepScore" )
        self._deepscores = val
        for d in self._deepscores:
            d.deepscoreset_id = self.id


    def __init__( self, *args, **kwargs ):
        SeeChangeBase.__init__( self )
        self._deepscores = None
        self.set_attributes_from_dict( kwargs )

    @orm.reconstructor
    def init_on_load( self ):
        SeeChangeBase.init_on_load( self )
        self._deepscores = None



    @classmethod
    def get_rb_cut( cls, method ):
        """Return the nominal cutoff for 'real' objects for a given method."""

        method = DeepscoreAlgorithmConverter.to_string( method )
        cuts = {
            'random': 0.5,
            'allperfect': 0.99,
            'RBbot-quiet-shadow-131-cut0.55': 0.55    # Dunno if this is a good cutoff, but it's what we use in tests
        }
        if method not in cuts:
            raise ValueError( f"Unknown deepscore method {method}" )

        return cuts[ method ]


    def get_upstream_ids( self, pgdb=None ):
        """Return the id upstreams of this DeepScoreSet object.

        Will be the MeasurementSet the DeepScoreSet object is associated with.

        """
        return [ ( MeasurementSet, self.measurementset_id ) ]

    def get_downstream_ids( self, pgdb=None ):
        """Return the downstreams of this DeepScoreSet object.

        Will be all DeepScore objects that are members of this set, plus
        any FakeAnalysis objects on injected-fakes subtractions that started with
        the same image and ref that this object's ancestors started with, and that
        went through the pipeline with the same parameters.

        """

        from models.fakeset import FakeAnalysis
        with PGDB( pgdb ) as pgdb:
            q = sql.SQL( "SELECT _id FROM fake_analysis WHERE orig_deepscore_set_id={me}" ).format( me=self.id )
            rows = pgdb.execute( q )
            downstreams = [ ( FakeAnalysis, row[0] ) for row in rows ]

            q = sql.SQL( "SELECT _id FROM deepscores WHERE deepscoreset_id={me}" ).format( me=self.id )
            rows = pgdb.execute( q )
            downstreams.extend( [ ( DeepScore,row[0] ) for row in rows ] )

        return downstreams


class DeepScore(Base, UUIDMixin, HasBitFlagBadness):
    """Contains the Deep Learning/Machine Learning algorithm score assigned to
    the corresponding measurements object. Each DeepScore object contains the
    result of a single algorithm, which is identified by the ___ attribute.
    """

    __tablename__ = 'deepscores'

    @declared_attr
    def __table_args__( cls ):  # noqa: N805
        return (
            UniqueConstraint('deepscoreset_id', 'index_in_sources', name='_deepscore_unique_uc'),
        )

    info = sa.Column(
        JSONB,
        nullable=True,
        server_default=None,
        doc=(
            "Optional additional information on this DeepScore. "
        )
    )

    deepscoreset_id = sa.Column(
        sa.ForeignKey( 'deepscore_sets._id', ondelete="CASCADE", name="score_score_set_id_fkey" ),
        nullable=False,
        index=True,
        doc="ID of the deepscore set this deepscore is a member of."
    )

    index_in_sources = sa.Column(
        sa.Integer,
        nullable=False,
        index=True,
        doc=( "Index in the source list (of detections in the difference image) that corresponds to the "
              "Measurements object this DeepScore is associated with." )
    )

    score = sa.Column(
        sa.REAL,
        nullable=False,
        doc="The score determined by the ML/DL algorithm used for this object."
    )

    def __init__(self, *args, **kwargs):
        SeeChangeBase.__init__(self) # don't pass kwargs as they could contain
                                     # non-column key-values
        HasBitFlagBadness.__init__(self)

        # manually set all properties (columns or not)
        self.set_attributes_from_dict(kwargs)

    @orm.reconstructor
    def init_on_load(self):
        SeeChangeBase.init_on_load(self)


    def __repr__(self):
        return f"<DeepScore {self.id} from Set {self.deepscoreset_id} ; score={self.score}>"


    def get_upstream_ids(self, session=None):
        """Get the id of the DeepScoreSet this is a member of."""
        return [ ( DeepScoreSet, self.deepscoreset_id ) ]

    def get_downstream_ids(self, session=None):
        """DeepScore objects have no downstreams."""
        return []
