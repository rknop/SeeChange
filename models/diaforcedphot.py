import numpy as np
import sqlalchemy as sa
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declared_attr

from models.base import Base, SeeChangeBase, UUIDMixin, HasBitFlagBadness
import models.object # OMG why did we name one of our classes Object????
from models.object import ObjectPosition
from models.image import Image
from models.enums_and_bitflags import measurements_badness_inverse


class DiaForcedPhot( Base, UUIDMixin, HasBitFlagBadness ):
    __tablename__ = 'dia_forced_photometry'

    @declared_attr
    def __table_args__( cls ):    # noqa: N805
        return (
            UniqueConstraint( 'object_id', 'subtraction_id', 'provenance_id', name='dia_forcedphot_unique' ),
        )

    object_id = sa.Column(
        sa.ForeignKey( 'objects._id', ondelete='RESTRICT', name='dia_forcedphot_object_id_fkey' ),
        nullable = False,
        index = True,
        doc = "ID of the object this is forced photometry for"
    )

    object_position_id = sa.Column(
        sa.ForeignKey( 'object_positions._id', ondelete='RESTRICT', name='dia_forcedphot_object_position_id_fkey' ),
        nullable = True,
        index = True,
        doc = "ID (if any) of the object position used for this forced photometry."
    )

    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances._id', ondelete='CASCADE', name='dia_forcedphot_provenance_id_fkey' ),
        nullable = False,
        index = True,
        doc = ( "ID of the Provenance of this forced photometry point" )
    )

    subtraction_id = sa.Column(
        sa.ForeignKey( 'images._id', ondelete='RESTRICT', name='dia_forcedphot_subtraction_id_fkey' ),
        nullable = False,
        index = True,
        doc = ( "ID of the subtraction this forced phot was performed on." )
    )

    # Note: the NaN default here is just because I had an existing database to which
    #   these columns were going to be added, so I didn't want the not null to kill
    #   it.  Yeah, I will need to manually go back and fix those columns later (if
    #   I decide I care).
    x = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        default = np.nan,
        server_default = sa.sql.elements.TextClause( "'NaN'" ),
        doc = ( "x position (0-indexed, n.0=center of pixel) on image given by subtraction_id" )
    )

    y = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        default = np.nan,
        server_default = sa.sql.elements.TextClause( "'NaN'" ),
        doc = ( "y position (0-indexed, n.0=center of pixel) on image given by subtraction_id" )
    )

    flux_psf = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        doc = ( "PSF flux, in dn.  Need to use the zeropoint to turn this into something standard. "
                "WARNING: right now for zogy we don't use the right PSF for phtometry!" )
    )

    flux_psf_err = sa.Column(
        sa.REAL,
        nullable = False,
        index = False,
        doc = ( "Uncertainty on psf_flux." ),
    )

    flux_apertures = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable = False,
        index = False,
        doc = ( "Aperture fluxes, in dn.  Does not include aperture corrections.  They are in the apertures "
                "defined in the zero_point record.  WARNING: these aperture correctoins are not right for zogy." )
    )

    flux_apertures_err = sa.Column(
        ARRAY( sa.REAL, zero_indexes=True ),
        nullable = False,
        index = False,
        doc = ( "Uncertainties on flux_apertures." )
    )

    def __init__( self, *args, **kwargs ):
        SeeChangeBase.__init__( self )
        HasBitFlagBadness.__init__( self )
        self.set_attributes_from_dict( kwargs )

    def _get_inverse_badness( self ):
        return measurements_badness_inverse

    def get_upstream_ids( self, pgdb=None ):
        upstrs = [ ( models.object.Object, self.object_id ),
                   ( Image, self.subtraction_id ) ]
        if self.object_position_id is not None:
            upstrs.append( [ ( ObjectPosition, self.object_position_id ) ] )
        return upstrs

    def get_downstream_ids( self, pgdb=None ):
        return []
