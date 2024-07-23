import sqlalchemy as sa

from models.base import Base, AutoIDMixin

class Alert( Base, AutoIDMixin ):
    __tablename__ = 'alerts'

    measurement_id = sa.Column(
        sa.ForeignKey( 'measurements.id', ondelete='CASCADE', name='alerts_measurements_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the measurement that this alert was sent for"
    )

    time_sent = sa.Column(
        sa.DateTime(timezone=True),
        default=None,
        nullable=True,
        doc="UTC time the alert was sent; NULL=it (probably) wasn't sent"
    )
    
    provenance_id = sa.Column(
        sa.ForeignKey( 'provenances.id', ondelete="CASCADE", name='alerts_provenance_id_fkey' ),
        nullable=False,
        index=True,
        doc="ID of the provenance of this alert"
    )
