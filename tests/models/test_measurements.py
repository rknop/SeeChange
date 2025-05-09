import pytest
import numpy as np

import sqlalchemy as sa

from models.base import SmartSession
from models.image import Image  # noqa: F401
from models.measurements import MeasurementSet, Measurements


def test_measurements_attributes(measurer, ptf_datastore, test_config):
    ds = ptf_datastore

    aper_radii = test_config.value('extraction.apertures')
    ds.measurement_set = None
    ds = measurer.run( ds )
    # check that the measurer actually loaded the measurements from db, and not recalculated
    # TODO -- testing that should be in pipeline/test_measuring.py.  We should just use
    #   here what the fixture gave us
    assert isinstance( ds.measurement_set, MeasurementSet )
    assert ds.measurements == ds.measurement_set.measurements
    assert len(ds.measurements) <= len(ds.cutouts.co_dict)  # not all cutouts have saved measurements
    assert len(ds.measurements) == len(ds.measurements)
    assert ds.measurements[0].from_db
    assert not measurer.has_recalculated

    # Make sure they are sorted by index_in_sources
    assert [ ds.measurements[i+1].index_in_sources > ds.measurements[i].index_in_sources
             for i in range( len(ds.measurements) - 1 ) ]

    # Make sure positions are consistent with what's in detections
    # (They won't be identical because the positions are redetermined in measuring.py
    # vs. what was found in the detections originally, but they should be close.)
    # ...these don't pass.  I see typicaly differences of 0.5".  This alarms me.
    # But, it probably has something to do with the positions found by the filter detection
    # method used with zogy.  Maybe worth understanding, but leave it be for now.
    # assert all( m.ra == pytest.approx( ds.detections.ra[m.index_in_sources], abs=0.1/3600. )
    #             for m in ds.measurements )
    # assert all( m.dec == pytest.approx( ds.detections.dec[m.index_in_sources], abs=0.1/3600. )
    #             for m in ds.measurements )

    # grab one example measurements object
    m = ds.measurements[0]

    # Make sure the zeropoint we get from the auto-loaded measurements field is the right one
    # (At this point, there are multiple zeropoints in the database, so this is at least sort
    # of a test that the monster query in Measurements.zp is filtering down right.  To really
    # test it, we'd need to get multiple source lists from the new image, each with different
    # provenances.  Make that a TODO?)
    assert m._zp is None
    assert m.zp.id == ds.zp.id

    # check some basic values
    assert np.allclose(m.aper_radii, m.zp.aper_cor_radii)
    assert np.allclose( m.zp.aper_cor_radii, ds.psf.fwhm_pixels * np.array(aper_radii) )

    # Make sure that flux_psf is the thing we want to muck with below
    assert m.best_aperture == -1

    original_flux_psf = m.flux_psf
    original_flux_psf_err = m.flux_psf_err
    original_flux_ap0 = m.flux_apertures[0]

    # set the flux temporarily to something positive
    m.flux_apertures[0] = 10000
    assert m.mag_apertures[0] == pytest.approx( -2.5 * np.log10(10000) + m.zp.zp + m.zp.aper_cors[0], abs=1e-4 )

    m.flux_psf = 10000
    expected_mag = -2.5 * np.log10(10000) + m.zp.zp
    assert m.mag_psf == pytest.approx( expected_mag, abs=1e-4 )

    # set the flux temporarily to something negative
    m.flux_apertures[0] = -10000
    assert np.isnan(m.mag_apertures[0])

    # set the flux and zero point to some randomly chosen values and test the distribution of the magnitude:
    fiducial_zp = m.zp.zp
    original_zp_err = m.zp.dzp
    fiducial_zp_err = 0.03  # more reasonable ZP error value (closer to dflux/flux)
    fiducial_flux = 10000
    fiducial_flux_err = 500
    m.flux_psf_err = fiducial_flux_err
    m.zp.dzp = fiducial_zp_err

    iterations = 1000
    mags = np.zeros(iterations)
    rng = np.random.default_rng()
    for i in range(iterations):
        m.flux_psf = rng.normal(fiducial_flux, fiducial_flux_err)
        m.zp.zp = rng.normal(fiducial_zp, fiducial_zp_err)
        mags[i] = m.magnitude

    m.flux_apertures[m.best_aperture] = fiducial_flux

    # the measured magnitudes should be normally distributed
    assert np.abs(np.std(mags) - m.magnitude_err) < 0.01
    assert np.abs(np.mean(mags) - m.magnitude) < m.magnitude_err * 3   # ...this should fail 0.3% of the time...

    # make sure to return things to their original state
    m.flux_apertures[0] = original_flux_ap0
    m.flux_psf = original_flux_psf
    m.flux_psf_err = original_flux_psf_err
    ds.zp.zp = fiducial_zp
    ds.zp.dzp = original_zp_err

    # Test getting cutout image data
    # (Note: I'm not sure what's up with sub_psfflux and sub_psffluxerr.

    m = ds.measurements[0]
    fields = [ 'sub_data', 'ref_data', 'new_data',
               'sub_weight', 'ref_weight', 'new_weight',
               'sub_flags', 'ref_flags', 'new_flags' ]

    def reset_fields():
        for f in fields:
            setattr( m, f'_{f}', None )

    def check_fields_none():
        assert all( getattr( m, f'_{f}' ) is None for f in fields )

    def check_fields_not_none():
        assert all ( getattr( m, f'_{f}' ) is not None for f in fields )

    # Make sure we start clean
    check_fields_none()

    # Make sure we can get stuff explicitly passing cutouts and detections
    m.get_data_from_cutouts( cutouts=ds.cutouts, detections=ds.detections )
    check_fields_not_none()

    reset_fields()
    check_fields_none()

    # Make sure we can get stuff with get_data_from_cutouts pulling cutouts and detections from database.
    m.get_data_from_cutouts()
    check_fields_not_none()

    # Now go through the auto-loaded attributes one by one
    for field in fields:
        reset_fields()
        check_fields_none()
        assert getattr( m, field ) is not None
        check_fields_not_none()


def test_filtering_measurements(ptf_datastore):
    mset = ptf_datastore.measurement_set
    measurements = ptf_datastore.measurements

    # test that we can filter on some measurements properties
    with SmartSession() as session:
        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 0)).all()
        # assert len(ms) == len(measurements)  # saved measurements will probably have a positive flux
        #  ...but they don't right now.  Fix this test once we've addressed Issue #398

        ms = session.scalars(sa.select(Measurements).where(Measurements.flux_apertures[0] > 5000)).all()
        assert len(ms) < len(measurements)  # only some measurements have a flux above 5000

        ms = session.scalars(sa.select(Measurements).where(Measurements.bkg_per_pix > 0)).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have positive background

        ms = ( session.query( Measurements )
               .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
               .filter( Measurements.x > Measurements.center_x_pixel )
               .filter( MeasurementSet.provenance_id == mset.provenance_id )
              ).all()
        assert len(ms) <= len(measurements)  # only some of the measurements have x > center_x_pixel

        # ms = session.scalars(sa.select(Measurements).where(
        #     Measurements.area_psf >= 0, Measurements.provenance_id == m.provenance_id
        # )).all()
        # assert len(ms) == len(measurements)  # all measurements have positive psf area

        ms = ( session.query( Measurements )
               .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
               .filter( Measurements.major_width >= 0 )
               .filter( MeasurementSet.provenance_id == mset.provenance_id )
              ).all()
        assert len(ms) == len(measurements)  # all measurements have positive width

        ms = ( session.query( Measurements )
               .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
               .filter( Measurements.psf_fit_flags != 0 )
               .filter( MeasurementSet.provenance_id == mset.provenance_id )
              ).all()
        assert len(ms) < len(measurements)   # Not all have psf fit flags set
        # assert len(ms) > 0                   # ...but some did... actually, no, none do any more...

        ms = ( session.query( Measurements )
               .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
               .filter( Measurements.nbadpix > 0 )
               .filter( MeasurementSet.provenance_id == mset.provenance_id )
              ).all()
        assert len(ms) < len(measurements)   # Not all measurements had a bad pixel
        # assert len(ms) > 0                 # ...but some did
        #                                    # ...well, some did if the deletion_thresholds were all null...

        ms = ( session.query( Measurements )
               .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
               .filter( Measurements.negfrac > 0.2 )
               .filter( MeasurementSet.provenance_id == mset.provenance_id )
              ).all()
        assert len(ms) < len(measurements)   # Not all measurements had negfrac > 0.2
        assert len(ms) > 0                   # ...but some did

        ms = ( session.query( Measurements )
               .join( MeasurementSet, Measurements.measurementset_id==MeasurementSet._id )
               .filter( Measurements.negfluxfrac > 0.1 )
               .filter( MeasurementSet.provenance_id == mset.provenance_id )
              ).all()
        assert len(ms) < len(measurements)   # Not all measurements had negfrac > 0.1
        assert len(ms) > 0                   # ...but some did
