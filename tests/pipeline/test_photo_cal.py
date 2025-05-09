import os
import pytest
import pathlib
import uuid

import numpy as np
import matplotlib.pyplot as plt

from models.base import CODE_ROOT
from models.zero_point import ZeroPoint

from tests.conftest import SKIP_WARNING_TESTS


def test_decam_photo_cal( decam_datastore_through_wcs, blocking_plots ):
    ds = decam_datastore_through_wcs
    photometor = ds._pipeline.photometor

    photometor.run(ds)
    assert photometor.has_recalculated

    if os.getenv('MAKE_PLOTS', False):  # skip this on github actions
        fig = plt.figure( figsize=(6, 8), dpi=150, layout='tight' )
        ax = fig.add_subplot( 2, 1, 1 )

        # This plot had too many points and it looked ugly.  So, pick out at most the top 100
        # low-dzp values
        xvals = photometor.catdata['MAG_BP'] - photometor.catdata['MAG_RP']
        yvals = photometor.individual_zps
        dyvals = photometor.individual_zpvars

        if len(xvals) > 100:
            dex = np.argsort( dyvals )[:100]
            xvals = xvals[dex]
            yvals = yvals[dex]
            dyvals = dyvals[dex]

        ax.errorbar( xvals, yvals, dyvals, linestyle='none', marker='.' )
        ax.plot( ax.get_xlim(), [ds.zp.zp, ds.zp.zp] )
        ax.set_xlabel( "Gaia m_BP - m_RP" )
        ax.set_ylabel( "zp" )
        ax.set_ylim( ( ds.zp.zp-0.3, ds.zp.zp+0.3 ) )

        ax = fig.add_subplot( 2, 1, 2 )
        ax.errorbar( photometor.individual_mags, photometor.individual_zps, np.sqrt(photometor.individual_zpvars),
                     linestyle='none', marker='.' )
        ax.plot( ax.get_xlim(), [ds.zp.zp, ds.zp.zp] )
        ax.set_xlabel( "Gaia m_G" )
        ax.set_ylabel( "zp" )
        ax.set_ylim( ( ds.zp.zp-0.3, ds.zp.zp+0.3 ) )

        ofpath = pathlib.Path( CODE_ROOT ) / 'tests/plots/test_decam_photo_cal.svg'
        plt.show(block=blocking_plots)
        fig.savefig( ofpath )

    # WORRY : zp + apercor (for the first aperture) is off from the
    # aperture-specific zeropoint that the lensgrinder pipeline
    # calculated for this image by 0.13 mags.  That was calibrated to
    # either DECaPS or PanSTARRS (investigate this), and it's
    # entirely possible that it's the lensgrinder zeropoint that is
    # off. <--- that comment was written for a different image.
    # investigate if it's still true for the image we're looking
    # at now.
    assert ds.zp.zp == pytest.approx( 30.228, abs=0.01 )
    assert ds.zp.dzp == pytest.approx( 4.93e-7, rel=0.1 )   # That number is absurd, but oh well
    assert ds.zp.aper_cor_radii == pytest.approx( [ 4.171, 8.343, 12.514, 20.857 ], abs=0.01 )
    assert ds.zp.aper_cors == pytest.approx( [ -0.264, -0.069, -0.025, 0. ], abs=0.01 )

    # Verify that it doesn't rerun if it doesn't have to
    ds.save_and_commit()
    ds.zp = None
    photometor.run(ds)
    assert not photometor.has_recalculated
    assert isinstance( ds.zp, ZeroPoint )

    # Verify that it will rerun if a parameter is changed
    ds.zp = None
    photometor.pars.test_parameter = uuid.uuid4().hex
    ds._provtag = None
    ds.edit_prov_tree( 'photocal', photometor.pars.get_critical_pars() )
    photometor.run(ds)
    assert photometor.has_recalculated


def test_warnings_and_exceptions(decam_datastore_through_wcs):
    ds = decam_datastore_through_wcs
    photometor = ds._pipeline.photometor

    if not SKIP_WARNING_TESTS:
        photometor.pars.inject_warnings = 1
        ds._provtag = None
        ds._pipeline.make_provenance_tree( ds )

        with pytest.warns(UserWarning) as record:
            photometor.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'photocal'." in str(w.message) for w in record)

    photometor.pars.inject_warnings = 0
    photometor.pars.inject_exceptions = 1
    ds.zp = None
    ds._provtag = None
    ds._pipeline.make_provenance_tree(ds)
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'photocal'."):
        ds = photometor.run( ds )
