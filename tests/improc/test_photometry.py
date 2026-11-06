import pytest

import numpy as np
from astropy.io import fits
import photutils.psf

from improc.photometry import photometry_and_diagnostics


# This test has a slow startup because making the psf palette takes ~20-30 seconds, and
#   then we run sextractor (fast) psfex (slow).  Overall a ~1 min fixture setup.
def test_photometry_and_diagnostics( psf_palette, round_psf_palette ):

    for isround, palette in zip( ( False, True ), ( psf_palette, round_psf_palette ) ):
        # Test photometry on something where we know all the fluxes and shapes
        # TODO : run a test with a much noisier psf palette
        with fits.open( palette.imagename ) as hdul:
            image = hdul[0].data
        with fits.open( palette.flagsname ) as hdul:
            mask = np.full_like( hdul[0].data, False, dtype=bool )
            mask[ hdul[0].data != 0 ] = True
        with fits.open( palette.weightname ) as hdul:
            weight = hdul[0].data
            noise = 1. / np.sqrt( hdul[0].data )
            mask[ weight <= 0 ] = True

        mesh = np.meshgrid( palette.xpos, palette.ypos )
        positions = [ ( x, y ) for x, y, in zip( mesh[0].flatten(), mesh[1].flatten() ) ]
        apers = [ 1.25, 2.5, 5. ]

        # ****
        # For debugging purposes, make a new image that has renderings of the PSF
        # Look at psf_image.fits, psf_psfs.fits, and psf_resid.fits in ds9 or something.
        # 0-12,000 is a good stretch for the first two.
        # -600 to 600 on the third one makes it 10x brigthened relative to the first two.
        # rendered_psfs = np.zeros_like( image )
        # for x,y in zip( mesh[0].flatten(), mesh[1].flatten() ):
        #     palette.psf.add_psf_to_image( rendered_psfs, x, y, 200000 )
        # fits.writeto( 'psf_image.fits', image, overwrite=True )
        # fits.writeto( 'psf_psfs.fits', rendered_psfs, overwrite=True )
        # fits.writeto( 'psf_resid.fits', image - rendered_psfs, overwrite=True )
        # import pdb; pdb.set_trace()
        # ****

        meas = photometry_and_diagnostics( image, noise, mask, positions, apers, psfobj=palette.psf )

        # Sky noise in the fixture is 5.  The FWHM is ~3.5 pixels (at
        #   least for the non-round version).  So, roughly, the
        #   varaince from the sky is ~π(3.5^2)(25.) = 306.25, or noise
        #   17.5.  The image was made with a gain of 1, so poisson noise
        #   is sqrt(200000.) = 447.2.  The quadrature sum is... 447.6.
        #   Heh.  And, that's really an overestimate of the true sky
        #   contribution for PSF photometry.  Just use 447.2.
        # WORRY.  Fluxes are coming in systematically low, with a median
        #   of -1σ and a mean of -0.3σ.  I do not know why.  The reduced
        #   χ² is OK (~0.6), but I'm distressed at the systematic
        #   offset.  The rendered PSFs mostly look right visually,
        #   though they get a little ragged at the edge, even with the
        #   5th order spatial variation the fixture allowed PSFEx, and
        #   they all sum to 1 (identically, by construction).  I used
        #   17x17 patches when rendering the origiunal image.  For a
        #   Gaussian with σ=1.5 (FVWM 3.5), the enclosed flux is more
        #   than 0.999.
        # ...Maybe I'm just worrying too much, because the realtive
        #   error is like .6% in the worst case.
        #
        fluxen = np.array( [ m.flux_psf for m in meas ] )
        resid = fluxen - 200000.
        redresid = resid / 447.2
        dfluxen = np.array( [ m.flux_psf_err for m in meas ] )
        assert redresid.min() > -3.
        assert redresid.max() < 3.
        assert np.median( redresid ) == pytest.approx( 0., abs=0.25 )
        assert redresid.mean() == pytest.approx( 0., abs=0.5 )
        assert np.all( np.isclose( dfluxen, 447.2, atol=3. ) )

        assert all( [ m.flux_apertures[2] > m.flux_apertures[1] for m in meas ] )
        assert all( [ m.flux_apertures[1] > m.flux_apertures[0] for m in meas ] )
        assert all( [ m.flux_apertures[2] / m.flux_psf == pytest.approx( 0.998, abs=0.02 ) for m in meas ] )
        assert all( [ m.flux_apertures_err[2] > m.flux_apertures_err[1] for m in meas ] )
        assert all( [ m.flux_apertures_err[1] > m.flux_apertures_err[0] for m in meas ] )
        assert all( [ m.bkg_per_pix == 0 for m in meas ] )
        assert all( [ ( m.aper_radii == np.array([1.25, 2.5, 5.]) ).all() for m in meas ] )
        assert all( [ m.center_x_pixel == pytest.approx( m.x, abs=0.5 ) for m in meas ] )
        assert all( [ m.center_y_pixel == pytest.approx( m.y, abs=0.5 ) for m in meas ] )
        # PSFPalette did not line up all things right at the center of pixels
        assert np.median( np.abs( [ m.center_x_pixel - m.x for m in meas ] ) ) == pytest.approx( 0.3, abs=0.1 )
        assert np.median( np.abs( [ m.center_y_pixel - m.y for m in meas ] ) ) == pytest.approx( 0.3, abs=0.1 )
        # Gaussian fit positions should be very close to psf fit positions
        assert all( [ m.gfit_x == pytest.approx( m.x, abs=0.01 ) for m in meas ] )
        assert all( [ m.gfit_y == pytest.approx( m.y, abs=0.01 ) for m in meas ] )

        # With non-round PSF palette:
        #   σx = 1.25 + 0.5 * ( x - 512 ) / 1024
        #   σy = 1.75 - 0.5 * ( x - 512 ) / 1024
        #   θ = 0. + π/2 * ( y - 512 ) / 1024
        #
        # Expect, for pure gaussians, without the convolving effect of pixelization:
        #   Lower left : σx = 1.0, σy = 2.0, θ = -π/4
        #   Lower right : σx = 1.5, σy = 1.5, θ = -π/4
        #   Middle : σx = 1.25, σy = 1.75, θ = 0.
        #   Upper left : σx = 1.0, σy = 2.0, θ = π/4
        #   Upper right : σx = 1.5, οy = 1.5, θ = π/4
        #
        # (For round, σx = σy = 1.25 )

        # Empirically, the major widths are systemtically about 0.05
        #   bigger than the σy values, and minor widths are about 0.08
        #   bigger than the σx values.  (In the round case, both are
        #   about 0.08 high.)  Is this the result of the convolution
        #   from pixellization?  That *will* have an effect.  Perhaps if
        #   I fit GaussianPRF instead of GaussianPSF it would be better?
        #   Not going to worry about it.
        majwid = np.array( [ m.major_width for m in meas ] )
        minwid = np.array( [ m.minor_width for m in meas ] )
        if isround:
            assert np.all( np.isclose( majwid, 2.35482 * 1.25, atol=0.085 ) )
            assert np.all( np.isclose( minwid, 2.35482 * 1.25, atol=0.085 ) )
            # posang will not be meaningful here
        else:
            posang = np.array( [ m.position_angle for m in meas ] )
            xposes = np.array( [ m.x for m in meas ] )
            yposes = np.array( [ m.y for m in meas ] )
            assert np.all( np.isclose( majwid, 2.35482 * ( 1.75 - 0.5 * (xposes - 512) / 1024. ), atol=0.07 ) )
            assert np.all( np.isclose( minwid, 2.35402 * ( 1.25 + 0.5 * (xposes - 512) / 1024. ), atol=0.10 ) )
            assert np.all( np.isclose( posang, ( np.pi / 2. * (yposes - 512) / 1024. ), atol=0.02 ) )

            # assert all( [ np.abs( m.major_width - 2.35482 * ( 1.75 - 0.5 * (m.x-512) / 1024. ) ) < 0.07
            #               for m in meas ] )
            # assert all( [ np.abs( m.minor_width - 2.35482 * ( 1.25 + 0.5 * (m.x-512) / 1024. ) ) < 0.10
            #               for m in meas ] )
            # assert all( [ np.abs( m.position_angle - ( np.pi / 2 * (m.y-512) / 1024 ) ) < 0.005
            #               for m in meas ] )

            # NEXT.  Make sure background subtraction works right, at least for a very basic
            #  background.  Should get all the same photometry back out.

            bgmeas = photometry_and_diagnostics( image + 20., noise, mask, positions, apers,
                                                 psfobj=palette.psf, dobgsub=True, innerrad=17., outerrad=20. )
            assert all( [ b.bkg_per_pix == pytest.approx( 20., abs=1. ) for b in bgmeas ] )
            # For a ring of inner radius 17, outer radius 20, area~350 pixels,
            #   expect bg to be good to 5./sqrt(350) = ~0.27
            # The average of the averages should be good to ~0.27 / sqrt(100) = ~0.027
            assert np.mean( [ b.bkg_per_pix - 20. for b in bgmeas ] ) == pytest.approx( 0., abs=0.03 )
            assert np.std( [ b.bkg_per_pix - 20. for b in bgmeas ] ) == pytest.approx( 0.27, rel=0.3 )
            assert all( [ b.flux_psf / m.flux_psf == pytest.approx( 1.0, 0.001 )
                          for b, m in zip( bgmeas, meas ) ] )
            assert all( [ b.flux_psf_err / m.flux_psf_err == pytest.approx( 1.0, 0.001 )
                          for b, m in zip( bgmeas, meas ) ] )
            for i in range(3):
                assert all( [ b.flux_apertures[i] / m.flux_apertures[i] == pytest.approx( 1.0, 0.001 )
                              for b, m in zip( bgmeas, meas ) ] )
            assert all( [ b.center_x_pixel == m.center_x_pixel for b, m in zip( bgmeas, meas ) ] )
            assert all( [ b.center_y_pixel == m.center_y_pixel for b, m in zip( bgmeas, meas ) ] )
            assert all( [ b.x == pytest.approx( m.x, abs=0.001 ) for b, m in zip( bgmeas, meas ) ] )
            assert all( [ b.y == pytest.approx( m.y, abs=0.001 ) for b, m in zip( bgmeas, meas ) ] )
            assert all( [ np.abs( m.major_width - b.major_width ) < 0.0025 for m, b in zip( meas, bgmeas ) ] )
            assert all( [ np.abs( m.minor_width - b.minor_width ) < 0.0025 for m, b in zip( meas, bgmeas ) ] )
            assert all( [ np.abs( m.position_angle - b.position_angle ) < 2e-5 for m, b in zip( meas, bgmeas ) ] )


def test_diagnostics( diagnostic_injections ):
    # Make a noise image, and add certain specific failure modes to make sure the diagnostics
    #   that we expect to catch these do in fact catch these.
    # Note that widths and position angles were tested with all our gaussians
    #   on the PSFPalette in test_photometry_and_diagnostics
    rng = np.random.default_rng( seed=42 )
    bg = 100.
    skynoise = 5.
    image = rng.normal( loc=bg, scale=skynoise, size=(1024, 1024) )
    noise = np.full_like( image, skynoise )
    mask = np.full_like( image, False, dtype=bool )

    # We'll be working in 51 by 51 patches, so it's a little bigger than
    #   the 41x41 patch that we'll be cutting out.  These are all
    #   created in the diagnostic_injections fixture function.

    sigma = 2.0
    fwhm = 4.70964
    inner = 4. * fwhm
    outer = 5. * fwhm
    # Note!  slow_but_right here is actually not right, because the
    #   image simulator created its psfs using gaussian sampling rather
    #   than gaussian integration.  So, we're better approximating those
    #   psfs with slow_but_right=false!  Good news, because it's faster
    #   that way.
    images, masks, positions, expected = diagnostic_injections( sigma, wid=51, slow_but_right=False )
    for im, ma, po in zip( images, masks, positions ):
        image[ po[1]:po[1]+51, po[0]:po[0]+51 ] += im
        mask[ po[1]:po[1]+51, po[0]:po[0]+51 ] = mask[ po[1]:po[1]+51, po[0]:po[0]+51 ] | ma

    # Need positions at center of clips for use in photometry_and_diagnostics
    positions = [ ( i+51//2, j+51//2 ) for (i,j) in positions ]
    psf = photutils.psf.CircularGaussianPSF( flux=1., fwhm=fwhm )
    measurements = photometry_and_diagnostics( image, noise, mask, positions, [ fwhm, 2*fwhm ],
                                               photutils_psf=psf, fwhm_pixels=fwhm,
                                               dobgsub=True, innerrad=inner, outerrad=outer )
    for i, (exp, m) in enumerate( zip( expected, measurements ) ):
        for attr, val in exp.items():
            m_attr = getattr( m, attr )
            if isinstance( val, tuple ):
                if val[0] == 'lt':
                    assert m_attr < val[1]
                elif val[0] == 'gt':
                    assert m_attr > val[1]
                elif val[0] == 'and':
                    assert m_attr & val[1]
                elif val[0] == 'notand':
                    assert m_attr & val[1] == 0
                else:
                    raise ValueError( f"Unknown comparator {val[0]}" )
            else:
                assert m_attr == val
