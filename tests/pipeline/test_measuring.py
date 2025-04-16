import pytest
import uuid

import numpy as np
import astropy.wcs

from improc.tools import pepper_stars
from models.world_coordinates import WorldCoordinates
from models.zero_point import ZeroPoint

from pipeline.data_store import DataStore
from pipeline.detection import Detector

from tests.conftest import SKIP_WARNING_TESTS


def test_measuring():
    # Make a fake image for purposes of testing measurements.  We're
    #  going to make a ref image that has a bunch of stars on it, and a
    #  new image that is the ref image plus other stuff on it that we
    #  want to find.  Then we'll run the naively-subtrated new image
    #  through detection to pull out stuff, and cutting and
    #  measurements, to make sure things come out more or less as
    #  expected.

    nx = 1024
    ny = 1024
    refskysig = 10.
    newskysig = 40.
    nstars = 500
    sig = 1.2
    fwhm = 2.35482 * sig

    # Use the same seed for rng for ref and new so the stars show up a the same place
    refimg, refvar = pepper_stars( xsize=nx, ysize=ny, skynoise=refskysig, seeing=fwhm,
                                   nstars=nstars, rng=np.default_rng( 42 ), noiserng=np.default_rng( 64738 ) )
    newimg, newvar = pepper_stars( xsize=nx, ysize=ny, skynoise=newskysig, seeing=fwhm,
                                   nstars=nstars, rng=np.default_rng( 42 ), noiserng=np.default_rng( 137 ) )
    flagim = np.zeros_like( refimg, dtype=np.int16 )

    ### 
    # First set of objects: at y=50.2, range of xs, put down basic Gaussians

    gaussy = 50.2
    gaussxs = 50.2 + 25.353 * np.arange( 36 )
    halfwid = 2 * int( np.ceil( 5. * fwhm ) )
    xvals, yvals = np.meshgrid( range(-halfwid, halfwid+1), range(-halfwid, halfwid+1) )
    # The 5. is sqrt( π * FWHM² ) for gauss_sig = 1.2
    # So, the first one should be a 1σ object; go in increments of 0.5σ
    gaussfluxen = 5. * newskysig * ( np.arange(36) / 2. + 1. )

    y = gaussy
    for x, flux in zip( gaussxs, gaussfluxen ):
        ix = int( np.floor( x + 0.5 ) )
        iy = int( np.floor( y + 0.5 ) )
        curxvals = xvals + ( ix - x )
        curywvals = yvals + ( iy - y )
        star = flux / ( 2. * np.pi * gauss_sig**2 ) * np.exp( -(curxvals**2 + curyvals**2) / ( 2. * gauss_sig**2 ) )
        varstar = np.zeros_like( star )
        varstar[ star >=0. ] = star[ star >=0. ]
        newim[ iy-halfwid:iy+halfwid+1, ix-halfwid:ix+halfwid+1 ] += star
        newvar[ iy-halfwid:iy+halfwid+1, ix-halfwid:ix+halfwid+1 ] += varstar


    refimobj = Image()
    refimobj.data = refimg
    refimobj.weight = 1. / refvar
    refimobj.flags = flagim
    refds = DataStore( refimobj )    
    ###
    # Just run extraction on the ref image.  We really only need this to
    #   get a PSF.  We're going to use its PSF for everything, since by
    #   construction it should all be the same.
    extractor = Detector( measure_psf=True, apers=[1.0, 2.0, 5.0] )
    refds = extractor.run( refds )

    import pdb; pdb.set_trace()
        
    ###
    # Subtract
    subim = newim - refim
    subvar = newvar + refvar
    subflg = newflg
        
    ###
    # WCS : make a WCS that's just (ra,dec) = (0, 0) at the center of the image with 1 arcsec/pix,
    #   because we don't really care about the WCS here.  (But, the functions need something to
    #   chew on.)

    apwcs = astropy.wcs.WCS( naxis=2 )
    apwcs.wcs.crpix = [ nx / 2., ny / 2. ]
    apwcs.wcs.crval = [ 0., 0. ]
    apwcs.wcs.cdelt = [ 1./3600., 1./3600. ]
    wcs = WorldCoordinates()
    wcs.wcs = apwcs

    ###
    # ZeroPoint : gratuitous
    zp = ZeroPoint( wcs_id=wcs.id, zp=25., dzp=0.01.,
                    aper_cor_radii=[ fhwm, 2.*fwhm, 5.*fwhm ],
                    aper_cors=[ -0.02, 0., 0. ] )
    
                    
                    
    
    
    
    

def test_warnings_and_exceptions( decam_datastore_through_cutouts ):
    ds = decam_datastore_through_cutouts
    measurer = ds._pipeline.measurer

    if not SKIP_WARNING_TESTS:
        measurer.pars.inject_warnings = 1
        ds._pipeline.make_provenance_tree( ds )

        with pytest.warns(UserWarning) as record:
            measurer.run( ds )
        assert len(record) > 0
        assert any("Warning injected by pipeline parameters in process 'measuring'." in str(w.message) for w in record)

    measurer.pars.inject_exceptions = 1
    measurer.pars.inject_warnings = 0
    ds.measurement_set = None
    ds._pipeline.make_provenance_tree( ds )
    with pytest.raises(Exception, match="Exception injected by pipeline parameters in process 'measuring'."):
        ds = measurer.run( ds )
