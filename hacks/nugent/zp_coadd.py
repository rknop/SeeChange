#!/usr/bin/env python3
"""Calibrate ZP + FWHM of a SWarp coadd using the LS4 Gaia calibration.

Reuses the local-healpix Gaia DR3 query and the LS4Cam g/i/z polynomial
from ``zp_gaia_ls4cam.py`` (same calibration used per-chip), but adapted
for a coadd: explicit ``--filter`` (coadd filename carries none), a
weight-derived mask, a representative epoch for proper motion, and the
WCS straight from the coadd header.  Writes the solution into the coadd
header in place (``<F>_ZP``, ``SEEING``, ``PIXSCALE``, ``LMT_MG`` …) and
a ``<coadd>.zp.cat`` of matched stars, so downstream (e.g. diff_zogy.py)
can report calibrated magnitudes.

Usage:
    zp_coadd.py --coadd i-band-apr21_24/coadd.fits --filter i \
                --weight i-band-apr21_24/coadd.weight.fits
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import astropy.time
import astropy.units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import sep

# Reuse the Gaia calibration machinery from the per-chip ZP script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zp_gaia_ls4cam import query_gaia_dr3, ls4cam_gaia_to_instrument  # noqa: E402

try:
    sep.set_extract_pixstack(5_000_000)   # 5040^2 coadds are crowded
except Exception:
    pass

GAIA_EPOCH = astropy.time.Time("2016.0", format="jyear", scale="tdb")
SEP_APER_HASMASKED = 0x10


def first_2d(hdul):
    for i, h in enumerate(hdul):
        if h.data is not None and getattr(h.data, "ndim", 0) == 2:
            return i
    raise SystemExit("no 2-D HDU")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coadd", required=True)
    ap.add_argument("--filter", required=True, choices=["g", "i", "z"])
    ap.add_argument("--weight", default=None, help="coadd.weight.fits (≤0 → masked)")
    ap.add_argument("--epoch", type=float, default=2026.35,
                    help="representative Julian-year epoch for Gaia PM")
    ap.add_argument("--det-thresh", type=float, default=5.0)
    ap.add_argument("--match-arcsec", type=float, default=1.0)
    ap.add_argument("--min-snr", type=float, default=10.0)
    ap.add_argument("--max-g-mag", type=float, default=20.0)
    ap.add_argument("--min-g-mag", type=float, default=10.0)
    args = ap.parse_args()

    coadd = Path(args.coadd)
    hdul = fits.open(coadd, mode="readonly")
    idx = first_2d(hdul)
    hdr = hdul[idx].header
    data = hdul[idx].data.astype(np.float64)
    ny, nx = data.shape
    wcs = WCS(hdr)
    print(f"coadd: {coadd}  shape={data.shape} filter={args.filter}")

    # weight → mask
    if args.weight and os.path.exists(args.weight):
        with fits.open(args.weight) as wh:
            w = wh[first_2d(wh)].data
        mask = (~np.isfinite(w)) | (w <= 0)
        print(f"  mask from weight: {int(mask.sum()):,} bad px ({100*mask.mean():.1f}%)")
    else:
        mask = np.zeros(data.shape, bool)

    # ---- SEP extraction (mirror zp_gaia_ls4cam.extract_sources) ----
    d = data.copy()
    bkg = sep.Background(d, mask=mask)
    sub = d - bkg
    sky_sig = float(bkg.globalrms)
    err = bkg.rms()
    objs = sep.extract(sub, args.det_thresh, err=err, mask=mask)
    a = np.clip(np.where(np.isfinite(objs["a"]) & (objs["a"] > 0), objs["a"], 1.0), 1e-3, None)
    b = np.where(np.isfinite(objs["b"]) & (objs["b"] > 0) & (objs["b"] <= a), objs["b"], np.minimum(a, 1.0))
    theta = np.clip(np.asarray(objs["theta"], float), -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    kron, _ = sep.kron_radius(sub, objs["x"], objs["y"], a, b, theta, 6.0, mask=mask)
    kron = np.where(np.isfinite(kron) & (kron > 0), kron, 2.5)
    flux, fluxerr, sflag = sep.sum_ellipse(sub, objs["x"], objs["y"], a, b, theta,
                                           2.5 * kron, subpix=1, mask=mask, err=err)
    fwhm = 2.0 * np.sqrt(np.log(2.0)) * np.sqrt(a * b)
    ixp = np.clip(np.round(objs["x"]).astype(int), 0, nx - 1)
    iyp = np.clip(np.round(objs["y"]).astype(int), 0, ny - 1)
    good = (np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
            & (flux > args.min_snr * fluxerr)
            & ((sflag & SEP_APER_HASMASKED) == 0) & (~mask[iyp, ixp]))
    sx, sy, sflux, sfwhm = objs["x"][good], objs["y"][good], flux[good], fwhm[good]
    print(f"  SEP: {len(objs)} total → {good.sum()} clean (S/N>{args.min_snr}); sky_rms={sky_sig:.3g}")
    if good.sum() < 10:
        raise SystemExit("too few clean sources")

    # ---- Gaia query over the field ----
    corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    cc = [wcs.pixel_to_world(x, y) for x, y in corners]
    center = wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    radius = max(center.separation(c).deg for c in cc) * 1.02
    print(f"  field: ({center.ra.deg:.4f},{center.dec.deg:+.4f}) r={radius:.3f}°")
    cat = query_gaia_dr3(center.ra.deg, center.dec.deg, radius,
                         max_g_mag=args.max_g_mag, min_g_mag=args.min_g_mag)
    if cat is None or len(cat) == 0:
        raise SystemExit("no Gaia stars")
    print(f"  Gaia DR3 in field: {len(cat)}")

    # PM to representative epoch
    epoch = astropy.time.Time(args.epoch, format="jyear", scale="tdb")
    pmra = np.nan_to_num(np.asarray(cat["pmra"], float))
    pmdec = np.nan_to_num(np.asarray(cat["pmdec"], float))
    cat_c = SkyCoord(ra=np.asarray(cat["ra"]) * u.deg, dec=np.asarray(cat["dec"]) * u.deg,
                     distance=Distance(1 * u.kpc),
                     pm_ra_cosdec=pmra * u.mas/u.yr, pm_dec=pmdec * u.mas/u.yr,
                     obstime=GAIA_EPOCH).apply_space_motion(epoch)

    # synthetic LS4 mag for this filter
    icat, icaterr = ls4cam_gaia_to_instrument(
        args.filter,
        np.asarray(cat["MAG_G"]), np.asarray(cat["MAGERR_G"]),
        np.asarray(cat["MAG_BP"]), np.asarray(cat["MAGERR_BP"]),
        np.asarray(cat["MAG_RP"]), np.asarray(cat["MAGERR_RP"]))

    # match SEP ↔ Gaia
    src = wcs.pixel_to_world(sx, sy)
    i_g, d2d, _ = src.match_to_catalog_sky(cat_c)
    sel = d2d < args.match_arcsec * u.arcsec
    n_match = int(sel.sum())
    print(f"  matched ({args.match_arcsec}\"): {n_match}")
    if n_match < 10:
        raise SystemExit("too few matches")

    m_flux = sflux[sel]; m_icat = icat[i_g[sel]]; m_fwhm = sfwhm[sel]
    finite = np.isfinite(m_icat) & (m_flux > 0)
    zp_vals = m_icat[finite] + 2.5 * np.log10(m_flux[finite])
    nuse = int(finite.sum())
    zp_mean, zp_med, zp_std = sigma_clipped_stats(zp_vals, sigma=3.0, maxiters=5)
    zp = float(zp_med); zp_err = float(zp_std / np.sqrt(max(nuse - 1, 1)))

    pm = wcs.pixel_scale_matrix
    pixscale = float(np.sqrt(abs(np.linalg.det(pm))) * 3600.0)
    fwhm_px = float(np.median(m_fwhm))
    fwhm_arc = fwhm_px * pixscale
    lmt = float(-2.5 * np.log10(5.0 * sky_sig * np.sqrt(np.pi) * fwhm_px) + zp)

    print(f"  >>> {args.filter}: ZP={zp:.3f}±{zp_err:.3f} (N={nuse}, scatter {zp_std:.3f})  "
          f"FWHM={fwhm_arc:.2f}\" ({fwhm_px:.2f}px)  LMT5≈{lmt:.2f}")

    # write header in place
    hdul.close()
    with fits.open(coadd, mode="update") as hu:
        k = first_2d(hu); H = hu[k].header
        F = args.filter.upper()
        H[f"{F}_ZP"] = (zp, f"Gaia/LS4Cam {args.filter} zeropoint")
        H[f"{F}_ZPERR"] = (zp_err, "ZP s.e.m.")
        H["ZP_NSTAR"] = (nuse, "stars used for ZP")
        H["ZP_SCAT"] = (float(zp_std), "ZP sigma-clipped scatter")
        H["SEEING"] = (fwhm_arc, "median FWHM of matched stars (arcsec)")
        H["FWHMPIX"] = (fwhm_px, "median FWHM (pixels)")
        H["PIXSCALE"] = (pixscale, "arcsec/pixel")
        H["LMT_MG"] = (lmt, "5-sigma limiting mag (approx)")
        H["ZP_CAT"] = ("GAIADR3+LS4CAM", "ZP calibration source")
        H["ZP_EPOCH"] = (args.epoch, "Gaia PM epoch (jyear)")
        hu.flush()
    # matched-star catalog
    cat_path = str(coadd).replace(".fits", ".zp.ecsv")
    from astropy.table import Table
    Table({"x": sx[sel][finite], "y": sy[sel][finite],
           "ra": src.ra.deg[sel][finite], "dec": src.dec.deg[sel][finite],
           "flux": m_flux[finite], "mag_cat": m_icat[finite],
           "zp_i": zp_vals}).write(cat_path, format="ascii.ecsv", overwrite=True)
    print(f"  wrote header + {cat_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
