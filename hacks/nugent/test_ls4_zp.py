#!/usr/bin/env python3
"""Standalone end-to-end test of the LS4 zeropoint pipeline.

This script exercises the new ``LS4Cam.gaia_dr3_to_instrument_mag`` method
on a single image:

  1. Loads the WCS from the ``*.wcs_*.txt`` sidecar produced by
     astrometry.net.
  2. Loads the SExtractor source list from ``*.sources_*.fits``.
  3. Loads the combined per-image mask ``*.mask.fits.fz`` (must exist
     already; see ``hacks/combine_mask_and_flags.py``).
  4. Queries Gaia DR3 in the field (DataLab if available, astroquery
     otherwise).
  5. Calls ``LS4Cam().gaia_dr3_to_instrument_mag(filter, catdata)`` to
     get synthetic LS4-filter magnitudes for the catalog stars.
  6. Matches catalog ↔ source-list by RA/Dec (with Gaia proper motion
     applied to the image MJD), rejecting source-list entries whose
     central pixel falls on a flagged pixel of the combined mask.
  7. Computes the zeropoint as the sigma-clipped median of
     ``trans_mag + 2.5 * log10(flux)``.

This is *not* yet plumbed into ``pipeline.photo_cal``; it's a standalone
sanity check until the LS4 g/i/z polynomial coefficients are recalibrated.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

# Quiet astropy/FITS chatter
warnings.filterwarnings("ignore")

import numpy as np
import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS


# Make sure the SeeChange package import path is visible regardless of cwd.
# This file lives at hacks/nugent/, so the SeeChange repo root is two levels up.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.ls4cam import LS4Cam  # noqa: E402


DEFAULT_IMAGE = "/Users/nugent/claude/ls4/test/ls4_20260410_000202_SE_C_g_Sci_D2X7RI.image.fits.fz"
GAIA_EPOCH = astropy.time.Time("2016.0", format="jyear", scale="tdb")
PRIOR_ZP_FOR_REFERENCE = 26.7  # what zp.py reported on this image (Legacy DR10 g)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_from_filename(path):
    m = re.search(r"_(g|i|z)_Sci_", Path(path).name)
    if not m:
        raise SystemExit(f"Cannot infer filter from filename {Path(path).name}")
    return m.group(1)


def load_wcs_sidecar(path):
    r"""Astrometry.net writes its WCS solution as 80-char FITS cards joined
    by *literal* '\n' (backslash + n) sequences, not real newlines.
    """
    raw = Path(path).read_text()
    parts = re.split(r"\\n", raw)
    if len(parts) == 1:
        # Fallback: real newlines
        parts = raw.splitlines()
    parts = [p for p in parts if p.strip()]
    flat = "".join(p.ljust(80) for p in parts)
    return WCS(fits.Header.fromstring(flat))


def first_2d(hdul):
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdu.data, hdu.header
    raise SystemExit("No 2-D image HDU")


def find_companion(image_path, glob_pattern):
    """Find a companion file next to ``image_path``.

    The glob is anchored to the image basename (the part before
    ``.image.fits.fz``) so that, e.g., a directory containing companion
    files for many chips of the same exposure resolves unambiguously to
    the right one.
    """
    image_path = Path(image_path)
    stem = image_path.name
    if stem.endswith(".image.fits.fz"):
        stem = stem[: -len(".image.fits.fz")]
    elif stem.endswith(".image.fits"):
        stem = stem[: -len(".image.fits")]
    else:
        stem = image_path.stem
    matches = sorted(image_path.parent.glob(stem + glob_pattern))
    if not matches:
        raise SystemExit(
            f"No file matching {stem + glob_pattern} next to {image_path}"
        )
    return matches[0]


def field_center_and_radius(wcs, nx, ny):
    """Return (SkyCoord_center, radius_deg) using the four image corners."""
    corners_pix = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    corner_coords = [wcs.pixel_to_world(x, y) for x, y in corners_pix]
    center = wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    seps = [center.separation(c).deg for c in corner_coords]
    return center, max(seps) * 1.05  # 5% pad


# ---------------------------------------------------------------------------
# Gaia DR3 query (DataLab preferred, astroquery fallback)
# ---------------------------------------------------------------------------

def query_gaia_dr3(ra_deg, dec_deg, radius_deg, max_g_mag=22.0, min_g_mag=10.0):
    """Return an astropy.table.Table with columns:
       ra, dec, pmra, pmdec, MAG_G/MAGERR_G, MAG_BP/MAGERR_BP, MAG_RP/MAGERR_RP.
    """
    try:
        from dl import queryClient, helpers  # noqa: F401
        return _query_gaia_via_datalab(ra_deg, dec_deg, radius_deg, max_g_mag, min_g_mag)
    except ImportError:
        return _query_gaia_via_astroquery(ra_deg, dec_deg, radius_deg, max_g_mag, min_g_mag)


def _convert_gaia_columns(t):
    """In-place: rename Gaia DR3 columns to the schema our code expects.

    Gaia returns ``phot_*_mean_mag`` (mag) and ``phot_*_mean_flux_over_error``
    (S/N).  Convert the latter to magnitude error via ``1.0857 / SN`` and
    rename to ``MAG_*`` / ``MAGERR_*``.
    """
    rename_map = {
        "phot_g_mean_mag":  "MAG_G",
        "phot_bp_mean_mag": "MAG_BP",
        "phot_rp_mean_mag": "MAG_RP",
        "phot_g_mean_flux_over_error":  "MAGERR_G",
        "phot_bp_mean_flux_over_error": "MAGERR_BP",
        "phot_rp_mean_flux_over_error": "MAGERR_RP",
    }
    for old, new in rename_map.items():
        if old in t.colnames:
            t.rename_column(old, new)
    # Convert SN -> mag err for the *err columns
    for col in ("MAGERR_G", "MAGERR_BP", "MAGERR_RP"):
        if col in t.colnames:
            t[col] = 1.0857 / np.asarray(t[col], dtype=np.float64)
    return t


def _query_gaia_via_datalab(ra_deg, dec_deg, radius_deg, max_g, min_g):
    from dl import queryClient, helpers
    sql = (
        "SELECT ra, dec, pmra, pmdec, "
        "phot_g_mean_mag, phot_g_mean_flux_over_error, "
        "phot_bp_mean_mag, phot_bp_mean_flux_over_error, "
        "phot_rp_mean_mag, phot_rp_mean_flux_over_error "
        "FROM gaia_dr3.gaia_source "
        f"WHERE q3c_radial_query(ra, dec, {ra_deg}, {dec_deg}, {radius_deg}) "
        f"AND phot_g_mean_mag BETWEEN {min_g} AND {max_g} "
        "AND phot_bp_mean_mag IS NOT NULL "
        "AND phot_rp_mean_mag IS NOT NULL"
    )
    df = helpers.utils.convert(queryClient.query(sql=sql), "pandas")
    from astropy.table import Table
    return _convert_gaia_columns(Table.from_pandas(df))


def _query_gaia_via_astroquery(ra_deg, dec_deg, radius_deg, max_g, min_g):
    from astroquery.gaia import Gaia
    adql = (
        "SELECT ra, dec, pmra, pmdec, "
        "phot_g_mean_mag, phot_g_mean_flux_over_error, "
        "phot_bp_mean_mag, phot_bp_mean_flux_over_error, "
        "phot_rp_mean_mag, phot_rp_mean_flux_over_error "
        "FROM gaiadr3.gaia_source "
        "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"                 CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) "
        f"AND phot_g_mean_mag BETWEEN {min_g} AND {max_g} "
        "AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL"
    )
    job = Gaia.launch_job_async(adql, dump_to_file=False)
    return _convert_gaia_columns(job.get_results())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                    help=f"Path to LS4 *.image.fits.fz (default: {DEFAULT_IMAGE})")
    ap.add_argument("--match-arcsec", type=float, default=1.0,
                    help="Maximum sky-match radius (arcsec)")
    ap.add_argument("--mask-clear-radius", type=int, default=0,
                    help="If >0, also reject sources where any pixel within "
                         "this many pixels of the source center is flagged.")
    ap.add_argument("--min-class-star", type=float, default=0.5,
                    help="SExtractor CLASS_STAR cut")
    ap.add_argument("--snr-min", type=float, default=5.0,
                    help="Minimum source flux/fluxerr")
    ap.add_argument("--max-g-mag", type=float, default=20.0,
                    help="Faint limit on Gaia G mag for the catalog query")
    ap.add_argument("--min-g-mag", type=float, default=10.0,
                    help="Bright limit on Gaia G mag for the catalog query")
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    print(f"image:        {image_path}")

    # ---- Companion files ------------------------------------------------
    sources_path = find_companion(image_path, ".sources_*.fits")
    wcs_txt = find_companion(image_path, ".wcs_*.txt")
    mask_path = (image_path.parent /
                 image_path.name.replace(".image.fits.fz", ".mask.fits.fz"))
    if not mask_path.exists():
        print("  mask:       (none — proceeding without mask filtering)")
        mask_path = None
    print(f"  sources:    {sources_path.name}")
    print(f"  wcs:        {wcs_txt.name}")
    if mask_path is not None:
        print(f"  mask:       {mask_path.name}")

    # ---- Filter, MJD, WCS, image dims ----------------------------------
    filt = filter_from_filename(image_path)
    with fits.open(image_path) as hdul:
        img_data, img_hdr = first_2d(hdul)
    ny, nx = img_data.shape
    img_mjd = astropy.time.Time(img_hdr["DATE-OBS"], scale="utc", format="isot").mjd
    print(f"  filter:     {filt}")
    print(f"  MJD:        {img_mjd:.6f}")
    print(f"  image size: {nx} x {ny}")

    wcs = load_wcs_sidecar(wcs_txt)
    center, radius_deg = field_center_and_radius(wcs, nx, ny)
    print(f"  field cen:  RA={center.ra.deg:.4f}, Dec={center.dec.deg:+.4f}")
    print(f"  query rad:  {radius_deg:.4f} deg ({radius_deg*60:.1f} arcmin)")

    # ---- Sources --------------------------------------------------------
    with fits.open(sources_path) as h:
        sources = h["LDAC_OBJECTS"].data
    n_extracted = len(sources)
    src_x = np.asarray(sources["XWIN_IMAGE"], dtype=np.float64)
    src_y = np.asarray(sources["YWIN_IMAGE"], dtype=np.float64)
    src_flux = np.asarray(sources["FLUX_APER"][:, -1], dtype=np.float64)
    src_efx  = np.asarray(sources["FLUXERR_APER"][:, -1], dtype=np.float64)
    src_flags = np.asarray(sources["FLAGS"], dtype=np.int32)
    src_class = np.asarray(sources["CLASS_STAR"], dtype=np.float64)

    qc = (
        (src_flags == 0)
        & (src_class > args.min_class_star)
        & np.isfinite(src_flux) & np.isfinite(src_efx)
        & (src_efx > 0)
        & (src_flux > args.snr_min * src_efx)
    )
    n_after_qc = int(qc.sum())
    print(f"  sources:    extracted={n_extracted}, after QC={n_after_qc}")

    src_x = src_x[qc]; src_y = src_y[qc]
    src_flux = src_flux[qc]; src_efx = src_efx[qc]

    # ---- Mask rejection (skipped if no combined mask available) -------
    if mask_path is not None:
        with fits.open(mask_path) as h:
            mask, _ = first_2d(h)

        # SExtractor coords are 1-based; numpy is 0-based.  Round to nearest pixel.
        src_xi = np.clip(np.rint(src_x - 1).astype(int), 0, nx - 1)
        src_yi = np.clip(np.rint(src_y - 1).astype(int), 0, ny - 1)
        on_bad = mask[src_yi, src_xi] != 0

        if args.mask_clear_radius > 0:
            R = args.mask_clear_radius
            for k in np.where(~on_bad)[0]:  # only check survivors of central-pixel cut
                yi, xi = src_yi[k], src_xi[k]
                y0, y1 = max(0, yi - R), min(ny, yi + R + 1)
                x0, x1 = max(0, xi - R), min(nx, xi + R + 1)
                if (mask[y0:y1, x0:x1] != 0).any():
                    on_bad[k] = True

        n_rej_mask = int(on_bad.sum())
        keep = ~on_bad
        src_x = src_x[keep]; src_y = src_y[keep]
        src_flux = src_flux[keep]; src_efx = src_efx[keep]
        print(f"  mask:       rejected={n_rej_mask}, surviving={int(keep.sum())}")
    else:
        n_rej_mask = 0

    # Source sky positions for matching
    src_sky = wcs.pixel_to_world(src_x - 1, src_y - 1)

    # ---- Gaia DR3 query -------------------------------------------------
    print("  querying Gaia DR3 (this can take 30s+) ...")
    cat = query_gaia_dr3(center.ra.deg, center.dec.deg, radius_deg,
                         max_g_mag=args.max_g_mag, min_g_mag=args.min_g_mag)
    n_gaia = len(cat)
    print(f"  Gaia DR3 in field: {n_gaia}")

    # Apply proper motion to the image MJD so the catalog positions match.
    cat_pmra = np.nan_to_num(np.asarray(cat["pmra"], dtype=np.float64))
    cat_pmdec = np.nan_to_num(np.asarray(cat["pmdec"], dtype=np.float64))
    cat_coords = SkyCoord(
        ra=np.asarray(cat["ra"]) * u.deg,
        dec=np.asarray(cat["dec"]) * u.deg,
        distance=Distance(1 * u.kpc),  # nominal
        pm_ra_cosdec=cat_pmra * u.mas / u.yr,
        pm_dec=cat_pmdec * u.mas / u.yr,
        obstime=GAIA_EPOCH,
    ).apply_space_motion(astropy.time.Time(img_mjd, format="mjd", scale="tdb"))

    # ---- Match ----------------------------------------------------------
    idx, d2d, _ = src_sky.match_to_catalog_sky(cat_coords)
    sel = d2d < args.match_arcsec * u.arcsec
    n_matched = int(sel.sum())
    print(f"  matched ({args.match_arcsec:.1f}″): {n_matched}")
    if n_matched < 5:
        raise SystemExit(f"Too few matches ({n_matched}); aborting.")

    # ---- Build catdata for LS4Cam method --------------------------------
    matched_cat = cat[idx[sel]]
    catdata = {
        "MAG_G":     np.asarray(matched_cat["MAG_G"], dtype=np.float64),
        "MAGERR_G":  np.asarray(matched_cat["MAGERR_G"], dtype=np.float64),
        "MAG_BP":    np.asarray(matched_cat["MAG_BP"], dtype=np.float64),
        "MAGERR_BP": np.asarray(matched_cat["MAGERR_BP"], dtype=np.float64) * u.mag,
        "MAG_RP":    np.asarray(matched_cat["MAG_RP"], dtype=np.float64),
        "MAGERR_RP": np.asarray(matched_cat["MAGERR_RP"], dtype=np.float64) * u.mag,
    }
    src_flux_m = src_flux[sel]

    # ---- Transform Gaia -> LS4 mag, then per-source ZP ------------------
    ls4 = LS4Cam()
    if getattr(ls4, "LS4_GAIA_TRNS_PLACEHOLDER", False):
        print("  WARNING: using PLACEHOLDER DECam g/i/z polynomial coefficients.")
    trans_mag, trans_magerr = ls4.gaia_dr3_to_instrument_mag(filt, catdata)

    finite = np.isfinite(trans_mag) & np.isfinite(src_flux_m) & (src_flux_m > 0)
    zps = np.asarray(trans_mag[finite]) + 2.5 * np.log10(src_flux_m[finite])
    n_used = int(finite.sum())

    zp_mean, zp_med, zp_std = sigma_clipped_stats(zps, sigma=3.0, maxiters=5)
    zp_err = zp_std / np.sqrt(max(n_used - 1, 1))

    # ---- Report --------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  filter:                 {filt}")
    print(f"  MJD:                    {img_mjd:.5f}")
    print(f"  field center (deg):     ({center.ra.deg:.4f}, {center.dec.deg:+.4f})")
    print(f"  query radius (deg):     {radius_deg:.4f}")
    print(f"  sources extracted:      {n_extracted}")
    print(f"    after source QC:      {n_after_qc}")
    print(f"    rejected by mask:     {n_rej_mask}")
    print(f"  Gaia DR3 stars:         {n_gaia}")
    print(f"  matched:                {n_matched}")
    print(f"    used in ZP fit:       {n_used}")
    print(f"  ZP (sigma-clip median): {zp_med:.3f} mag")
    print(f"  ZP scatter (std):       {zp_std:.3f} mag")
    print(f"  ZP s.e.m.:              {zp_err:.3f} mag")
    print(f"  prior result (zp.py):   ~{PRIOR_ZP_FOR_REFERENCE:.1f} mag (LS DR10 g)")
    print("=" * 60)


if __name__ == "__main__":
    main()
