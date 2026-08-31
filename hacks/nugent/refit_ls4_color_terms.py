#!/usr/bin/env python3
"""Refit the LS4 ``gaia_dr3_to_instrument_mag`` polynomial coefficients.

Iteratively fits a degree-3 polynomial in Gaia (BP - RP) color, with
LS DR10 PSF magnitudes as the truth.  Per-image zero-points are
fitted as nuisance parameters (sigma-clipped median residual per
image), then the global polynomial is re-fitted to the per-image-ZP-
subtracted residuals.  Three iterations.

Inputs:  a directory of complete (image + sources + wcs) trios for a
single LS4 chip+filter combination, e.g. /ls4/test/ for SE_C g-band
after running ``hacks/pull_chip_files.py``.

Outputs:
  * Final polynomial coefficient line printed verbatim, ready to paste
    into ``models/ls4cam.py:transformations`` for the chosen filter.
  * ``--outdir`` PNGs: residual-vs-color (before/after) and per-image
    ZP histogram.
  * ``--outdir`` CSV summary with per-image ZP & per-iteration RMS.
"""

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Gaia DR3 reference epoch
GAIA_EPOCH = astropy.time.Time("2016.0", format="jyear", scale="tdb")

# Mag-error conversion: MAGERR = 1.0857 / SN
MAG_ERR_K = 1.0857


# ---------------------------------------------------------------------------
# Helpers (mostly copied from hacks/test_ls4_zp.py)
# ---------------------------------------------------------------------------

def filter_from_filename(name):
    m = re.search(r"_(g|i|z)_Sci_", name)
    return m.group(1) if m else None


def chip_from_filename(name):
    m = re.search(r"_([NS][EW]_[A-H])_(?:g|i|z)_Sci_", name)
    return m.group(1) if m else None


def find_companion(image_path, suffix_glob):
    image_path = Path(image_path)
    stem = image_path.name
    if stem.endswith(".image.fits.fz"):
        stem = stem[: -len(".image.fits.fz")]
    elif stem.endswith(".image.fits"):
        stem = stem[: -len(".image.fits")]
    matches = sorted(image_path.parent.glob(stem + suffix_glob))
    return matches[0] if matches else None


def load_wcs_sidecar(path):
    raw = Path(path).read_text()
    parts = re.split(r"\\n", raw)
    if len(parts) == 1:
        parts = raw.splitlines()
    parts = [p for p in parts if p.strip()]
    flat = "".join(p.ljust(80) for p in parts)
    return WCS(fits.Header.fromstring(flat))


def first_2d(hdul):
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdu.data, hdu.header
    return None, None


def field_center_and_radius(wcs, nx, ny, pad=1.05):
    corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    coords = [wcs.pixel_to_world(x, y) for x, y in corners]
    center = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    rad = max(center.separation(c).deg for c in coords) * pad
    return center, rad


# ---------------------------------------------------------------------------
# DataLab queries
# ---------------------------------------------------------------------------

def _retry(fn, attempts=3, sleep=2.0):
    last = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            print(f"  query attempt {k + 1}/{attempts} failed: {exc}", flush=True)
            time.sleep(sleep)
    raise RuntimeError(f"all {attempts} attempts failed: {last}")


def query_gaia_dr3(ra_deg, dec_deg, radius_deg, max_g=21.0, min_g=12.0):
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
    df = _retry(lambda: helpers.utils.convert(queryClient.query(sql=sql), "pandas"))
    df["MAG_G"] = df["phot_g_mean_mag"]
    df["MAG_BP"] = df["phot_bp_mean_mag"]
    df["MAG_RP"] = df["phot_rp_mean_mag"]
    df["MAGERR_G"] = MAG_ERR_K / df["phot_g_mean_flux_over_error"]
    df["MAGERR_BP"] = MAG_ERR_K / df["phot_bp_mean_flux_over_error"]
    df["MAGERR_RP"] = MAG_ERR_K / df["phot_rp_mean_flux_over_error"]
    return df


def query_ls_dr10(ra_deg, dec_deg, radius_deg, filt, max_mag=21.0, min_mag=12.0):
    """Fetch LS DR10 PSF stars in the field with all four mags + errors."""
    from dl import queryClient, helpers
    mag_col = f"mag_{filt}"   # mag_g, mag_i, mag_z
    sql = (
        "SELECT ra, dec, "
        "mag_g, mag_r, mag_i, mag_z, "
        "flux_ivar_g, flux_ivar_r, flux_ivar_i, flux_ivar_z, "
        "flux_g, flux_r, flux_i, flux_z "
        "FROM ls_dr10.tractor "
        f"WHERE q3c_radial_query(ra, dec, {ra_deg}, {dec_deg}, {radius_deg}) "
        f"AND type = 'PSF' "
        f"AND {mag_col} BETWEEN {min_mag} AND {max_mag} "
        f"AND {mag_col} IS NOT NULL"
    )
    df = _retry(lambda: helpers.utils.convert(queryClient.query(sql=sql), "pandas"))
    return df


# ---------------------------------------------------------------------------
# Main fit
# ---------------------------------------------------------------------------

def discover_trios(indir, chip=None, filt=None):
    """Yield (image_path, sources_path, wcs_path, chip, filt) for every complete trio."""
    indir = Path(indir)
    for img in sorted(indir.glob("ls4_*.image.fits.fz")):
        name = img.name
        c = chip_from_filename(name)
        f = filter_from_filename(name)
        if chip and c != chip:
            continue
        if filt and f != filt:
            continue
        srcs = find_companion(img, ".sources_*.fits")
        wcs = find_companion(img, ".wcs_*.txt")
        if srcs is None or wcs is None:
            continue
        yield img, srcs, wcs, c, f


def extract_image_records(image_path, sources_path, wcs_path, image_id,
                          source_qc_min_class=0.5, source_qc_min_snr=5.0,
                          mask_clear_radius=0):
    """Return a dict of per-source numpy arrays for one image, plus the
    image's MJD and field center+radius (all needed downstream).
    Returns ``None`` on any unrecoverable file-level error.

    If a sibling ``*.mask.fits.fz`` is present, sources whose central
    pixel (or any pixel within ``mask_clear_radius`` if set > 0) is
    nonzero are dropped.  A counter ``n_rej_mask`` is included in the
    returned dict.
    """
    try:
        with fits.open(image_path) as hdul:
            _, hdr = first_2d(hdul)
        if "DATE-OBS" not in hdr:
            return None
        mjd = astropy.time.Time(hdr["DATE-OBS"], scale="utc", format="isot").mjd
        nx = int(hdr.get("NAXIS1") or hdr.get("ZNAXIS1"))
        ny = int(hdr.get("NAXIS2") or hdr.get("ZNAXIS2"))
        wcs = load_wcs_sidecar(wcs_path)
        center, radius = field_center_and_radius(wcs, nx, ny)
        with fits.open(sources_path) as h:
            srcs = h["LDAC_OBJECTS"].data
        flux_aper = np.asarray(srcs["FLUX_APER"][:, -1], dtype=np.float64)
        fluxerr = np.asarray(srcs["FLUXERR_APER"][:, -1], dtype=np.float64)
        flags = np.asarray(srcs["FLAGS"], dtype=np.int32)
        cls = np.asarray(srcs["CLASS_STAR"], dtype=np.float64)
        x = np.asarray(srcs["XWIN_IMAGE"], dtype=np.float64)
        y = np.asarray(srcs["YWIN_IMAGE"], dtype=np.float64)
        qc = (
            (flags == 0)
            & (cls > source_qc_min_class)
            & np.isfinite(flux_aper) & np.isfinite(fluxerr)
            & (fluxerr > 0)
            & (flux_aper > source_qc_min_snr * fluxerr)
        )
        if qc.sum() < 5:
            return None
        x = x[qc]; y = y[qc]
        flux_aper = flux_aper[qc]

        # Optional: per-image mask filtering.  Uses the sibling
        # *.mask.fits.fz produced by combine_mask_and_flags.py.
        n_rej_mask = 0
        mask_path = (Path(image_path).parent /
                     Path(image_path).name.replace(".image.fits.fz", ".mask.fits.fz"))
        if mask_path.exists() and len(x) > 0:
            with fits.open(mask_path) as h:
                mask, _ = first_2d(h)
            xi = np.clip(np.rint(x - 1).astype(int), 0, nx - 1)
            yi = np.clip(np.rint(y - 1).astype(int), 0, ny - 1)
            on_bad = mask[yi, xi] != 0
            if mask_clear_radius > 0:
                R = int(mask_clear_radius)
                for k in np.where(~on_bad)[0]:
                    yy = yi[k]; xx = xi[k]
                    y0, y1 = max(0, yy - R), min(ny, yy + R + 1)
                    x0, x1 = max(0, xx - R), min(nx, xx + R + 1)
                    if (mask[y0:y1, x0:x1] != 0).any():
                        on_bad[k] = True
            keep = ~on_bad
            n_rej_mask = int(on_bad.sum())
            x = x[keep]; y = y[keep]
            flux_aper = flux_aper[keep]
            if len(x) < 5:
                return None

        return {
            "image_id": image_id,
            "image_path": str(image_path),
            "mjd": mjd,
            "wcs": wcs,
            "center": center,
            "radius": radius,
            "src_x": x,
            "src_y": y,
            "src_flux": flux_aper,
            "n_rej_mask": n_rej_mask,
            "mask_used": mask_path.exists(),
        }
    except Exception as exc:
        print(f"  WARN extracting {image_path.name}: {exc}", flush=True)
        return None


def union_query_box(records):
    """Return (center_ra, center_dec, max_radius) covering all per-image fields."""
    centers_ra = np.array([r["center"].ra.deg for r in records])
    centers_dec = np.array([r["center"].dec.deg for r in records])
    radii = np.array([r["radius"] for r in records])
    cra = centers_ra.mean()
    cdec = centers_dec.mean()
    union_centre = SkyCoord(ra=cra * u.deg, dec=cdec * u.deg)
    per_centers = SkyCoord(ra=centers_ra * u.deg, dec=centers_dec * u.deg)
    seps = union_centre.separation(per_centers).deg
    max_radius = (seps + radii).max() + 0.05
    return cra, cdec, max_radius


def cluster_pointings(records, link_arcsec_deg=0.3):
    """Greedy cluster image records by sky location.

    Two images are in the same cluster if their centers are within
    ``link_arcsec_deg`` (degrees) of any existing cluster member.

    Returns: list of dicts:
        {
          'records':   [record, ...],
          'center_ra': float,   # mean center RA in deg
          'center_dec':float,
          'radius':    float,   # degrees, covering all members + their fov
        }
    """
    clusters = []  # list of dicts {coord_list (deg), members}
    for rec in records:
        cra, cdec = rec["center"].ra.deg, rec["center"].dec.deg
        c = SkyCoord(ra=cra * u.deg, dec=cdec * u.deg)
        placed = False
        for cl in clusters:
            # Compare to mean of cluster so far
            mean_c = SkyCoord(ra=np.mean([m["center"].ra.deg for m in cl["members"]]) * u.deg,
                              dec=np.mean([m["center"].dec.deg for m in cl["members"]]) * u.deg)
            if c.separation(mean_c).deg <= link_arcsec_deg:
                cl["members"].append(rec)
                placed = True
                break
        if not placed:
            clusters.append({"members": [rec]})
    # Compute geometry of each cluster
    out = []
    for cl in clusters:
        recs = cl["members"]
        ra = np.mean([r["center"].ra.deg for r in recs])
        dec = np.mean([r["center"].dec.deg for r in recs])
        cen = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        rmax = max(
            cen.separation(SkyCoord(ra=r["center"].ra, dec=r["center"].dec)).deg
            + r["radius"]
            for r in recs
        ) + 0.05
        out.append({"records": recs, "center_ra": ra, "center_dec": dec, "radius": rmax})
    return out


def crossmatch_gaia_dr10(gaia_df, dr10_df, max_arcsec=0.5):
    """Return a merged DataFrame of unique stars matched between Gaia and DR10."""
    import pandas as pd
    gc = SkyCoord(ra=np.asarray(gaia_df["ra"]) * u.deg,
                  dec=np.asarray(gaia_df["dec"]) * u.deg)
    dc = SkyCoord(ra=np.asarray(dr10_df["ra"]) * u.deg,
                  dec=np.asarray(dr10_df["dec"]) * u.deg)
    idx, d2d, _ = gc.match_to_catalog_sky(dc)
    sel = d2d < max_arcsec * u.arcsec
    out = pd.DataFrame({
        "ra":         np.asarray(gaia_df["ra"])[sel.value if hasattr(sel, "value") else sel],
        "dec":        np.asarray(gaia_df["dec"])[sel.value if hasattr(sel, "value") else sel],
        "pmra":       np.nan_to_num(np.asarray(gaia_df["pmra"])[sel.value if hasattr(sel, "value") else sel]),
        "pmdec":      np.nan_to_num(np.asarray(gaia_df["pmdec"])[sel.value if hasattr(sel, "value") else sel]),
        "MAG_G":      np.asarray(gaia_df["MAG_G"])[sel.value if hasattr(sel, "value") else sel],
        "MAG_BP":     np.asarray(gaia_df["MAG_BP"])[sel.value if hasattr(sel, "value") else sel],
        "MAG_RP":     np.asarray(gaia_df["MAG_RP"])[sel.value if hasattr(sel, "value") else sel],
        "mag_g_dr10": np.asarray(dr10_df["mag_g"].iloc[idx])[sel.value if hasattr(sel, "value") else sel],
        "mag_r_dr10": np.asarray(dr10_df["mag_r"].iloc[idx])[sel.value if hasattr(sel, "value") else sel],
        "mag_i_dr10": np.asarray(dr10_df["mag_i"].iloc[idx])[sel.value if hasattr(sel, "value") else sel],
        "mag_z_dr10": np.asarray(dr10_df["mag_z"].iloc[idx])[sel.value if hasattr(sel, "value") else sel],
    })
    return out


def project_to_mjd(combined_df, target_mjd):
    """Return SkyCoord of combined catalog projected to target_mjd via Gaia PMs."""
    sc = SkyCoord(
        ra=np.asarray(combined_df["ra"]) * u.deg,
        dec=np.asarray(combined_df["dec"]) * u.deg,
        distance=Distance(1 * u.kpc),
        pm_ra_cosdec=np.asarray(combined_df["pmra"]) * u.mas / u.yr,
        pm_dec=np.asarray(combined_df["pmdec"]) * u.mas / u.yr,
        obstime=GAIA_EPOCH,
    )
    return sc.apply_space_motion(astropy.time.Time(target_mjd, format="mjd", scale="tdb"))


# ---------------------------------------------------------------------------
# Multi-color fit + 1-D marginalization helpers (used only with --multi-color)
# ---------------------------------------------------------------------------

def gen_total_degree_monomials(n_vars, max_deg):
    """Yield exponent tuples (e1, ..., e_n) with sum(e) <= max_deg."""
    from itertools import combinations_with_replacement
    for total in range(max_deg + 1):
        for combo in combinations_with_replacement(range(n_vars), total):
            exps = [0] * n_vars
            for c in combo:
                exps[c] += 1
            yield tuple(exps)


def build_multivar_features(arrays, monomials):
    """Return (N, len(monomials)) feature matrix for the listed monomials."""
    cols = []
    for exps in monomials:
        col = np.ones_like(arrays[0])
        for arr, e in zip(arrays, exps):
            if e > 0:
                col = col * (arr ** e)
        cols.append(col)
    return np.column_stack(cols)


def multi_color_fit(cat_df, n_iter=3, poly_degree=3):
    """Fit M = polynomial in (BP-RP, g-r, r-i, i-z) of total degree poly_degree.

    Returns (coeffs, monomials, mask, rms) where ``mask`` is the
    sigma-clipped-survivor index into cat_df.
    """
    bp_rp = cat_df["bp_rp"].values
    gr = cat_df["gr"].values
    ri = cat_df["ri"].values
    iz = cat_df["iz"].values
    target = cat_df["target"].values

    monomials = list(gen_total_degree_monomials(4, poly_degree))
    print(f"  multi-color monomials: {len(monomials)} (4 features, deg <= {poly_degree})")
    X = build_multivar_features([bp_rp, gr, ri, iz], monomials)

    mask = np.ones(len(target), dtype=bool)
    for it in range(n_iter):
        coeffs, *_ = np.linalg.lstsq(X[mask], target[mask], rcond=None)
        pred_all = X @ coeffs
        residuals = target - pred_all
        clipped = sigma_clip(residuals, sigma=3.0, maxiters=3, masked=True)
        new_mask = ~clipped.mask
        rms = float(np.sqrt(((residuals[new_mask]) ** 2).mean()))
        print(f"  iter {it}: n_used={int(new_mask.sum())}/{len(new_mask)}  "
              f"multi-color RMS={rms:.4f} mag")
        mask = new_mask
    return coeffs, monomials, mask, rms


def marginalize_to_1d(cat_df, mask, monomials, coeffs, poly_degree=3, n_grid=1000):
    """Collapse the multi-color fit to a 1-D polynomial in (BP-RP).

    Strategy:
      1. Fit conditional means E[g-r | BP-RP], E[r-i | BP-RP], E[i-z | BP-RP]
         as deg-2 polynomials in BP-RP (sigma-clipped, on the same survivor
         mask as the multi-color fit).
      2. Build a dense (BP-RP) grid spanning the data range.
      3. Evaluate the multi-color polynomial along the grid using the
         conditional means.
      4. Fit a deg=poly_degree 1-D polynomial in BP-RP to the grid values,
         weighted by the BP-RP histogram of the calibration sample (so the
         production polynomial is most accurate where most stars live).

    Returns (poly_ascending, cond_polys_dict, grid, target_grid, weights_grid,
             grid_fit_rms).
    """
    bp_rp = cat_df["bp_rp"].values[mask]
    gr = cat_df["gr"].values[mask]
    ri = cat_df["ri"].values[mask]
    iz = cat_df["iz"].values[mask]

    # 1. Conditional means via deg-2 sigma-clipped 1-D fits
    cond_polys = {}
    for name, arr in (("gr", gr), ("ri", ri), ("iz", iz)):
        m = np.ones(len(bp_rp), dtype=bool)
        cp = np.polyfit(bp_rp, arr, 2)
        for _ in range(2):
            r = arr - np.polyval(cp, bp_rp)
            m = ~sigma_clip(r, sigma=3.0, maxiters=3, masked=True).mask
            cp = np.polyfit(bp_rp[m], arr[m], 2)
        cond_polys[name] = cp
        print(f"  E[{name}|bp_rp] poly2 = "
              + " ".join(f"{c:+.4f}" for c in cp))

    # 2. Dense grid
    grid_lo = max(bp_rp.min(), -0.5)
    grid_hi = min(bp_rp.max(), 3.5)
    bp_rp_grid = np.linspace(grid_lo, grid_hi, n_grid)
    gr_grid = np.polyval(cond_polys["gr"], bp_rp_grid)
    ri_grid = np.polyval(cond_polys["ri"], bp_rp_grid)
    iz_grid = np.polyval(cond_polys["iz"], bp_rp_grid)

    # 3. Evaluate 2-D polynomial along the grid
    X_grid = build_multivar_features([bp_rp_grid, gr_grid, ri_grid, iz_grid],
                                     monomials)
    target_grid = X_grid @ coeffs

    # 4. Density-weighted 1-D polynomial fit
    counts, edges = np.histogram(bp_rp, bins=50, range=(grid_lo, grid_hi))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.interp(bp_rp_grid, bin_centers, counts)
    w = np.clip(w, 1e-6, None)
    new_poly_desc = np.polyfit(bp_rp_grid, target_grid, deg=poly_degree,
                               w=np.sqrt(w))
    new_poly_asc = new_poly_desc[::-1]

    # Quality: how well does the 1-D collapse approximate the 2-D evaluation?
    grid_resid = target_grid - np.polyval(new_poly_desc, bp_rp_grid)
    grid_rms_weighted = float(
        np.sqrt((grid_resid ** 2 * w).sum() / w.sum())
    )
    print(f"  marginalized 1-D collapse RMS (density-weighted): "
          f"{grid_rms_weighted:.5f} mag")

    return new_poly_asc, cond_polys, bp_rp_grid, target_grid, w, grid_rms_weighted


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--indir", default="/ls4/test",
                    help="Directory of complete trios.")
    ap.add_argument("--chip", default=None,
                    help="Restrict to one chip (e.g. SE_C).")
    ap.add_argument("--filter", dest="filt", default=None,
                    help="Restrict to one filter (g/i/z).")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Cap on number of images (for quick testing).")
    ap.add_argument("--match-arcsec", type=float, default=1.0,
                    help="Source<->catalog match radius (arcsec).")
    ap.add_argument("--mask-clear-radius", type=int, default=0,
                    help="If >0, also reject sources where any pixel within "
                         "this many pixels of the source center is flagged. "
                         "Per-image mask is read from sibling *.mask.fits.fz.")
    ap.add_argument("--n-iter", type=int, default=3,
                    help="Number of fit iterations (per-image ZP -> global poly).")
    ap.add_argument("--poly-degree", type=int, default=3,
                    help="Polynomial degree (DECam template uses 3).")
    ap.add_argument("--multi-color", action="store_true",
                    help="Use DR10 (g-r), (r-i), (i-z) colors as additional "
                         "features during fit; marginalize to a 1-D Gaia-only "
                         "polynomial for production. Production interface in "
                         "models/ls4cam.py is unchanged.")
    ap.add_argument("--photometric-zp-window", type=float, default=0.10,
                    help="If >0, restrict the calibration sample to images "
                         "within this many mag of the mode of per-image ZPs "
                         "(robust median).  Set to a large number (e.g. 999) "
                         "to disable filtering.")
    ap.add_argument("--outdir", default="/seechange/hacks/nugent/refit_outputs",
                    help="Where to put PNG plots and CSV summary.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip the global fit; just print plumbing summary.")
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    if not indir.is_dir():
        sys.exit(f"missing indir: {indir}")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Step 1: discover trios ----------------------------------
    trios = list(discover_trios(indir, chip=args.chip, filt=args.filt))
    if args.max_images is not None:
        trios = trios[: args.max_images]
    if not trios:
        sys.exit(f"No trios discovered in {indir} (chip={args.chip}, filt={args.filt})")
    inferred_chip = trios[0][3]
    inferred_filt = trios[0][4]
    print(f"discovered {len(trios)} trios, chip={inferred_chip}, filter={inferred_filt}")

    # ---------- Step 2: per-image extraction (no catalog match yet) -----
    print("extracting per-image data ...", flush=True)
    records = []
    for i, (img, srcs, wcs, c, f) in enumerate(trios):
        rec = extract_image_records(img, srcs, wcs, image_id=i,
                                    mask_clear_radius=args.mask_clear_radius)
        if rec is None:
            continue
        records.append(rec)
    if len(records) < 3:
        sys.exit(f"Only {len(records)} images survived QC; aborting.")
    n_with_mask = sum(1 for r in records if r.get("mask_used"))
    n_rej_total = sum(r.get("n_rej_mask", 0) for r in records)
    n_src_total = sum(len(r["src_x"]) for r in records)
    print(f"  {len(records)} images survived source QC")
    print(f"  {n_with_mask}/{len(records)} images had a *.mask.fits.fz applied "
          f"(rejected {n_rej_total} sources, {n_src_total} survive)")

    # ---------- Step 3: cluster images by sky location, query per cluster
    clusters = cluster_pointings(records, link_arcsec_deg=0.3)
    print(f"clustered {len(records)} images into {len(clusters)} sky pointings")
    for k, cl in enumerate(clusters):
        print(f"  cluster {k}: {len(cl['records'])} images at "
              f"RA={cl['center_ra']:.3f}, Dec={cl['center_dec']:+.3f}, "
              f"radius={cl['radius']:.3f} deg")

    import pandas as pd
    combined_per_cluster = {}   # cluster_idx -> combined DataFrame
    for k, cl in enumerate(clusters):
        print(f"querying Gaia DR3 (cluster {k}/{len(clusters)})...", flush=True)
        t0 = time.time()
        gaia_df = query_gaia_dr3(cl["center_ra"], cl["center_dec"], cl["radius"],
                                 max_g=21.0, min_g=12.0)
        print(f"  {len(gaia_df)} Gaia stars in {time.time() - t0:.1f}s")
        print(f"querying LS DR10 (cluster {k}/{len(clusters)})...", flush=True)
        t0 = time.time()
        dr10_df = query_ls_dr10(cl["center_ra"], cl["center_dec"], cl["radius"],
                                filt=inferred_filt, max_mag=21.0, min_mag=12.0)
        print(f"  {len(dr10_df)} DR10 PSF stars in {time.time() - t0:.1f}s")
        combined = crossmatch_gaia_dr10(gaia_df, dr10_df, max_arcsec=0.5).reset_index(drop=True)
        combined = combined.dropna(subset=["MAG_G", "MAG_BP", "MAG_RP",
                                           f"mag_{inferred_filt}_dr10"]).reset_index(drop=True)
        print(f"  cluster {k} cross-matched + cleaned: {len(combined)} stars")
        combined_per_cluster[k] = combined
        # Mark each member record with its cluster id so step 5 can pick the right combined table
        for rec in cl["records"]:
            rec["_cluster"] = k

    # ---------- Step 5: per-image source <-> combined-catalog match -----
    print("matching sources per image ...", flush=True)
    rows = {  # long-format columns
        "image_id":  [],
        "mag_truth": [],   # LS DR10 in chosen filter
        "bp_rp":     [],
        "inst_mag":  [],
    }
    n_matched_per_image = []
    # Cache per (cluster_id, mjd) — proper-motion projection is the expensive bit.
    cat_sky_cache = {}
    for rec in records:
        cluster_id = rec["_cluster"]
        combined = combined_per_cluster[cluster_id]
        key = (cluster_id, rec["mjd"])
        if key not in cat_sky_cache:
            cat_sky_cache[key] = project_to_mjd(combined, rec["mjd"])
        cat_sky = cat_sky_cache[key]
        src_sky = rec["wcs"].pixel_to_world(rec["src_x"] - 1, rec["src_y"] - 1)
        idx, d2d, _ = src_sky.match_to_catalog_sky(cat_sky)
        sel = d2d < args.match_arcsec * u.arcsec
        if int(sel.sum()) < 3:
            n_matched_per_image.append(0)
            continue
        n_matched_per_image.append(int(sel.sum()))
        matched = combined.iloc[idx[sel]]
        truth_col = f"mag_{inferred_filt}_dr10"
        rows["image_id"].extend([rec["image_id"]] * int(sel.sum()))
        rows["mag_truth"].extend(np.asarray(matched[truth_col]).tolist())
        rows["bp_rp"].extend((np.asarray(matched["MAG_BP"]) - np.asarray(matched["MAG_RP"])).tolist())
        sel_idx = sel.value if hasattr(sel, "value") else sel
        rows["inst_mag"].extend((-2.5 * np.log10(rec["src_flux"][sel_idx])).tolist())
    print(f"  median matches/image: {int(np.median(n_matched_per_image))}, "
          f"total matched stars: {len(rows['image_id'])}")

    if args.dry_run:
        print("[dry-run] skipping fit; first 5 rows:")
        for i in range(min(5, len(rows["image_id"]))):
            print(f"  img={rows['image_id'][i]} bp_rp={rows['bp_rp'][i]:.3f} "
                  f"truth={rows['mag_truth'][i]:.3f} inst={rows['inst_mag'][i]:.3f}")
        return 0

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["bp_rp"]) & np.isfinite(df["mag_truth"]) & np.isfinite(df["inst_mag"])]
    df = df.reset_index(drop=True)
    print(f"  fittable rows: {len(df)}")

    # ---------- Step 5b: per-image ZP from mag_truth - inst_mag ---------
    # zp_per_star = trans_mag - inst_mag.  If our calibration is right,
    # trans_mag = mag_truth, so zp = mag_truth - inst_mag.  This is the same
    # quantity zp.py used historically.
    df["zp_per_star"] = df["mag_truth"] - df["inst_mag"]
    zp_per_image = (
        df.groupby("image_id")["zp_per_star"]
          .agg(lambda s: float(sigma_clipped_stats(np.asarray(s), sigma=3.0, maxiters=5)[1]))
    )
    df["zp_img"] = df["image_id"].map(zp_per_image)
    print("\n=== validation: per-image ZPs from mag_truth - inst_mag ===")
    print(f"  median ZP across {len(zp_per_image)} images:  {zp_per_image.median():.3f}")
    print(f"  std    ZP across {len(zp_per_image)} images:  {zp_per_image.std():.3f}")

    # ---------- Step 5c: identify photometric images & clusters ---------
    # Robust mode of per-image ZPs is the photometric reference.  Images
    # within ±photometric_zp_window of it are "likely photometric"; cloudy
    # nights typically sit several tenths to a magnitude lower.
    if args.photometric_zp_window > 0 and len(zp_per_image) >= 5:
        zp_arr = zp_per_image.to_numpy()
        zp_mode_mean, zp_mode, zp_mode_std = sigma_clipped_stats(
            zp_arr, sigma=2.0, maxiters=5
        )
        is_phot = (np.abs(zp_arr - zp_mode) <= args.photometric_zp_window)
        photometric_image_ids = set(np.asarray(zp_per_image.index)[is_phot])
        photometric_clusters = {
            rec["_cluster"] for rec in records
            if rec["image_id"] in photometric_image_ids
        }
        print(f"\n=== photometric filter (window=±{args.photometric_zp_window:.2f} mag) ===")
        print(f"  photometric ZP mode (sigma-clipped median): {zp_mode:.3f}")
        print(f"  photometric images:   {len(photometric_image_ids)} / {len(records)}")
        print(f"  photometric clusters: {len(photometric_clusters)} / {len(clusters)}")
    else:
        photometric_image_ids = set(np.asarray(zp_per_image.index))
        photometric_clusters = set(combined_per_cluster.keys())

    # ---------- Step 6: catalog-to-catalog polynomial fit ---------------
    #
    # The production formula in models/ls4cam.py is:
    #     MAG_LS4 = MAG_G - poly(BP - RP)
    # Substituting our truth assumption MAG_LS4 ≈ MAG_DR10:
    #     poly(BP - RP) = MAG_G - MAG_DR10_g
    # Pure catalog-to-catalog relationship -- image data is not needed.

    truth_col = f"mag_{inferred_filt}_dr10"
    cat_rows = []
    for k in photometric_clusters:
        c = combined_per_cluster[k]
        # Carry all four DR10 mags so multi-color mode has them too.
        keep_cols = ["MAG_G", "MAG_BP", "MAG_RP",
                     "mag_g_dr10", "mag_r_dr10", "mag_i_dr10", "mag_z_dr10"]
        block = c[[col for col in keep_cols if col in c.columns]].copy()
        cat_rows.append(block)
    cat_df = pd.concat(cat_rows, ignore_index=True).dropna()
    cat_df["bp_rp"] = cat_df["MAG_BP"] - cat_df["MAG_RP"]
    cat_df["target"] = cat_df["MAG_G"] - cat_df[truth_col]
    cat_df = cat_df[(cat_df["bp_rp"] > -0.5) & (cat_df["bp_rp"] < 4.0)]
    print(f"\n=== catalog-to-catalog fit (deg {args.poly_degree}) ===")
    print(f"  catalog stars from {len(photometric_clusters)} photometric clusters: {len(cat_df)}")

    if args.multi_color:
        # Add DR10 colors and apply per-color quality cuts to drop cross-match
        # outliers and saturated-star noise.
        cat_df["gr"] = cat_df["mag_g_dr10"] - cat_df["mag_r_dr10"]
        cat_df["ri"] = cat_df["mag_r_dr10"] - cat_df["mag_i_dr10"]
        cat_df["iz"] = cat_df["mag_i_dr10"] - cat_df["mag_z_dr10"]
        before = len(cat_df)
        cat_df = cat_df[
            (cat_df["gr"] > 0.0) & (cat_df["gr"] < 2.5)
            & (cat_df["ri"] > -0.5) & (cat_df["ri"] < 2.0)
            & (cat_df["iz"] > -0.5) & (cat_df["iz"] < 1.5)
            & np.isfinite(cat_df["gr"]) & np.isfinite(cat_df["ri"]) & np.isfinite(cat_df["iz"])
        ].reset_index(drop=True)
        print(f"  after DR10 color quality cuts: {len(cat_df)}  (dropped {before - len(cat_df)})")

        coeffs_2d, monomials, mask_2d, rms_2d = multi_color_fit(
            cat_df, n_iter=args.n_iter, poly_degree=args.poly_degree
        )

        # Marginalize back to 1-D in (BP-RP) for production
        print("\n=== marginalize multi-color fit -> 1-D (BP-RP) polynomial ===")
        poly, cond_polys, bp_rp_grid, target_grid, weights_grid, grid_rms = \
            marginalize_to_1d(cat_df, mask_2d, monomials, coeffs_2d,
                              poly_degree=args.poly_degree)
        # Pin the same names the rest of the script uses for plotting
        bp_rp_cat = cat_df["bp_rp"].values
        target_cat = cat_df["target"].values
        ok = mask_2d
        rms_cat = rms_2d
        final_rms = rms_2d
        rms_history = [rms_2d]
    else:
        bp_rp_cat = cat_df["bp_rp"].values
        target_cat = cat_df["target"].values
        poly = np.polyfit(bp_rp_cat, target_cat, deg=args.poly_degree)[::-1]
        for it in range(args.n_iter):
            pred = np.polyval(poly[::-1], bp_rp_cat)
            clipped = sigma_clip(target_cat - pred, sigma=3.0, maxiters=3, masked=True)
            ok = ~clipped.mask
            poly = np.polyfit(bp_rp_cat[ok], target_cat[ok], deg=args.poly_degree)[::-1]
            rms_cat = float(np.sqrt(((target_cat[ok] - np.polyval(poly[::-1], bp_rp_cat[ok])) ** 2).mean()))
            print(f"  iter {it}: n_used={int(ok.sum())}/{len(ok)}  catalog-fit RMS={rms_cat:.4f} mag")
        rms_history = [rms_cat]
        final_rms = rms_cat
        bp_rp_grid = None
        target_grid = None
        grid_rms = None

    # ---------- Step 7: report ------------------------------------------
    # The previously-fitted g coefficients (1-D Gaia-only, no photometric
    # filter) — for side-by-side comparison.
    PREVIOUS_G_COEFFS = np.array([-0.092679, +0.165865, -0.728677, +0.156710])
    coeffs_str = ", ".join(f"{c:+.6f}" for c in poly)
    print()
    print("=" * 64)
    print(f"  filter:               {inferred_filt}")
    print(f"  chip:                 {inferred_chip}")
    print(f"  images:               {len(records)} ({len(photometric_image_ids)} photometric)")
    print(f"  catalog stars used:   {int(ok.sum())} / {len(cat_df)}")
    print(f"  mode:                 {'multi-color (DR10 g/r/i/z)' if args.multi_color else '1-D Gaia-only'}")
    print(f"  catalog-fit RMS:      {final_rms:.4f} mag")
    if args.multi_color and grid_rms is not None:
        print(f"  collapse RMS:         {grid_rms:.5f} mag (1-D approximation quality)")
    print(f"  per-image ZP median:  {zp_per_image.median():.3f} mag")
    print(f"  per-image ZP std:     {zp_per_image.std():.3f} mag (across all images)")
    print()
    if inferred_filt == "g":
        # Side-by-side comparison at typical color values
        print("  comparison vs current g coeffs in models/ls4cam.py:")
        print("    BP-RP    previous   new       delta")
        for c in (0.0, 0.5, 1.0, 1.5, 2.0):
            prev = float(np.polyval(PREVIOUS_G_COEFFS[::-1], c))
            new_ = float(np.polyval(poly[::-1], c))
            print(f"    {c:5.2f}   {prev:+8.4f}  {new_:+8.4f}  {(new_ - prev):+8.4f}")
        print()
    print("  PASTE INTO models/ls4cam.py transformations dict:")
    print(f"      '{inferred_filt}': np.array([ {coeffs_str} ]),")
    print("=" * 64)

    # ---------- Step 8: diagnostics -------------------------------------
    # Plot 1: catalog (BP-RP) vs (MAG_G - mag_truth) with fitted polynomial
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    color_grid = np.linspace(bp_rp_cat.min(), bp_rp_cat.max(), 400)
    axes[0].scatter(bp_rp_cat, target_cat, s=2, alpha=0.15, color="0.4")
    axes[0].plot(color_grid, np.polyval(poly[::-1], color_grid),
                 "r-", lw=2, label="fitted poly (deg "
                                   f"{args.poly_degree})")
    axes[0].set_xlabel("Gaia BP - RP")
    axes[0].set_ylabel(f"MAG_G - MAG_DR10_{inferred_filt}")
    axes[0].set_title(f"catalog-to-catalog: chip {inferred_chip}, {inferred_filt}")
    axes[0].set_ylim(np.percentile(target_cat, [1, 99]))
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Right: residual after polynomial subtraction
    resid = target_cat - np.polyval(poly[::-1], bp_rp_cat)
    axes[1].scatter(bp_rp_cat, resid, s=2, alpha=0.15, color="0.4")
    axes[1].axhline(0, color="r", lw=1)
    axes[1].set_xlabel("Gaia BP - RP")
    axes[1].set_ylabel(f"(MAG_G - MAG_DR10_{inferred_filt}) - poly")
    axes[1].set_title(f"residual: RMS={final_rms:.4f} mag, N={len(resid):,}")
    axes[1].set_ylim(np.percentile(resid, [1, 99]))
    axes[1].grid(alpha=0.3)
    p1 = outdir / f"{inferred_filt}_residual_vs_color.png"
    fig.savefig(p1, dpi=120)
    print(f"  wrote {p1}")

    # Plot 2: per-image ZP histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    zps = list(zp_per_image.values)
    ax2.hist(zps, bins=40, color="C0", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("per-image ZP (mag)")
    ax2.set_ylabel("# images")
    ax2.set_title(f"chip {inferred_chip}, {inferred_filt}: "
                  f"median={np.median(zps):.3f}, std={np.std(zps):.3f}")
    ax2.axvline(np.median(zps), color="red", lw=1, label="median")
    ax2.legend()
    p2 = outdir / f"{inferred_filt}_zp_per_image.png"
    fig2.savefig(p2, dpi=120)
    print(f"  wrote {p2}")

    # ----- Plot 3 (multi-color only): residual binned in DR10 colors --
    if args.multi_color and bp_rp_grid is not None:
        # 3a: marginalized 1-D function vs the production polynomial
        fig3, ax3 = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax3.plot(bp_rp_grid, target_grid, "k-", lw=2,
                 label="2-D fit evaluated along E[DR10|BP-RP]")
        ax3.plot(bp_rp_grid, np.polyval(poly[::-1], bp_rp_grid),
                 "r--", lw=1.5, label=f"deg-{args.poly_degree} 1-D fit (production)")
        ax3.set_xlabel("Gaia BP - RP")
        ax3.set_ylabel(f"MAG_G - MAG_DR10_{inferred_filt}")
        ax3.set_title(f"marginalized 1-D collapse "
                      f"(approx RMS = {grid_rms:.4f} mag)")
        ax3.legend()
        ax3.grid(alpha=0.3)
        p3 = outdir / f"{inferred_filt}_marginalized_1d.png"
        fig3.savefig(p3, dpi=120)
        print(f"  wrote {p3}")

        # 3b: residual after the multi-color fit, binned in each DR10 color.
        # Confirms the multi-color terms picked up real structure (good)
        # rather than just absorbed noise (bad).
        bp_rp_d = cat_df["bp_rp"].values
        gr_d = cat_df["gr"].values
        ri_d = cat_df["ri"].values
        iz_d = cat_df["iz"].values
        target_d = cat_df["target"].values
        pred_2d_full = build_multivar_features([bp_rp_d, gr_d, ri_d, iz_d], monomials) @ coeffs_2d
        resid_2d = target_d - pred_2d_full
        # Compare with pure 1-D fit residual
        poly_1d_only = np.polyfit(bp_rp_d[ok], target_d[ok], deg=args.poly_degree)[::-1]
        pred_1d_full = np.polyval(poly_1d_only[::-1], bp_rp_d)
        resid_1d = target_d - pred_1d_full
        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5),
                                   constrained_layout=True, sharey=True)
        for ax, (name, x_arr) in zip(axes4, [("g-r", gr_d), ("r-i", ri_d), ("i-z", iz_d)]):
            ax.scatter(x_arr[ok], resid_1d[ok], s=2, alpha=0.10, color="C0",
                       label="1-D fit residual")
            ax.scatter(x_arr[ok], resid_2d[ok], s=2, alpha=0.10, color="C3",
                       label="multi-color fit residual")
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel(f"DR10 {name}")
            ax.set_ylabel("M - prediction (mag)")
            ax.set_ylim(-0.2, 0.2)
            ax.set_title(f"residual vs {name}")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        p4 = outdir / f"{inferred_filt}_color_residual_check.png"
        fig4.savefig(p4, dpi=120)
        print(f"  wrote {p4}")

    # CSV summary
    summary = {
        "filter": inferred_filt,
        "chip": inferred_chip,
        "n_images": len(records),
        "n_photometric_images": len(photometric_image_ids),
        "n_stars_total": len(df),
        "n_stars_used": int(ok.sum()),
        "mode": "multi-color" if args.multi_color else "1-D Gaia-only",
        "rms_mag": rms_history[-1],
        "collapse_rms_mag": float(grid_rms) if grid_rms is not None else "",
        "zp_median": float(np.median(zps)),
        "zp_std": float(np.std(zps)),
    }
    for i, c in enumerate(poly):
        summary[f"poly_c{i}"] = float(c)
    csv = outdir / f"{inferred_filt}_fit_summary.csv"
    import pandas as pd
    pd.DataFrame([summary]).to_csv(csv, index=False)
    print(f"  wrote {csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
