#!/usr/bin/env python3
"""Gaia DR3 + LS4Cam-polynomial photometric zeropoint, mask-aware.

For a single LS4 image (typically ``*.image.w.fits.fz`` produced by
``solve_ls4.wcs``):

  1. SEP source extraction with the combined per-image mask passed in.
  2. **Reject any source whose Kron photometric aperture overlaps a
     masked pixel** (SEP's ``APER_HASMASKED`` flag bit = 0x10), and any
     source whose central pixel is masked.  This is the "stars on top of
     masked pixels" exclusion the user asked for — implemented at the
     aperture level rather than the central-pixel level so PSF wings
     touching a saturated or vignetted pixel still get tossed.
  3. Query Gaia DR3 in the field (DataLab preferred, astroquery fallback).
  4. Apply Gaia proper motion to the image MJD.
  5. Match source RA/Dec → Gaia (1.0″ default).
  6. Transform Gaia (G, BP, RP) → LS4 instrumental mag via the refit
     BP-RP polynomial.  Coefficients copied inline from
     ``models/ls4cam.py:645-668`` (Perlmutter doesn't have the SeeChange
     package installed).
  7. Per-star ZP = ``trans_mag + 2.5*log10(flux)``; image ZP = sigma-
     clipped median.
  8. Write ZP + image-quality keys back into the input image's WCS-
     bearing HDU header, and a per-image ``*.image.w.fits.cat`` BinTable
     of matched stars (extension uses ``.cat`` in place of ``.fz``,
     matching the existing ``zp.py`` convention so consumers don't
     need to learn a new filename).

Usage
-----
::

    zp_gaia_ls4cam.py <image.w.fits.fz> [mask.fits.fz]
        --detection-threshold 5.0
        --match-arcsec 1.0
        --min-snr 10.0
        --max-g-mag 20.0
        --min-g-mag 10.0

If the mask argument is omitted, no mask-based rejection happens and SEP
runs without a mask (useful for sanity tests; production always passes
the mask).
"""

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Quiet astropy chatter — there's a lot of "the WCS transformation has
# more axes (2) than the image it is associated with (0)" noise from the
# pixel_to_world step on minimal headers; not interesting.
warnings.filterwarnings("ignore")

try:
    import astropy.time
    import astropy.units as u
    from astropy.coordinates import Distance, SkyCoord
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from astropy.wcs import WCS
    import sep
except ImportError as exc:
    print(f"Missing required library: {exc}")
    print("Install with: pip install astropy sep astro-datalab")
    sys.exit(1)


# -------------------------------------------------------------------- constants

# 0.2: FWHM = 2*sqrt(2 ln2)*sigma.  0.1 used 2*sqrt(ln2)*sigma, so its SEEING is
#      low by sqrt(2) and its LMT_MG 0.376 mag too faint; ZPs are unaffected.
CODE_VERSION = "zp_gaia_ls4cam 0.2"
GAIA_EPOCH = astropy.time.Time("2016.0", format="jyear", scale="tdb")

# SEP flag bits.  In SEP <= 1.2.x there's no public constant for this, so
# we use the hard-coded value from sep.h.  APER_HASMASKED is set by
# sep.sum_ellipse / sum_circle whenever any pixel inside the aperture
# (or rejected by the mask) contributed to the photometry.
SEP_APER_HASMASKED = 0x10

# Bump SEP's per-image active-pixel stack.  Default is 300_000 which
# overflows on a handful of crowded fields (e.g. the SE_A frames at
# 08:25 UT 20260429); 3M is comfortably above the worst real case while
# still bounded.
try:
    sep.set_extract_pixstack(3_000_000)
except Exception:
    pass  # older sep versions don't have it; ignore.

# LS4Cam Gaia BP-RP polynomial coefficients (degree-3) — copied verbatim
# from models/ls4cam.py:645-668.  Refit by hacks/nugent/refit_ls4_color_terms.py
# --multi-color, marginalized to a 1-D Gaia-only production polynomial.
LS4_TRNS = {
    "g": np.array([+0.363464, -0.862719, -0.020064, +0.007459]),
    "i": np.array([-0.589334, +1.201847, -0.352709, +0.045252]),
    "z": np.array([-0.662140, +1.290285, -0.264194, +0.034393]),
}


# -------------------------------------------------------------------- helpers

def filter_from_filename(path):
    """Infer LS4 filter (g/i/z) from filename.  Returns None if unmatched."""
    m = re.search(r"_[NS][EW]_[A-Z]+_(g|i|z)_Sci_", Path(path).name)
    return m.group(1) if m else None


def chip_from_filename(path):
    """Infer CCD_LOC (e.g. 'NW_H') from filename.  Returns None if unmatched."""
    m = re.search(r"_([NS][EW]_[A-H])_(?:g|i|z)_Sci_", Path(path).name)
    return m.group(1) if m else None


def _open_image(path):
    """Return ``(hdulist, hdu_index)`` for the HDU carrying the science data.

    Astrometry.net's ``solve-field -N <out.fits.fz>`` keeps the input HDU
    layout, so a SeeChange ``.image.fits.fz`` round-trips as
    ``[PrimaryHDU(empty), CompImageHDU(data + WCS)]``.  We pick the
    first 2-D HDU and return its index so callers can mutate its header
    in place before writeto().
    """
    hdulist = fits.open(path, mode="readonly")
    for i, hdu in enumerate(hdulist):
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdulist, i
    hdulist.close()
    raise SystemExit(f"No 2-D image HDU in {path}")


def _read_mask(path):
    """Return the mask array (uint16/int) from a FITS / FITS.fz file."""
    with fits.open(path) as h:
        for hdu in h:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                return np.asarray(hdu.data)
    raise SystemExit(f"No 2-D image HDU in mask {path}")


# -------------------------------------------------------------------- Gaia query

# Search order for Gaia DR3 healpix-partitioned FITS files.
# pscratch is faster than CFS (closer to compute nodes); we try it first
# and fall back to CFS per-file so this works during a partial copy.
# Override with env: GAIA_LOCAL_DIRS='/path1:/path2:...'.
GAIA_LOCAL_DIRS = tuple(
    p for p in (os.environ.get(
        "GAIA_LOCAL_DIRS",
        "/pscratch/sd/n/nugent/healpix:/global/cfs/projectdirs/cosmo/data/gaia/dr3/healpix",
    ).split(":")) if p
)
GAIA_LOCAL_NSIDE = 32          # matches healpix-<idx:05d>.fits nside=32 NESTED layout
GAIA_LOCAL_COLS = (
    "RA", "DEC", "PMRA", "PMDEC",
    "PHOT_G_MEAN_MAG", "PHOT_G_MEAN_FLUX_OVER_ERROR",
    "PHOT_BP_MEAN_MAG", "PHOT_BP_MEAN_FLUX_OVER_ERROR",
    "PHOT_RP_MEAN_MAG", "PHOT_RP_MEAN_FLUX_OVER_ERROR",
)


def _query_gaia_local(ra_deg, dec_deg, radius_deg, max_g_mag, min_g_mag):
    """Query the local Gaia DR3 healpix-partitioned FITS files.

    Returns an astropy Table with the same columns/units as the
    DataLab/astroquery code path (post-_convert_gaia_columns), or None
    if the local catalog directory is unreachable.

    Cuts: brightness window on G, plus finite BP/RP (required by the
    LS4Cam BP-RP polynomial).  Exact distance cut is applied here so
    the caller never has to revisit it.
    """
    if not any(os.path.isdir(d) for d in GAIA_LOCAL_DIRS):
        return None
    try:
        from astropy_healpix import HEALPix
        import astropy.units as _u
    except ImportError:
        return None

    hp = HEALPix(nside=GAIA_LOCAL_NSIDE, order="nested")
    pix = hp.cone_search_lonlat(ra_deg * _u.deg, dec_deg * _u.deg,
                                radius_deg * _u.deg)
    chunks = []
    for p in pix:
        fp = None
        for d in GAIA_LOCAL_DIRS:
            cand = f"{d}/healpix-{int(p):05d}.fits"
            if os.path.exists(cand):
                fp = cand
                break
        if fp is None:
            continue
        with fits.open(fp, memmap=False) as hdul:
            d = hdul[1].data
            sel = ((d["PHOT_G_MEAN_MAG"] >= min_g_mag)
                   & (d["PHOT_G_MEAN_MAG"] <= max_g_mag)
                   & np.isfinite(d["PHOT_BP_MEAN_MAG"])
                   & np.isfinite(d["PHOT_RP_MEAN_MAG"]))
            chunks.append({c: np.asarray(d[c][sel]) for c in GAIA_LOCAL_COLS})
    if not chunks:
        return None
    cat = {c: np.concatenate([k[c] for k in chunks]) for c in GAIA_LOCAL_COLS}

    # Exact-disc cut via spherical law of cosines
    cosrad = np.cos(np.radians(radius_deg))
    cra, cdec = np.radians(ra_deg), np.radians(dec_deg)
    sra, sdec = np.radians(cat["RA"]), np.radians(cat["DEC"])
    cosd = (np.sin(cdec) * np.sin(sdec)
            + np.cos(cdec) * np.cos(sdec) * np.cos(sra - cra))
    in_disc = cosd >= cosrad
    cat = {c: cat[c][in_disc] for c in GAIA_LOCAL_COLS}

    # Build a Table in the schema the caller (and the LS4Cam transform)
    # expect: lowercase ra/dec/pmra/pmdec + MAG_G/BP/RP + MAGERR_G/BP/RP.
    # Convert flux_over_error -> mag err with floor (some sources have SN=0).
    def fe_to_magerr(arr):
        arr = np.asarray(arr, dtype=np.float64)
        out = np.full_like(arr, 1.0)
        good = arr > 0
        out[good] = 1.0857 / arr[good]
        return out

    t = Table()
    t["ra"]        = cat["RA"]
    t["dec"]       = cat["DEC"]
    t["pmra"]      = cat["PMRA"]
    t["pmdec"]     = cat["PMDEC"]
    t["MAG_G"]     = cat["PHOT_G_MEAN_MAG"]
    t["MAGERR_G"]  = fe_to_magerr(cat["PHOT_G_MEAN_FLUX_OVER_ERROR"])
    t["MAG_BP"]    = cat["PHOT_BP_MEAN_MAG"]
    t["MAGERR_BP"] = fe_to_magerr(cat["PHOT_BP_MEAN_FLUX_OVER_ERROR"])
    t["MAG_RP"]    = cat["PHOT_RP_MEAN_MAG"]
    t["MAGERR_RP"] = fe_to_magerr(cat["PHOT_RP_MEAN_FLUX_OVER_ERROR"])
    return t


def _convert_gaia_columns(t):
    """Rename Gaia DR3 columns to MAG_* / MAGERR_* (mag units) in place."""
    rename_map = {
        "phot_g_mean_mag":               "MAG_G",
        "phot_bp_mean_mag":              "MAG_BP",
        "phot_rp_mean_mag":              "MAG_RP",
        "phot_g_mean_flux_over_error":   "MAGERR_G",
        "phot_bp_mean_flux_over_error":  "MAGERR_BP",
        "phot_rp_mean_flux_over_error":  "MAGERR_RP",
    }
    for old, new in rename_map.items():
        if old in t.colnames:
            t.rename_column(old, new)
    # Gaia gives flux/flux_err (S/N); convert to magnitude error.
    for col in ("MAGERR_G", "MAGERR_BP", "MAGERR_RP"):
        if col in t.colnames:
            t[col] = 1.0857 / np.asarray(t[col], dtype=np.float64)
    return t


def query_gaia_dr3(ra_deg, dec_deg, radius_deg,
                   max_g_mag=20.0, min_g_mag=10.0, max_retries=2):
    """Return Table(ra, dec, pmra, pmdec, MAG_G/BP/RP, MAGERR_G/BP/RP).

    Order of attempt: local healpix files on /global/cfs (fast, no
    throttling), then NOIRLab DataLab, then astroquery.  The local
    path needs astropy_healpix + the dr3/healpix/ tree to be available;
    otherwise it returns None silently and we fall through to network.
    """
    local = _query_gaia_local(ra_deg, dec_deg, radius_deg, max_g_mag, min_g_mag)
    if local is not None and len(local) > 0:
        return local

    last_exc = None
    for attempt in range(max_retries):
        try:
            from dl import helpers, queryClient
            sql = (
                "SELECT ra, dec, pmra, pmdec, "
                "phot_g_mean_mag, phot_g_mean_flux_over_error, "
                "phot_bp_mean_mag, phot_bp_mean_flux_over_error, "
                "phot_rp_mean_mag, phot_rp_mean_flux_over_error "
                "FROM gaia_dr3.gaia_source "
                f"WHERE q3c_radial_query(ra, dec, {ra_deg}, {dec_deg}, {radius_deg}) "
                f"AND phot_g_mean_mag BETWEEN {min_g_mag} AND {max_g_mag} "
                "AND phot_bp_mean_mag IS NOT NULL "
                "AND phot_rp_mean_mag IS NOT NULL"
            )
            df = helpers.utils.convert(queryClient.query(sql=sql), "pandas")
            return _convert_gaia_columns(Table.from_pandas(df))
        except ImportError:
            break  # fall through to astroquery
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))  # 2s, 4s
                continue
            print(f"  WARN: DataLab failed after {max_retries} tries ({exc})")

    # astroquery fallback
    try:
        from astroquery.gaia import Gaia
        adql = (
            "SELECT ra, dec, pmra, pmdec, "
            "phot_g_mean_mag, phot_g_mean_flux_over_error, "
            "phot_bp_mean_mag, phot_bp_mean_flux_over_error, "
            "phot_rp_mean_mag, phot_rp_mean_flux_over_error "
            "FROM gaiadr3.gaia_source "
            "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
            f"                 CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})) "
            f"AND phot_g_mean_mag BETWEEN {min_g_mag} AND {max_g_mag} "
            "AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL"
        )
        job = Gaia.launch_job_async(adql, dump_to_file=False)
        return _convert_gaia_columns(job.get_results())
    except ImportError:
        if last_exc:
            raise last_exc
        raise SystemExit("Neither astro-datalab nor astroquery is installed.")


# -------------------------------------------------------------------- transform

def ls4cam_gaia_to_instrument(filter_short, mag_g, magerr_g,
                              mag_bp, magerr_bp, mag_rp, magerr_rp):
    """Apply the LS4Cam (Gaia BP-RP polynomial) transformation.

    Same logic as ``models/ls4cam.py:gaia_dr3_to_instrument_mag``;
    inlined here so this script has no SeeChange-package dependency.
    """
    if filter_short not in LS4_TRNS:
        raise ValueError(f"LS4Cam transformation undefined for filter '{filter_short}'")
    trns = LS4_TRNS[filter_short]
    fitorder = len(trns) - 1

    color = np.asarray(mag_bp, dtype=np.float64) - np.asarray(mag_rp, dtype=np.float64)
    colorerr = np.sqrt(
        np.asarray(magerr_bp, dtype=np.float64) ** 2
        + np.asarray(magerr_rp, dtype=np.float64) ** 2
    )

    colton = color[:, None] ** np.arange(0, fitorder + 1, 1)
    coltonm1 = np.zeros_like(colton)
    coltonm1[:, 1:] = color[:, None] ** np.arange(0, fitorder, 1)
    coltonerr = np.zeros_like(colton)
    coltonerr[:, 1:] = (
        np.arange(1, fitorder + 1, 1) * coltonm1[:, 1:] * colorerr[:, None]
    )

    trans_mag = np.asarray(mag_g, dtype=np.float64) - (trns[None, :] * colton).sum(axis=1)
    trans_magerr = np.sqrt(
        np.asarray(magerr_g, dtype=np.float64) ** 2
        + (trns[None, :] * coltonerr).sum(axis=1) ** 2
    )
    return trans_mag, trans_magerr


# -------------------------------------------------------------------- main

class GaiaZP:
    """Encapsulates the whole per-image ZP measurement."""

    def __init__(self, image_path, mask_path=None, **kwargs):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path) if mask_path is not None else None
        self.detection_threshold = kwargs.get("detection_threshold", 5.0)
        self.match_arcsec = kwargs.get("match_arcsec", 1.0)
        self.min_snr = kwargs.get("min_snr", 10.0)
        self.max_g_mag = kwargs.get("max_g_mag", 20.0)
        self.min_g_mag = kwargs.get("min_g_mag", 10.0)

        # Open image; remember which HDU carried the data so we can mutate
        # its header in place before writeto().
        self.hdulist, self.hdu_idx = _open_image(self.image_path)
        self.header = self.hdulist[self.hdu_idx].header
        self.data = self.hdulist[self.hdu_idx].data.astype(np.float64)
        print(f"image: {self.image_path}  (HDU {self.hdu_idx}, "
              f"shape={self.data.shape})")

        # Filter — header is often UNKNOWN; fall back to filename
        f = str(self.header.get("FILTER", "")).lower().strip()
        if f in ("", "unknown"):
            f = filter_from_filename(self.image_path) or "unknown"
        self.filter_short = f
        self.chip = chip_from_filename(self.image_path) or "?"
        print(f"  filter={self.filter_short}, chip={self.chip}")

        # WCS
        try:
            self.wcs = WCS(self.header)
            if not self.wcs.has_celestial:
                raise ValueError("WCS has no celestial axes")
        except Exception as exc:
            raise SystemExit(f"Bad WCS in {self.image_path}: {exc}")

        # MJD — DATE-OBS is the SeeChange convention
        try:
            self.mjd = astropy.time.Time(
                self.header["DATE-OBS"], scale="utc", format="isot"
            ).mjd
        except Exception:
            self.mjd = float(self.header.get("MJD-OBS", 0.0))
        if self.mjd <= 0:
            raise SystemExit(f"No usable MJD in {self.image_path}")

        # Mask
        if self.mask_path is None:
            print("  mask: (none — running unmasked)")
            self.mask = np.zeros_like(self.data, dtype=np.uint16)
        else:
            self.mask = _read_mask(self.mask_path).astype(np.uint16, copy=False)
            if self.mask.shape != self.data.shape:
                raise SystemExit(
                    f"Mask shape {self.mask.shape} != image shape {self.data.shape}"
                )
            n_bad = int((self.mask != 0).sum())
            print(f"  mask: {self.mask_path.name}  "
                  f"({n_bad:,} bad px, {100*n_bad/self.mask.size:.2f}%)")

    # ---------------------------------------------------------- extract

    def extract_sources(self):
        sep_mask = (self.mask != 0)
        data = self.data.copy()  # SEP modifies its input array
        bkg = sep.Background(data, mask=sep_mask)
        sub = data - bkg
        self.med_sky = float(np.median(bkg.back()))
        self.sky_sigma = float(bkg.globalrms)
        err_map = bkg.rms()  # 2-D RMS map for proper fluxerr propagation

        try:
            objs = sep.extract(
                sub, self.detection_threshold,
                err=err_map, mask=sep_mask,
            )
        except Exception as exc:
            raise SystemExit(f"SEP extract failed: {exc}")

        if len(objs) == 0:
            raise SystemExit("SEP found no sources")

        # SEP's _assert_ok rejects the whole batch if any object has
        # |theta| >= pi/2 (it requires strict inequality), b > a, or a/b
        # non-positive.  Even one ±pi/2 object — which sep.extract can
        # legitimately produce on rare highly-elongated sources — blows
        # up sum_ellipse for everyone.  Clip / sanitize per-object.
        a = np.asarray(objs["a"], dtype=np.float64)
        b = np.asarray(objs["b"], dtype=np.float64)
        theta = np.asarray(objs["theta"], dtype=np.float64)
        theta = np.clip(theta, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
        # Guard against degenerate or unordered axes
        a = np.where(np.isfinite(a) & (a > 0), a, 1.0)
        b = np.where(np.isfinite(b) & (b > 0) & (b <= a), b, np.minimum(a, 1.0))

        # Kron ellipse photometry (mask-aware)
        kron, krflag = sep.kron_radius(
            sub, objs["x"], objs["y"], a, b, theta, 6.0, mask=sep_mask,
        )
        kron = np.where(np.isfinite(kron) & (kron > 0), kron, 2.5)
        try:
            flux, fluxerr, sep_flag = sep.sum_ellipse(
                sub, objs["x"], objs["y"], a, b, theta,
                2.5 * kron, subpix=1, mask=sep_mask, err=err_map,
            )
        except Exception as exc:
            print(f"  WARN: sum_ellipse failed ({exc}); falling back to sum_circle")
            # Use a per-source circular aperture scaled by the source's
            # semi-major axis (still mask-aware so APER_HASMASKED works).
            radius = np.maximum(2.5, 2.5 * kron * np.sqrt(a * b))
            flux, fluxerr, sep_flag = sep.sum_circle(
                sub, objs["x"], objs["y"], radius, subpix=1, mask=sep_mask,
                err=err_map,
            )
        # Gaussian FWHM from the second-moment axes (a = b = sigma for a Gaussian)
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(a * b)

        # Mask-based rejections.  Sources we drop:
        #   - APER_HASMASKED: any pixel in the 2.5*Kron aperture was masked
        #     (this is the "star overlaps a masked pixel" cut the user asked
        #     for; it adapts to each star's own size).
        #   - central pixel masked (cheap secondary check; usually a
        #     subset of APER_HASMASKED but harmless to keep explicit).
        ix = np.clip(np.round(objs["x"]).astype(int), 0, self.data.shape[1] - 1)
        iy = np.clip(np.round(objs["y"]).astype(int), 0, self.data.shape[0] - 1)
        center_bad = self.mask[iy, ix] != 0
        aper_bad = (sep_flag & SEP_APER_HASMASKED) != 0

        finite = np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0)
        snr_ok = finite & (flux > self.min_snr * fluxerr)

        keep = snr_ok & ~aper_bad & ~center_bad

        n_total = len(objs)
        n_rej_aper = int((snr_ok & aper_bad).sum())
        n_rej_center = int((snr_ok & ~aper_bad & center_bad).sum())
        n_rej_snr = int((~snr_ok).sum())
        n_keep = int(keep.sum())
        print(f"  SEP: total={n_total}  "
              f"rej_aper_mask={n_rej_aper}  rej_center_mask={n_rej_center}  "
              f"rej_snr={n_rej_snr}  keep={n_keep}")
        if n_keep < 5:
            raise SystemExit(f"Only {n_keep} clean source(s) — too few for ZP")

        self.n_rej_aper = n_rej_aper
        self.n_rej_center = n_rej_center

        self.sources = Table()
        self.sources["x"] = objs["x"][keep]
        self.sources["y"] = objs["y"][keep]
        self.sources["flux"] = flux[keep]
        self.sources["fluxerr"] = fluxerr[keep]
        self.sources["fwhm"] = fwhm[keep]
        self.median_fwhm_all = float(np.median(self.sources["fwhm"]))
        print(f"  sky: med={self.med_sky:.2f}  rms={self.sky_sigma:.2f}  "
              f"FWHM(all)={self.median_fwhm_all:.2f} px")

    # ---------------------------------------------------------- match

    def query_and_match(self):
        ny, nx = self.data.shape
        # Field center & radius from image corners
        corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
        ccoords = [self.wcs.pixel_to_world(x, y) for x, y in corners]
        center = self.wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
        radius_deg = max(center.separation(c).deg for c in ccoords) * 1.05
        print(f"  field: center=({center.ra.deg:.4f}, {center.dec.deg:+.4f})  "
              f"radius={radius_deg:.4f} deg")

        cat = query_gaia_dr3(
            center.ra.deg, center.dec.deg, radius_deg,
            max_g_mag=self.max_g_mag, min_g_mag=self.min_g_mag,
        )
        if cat is None or len(cat) == 0:
            raise SystemExit("Gaia DR3 query returned no stars")
        print(f"  Gaia DR3 in field: {len(cat)}")

        # Proper motion → image MJD (NaN PMs treated as zero)
        pmra = np.nan_to_num(np.asarray(cat["pmra"], dtype=np.float64))
        pmdec = np.nan_to_num(np.asarray(cat["pmdec"], dtype=np.float64))
        cat_coords = SkyCoord(
            ra=np.asarray(cat["ra"]) * u.deg,
            dec=np.asarray(cat["dec"]) * u.deg,
            distance=Distance(1 * u.kpc),
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            obstime=GAIA_EPOCH,
        ).apply_space_motion(astropy.time.Time(self.mjd, format="mjd", scale="tdb"))

        # SEP pixel coords are 0-based already (extract returns 0-based).
        src_sky = self.wcs.pixel_to_world(self.sources["x"], self.sources["y"])
        idx, d2d, _ = src_sky.match_to_catalog_sky(cat_coords)
        sel = d2d < self.match_arcsec * u.arcsec
        n_matched = int(sel.sum())
        print(f"  matched ({self.match_arcsec:.1f}\"): {n_matched}")
        if n_matched < 5:
            raise SystemExit(f"Only {n_matched} matched stars — too few for ZP")

        self.matched_cat = cat[idx[sel]]
        self.matched_src = self.sources[sel]
        self.matched_dist = d2d[sel].arcsec
        # RA/Dec of the matched catalog stars (PM-applied), for the .cat
        self.matched_ra = cat_coords.ra.deg[idx[sel]]
        self.matched_dec = cat_coords.dec.deg[idx[sel]]

    # ---------------------------------------------------------- ZP

    def fit_zeropoint(self):
        trans_mag, trans_magerr = ls4cam_gaia_to_instrument(
            self.filter_short,
            self.matched_cat["MAG_G"], self.matched_cat["MAGERR_G"],
            self.matched_cat["MAG_BP"], self.matched_cat["MAGERR_BP"],
            self.matched_cat["MAG_RP"], self.matched_cat["MAGERR_RP"],
        )
        flux = np.asarray(self.matched_src["flux"], dtype=np.float64)
        finite = np.isfinite(trans_mag) & np.isfinite(flux) & (flux > 0)
        zps = trans_mag[finite] + 2.5 * np.log10(flux[finite])
        n_used = int(finite.sum())
        if n_used < 5:
            raise SystemExit(f"Only {n_used} finite ZP values — too few")
        zp_mean, zp_med, zp_std = sigma_clipped_stats(zps, sigma=3.0, maxiters=5)
        self.zp = float(zp_med)
        self.zp_std = float(zp_std)
        self.zp_err = float(zp_std / np.sqrt(max(n_used - 1, 1)))
        self.n_zp = n_used
        self.trans_mag = trans_mag
        self.median_fwhm_matched = float(np.median(self.matched_src["fwhm"]))
        print(f"  ZP={self.zp:.3f} +/- {self.zp_err:.3f}  "
              f"(scatter {self.zp_std:.3f}, n={n_used})")
        print(f"  FWHM(matched)={self.median_fwhm_matched:.2f} px")

    # ---------------------------------------------------------- header

    def update_header_and_write(self):
        # Pixel scale from WCS
        try:
            pm = self.wcs.pixel_scale_matrix
            pixel_scale = float(np.sqrt(np.abs(np.linalg.det(pm))) * 3600.0)
        except Exception:
            pixel_scale = 1.0
        # 1-sigma limiting magnitude in a 1-FWHM-radius circle
        try:
            n_pix = np.pi * (self.median_fwhm_matched ** 2)  # aperture area in px
            lmt_mg = -2.5 * np.log10(np.sqrt(n_pix) * self.sky_sigma) + self.zp
        except Exception:
            lmt_mg = float("nan")

        h = self.header
        zp_key = f"{self.filter_short.upper()}_ZP"
        h["SEEING"]   = (self.median_fwhm_matched, "Median FWHM from matched stars (px)")
        h["MEDSKY"]   = (self.med_sky, "Median sky background")
        h["SKYSIG"]   = (self.sky_sigma, "1-sigma sky")
        h["PIXSCALE"] = (pixel_scale, "Pixel scale (arcsec/px)")
        h["LMT_MG"]   = (lmt_mg, "1-sigma limiting mag")
        h[zp_key]     = (self.zp, f"Zeropoint for {self.filter_short} filter")
        h[f"{zp_key}_ERR"] = (self.zp_err, f"ZP s.e.m. for {self.filter_short}")
        h["ZP_NSTAR"] = (self.n_zp, "Stars used for ZP")
        h["ZP_SCAT"]  = (self.zp_std, "Sigma-clipped scatter of per-star ZP")
        h["ZP_MATCH"] = (self.match_arcsec, "Match radius (arcsec)")
        h["ZP_CAT"]   = ("GAIADR3+LS4CAM", "Photometric catalog/transformation")
        h["ZP_MSKR"]  = (self.n_rej_aper, "Sources rejected by APER_HASMASKED")
        h["ZP_MSKC"]  = (self.n_rej_center, "Sources rejected by central-pixel mask")
        h["ZP_VER"]   = (CODE_VERSION, "ZP code version")

        # Write back — header was a reference to hdulist[hdu_idx].header, so
        # the in-place mutations are already present.
        self.hdulist.writeto(self.image_path, overwrite=True)
        print(f"  wrote header to {self.image_path.name}")

    def write_catalog(self):
        cat_path = self.image_path.with_suffix(".cat")
        primary = fits.PrimaryHDU()
        cols = [
            fits.Column(name="X_IMAGE",  format="E",
                        array=np.asarray(self.matched_src["x"])),
            fits.Column(name="Y_IMAGE",  format="E",
                        array=np.asarray(self.matched_src["y"])),
            fits.Column(name="RA",       format="D", unit="deg",
                        array=self.matched_ra),
            fits.Column(name="DEC",      format="D", unit="deg",
                        array=self.matched_dec),
            fits.Column(name="FLUX",     format="E",
                        array=np.asarray(self.matched_src["flux"])),
            fits.Column(name="FLUXERR",  format="E",
                        array=np.asarray(self.matched_src["fluxerr"])),
            fits.Column(name="FWHM",     format="E", unit="pix",
                        array=np.asarray(self.matched_src["fwhm"])),
            fits.Column(name="MAG_G",    format="E",
                        array=np.asarray(self.matched_cat["MAG_G"])),
            fits.Column(name="MAG_BP",   format="E",
                        array=np.asarray(self.matched_cat["MAG_BP"])),
            fits.Column(name="MAG_RP",   format="E",
                        array=np.asarray(self.matched_cat["MAG_RP"])),
            fits.Column(name="TRANS_MAG", format="E",
                        array=np.asarray(self.trans_mag)),
            fits.Column(name="MATCH_DIST", format="E", unit="arcsec",
                        array=np.asarray(self.matched_dist)),
        ]
        tbl = fits.BinTableHDU.from_columns(cols, name="CATALOG")
        tbl.header["FILTER"]  = self.filter_short
        tbl.header["CCD_LOC"] = self.chip
        tbl.header["ZP"]      = (self.zp, "Per-star ZP image median")
        tbl.header["ZP_ERR"]  = (self.zp_err, "ZP s.e.m.")
        tbl.header["ZP_CAT"]  = "GAIADR3+LS4CAM"
        fits.HDUList([primary, tbl]).writeto(cat_path, overwrite=True)
        print(f"  wrote catalog: {cat_path.name}  ({len(self.matched_src)} stars)")

    def run(self):
        self.extract_sources()
        self.query_and_match()
        self.fit_zeropoint()
        self.update_header_and_write()
        self.write_catalog()
        return True

    def __del__(self):
        try:
            self.hdulist.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("fits_file", help="LS4 WCS-solved image (*.image.w.fits.fz)")
    ap.add_argument("mask_file", nargs="?", default=None,
                    help="Combined per-image mask (*.mask.fits.fz); optional")
    ap.add_argument("--detection-threshold", "-t", type=float, default=5.0,
                    help="SEP detection threshold (sigma)")
    ap.add_argument("--match-arcsec", "-r", type=float, default=1.0,
                    help="Source-catalog match radius (arcsec)")
    ap.add_argument("--min-snr", type=float, default=10.0,
                    help="Minimum source S/N to keep")
    ap.add_argument("--max-g-mag", type=float, default=20.0,
                    help="Faint G-mag limit on Gaia query")
    ap.add_argument("--min-g-mag", type=float, default=10.0,
                    help="Bright G-mag limit on Gaia query")
    args = ap.parse_args()

    if not Path(args.fits_file).exists():
        sys.exit(f"ERROR: image {args.fits_file} not found")
    if args.mask_file is not None and not Path(args.mask_file).exists():
        sys.exit(f"ERROR: mask {args.mask_file} not found")

    zp = GaiaZP(
        args.fits_file, args.mask_file,
        detection_threshold=args.detection_threshold,
        match_arcsec=args.match_arcsec,
        min_snr=args.min_snr,
        max_g_mag=args.max_g_mag,
        min_g_mag=args.min_g_mag,
    )
    return 0 if zp.run() else 1


if __name__ == "__main__":
    sys.exit(main())
