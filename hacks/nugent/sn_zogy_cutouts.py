#!/usr/bin/env python3
"""ZOGY light curve of one SN from SeeChange image cutouts against a DECam reference.

Same machinery as sn_zogy_lc.py, imported from it (TAN grid, Gaia
registration, empirical PSF, Gaia/LS4Cam PSF-fit zero point, ZOGY via
code/zogylib, light curve).  What differs is the input:

  science  SeeChange cutouts: <stem>.image/.weight/.flags.fits.fz, WCS in
           the <stem>.wcs_*.txt sidecar (e.g. Rob Knop's 256x256 cutouts
           around an SN).  The reprocessed frames carry no row stripes (row
           medians sit at the noise level) and their weights match the pixel
           RMS, so there is no destriping here.
  ref      DECam CP instcal frames (*_ooi_*, with *_oow_* and *_ood_*): the
           CCD holding the SN is cut out, sky-subtracted, smoothed with a
           Gaussian to a common REF_FWHM (well sampled on the 1"/px grid and
           matched between frames), registered to Gaia, resampled onto the
           grid and calibrated with the same LS4Cam Gaia->g transform as the
           LS4 frames (LS4 g ~ DECam g: negligible colour term); then coadded.

Backgrounds are constants per frame (a mesh background would eat host
light); the residual plane between each frame and the reference is fitted
off sources and removed just before ZOGY.

Stages (each reads the previous one's outputs in --workdir):
  prep  cutouts -> prep/<night>_<expid>_<chip>.fits, DECam -> prep_ref/decam_*.fits
  ref   coadd of prep_ref/ -> ref/ref.fits
  diff  per LS4 frame: background match (plane + local N-R mesh), ZOGY,
        controls, fakes; also the photon
        noise of SN + host (gain from sky level vs pixel variance over all
        frames) and the PSF-flux ratio of field stars new/ref
  lc    sn_zogy_lc.py lc --null-before MJD (pre-explosion frames are the null set)

Usage:
  sn_zogy_cutouts.py prep --cutouts 'ls4/*.image.fits.fz' --decam 'decam/*/*_ooi_*.fits.fz' \\
                          --ra 191.681309 --dec -40.025125 --workdir W
  sn_zogy_cutouts.py ref  --workdir W
  sn_zogy_cutouts.py diff --workdir W
  sn_zogy_lc.py lc --workdir W --disc-mjd 61184.397 --null-before 61183.4 --name "SN 2026nmw"
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
import sep

import sn_zogy_lc as base

# The cutouts are 256 x 1.018" px, centred on the SN to < 1 px: a 240" grid
# keeps ~9" of margin.  Even side, as in sn_zogy_lc.py.  The base functions
# read these module globals at call time (workers are forked, so they see
# them too).
base.NPIX = 240
base.C = base.NPIX // 2
base.R_STARS = 110
NPIX, C, SCALE, ZP_OUT, PSF_HALF = base.NPIX, base.C, base.SCALE, base.ZP_OUT, base.PSF_HALF

REF_FWHM = 2.2           # arcsec: DECam smoothed to this (2.2 px; LS4 frames are 2.4-3.5")
# characterize() settings.  A 240" field has only ~15 PSF stars, so the PSF
# wings are annulus medians (wing="azimuthal"; see characterize).
# DECam saturates near G ~ 16: its astrometric / PSF / ZP stars start fainter.
DECAM_CHAR = dict(ast_g=(16.0, 20.0), psf_g=(16.5, 20.0), zp_g=(16.5, 19.5), g_bright=19.0,
                  wing="azimuthal")
# LS4: only ~4 usable stars at G < 17, so the S/N predictor's ZP also takes
# G 17-18 (S/N >~ 30 in a 60 s frame)
LS4_CHAR = dict(g_bright=18.0, wing="azimuthal")
GAIA_RAD = 0.07          # deg: covers the cutout corners and the DECam crops


def pool(n):
    return ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork"))


# ------------------------------------------------------------------ helpers

def grid_to_input(win, gw, n=25):
    """Input-pixel coordinates (full distortion) of every grid pixel, [y, x]."""
    cg = np.linspace(0, NPIX - 1, n)
    cx, cy = np.meshgrid(cg, cg)
    ra, dec = gw.wcs_pix2world(cx.ravel(), cy.ravel(), 0)
    xin, yin = win.all_world2pix(ra, dec, 0, quiet=True)
    sx = RectBivariateSpline(cg, cg, xin.reshape(cx.shape))
    sy = RectBivariateSpline(cg, cg, yin.reshape(cx.shape))
    fg = np.arange(NPIX, dtype=float)
    return sx(fg, fg), sy(fg, fg)


def flat_sky(img, bad, bw):
    """Constant sky = clipped median of source-masked pixels, the pixel RMS
    there, and the source mask.  The mesh background is for detection only."""
    a = np.ascontiguousarray(np.where(bad, 0.0, img))
    bk = sep.Background(a, mask=bad, bw=bw, bh=bw)
    _, seg = sep.extract(np.ascontiguousarray(a - bk.back()), 1.5, err=bk.globalrms, mask=bad,
                         segmentation_map=True, minarea=5)
    m = ndimage.binary_dilation(seg > 0, iterations=4) | bad
    _, med, _ = sigma_clipped_stats(img[~m], sigma=3.0, maxiters=5)
    return float(med), float(1.4826 * np.median(np.abs(img[~m] - med))), m


def register(sub, bad, gaia, mjd, win, x0, y0, g_range, iso, wsig, tol=3.0):
    """Affine correction of `win` from windowed centroids of Gaia stars, as
    in sn_zogy_lc.prep_one.  `sub` is the sky-subtracted crop whose pixel
    (0, 0) is input pixel (x0, y0).  Returns (cx, cy, rms, n) with
    true = predicted + c[0] + c[1]*lx + c[2]*ly in crop coordinates."""
    H, W = sub.shape
    gxi, gyi = base.gaia_xy(gaia, mjd, win, sip=True)
    G = np.asarray(gaia["G"], float)
    lx, ly = gxi - x0, gyi - y0
    cand = ((lx > 10) & (lx < W - 10) & (ly > 10) & (ly < H - 10) &
            (G > g_range[0]) & (G < g_range[1]))
    cand &= base.isolated(gxi, gyi, gxi, gyi, G, G, iso) if cand.any() else cand
    cand &= ~ndimage.binary_dilation(bad, iterations=3)[
        np.clip(np.round(ly).astype(int), 0, H - 1), np.clip(np.round(lx).astype(int), 0, W - 1)]
    wx, wy, wf = sep.winpos(np.ascontiguousarray(sub), lx[cand], ly[cand], wsig)
    ok = (wf == 0) & (np.hypot(wx - lx[cand], wy - ly[cand]) < tol)
    px, py = lx[cand][ok], ly[cand][ok]
    ddx, ddy = (wx - lx[cand])[ok], (wy - ly[cand])[ok]
    if px.size < 10:
        raise RuntimeError(f"only {px.size} astrometric stars")
    A = np.column_stack([np.ones_like(px), px, py])
    use = np.ones(px.size, bool)
    for _ in range(4):
        cxf, *_ = np.linalg.lstsq(A[use], ddx[use], rcond=None)
        cyf, *_ = np.linalg.lstsq(A[use], ddy[use], rcond=None)
        res = np.hypot(ddx - A @ cxf, ddy - A @ cyf)
        use = res < max(3.0 * np.median(res[use]) * 1.4826, 0.15)
    return cxf, cyf, float(np.sqrt(np.mean(res[use] ** 2))), int(use.sum())


def resample(crop, cbad, XIN, YIN, x0, y0, cxf, cyf, grow):
    lx, ly = XIN - x0, YIN - y0
    lxg = lx + cxf[0] + cxf[1] * lx + cxf[2] * ly
    lyg = ly + cyf[0] + cyf[1] * lx + cyf[2] * ly
    out = ndimage.map_coordinates(crop, [lyg, lxg], order=3, mode="constant", cval=0.0)
    obad = ndimage.map_coordinates(cbad.astype(np.float32), [lyg, lxg], order=1,
                                   mode="constant", cval=1.0) > 1e-3
    obad = ndimage.binary_dilation(obad, iterations=grow)
    out[obad] = np.nan
    return out, obad


def finite(v, fill=-99.0):
    return float(v) if np.isfinite(v) else fill


def frame_header(gw, mjd, ch, s, extra):
    """Grid header with the keys sn_zogy_lc.diff_one reads, plus `extra`."""
    oh = gw.to_header()
    for key, val, com in (
            ("MJD-OBS", mjd, "frame MJD (start)"), ("ZP_OUT", ZP_OUT, "pixel units: uJy"),
            ("ZP_NAT", ch["zp"], "native PSF-fit zero point (Gaia/LS4Cam)"),
            ("ZP_ERR", ch["zp_err"], "ZP s.e.m."), ("ZP_SCAT", ch["zp_scat"], ""),
            ("ZP_N", ch["n_zp"], "ZP stars"),
            ("SKYSIG", ch["sky"] * s, "sky RMS per px (uJy)"),
            ("FWHM_PX", ch["fwhm_px"], "empirical PSF FWHM (px)"),
            ("NPSF", ch["n_psf"], "PSF stars"),
            ("ASTDX", finite(ch["ast_dx"]), "post-resample median dx vs Gaia (px)"),
            ("ASTDY", finite(ch["ast_dy"]), "post-resample median dy vs Gaia (px)"),
            ("ASTRMS", finite(ch["ast_rms"]), "post-resample RMS vs Gaia (px)")) + extra:
        oh[key] = (val, com)
    return oh


# ------------------------------------------------------------------ prep

def cutout_wcs(stem):
    txt = open(glob.glob(stem + ".wcs_*.txt")[0]).read().replace("\\n", "\n")
    return WCS(fits.Header.fromstring(txt, sep="\n"))


def prep_cutout(job):
    path, gaia, ra0, dec0, outdir = job
    m = re.match(r"ls4_(\d{8})_(\d{6})_([A-Z]{2}_[A-Z])_(\w+?)_", os.path.basename(path))
    night, expid, chip, band = m.groups()
    tag = f"{night}_{expid}_{chip}"
    try:
        stem = path[:-len(".image.fits.fz")]
        with fits.open(path) as h:
            img = h[1].data.astype(np.float64)
            hdr = h[1].header.copy()
        wgt = fits.getdata(stem + ".weight.fits.fz", 1).astype(np.float64)
        flg = fits.getdata(stem + ".flags.fits.fz", 1)
        bad = np.ascontiguousarray((flg != 0) | ~(wgt > 0) | ~np.isfinite(img))
        win = cutout_wcs(stem)
        mjd = float(Time(hdr["DATE-OBS"], scale="utc").mjd)
        gw = base.grid_wcs(ra0, dec0)

        sky, sig, srcm = flat_sky(img, bad, 64)
        wsig = float(np.median(1.0 / np.sqrt(wgt[~srcm])))
        crop = np.where(bad, 0.0, img - sky)
        cxf, cyf, reg_rms, reg_n = register(crop, bad, gaia, mjd, win, 0, 0, (13.5, 18.0), 8.0, 1.3)
        XIN, YIN = grid_to_input(win, gw)
        outimg, obad = resample(crop, bad, XIN, YIN, 0, 0, cxf, cyf, 2)

        ch = base.characterize(outimg, obad, gaia, mjd, gw, **LS4_CHAR)
        s = 10 ** (0.4 * (ZP_OUT - ch["zp"]))
        rr = np.hypot(*np.meshgrid(np.arange(-8, 9), np.arange(-8, 9)))
        snbad = float(obad[C - 8:C + 9, C - 8:C + 9][rr <= 6].mean())
        oh = frame_header(gw, mjd, ch, s, (
            ("NIGHT", night, ""), ("EXPID", expid, ""), ("CHIP", chip, ""), ("BAND", band, ""),
            ("ZP_HDR", -99.0, "no header ZP on SeeChange cutouts"),
            ("EXPTIME", float(hdr["EXPTIME"]), ""),
            ("SKYDN", sky, "constant sky level removed (DN)"),
            ("SIGDN", sig, "sky pixel RMS (DN)"),
            ("WSIGDN", wsig, "median weight-map sigma (DN)"),
            ("REGRMS", reg_rms, "registration residual RMS (input px)"),
            ("REGN", reg_n, "registration stars"),
            ("SNBAD", snbad, "flagged fraction within 6 px of SN"),
            ("SRCFRAME", os.path.basename(stem)[:68], "")))
        base.write_grid(f"{outdir}/{tag}.fits", outimg * s, obad, ch["psf"], oh)
        return dict(tag=tag, kind="ls4", ok=True, mjd=mjd, zp=ch["zp"], zp_err=ch["zp_err"],
                    n_zp=ch["n_zp"], n_psf=ch["n_psf"], sky=ch["sky"] * s,
                    fwhm=ch["fwhm_px"] * SCALE, reg_rms=reg_rms, reg_n=reg_n,
                    ast_rms=ch["ast_rms"], snbad=snbad, sky_dn=sky, sig_dn=sig, wsig_dn=wsig)
    except Exception as e:
        return dict(tag=tag, kind="ls4", ok=False, err=f"{type(e).__name__}: {e}")


def prep_decam(job):
    path, gaia, ra0, dec0, outdir = job
    m = re.match(r"c4d_(\d{6})_(\d{6})_ooi_(\w)_", os.path.basename(path))
    tag = f"decam_{m.group(1)}_{m.group(2)}"
    try:
        with fits.open(path) as h:
            ph = h[0].header.copy()
            for k, hd in enumerate(h[1:], 1):
                hh = hd.header
                w = WCS(hh)
                nx, ny = hh["NAXIS1"], hh["NAXIS2"]
                x, y = w.wcs_world2pix(ra0, dec0, 0)
                if 0 <= x < nx and 0 <= y < ny:
                    ext, hdr, full = hh["EXTNAME"], hh.copy(), hd.data.astype(np.float64)
                    break
            else:
                raise RuntimeError("SN on no CCD")

        # weight / DQ by HDU index (extension versions differ between CP releases)
        def sibling(kind):
            with fits.open(path.replace("_ooi_", f"_{kind}_")) as hs:
                if hs[k].header.get("EXTNAME") != ext:
                    raise RuntimeError(f"{kind} HDU {k} is {hs[k].header.get('EXTNAME')}, not {ext}")
                return hs[k].data
        wt = sibling("oow").astype(np.float64)
        dq = sibling("ood")
        pix = float(np.mean(proj_plane_pixel_scales(w)) * 3600.0)
        mjd = float(ph["MJD-OBS"])
        fwhm_nat = float(hdr.get("FWHM", np.nan)) * pix
        if not 0.5 < fwhm_nat < 2.1:
            raise RuntimeError(f"CP seeing {fwhm_nat:.2f}\" unusable for REF_FWHM {REF_FWHM}\"")
        gw = base.grid_wcs(ra0, dec0)

        half = int(NPIX * SCALE / 2 / pix) + 60
        xi, yi = int(round(float(x))), int(round(float(y)))
        x0, x1 = max(0, xi - half), min(nx, xi + half + 1)
        y0, y1 = max(0, yi - half), min(ny, yi + half + 1)
        img = full[y0:y1, x0:x1]
        cbad = np.ascontiguousarray((dq[y0:y1, x0:x1] != 0) | ~(wt[y0:y1, x0:x1] > 0) |
                                    ~np.isfinite(img))
        sky, sig, _ = flat_sky(img, cbad, 128)
        crop = np.where(cbad, 0.0, img - sky)
        cxf, cyf, reg_rms, reg_n = register(crop, cbad, gaia, mjd, w, x0, y0, DECAM_CHAR["ast_g"],
                                            8.0 / pix, fwhm_nat / pix / 2.3548)

        # smooth to REF_FWHM by normalised convolution (flagged pixels carry
        # no weight; where they dominate the kernel the output is flagged)
        ks = np.sqrt(REF_FWHM ** 2 - fwhm_nat ** 2) / 2.3548 / pix
        good = (~cbad).astype(np.float64)
        num = ndimage.gaussian_filter(crop * good, ks, mode="constant")
        den = ndimage.gaussian_filter(good, ks, mode="constant")
        sbad = den < 0.97
        sm = np.where(sbad, 0.0, num / np.maximum(den, 1e-6)) * (SCALE / pix) ** 2

        XIN, YIN = grid_to_input(w, gw)
        outimg, obad = resample(sm, sbad, XIN, YIN, x0, y0, cxf, cyf, 1)
        ch = base.characterize(outimg, obad, gaia, mjd, gw, **DECAM_CHAR)
        s = 10 ** (0.4 * (ZP_OUT - ch["zp"]))
        oh = frame_header(gw, mjd, ch, s, (
            ("CHIP", f"DECam_{ext}", ""), ("BAND", ph.get("BAND", "g"), ""),
            ("EXPTIME", float(ph["EXPTIME"]), ""), ("CPSEEING", fwhm_nat, "CP FWHM (arcsec)"),
            ("KERNSIG", ks, "Gaussian smoothing sigma (native px)"),
            ("SKYDN", sky, "constant sky level removed (native)"),
            ("REGRMS", reg_rms, "registration residual RMS (native px)"),
            ("REGN", reg_n, "registration stars"),
            ("COVER", float((~obad).mean()), "unflagged fraction of the grid"),
            ("SRCFRAME", os.path.basename(path)[:68], "")))
        base.write_grid(f"{outdir}/{tag}.fits", outimg * s, obad, ch["psf"], oh)
        return dict(tag=tag, kind="decam", ok=True, mjd=mjd, zp=ch["zp"], zp_err=ch["zp_err"],
                    n_zp=ch["n_zp"], n_psf=ch["n_psf"], sky=ch["sky"] * s,
                    fwhm=ch["fwhm_px"] * SCALE, reg_rms=reg_rms * pix / SCALE, reg_n=reg_n,
                    ast_rms=ch["ast_rms"], cover=float((~obad).mean()), seeing=fwhm_nat)
    except Exception as e:
        return dict(tag=tag, kind="decam", ok=False, err=f"{type(e).__name__}: {e}")


def cmd_prep(a):
    from zp_gaia_ls4cam import query_gaia_dr3, ls4cam_gaia_to_instrument
    for d in ("prep", "prep_ref"):
        os.makedirs(f"{a.workdir}/{d}", exist_ok=True)
    cuts = sorted(glob.glob(a.cutouts))
    dec_ = sorted(glob.glob(a.decam))
    cat = query_gaia_dr3(a.ra, a.dec, GAIA_RAD, max_g_mag=20.5, min_g_mag=8.0)
    gm, _ = ls4cam_gaia_to_instrument(
        a.band, np.asarray(cat["MAG_G"]), np.asarray(cat["MAGERR_G"]),
        np.asarray(cat["MAG_BP"]), np.asarray(cat["MAGERR_BP"]),
        np.asarray(cat["MAG_RP"]), np.asarray(cat["MAGERR_RP"]))
    gaia = Table({"ra": np.asarray(cat["ra"], float), "dec": np.asarray(cat["dec"], float),
                  "pmra": np.asarray(cat["pmra"], float), "pmdec": np.asarray(cat["pmdec"], float),
                  "G": np.asarray(cat["MAG_G"], float), "gmag": np.asarray(gm, float)})
    gaia.write(f"{a.workdir}/gaia.ecsv", format="ascii.ecsv", overwrite=True)
    json.dump(dict(ra=a.ra, dec=a.dec, band=a.band, npix=NPIX, scale=SCALE, ref_fwhm=REF_FWHM),
              open(f"{a.workdir}/grid.json", "w"))
    print(f"Gaia: {len(gaia)} stars within {GAIA_RAD} deg; {len(cuts)} cutouts, "
          f"{len(dec_)} DECam frames", flush=True)

    jobs = ([(prep_cutout, (p, gaia, a.ra, a.dec, f"{a.workdir}/prep")) for p in cuts] +
            [(prep_decam, (p, gaia, a.ra, a.dec, f"{a.workdir}/prep_ref")) for p in dec_])
    with pool(a.nproc) as ex:
        res = list(ex.map(_call, jobs))
    keys = ["tag", "kind", "ok", "mjd", "zp", "zp_err", "n_zp", "n_psf", "sky", "fwhm", "reg_rms",
            "reg_n", "ast_rms", "snbad", "sky_dn", "sig_dn", "wsig_dn", "cover", "seeing", "err"]
    with open(f"{a.workdir}/prep.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in res:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in sorted(res, key=lambda r: (r["kind"], r.get("mjd", 0))):
        if r["ok"]:
            extra = (f"cover {r['cover']:.2f}  CP seeing {r['seeing']:.2f}\"" if r["kind"] == "decam"
                     else f"sigDN {r['sig_dn']:.2f} (weight {r['wsig_dn']:.2f})  snbad {r['snbad']:.2f}")
            print(f"  {r['tag']:24s} zp {r['zp']:.3f}+-{r['zp_err']:.3f} (N={r['n_zp']}, "
                  f"psf {r['n_psf']})  sky {r['sky']:.3f} uJy  fwhm {r['fwhm']:.2f}\"  "
                  f"reg {r['reg_rms']:.3f}px/{r['reg_n']}  ast {r['ast_rms']:.3f}  {extra}")
        else:
            print(f"  {r['tag']:24s} FAILED  {r['err']}")


def _call(fj):
    return fj[0](fj[1])


# ------------------------------------------------------------------ ref

def cmd_ref(a):
    g = json.load(open(f"{a.workdir}/grid.json"))
    gw = base.grid_wcs(g["ra"], g["dec"])
    gaia = Table.read(f"{a.workdir}/gaia.ecsv", format="ascii.ecsv")
    os.makedirs(f"{a.workdir}/ref", exist_ok=True)
    frames, names, mjds = [], [], []
    for p in sorted(glob.glob(f"{a.workdir}/prep_ref/*.fits")):
        img, bad, psf, hdr = base.read_grid(p)
        frames.append((img, bad, hdr["SKYSIG"]))
        names.append(os.path.basename(p)[6:-5])
        mjds.append(hdr["MJD-OBS"])
    if len(frames) < 2:
        raise SystemExit(f"only {len(frames)} DECam frames")
    ref, bad, nuse = base.coadd(frames, min_use=2)
    mjd = float(np.mean(mjds))
    ch = base.characterize(ref, bad, gaia, mjd, gw, **DECAM_CHAR)
    s = 10 ** (0.4 * (ZP_OUT - ch["zp"]))       # renormalise exactly to ZP_OUT
    h = gw.to_header()
    h["MJD-OBS"] = (mjd, "mean MJD of members")
    h["NMEMBER"] = len(frames)
    h["SKYSIG"] = (ch["sky"] * s, "sky RMS per px (uJy)")
    h["FWHM_PX"] = ch["fwhm_px"]
    h["ZP_RENRM"] = (float(-2.5 * np.log10(s)), "mag renormalisation applied")
    h["ZP_N"] = ch["n_zp"]
    h["ZP_SCAT"] = ch["zp_scat"]
    h["NPSF"] = ch["n_psf"]
    h["MEMBERS"] = ",".join(names)[:68]
    base.write_grid(f"{a.workdir}/ref/ref.fits", ref * s, bad, ch["psf"], h)
    fits.writeto(f"{a.workdir}/ref/ref_nuse.fits", nuse.astype(np.int16), overwrite=True)
    print(f"reference: {len(frames)} DECam frames ({', '.join(names)})")
    print(f"  sky {ch['sky']*s:.4f} uJy  fwhm {ch['fwhm_px']*SCALE:.2f}\"  renorm "
          f"{-2.5*np.log10(s):+.4f} mag  (ZP N={ch['n_zp']}, scat {ch['zp_scat']:.3f}, "
          f"PSF stars {ch['n_psf']})  frames used at SN {nuse[C, C]}  flagged {bad.mean():.3f}")


# ------------------------------------------------------------------ diff

def psf_flux(img, psf, x, y, fh=base.FIT_HALF):
    i, j = int(round(x)), int(round(y))
    sl = np.s_[j - fh:j + fh + 1, i - fh:i + fh + 1]
    st = img[sl]
    if st.shape != (2 * fh + 1,) * 2 or not np.isfinite(st).all():
        return np.nan
    p = base.shift_psf(psf, x - i, y - j)[PSF_HALF - fh:PSF_HALF + fh + 1,
                                          PSF_HALF - fh:PSF_HALF + fh + 1]
    A = np.column_stack([p.ravel(), np.ones(p.size)])
    (f, _), *_ = np.linalg.lstsq(A, st.ravel(), rcond=None)
    return float(f)


def match_plane(N, R, mask):
    """Plane a + b*(x-C) + c*(y-C) fitted (clipped) to N - R off sources."""
    yy, xx = np.indices(N.shape)
    D = N - R
    use = ~mask & np.isfinite(D)
    A = np.column_stack([np.ones(use.sum()), xx[use] - C, yy[use] - C])
    d = D[use]
    keep = np.ones(d.size, bool)
    for _ in range(5):
        cf, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        r = d - A @ cf
        keep = np.abs(r) < 3.0 * 1.4826 * np.median(np.abs(r[keep]))
    return cf, cf[0] + cf[1] * (xx - C) + cf[2] * (yy - C)


def local_mask(R, rbad, rsn=10):
    """What the local background of N - R must not see: compact structure in
    the reference (stars, knots, the galaxy core -- where PSF mismatch leaves
    residuals), flagged pixels, and the SN itself (r <= rsn px)."""
    Rf = np.nan_to_num(R)
    Rs = ndimage.gaussian_filter(Rf, 2.0)
    m = (Rf - ndimage.median_filter(Rf, 9)) > 1.0
    m |= (Rs - ndimage.median_filter(Rs, 25)) > 2.0
    yy, xx = np.indices(R.shape)
    return ndimage.binary_dilation(m, iterations=3) | rbad | (np.hypot(xx - C, yy - C) <= rsn)


def match_local(D, mask, bw):
    """Mesh background (bw px, 3x3 median filtered) of D = N - R; meshes that
    are mostly masked -- the SN hole -- are interpolated from their neighbours."""
    m = np.ascontiguousarray(mask | ~np.isfinite(D))
    return sep.Background(np.ascontiguousarray(np.where(m, 0.0, D)), mask=m,
                          bw=bw, bh=bw, fw=3, fh=3).back()


def host_controls(R, rbad, rmin=9, rmax=60, step=3):
    """Host-disk positions whose (PSF-scale smoothed) reference surface
    brightness is within 0.6-1.6x that at the SN, off compact structure,
    rmin-rmax px from the SN.  ZOGY flux there should be ~0 in every frame;
    it is the check that the host subtracts (blank controls cannot see it)."""
    Rf = np.nan_to_num(R)
    Rs = ndimage.gaussian_filter(Rf, 2.0)
    compact = ndimage.binary_dilation((Rf - ndimage.median_filter(Rf, 9)) > 1.0, iterations=3)
    yy, xx = np.indices(R.shape)
    r = np.hypot(xx - C, yy - C)
    ok = ((Rs > 0.6 * Rs[C, C]) & (Rs < 1.6 * Rs[C, C]) & (r >= rmin) & (r < rmax) & ~compact &
          ~rbad & (xx % step == 0) & (yy % step == 0))
    return np.column_stack(np.where(ok)[::-1])


def cmd_diff(a):
    for d in ("diff", "diffin"):
        os.makedirs(f"{a.workdir}/{d}", exist_ok=True)
    refpath = f"{a.workdir}/ref/ref.fits"
    R, rbad, pR, rh = base.read_grid(refpath)
    work = np.ascontiguousarray(np.where(rbad, 0.0, R))
    bk = sep.Background(work, mask=rbad, bw=48, bh=48)
    _, seg = sep.extract(work - bk.back(), 1.5, err=bk.globalrms, mask=rbad,
                         segmentation_map=True, minarea=5)
    src = ndimage.binary_dilation(seg > 0, iterations=6)
    ctrl = base.control_positions(work, rbad, n=150, rmin=12, rmax=104, sep_px=9)
    np.savetxt(f"{a.workdir}/diff/controls.txt", ctrl, fmt="%d")
    hctl = host_controls(R, rbad)
    np.savetxt(f"{a.workdir}/diff/host_controls.txt", hctl, fmt="%d")
    print(f"{len(ctrl)} blank-sky control positions, {len(hctl)} host-disk control positions")

    # field stars for the new/ref flux-ratio check: unsaturated in DECam
    g = json.load(open(f"{a.workdir}/grid.json"))
    gw = base.grid_wcs(g["ra"], g["dec"])
    gaia = Table.read(f"{a.workdir}/gaia.ecsv", format="ascii.ecsv")
    G = np.asarray(gaia["G"], float)

    # Match every frame's background to the reference, then ZOGY (base.diff_one).
    # A plane off sources is not enough: over the host the frames sit below
    # the reference by ~2-3% of the sky level plus ~5% of the host, which a
    # plane fitted to the blank field cannot see (ZOGY flux at host-disk
    # positions came out -4 to -70 uJy while blank sky averaged zero).  So,
    # with --bg local, a mesh background of N - R that excludes only compact
    # structure and the SN is removed as well: N - R is free of the host to
    # first order, so this takes out no host light, only the mismatch.
    lmask = local_mask(R, rbad)
    paths = sorted(glob.glob(f"{a.workdir}/prep/*.fits"))
    info = {}
    for p in paths:
        N, nbad, pN, nh = base.read_grid(p)
        cf, plane = match_plane(N, R, src | nbad | rbad)
        N = N - plane
        nh["PLANE0"] = (float(cf[0]), "plane matched to ref: N -= a + b(x-C) + c(y-C)")
        nh["PLANEX"] = float(cf[1])
        nh["PLANEY"] = float(cf[2])
        bsn = 0.0
        if a.bg == "local":
            loc = match_local(N - R, lmask | nbad, a.local_bw)
            N = N - loc
            bsn = float(loc[C, C])
            nh["LOCALBW"] = (a.local_bw, "local N-R mesh background removed (px)")
            nh["BKG_SN"] = (bsn, "local background removed at the SN (uJy/px)")
        tag = os.path.basename(p)[:-5]
        base.write_grid(f"{a.workdir}/diffin/{tag}.fits", N, nbad, pN, nh)

        gx, gy = base.gaia_xy(gaia, nh["MJD-OBS"], gw)
        sel = ((G > 16.5) & (G < 18.5) & (np.hypot(gx - C, gy - C) < 105) &
               base.isolated(gx, gy, gx, gy, G, G, 7.0))
        rat = []
        for k in np.where(sel)[0]:
            fn, fr = psf_flux(N, pN, gx[k], gy[k]), psf_flux(R, pR, gx[k], gy[k])
            if np.isfinite(fn) and np.isfinite(fr) and fr > 0:
                rat.append(fn / fr)
        rat = np.array(rat)
        mr = float(np.median(rat)) if rat.size else np.nan
        er = float(1.2533 * 1.4826 * np.median(np.abs(rat - mr)) / np.sqrt(rat.size)) if rat.size > 2 else np.nan
        # PSF-weighted signal under the SN (host + SN, uJy/px) for photon noise
        S = np.clip(np.nan_to_num(N[C - PSF_HALF:C + PSF_HALF + 1, C - PSF_HALF:C + PSF_HALF + 1]), 0, None)
        info[tag] = dict(plane0=float(cf[0]), planex=float(cf[1]), planey=float(cf[2]),
                         bkg_sn=bsn, bkg_sn_flux=bsn / float((pN ** 2).sum()),
                         star_ratio=mr, star_ratio_err=er, n_star=int(rat.size),
                         sum_p2s=float((pN ** 2 * S).sum()), sum_p2=float((pN ** 2).sum()),
                         uJy_per_dn=10 ** (0.4 * (ZP_OUT - nh["ZP_NAT"])),
                         sky_dn=nh["SKYDN"], sig_dn=nh["SIGDN"], chip=nh["CHIP"])

    # gain from sky level vs pixel variance over the frames of the main chip
    chips = [v["chip"] for v in info.values()]
    main = max(set(chips), key=chips.count)
    sk = np.array([v["sky_dn"] for v in info.values() if v["chip"] == main])
    va = np.array([v["sig_dn"] ** 2 for v in info.values() if v["chip"] == main])
    (slope, rn2), *_ = np.linalg.lstsq(np.column_stack([sk, np.ones_like(sk)]), va, rcond=None)
    gain = 1.0 / slope
    print(f"gain from {len(sk)} {main} frames: {gain:.3f} e-/DN, read noise {np.sqrt(max(rn2, 0)):.2f} DN "
          f"(var = sky/g + rn^2; residual RMS {np.std(va - slope * sk - rn2) / np.mean(va):.3f} rel)")
    for v in info.values():
        # var(F) = sum P^2 var_i / (sum P^2)^2, var_i = S_i[uJy] * (uJy/DN) / g
        v["src_err"] = float(np.sqrt(v["uJy_per_dn"] * v["sum_p2s"] / gain) / v["sum_p2"])
        v["gain"] = gain

    jobs = [(f"{a.workdir}/diffin/{os.path.basename(p)}", refpath, ctrl, f"{a.workdir}/diff", a.nfake)
            for p in paths]
    with pool(a.nproc) as ex:
        res = list(ex.map(base.diff_one, jobs))
    extra = ["src_err", "star_ratio", "star_ratio_err", "n_star", "plane0", "planex", "planey",
             "bkg_sn", "bkg_sn_flux", "gain", "host_bias", "host_bias_err"]
    keys = ["tag", "ok", "chip", "mjd", "ref", "flux", "score", "alpha_std", "ctrl_bias",
            "ctrl_std", "n_ctrl", "fake_ratio", "fake_ratio_std", "fake_off", "n_fake",
            "pk_dx", "pk_dy", "sn_bad", "zp_nat", "zp_err", "zp_hdr", "fwhm", "sky",
            "ast_rms"] + extra + ["err"]
    hw = 100                                   # diff_one saves a +-100 px cut around C
    for r in res:
        if r["ok"]:
            r.update({k: info[r["tag"]][k] for k in extra[:-2]})
            with fits.open(f"{a.workdir}/diff/{r['tag']}.fits") as h:
                hv = h["ALPHA"].data[hctl[:, 1] - (C - hw), hctl[:, 0] - (C - hw)].astype(float)
            hv = hv[np.isfinite(hv)]
            r["host_bias"] = float(np.median(hv))
            # ~9 host positions (3-px steps) per PSF area are correlated
            r["host_bias_err"] = float(1.2533 * 1.4826 * np.median(np.abs(hv - np.median(hv))) /
                                       np.sqrt(max(len(hv) / 9.0, 1.0)))
    with open(f"{a.workdir}/diff.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in res:
            wr.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    for r in sorted(res, key=lambda r: r.get("mjd", 0)):
        if r["ok"]:
            print(f"  {r['tag']:22s} flux {r['flux']:+9.2f} uJy  ctrl {r['ctrl_bias']:+6.2f}+-"
                  f"{r['ctrl_std']:5.2f} (n={r['n_ctrl']})  src {r['src_err']:5.2f}  S {r['score']:+7.1f}  "
                  f"fake {r['fake_ratio']:.3f}+-{r['fake_ratio_std']:.3f} off {r['fake_off']:.2f}  "
                  f"pk ({r['pk_dx']:+.2f},{r['pk_dy']:+.2f})  stars new/ref {r['star_ratio']:.3f}"
                  f"+-{r['star_ratio_err']:.3f} (n={r['n_star']})  plane {r['plane0']:+.3f}  "
                  f"local bkg at SN {r['bkg_sn']:+.3f} uJy/px  host {r['host_bias']:+6.2f}"
                  f"+-{r['host_bias_err']:.2f}  "
                  f"{'SNBAD' if r['sn_bad'] else ''}")
        else:
            print(f"  {r['tag']:22s} FAILED  {r['err']}")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--cutouts", required=True, help="glob of <stem>.image.fits.fz")
    p.add_argument("--decam", required=True, help="glob of DECam *_ooi_* frames")
    p.add_argument("--band", default="g", choices=["g", "i", "z"])
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    sp.add_parser("ref")
    p = sp.add_parser("diff")
    p.add_argument("--nfake", type=int, default=30)
    p.add_argument("--bg", choices=["local", "plane"], default="local",
                   help="background match to the ref: plane off sources, + local N-R mesh")
    p.add_argument("--local-bw", type=int, default=24, help="local mesh size (px)")
    for p in sp.choices.values():
        p.add_argument("--workdir", required=True)
        p.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    {"prep": cmd_prep, "ref": cmd_ref, "diff": cmd_diff}[a.cmd](a)


if __name__ == "__main__":
    main()
