#!/usr/bin/env python3
"""ZOGY difference-imaging light curve of one SN from LS4 STDALN frames.

Standalone (no SeeChange pipeline/DB): SeeChange's ZOGY algorithm via the
vendored code/zogylib, as in diff_zogy.py.  Input is a coverage CSV from
dump_coverage.py (band, mjd, night, expid, chip, ..., path).

Stages (run in order; each reads the previous one's outputs in --workdir):

  prep  Resample every frame of --band onto one TAN grid centred on the SN
        (1000x1000, 1"/px, north up), registered to Gaia DR3: the frame's SIP
        WCS is corrected by an affine fit of measured star centroids against
        Gaia positions propagated to the frame epoch.  Then build an empirical
        PSF, calibrate a local zero point by PSF-fitting Gaia stars against the
        LS4Cam Gaia->instrument transform, and scale to ZP 23.9 (pixels in uJy,
        AB).  Noise is measured from the pixels: the STDALN weight maps are not
        usable for it (on SE_F their absolute scale is off by ~2400x in sigma).
        Before resampling, each amplifier is destriped: LS4 frames carry
        row-wise offsets that are uniform along an amp row -- pattern noise at
        ~0.5 sigma_sky per row plus discrete negative stripes of several sigma
        that are not in the masks.
  ref   Clipped inverse-variance mean of the frames before --ref-before into
        ref/ref.fits, plus a leave-one-out reference for each of them.
  diff  ZOGY of every frame against the reference (leave-one-out for the
        reference frames); forced PSF photometry (ZOGY alpha) at the SN and at
        blank control positions; fake point sources injected at control
        positions to check the flux normalisation and centring.
  lc    Light curve: flux = alpha at the SN; error = control scatter (+) a
        systematic floor from the reference-epoch null test (+) the ZP error.
        Magnitudes or 3-sigma limits, plot, cutout montage.

Usage:
  sn_zogy_lc.py prep --coverage cov.csv --band g --ra 191.6814 --dec -40.025261 --workdir W
  sn_zogy_lc.py ref  --workdir W --ref-before 61160
  sn_zogy_lc.py diff --workdir W
  sn_zogy_lc.py lc   --workdir W --disc-mjd 61184.397 --name "SN 2026nmw"
"""
import argparse
import csv
import glob
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
import sep

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("ZOGYLIB", "/pscratch/sd/n/nugent/ls4/code/zogylib"))

ZP_OUT = 23.9            # output zero point: pixel values in uJy (AB)
# Even on purpose: for odd sides ZOGY places the PSF at n//2+1 and its alpha
# map comes out shifted by (1, 1) px (the fake-injection test shows it).
NPIX = 1000              # grid side; the SN sits on pixel (NPIX//2, NPIX//2)
SCALE = 1.0              # arcsec / px
PSF_HALF = 15            # PSF stamp 31x31
FIT_HALF = 7             # PSF-fit photometry stamp 15x15
R_STARS = 420            # use PSF / ZP stars within this radius (px) of the SN
GAIA_EPOCH = Time("2016.0", format="jyear", scale="tdb")
C = NPIX // 2

try:
    sep.set_extract_pixstack(2_000_000)
    sep.set_sub_object_limit(4096)      # cloudy frames overflow the default 1024
except Exception:
    pass


# ------------------------------------------------------------------ helpers

def grid_wcs(ra, dec):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [C + 1.0, C + 1.0]
    w.wcs.cdelt = [-SCALE / 3600.0, SCALE / 3600.0]
    return w


def gaia_xy(gaia, mjd, wcs, origin=0, sip=False):
    """Gaia positions propagated to `mjd`, as pixel coords in `wcs`."""
    c = SkyCoord(ra=np.asarray(gaia["ra"]) * u.deg, dec=np.asarray(gaia["dec"]) * u.deg,
                 distance=Distance(1 * u.kpc),
                 pm_ra_cosdec=np.nan_to_num(np.asarray(gaia["pmra"], float)) * u.mas / u.yr,
                 pm_dec=np.nan_to_num(np.asarray(gaia["pmdec"], float)) * u.mas / u.yr,
                 obstime=GAIA_EPOCH).apply_space_motion(Time(mjd, format="mjd"))
    ra, dec = c.ra.deg, c.dec.deg
    if sip:
        return wcs.all_world2pix(ra, dec, origin, quiet=True)
    return wcs.wcs_world2pix(ra, dec, origin)


def stamp_fwhm(s):
    """FWHM from the area above half maximum (4x upsampled).  Second moments
    are no good here: the faint, extended wing of the LS4 PSF inflates them."""
    z = ndimage.zoom(s, 4, order=3)
    return float(2.0 * np.sqrt((z >= 0.5 * z.max()).sum() / 16.0 / np.pi))


def isolated(x, y, gx, gy, gG, G, rad):
    """True where no other Gaia star brighter than G+2.5 lies within `rad` px."""
    ok = np.ones(len(x), bool)
    for k in range(len(x)):
        d = np.hypot(gx - x[k], gy - y[k])
        ok[k] = not np.any((d > 0.5) & (d < rad) & (gG < G[k] + 2.5))
    return ok


def shift_psf(psf, dx, dy):
    return ndimage.shift(psf, (dy, dx), order=3, mode="constant")


class EmpPSF:
    """Serve an empirical (odd, centred, sum=1) PSF stamp to ZOGY.

    Mirrors models/psf.py:get_centered_psf for ZOGY's call pattern
    (x=y=None, offx/offy set by zogy): the odd clip's centre pixel lands at
    (nx//2, ny//2) + floor(offset).  Same shim as diff_zogy.py."""
    def __init__(self, stamp):
        assert stamp.shape[0] % 2 == 1 and stamp.shape[1] % 2 == 1
        self.stamp = (stamp / stamp.sum()).astype(np.float64)
        self.fwhm_pixels = stamp_fwhm(self.stamp)

    def get_centered_psf(self, nx, ny, x=None, y=None, offx=0., offy=0.,
                         flux=1.0, norm=True, **kw):
        s = self.stamp * flux
        sh, sw = s.shape
        padlowx = nx // 2 - sw // 2 + int(np.floor(offx))
        padlowy = ny // 2 - sh // 2 + int(np.floor(offy))
        out = np.zeros((ny, nx), dtype=np.float64)
        out[padlowy:padlowy + sh, padlowx:padlowx + sw] = s
        return out


def characterize(img, bad, gaia, mjd, gw, ast_g=(13.5, 18.0), psf_g=(13.5, 18.5),
                 zp_g=(14.0, 19.5), g_bright=17.0, wing="clip", xy=None, center=None):
    """Empirical PSF + PSF-fit Gaia zero point + sky RMS of a grid image.

    `img` is background-subtracted, NaN where `bad`.  The Gaia G ranges for
    astrometric, PSF and ZP stars (and the G limit of the stars that set the
    S/N predictor's ZP) default to LS4; deeper data (DECam) saturate at the
    bright end and want fainter ranges.  wing="azimuthal" replaces the PSF
    beyond r = 5 px by annulus medians (unbiased; use it when PSF stars are
    few).  For an image not on the grid (native pixels), pass the Gaia pixel
    positions as xy=(gx, gy) and the SN pixel as center=(x, y); gw is then
    unused.  Returns dict with psf, zp (native units), zp_err, n_zp, sky,
    fwhm_px, astrom offsets."""
    ny, nx = img.shape
    work = np.where(bad, 0.0, img)
    bkg = sep.Background(np.ascontiguousarray(work), mask=bad, bw=64, bh=64)
    sky = float(bkg.globalrms)

    gx, gy = gaia_xy(gaia, mjd, gw) if xy is None else xy
    G = np.asarray(gaia["G"], float)
    gm = np.asarray(gaia["gmag"], float)
    cx0, cy0 = (C, C) if center is None else center
    r = np.hypot(gx - cx0, gy - cy0)
    inside = ((gx > PSF_HALF + 2) & (gx < nx - PSF_HALF - 3) &
              (gy > PSF_HALF + 2) & (gy < ny - PSF_HALF - 3) & (r < R_STARS))
    badgrow = ndimage.binary_dilation(bad, iterations=3)

    # astrometric check: windowed centroids of bright stars vs Gaia
    sel = inside & (G > ast_g[0]) & (G < ast_g[1])
    sel &= isolated(gx, gy, gx, gy, G, G, 8.0) if sel.any() else sel
    ix, iy = np.round(gx).astype(int), np.round(gy).astype(int)
    sel &= ~badgrow[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
    dxs = dys = np.array([])
    if sel.sum() >= 5:
        wx, wy, wf = sep.winpos(np.ascontiguousarray(work), gx[sel], gy[sel], 1.3)
        ok = wf == 0
        dxs, dys = (wx - gx[sel])[ok], (wy - gy[sel])[ok]

    # PSF stars: isolated, unflagged, mid-bright
    psel = inside & (G > psf_g[0]) & (G < psf_g[1])
    psel &= isolated(gx, gy, gx, gy, G, G, 12.0) if psel.any() else psel
    stamps = []
    yy, xx = np.indices((2 * PSF_HALF + 1,) * 2)
    rr = np.hypot(xx - PSF_HALF, yy - PSF_HALF)
    for k in np.where(psel)[0]:
        i, j = ix[k], iy[k]
        sl = np.s_[j - PSF_HALF:j + PSF_HALF + 1, i - PSF_HALF:i + PSF_HALF + 1]
        if bad[sl][rr <= 10].any():
            continue
        st = np.where(bad[sl], 0.0, img[sl])
        st = st - np.median(st[rr >= 13])
        st = ndimage.shift(st, (j - gy[k], i - gx[k]), order=3, mode="constant")
        norm = st[rr <= 10].sum()
        if norm <= 0:
            continue
        stamps.append(st / norm)
    if len(stamps) < 8:
        raise RuntimeError(f"only {len(stamps)} PSF stars")
    psf = np.median(np.stack(stamps), axis=0)
    psf -= np.median(psf[rr >= 13])
    if wing == "azimuthal":
        # beyond the core, the median of each 1-px annulus.  Clipping noisy
        # single pixels at zero (below) leaves a positive pedestal that grows
        # with the sky noise: with ~15 PSF stars it held up to 30% of the sum,
        # and ZOGY then over-subtracted the host by up to 60 uJy.
        rb = np.floor(rr).astype(int)
        for k in range(5, PSF_HALF + 1):
            psf[rb == k] = np.median(psf[rb == k])
    psf[rr > 14] = 0.0
    psf[(rr > 4) & (psf < 0)] = 0.0
    psf /= psf.sum()

    # PSF-fit photometry of Gaia stars -> zero point
    zsel = inside & (G > zp_g[0]) & (G < zp_g[1]) & np.isfinite(gm)
    zsel &= isolated(gx, gy, gx, gy, G, G, 7.0) if zsel.any() else zsel
    fh = FIT_HALF
    fa, fm, fG = [], [], []
    sumP2 = float((psf ** 2).sum())
    for k in np.where(zsel)[0]:
        i, j = ix[k], iy[k]
        sl = np.s_[j - fh:j + fh + 1, i - fh:i + fh + 1]
        if bad[sl].any():
            continue
        p = shift_psf(psf, gx[k] - i, gy[k] - j)[PSF_HALF - fh:PSF_HALF + fh + 1,
                                                 PSF_HALF - fh:PSF_HALF + fh + 1]
        A = np.column_stack([p.ravel(), np.ones(p.size)])
        (a, b), *_ = np.linalg.lstsq(A, img[sl].ravel(), rcond=None)
        fa.append(a); fm.append(gm[k]); fG.append(G[k])
    fa, fm, fG = np.array(fa), np.array(fm), np.array(fG)
    # Select on the S/N *predicted* from the catalogue magnitude: cutting on
    # the measured S/N keeps stars that fluctuated bright and biases the ZP
    # high in noisy frames.  The predictor's ZP comes from G < g_bright
    # stars, which are well above any noise cut.
    br = (fG < g_bright) & (fa > 0)
    if br.sum() < 5:
        raise RuntimeError(f"only {br.sum()} bright ZP stars")
    zp0 = sigma_clipped_stats(fm[br] + 2.5 * np.log10(fa[br]), sigma=3.0, maxiters=5)[1]
    pred_snr = 10 ** (0.4 * (zp0 - fm)) / (sky / np.sqrt(sumP2))
    use = (pred_snr > 20) & (fa > 0)
    zps = fm[use] + 2.5 * np.log10(fa[use])
    if len(zps) < 10:
        raise RuntimeError(f"only {len(zps)} ZP stars")
    _, zmed, zstd = sigma_clipped_stats(zps, sigma=3.0, maxiters=5)
    return dict(psf=psf, zp=float(zmed), zp_err=float(zstd / np.sqrt(len(zps) - 1)),
                zp_scat=float(zstd), n_zp=int(len(zps)), n_psf=len(stamps), sky=sky,
                fwhm_px=stamp_fwhm(psf),
                ast_dx=float(np.median(dxs)) if dxs.size else np.nan,
                ast_dy=float(np.median(dys)) if dys.size else np.nan,
                ast_rms=float(np.sqrt(np.median(dxs ** 2 + dys ** 2))) if dxs.size else np.nan,
                n_ast=int(dxs.size))


def read_grid(path):
    with fits.open(path) as h:
        img = h[0].data.astype(np.float64)
        hdr = h[0].header.copy()
        bad = h["BAD"].data.astype(bool)
        psf = h["PSF"].data.astype(np.float64)
    img[bad] = np.nan
    return img, bad, psf, hdr


def write_grid(path, img, bad, psf, hdr):
    hl = fits.HDUList([fits.PrimaryHDU(np.where(bad, np.nan, img).astype(np.float32), hdr),
                       fits.ImageHDU(bad.astype(np.uint8), name="BAD"),
                       fits.ImageHDU(psf.astype(np.float32), name="PSF")])
    hl.writeto(path, overwrite=True)


# ------------------------------------------------------------------ prep

def destripe(img, bad, amp_split=1024):
    """Subtract row-wise offsets per amplifier; return (image, row-offset RMS).

    Each row's offset is the median of its source-masked, unflagged pixels
    after a 128-px background, high-passed over 31 rows so that smooth
    structure is left to the background model."""
    out = img.copy()
    rms = []
    for xs in (slice(0, amp_split), slice(amp_split, img.shape[1])):
        b = np.ascontiguousarray(bad[:, xs])
        if (~b).mean() < 0.2:                       # dead amp (SE_F left)
            continue
        a = np.ascontiguousarray(np.where(b, 0.0, img[:, xs]))
        bk = sep.Background(a, mask=b, bw=128, bh=128)
        d = np.ascontiguousarray(a - bk.back())
        _, seg = sep.extract(d, 1.5, err=bk.globalrms, mask=b, segmentation_map=True, minarea=5)
        m = ndimage.binary_dilation(seg > 0, iterations=4) | b
        dm = np.where(m, np.nan, d)
        n = (~m).sum(axis=1)
        row = np.full(d.shape[0], np.nan)
        ok = n > 100
        row[ok] = np.nanmedian(dm[ok], axis=1)
        if ok.sum() < 100:
            continue
        yy = np.arange(d.shape[0])
        filled = np.interp(yy, yy[ok], row[ok])
        hp = np.where(ok, filled - ndimage.median_filter(filled, 31, mode="nearest"), 0.0)
        out[:, xs] -= hp[:, None]
        rms.append(float(np.std(hp[ok])))
    return out, (float(np.mean(rms)) if rms else np.nan)


def prep_one(job):
    row, gaia, ra0, dec0, outdir = job
    tag = f"{row['night']}_{row['expid']}_{row['chip']}"
    out = f"{outdir}/{tag}.fits"
    try:
        base = row["path"].replace(".image.w.fits.fz", "")
        with fits.open(row["path"]) as h:
            k = next(i for i, hd in enumerate(h) if hd.data is not None and hd.data.ndim == 2)
            img = h[k].data.astype(np.float64)
            hdr = h[k].header.copy()
        msk = fits.getdata(base + ".mask.fits.fz", 1)
        flg = fits.getdata(base + ".flags.fits.fz", 1)
        wgt = fits.getdata(base + ".weight.fits.fz", 1)
        bad = (msk != 0) | (flg != 0) | ~(wgt > 0) | ~np.isfinite(img)
        img, row_rms = destripe(img, bad)
        win = WCS(hdr)
        mjd = float(row["mjd"])
        gw = grid_wcs(ra0, dec0)
        H, W_ = img.shape

        # output grid -> input pixels (SIP), on a coarse mesh then splined
        cg = np.linspace(0, NPIX - 1, 101)
        cx, cy = np.meshgrid(cg, cg)
        ra, dec = gw.wcs_pix2world(cx.ravel(), cy.ravel(), 0)
        xin, yin = win.all_world2pix(ra, dec, 0, quiet=True)
        sx = RectBivariateSpline(cg, cg, xin.reshape(cx.shape))
        sy = RectBivariateSpline(cg, cg, yin.reshape(cx.shape))
        fg = np.arange(NPIX, dtype=float)
        XIN, YIN = sx(fg, fg), sy(fg, fg)          # [y, x] ordering

        # crop the input to what the grid touches
        x0 = int(max(0, np.floor(XIN.min()) - 40)); x1 = int(min(W_, np.ceil(XIN.max()) + 41))
        y0 = int(max(0, np.floor(YIN.min()) - 40)); y1 = int(min(H, np.ceil(YIN.max()) + 41))
        if x1 - x0 < 100 or y1 - y0 < 100:
            raise RuntimeError("grid does not overlap the frame")
        crop = img[y0:y1, x0:x1].copy()
        cbad = np.ascontiguousarray(bad[y0:y1, x0:x1])
        _, med, _ = sigma_clipped_stats(crop[~cbad], sigma=3.0, maxiters=5)
        crop -= med
        crop[cbad] = 0.0

        # register: affine fit of measured centroids vs Gaia, in input pixels
        gxi, gyi = gaia_xy(gaia, mjd, win, sip=True)
        G = np.asarray(gaia["G"], float)
        bk = sep.Background(np.ascontiguousarray(crop), mask=cbad, bw=128, bh=128)
        sub = np.ascontiguousarray(crop - bk.back())
        lx, ly = gxi - x0, gyi - y0
        cand = ((lx > 10) & (lx < x1 - x0 - 10) & (ly > 10) & (ly < y1 - y0 - 10) &
                (G > 13.5) & (G < 18.0))
        cand &= isolated(gxi, gyi, gxi, gyi, G, G, 8.0) if cand.any() else cand
        cand &= ~ndimage.binary_dilation(cbad, iterations=3)[
            np.clip(np.round(ly).astype(int), 0, y1 - y0 - 1),
            np.clip(np.round(lx).astype(int), 0, x1 - x0 - 1)]
        wx, wy, wf = sep.winpos(sub, lx[cand], ly[cand], 1.3)
        ok = (wf == 0) & (np.hypot(wx - lx[cand], wy - ly[cand]) < 3.0)
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
        reg_rms = float(np.sqrt(np.mean(res[use] ** 2)))
        lxg, lyg = XIN - x0, YIN - y0
        lxg = lxg + cxf[0] + cxf[1] * lxg + cxf[2] * lyg
        lyg = lyg + cyf[0] + cyf[1] * (XIN - x0) + cyf[2] * (YIN - y0)

        # resample (cubic spline); bad map bilinear, grown by 2 px
        outimg = ndimage.map_coordinates(crop, [lyg, lxg], order=3, mode="constant", cval=0.0)
        obad = ndimage.map_coordinates(cbad.astype(np.float32), [lyg, lxg], order=1,
                                       mode="constant", cval=1.0) > 1e-3
        obad = ndimage.binary_dilation(obad, iterations=2)
        bko = sep.Background(np.ascontiguousarray(np.where(obad, 0.0, outimg)),
                             mask=obad, bw=128, bh=128)
        outimg = outimg - bko.back()
        outimg[obad] = np.nan

        ch = characterize(outimg, obad, gaia, mjd, gw)
        s = 10 ** (0.4 * (ZP_OUT - ch["zp"]))
        oh = gw.to_header()
        rr = np.hypot(*np.meshgrid(np.arange(-8, 9), np.arange(-8, 9)))
        snbad = float(obad[C - 8:C + 9, C - 8:C + 9][rr <= 6].mean())
        for key, val, com in (
                ("MJD-OBS", mjd, "frame MJD"), ("NIGHT", row["night"], ""),
                ("EXPID", row["expid"], ""), ("CHIP", row["chip"], ""),
                ("BAND", row["band"], ""), ("ZP_OUT", ZP_OUT, "pixel units: uJy"),
                ("ZP_NAT", ch["zp"], "native PSF-fit zero point (Gaia/LS4Cam)"),
                ("ZP_ERR", ch["zp_err"], "ZP s.e.m."), ("ZP_SCAT", ch["zp_scat"], ""),
                ("ZP_N", ch["n_zp"], "ZP stars"), ("ZP_HDR", float(row["zp"]), "header G_ZP"),
                ("SKYSIG", ch["sky"] * s, "sky RMS per px (uJy)"),
                ("FWHM_PX", ch["fwhm_px"], "empirical PSF FWHM (px)"),
                ("NPSF", ch["n_psf"], "PSF stars"),
                ("REGRMS", reg_rms, "registration residual RMS (input px)"),
                ("REGN", int(use.sum()), "registration stars"),
                ("REGDX", float(cxf[0]), "affine dx at crop origin"),
                ("REGDY", float(cyf[0]), "affine dy at crop origin"),
                ("ASTDX", ch["ast_dx"], "post-resample median dx vs Gaia (px)"),
                ("ASTDY", ch["ast_dy"], "post-resample median dy vs Gaia (px)"),
                ("ASTRMS", ch["ast_rms"], "post-resample RMS vs Gaia (px)"),
                ("SNBAD", snbad, "flagged fraction within 6 px of SN"),
                ("ROWRMS", row_rms, "RMS of removed row offsets (native)"),
                ("SRCFRAME", os.path.basename(row["path"])[:68], "")):
            oh[key] = (val, com)
        write_grid(out, outimg * s, obad, ch["psf"], oh)
        return dict(tag=tag, ok=True, mjd=mjd, zp=ch["zp"], zp_hdr=float(row["zp"]),
                    zp_err=ch["zp_err"], n_zp=ch["n_zp"], sky=ch["sky"] * s,
                    fwhm=ch["fwhm_px"] * SCALE, reg_rms=reg_rms, reg_n=int(use.sum()),
                    ast_rms=ch["ast_rms"], ast_dx=ch["ast_dx"], ast_dy=ch["ast_dy"],
                    snbad=snbad)
    except Exception as e:
        return dict(tag=tag, ok=False, err=f"{type(e).__name__}: {e}")


def cmd_prep(a):
    from zp_gaia_ls4cam import query_gaia_dr3, ls4cam_gaia_to_instrument
    os.makedirs(f"{a.workdir}/prep", exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.coverage)) if r["band"] == a.band]
    if a.only:
        rows = [r for r in rows if r["expid"] in a.only.split(",")]
    rad = NPIX * SCALE / 3600.0 / np.sqrt(2.0) * 1.05
    cat = query_gaia_dr3(a.ra, a.dec, rad, max_g_mag=20.5, min_g_mag=8.0)
    gm, _ = ls4cam_gaia_to_instrument(
        a.band, np.asarray(cat["MAG_G"]), np.asarray(cat["MAGERR_G"]),
        np.asarray(cat["MAG_BP"]), np.asarray(cat["MAGERR_BP"]),
        np.asarray(cat["MAG_RP"]), np.asarray(cat["MAGERR_RP"]))
    gaia = Table({"ra": np.asarray(cat["ra"], float), "dec": np.asarray(cat["dec"], float),
                  "pmra": np.asarray(cat["pmra"], float), "pmdec": np.asarray(cat["pmdec"], float),
                  "G": np.asarray(cat["MAG_G"], float), "gmag": np.asarray(gm, float)})
    gaia.write(f"{a.workdir}/gaia.ecsv", format="ascii.ecsv", overwrite=True)
    json.dump(dict(ra=a.ra, dec=a.dec, band=a.band, npix=NPIX, scale=SCALE),
              open(f"{a.workdir}/grid.json", "w"))
    print(f"Gaia: {len(gaia)} stars within {rad:.3f} deg; {len(rows)} {a.band} frames", flush=True)

    jobs = [(r, gaia, a.ra, a.dec, f"{a.workdir}/prep") for r in rows]
    with ProcessPoolExecutor(max_workers=a.nproc) as ex:
        res = list(ex.map(prep_one, jobs))
    keys = ["tag", "ok", "mjd", "zp", "zp_hdr", "zp_err", "n_zp", "sky", "fwhm",
            "reg_rms", "reg_n", "ast_rms", "ast_dx", "ast_dy", "snbad", "err"]
    with open(f"{a.workdir}/prep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in res:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in sorted(res, key=lambda r: r.get("mjd", 0)):
        if r["ok"]:
            print(f"  {r['tag']}  zp {r['zp']:.3f} (hdr {r['zp_hdr']:.3f}, N={r['n_zp']})  "
                  f"sky {r['sky']:.3f} uJy  fwhm {r['fwhm']:.2f}\"  reg {r['reg_rms']:.3f}px/"
                  f"{r['reg_n']}  ast {r['ast_dx']:+.3f},{r['ast_dy']:+.3f} rms {r['ast_rms']:.3f}  "
                  f"snbad {r['snbad']:.2f}")
        else:
            print(f"  {r['tag']}  FAILED  {r['err']}")


# ------------------------------------------------------------------ ref

def coadd(frames, min_use=3):
    """Clipped inverse-variance mean of (img, bad, sky) grid frames.

    Pixels kept by fewer than `min_use` frames are flagged.  min_use=2 is
    still safe against cosmic rays: with two frames a hit pulls the median
    halfway, so both values fail the clip and the pixel is flagged."""
    stack = np.stack([np.where(b, np.nan, im) for im, b, s in frames])
    sig = np.array([s for _, _, s in frames])[:, None, None]
    med = np.nanmedian(stack, axis=0)
    tol = 5.0 * np.sqrt(sig ** 2 + (0.1 * np.abs(med)) ** 2)
    keep = np.isfinite(stack) & (np.abs(stack - med) < tol)
    w = np.where(keep, 1.0 / sig ** 2, 0.0)
    wsum = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = (np.where(keep, stack, 0.0) * w).sum(0) / wsum
    nuse = keep.sum(0)
    bad = (nuse < min_use) | ~np.isfinite(ref)
    ref[bad] = np.nan
    return ref, bad, nuse


def make_ref(frames, gaia, mjd, gw):
    ref, bad, nuse = coadd([(f[0], f[1], f[2]) for f in frames])
    ch = characterize(ref, bad, gaia, mjd, gw)
    s = 10 ** (0.4 * (ZP_OUT - ch["zp"]))      # renormalise exactly to ZP_OUT
    return ref * s, bad, ch, s, nuse


def cmd_ref(a):
    g = json.load(open(f"{a.workdir}/grid.json"))
    gw = grid_wcs(g["ra"], g["dec"])
    gaia = Table.read(f"{a.workdir}/gaia.ecsv", format="ascii.ecsv")
    os.makedirs(f"{a.workdir}/ref", exist_ok=True)
    frames = []
    for p in sorted(glob.glob(f"{a.workdir}/prep/*.fits")):
        img, bad, psf, hdr = read_grid(p)
        if hdr["MJD-OBS"] < a.ref_before:
            frames.append((img, bad, hdr["SKYSIG"], hdr, os.path.basename(p)[:-5]))
    if len(frames) < 4:
        raise SystemExit(f"only {len(frames)} reference frames")
    mjd = float(np.mean([f[3]["MJD-OBS"] for f in frames]))
    print(f"reference: {len(frames)} frames before MJD {a.ref_before} (mean {mjd:.2f})")

    def save(name, ref, bad, ch, s, nuse, members):
        h = gw.to_header()
        h["MJD-OBS"] = (mjd, "mean MJD of members")
        h["NMEMBER"] = len(members)
        h["SKYSIG"] = (ch["sky"] * s, "sky RMS per px (uJy)")
        h["FWHM_PX"] = ch["fwhm_px"]
        h["ZP_RENRM"] = (float(-2.5 * np.log10(s)), "mag renormalisation applied")
        h["ZP_N"] = ch["n_zp"]
        h["ZP_SCAT"] = ch["zp_scat"]
        h["MEMBERS"] = ",".join(m.split("_")[1] for m in members)[:68]
        write_grid(f"{a.workdir}/ref/{name}.fits", ref, bad, ch["psf"], h)
        return h

    ref, bad, ch, s, nuse = make_ref(frames, gaia, mjd, gw)
    save("ref", ref, bad, ch, s, nuse, [f[4] for f in frames])
    fits.writeto(f"{a.workdir}/ref/ref_nuse.fits", nuse.astype(np.int16), overwrite=True)
    print(f"  full ref: sky {ch['sky']*s:.4f} uJy  fwhm {ch['fwhm_px']*SCALE:.2f}\"  "
          f"renorm {-2.5*np.log10(s):+.4f} mag  (ZP N={ch['n_zp']}, scat {ch['zp_scat']:.3f})  "
          f"n at SN = {nuse[C, C]}")
    for k, f in enumerate(frames):
        others = frames[:k] + frames[k + 1:]
        r2, b2, c2, s2, n2 = make_ref(others, gaia, mjd, gw)
        save(f"loo_{f[4]}", r2, b2, c2, s2, n2, [o[4] for o in others])
        print(f"  LOO {f[4]}: sky {c2['sky']*s2:.4f}  renorm {-2.5*np.log10(s2):+.4f}", flush=True)


# ------------------------------------------------------------------ diff

def zogy(R, N, pR, pN, sR, sN):
    from zogy import zogy_subtract
    return zogy_subtract(R.copy(), N.copy(), EmpPSF(pR), EmpPSF(pN), sR, sN, 1.0, 1.0)


def control_positions(ref, rbad, n=300, rmin=25, rmax=420, sep_px=14, seed=7):
    """Blank-sky positions: off detected sources and flagged pixels."""
    work = np.ascontiguousarray(np.where(rbad, 0.0, ref))
    bk = sep.Background(work, mask=rbad, bw=64, bh=64)
    _, seg = sep.extract(work - bk.back(), 2.0, err=bk.globalrms, mask=rbad,
                         segmentation_map=True, minarea=4)
    occupied = ndimage.binary_dilation((seg > 0) | rbad, iterations=8)
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(40000):
        if len(pts) >= n:
            break
        ang = rng.uniform(0, 2 * np.pi)
        rad = np.sqrt(rng.uniform(rmin ** 2, rmax ** 2))
        x, y = int(round(C + rad * np.cos(ang))), int(round(C + rad * np.sin(ang)))
        if occupied[y, x]:
            continue
        if pts and min(np.hypot(px - x, py - y) for px, py in pts) < sep_px:
            continue
        pts.append((x, y))
    return np.array(pts)


def diff_one(job):
    path, refpath, ctrl, outdir, nfake = job
    tag = os.path.basename(path)[:-5]
    try:
        N, nbad, pN, nh = read_grid(path)
        R, rbad, pR, rh = read_grid(refpath)
        sN, sR = float(nh["SKYSIG"]), float(rh["SKYSIG"])
        out = zogy(R, N, pR, pN, sR, sN)
        D, S, A = out["sub_image"], out["score_corr"], out["alpha"]
        anybad = nbad | rbad
        g = ~anybad & np.isfinite(S)
        _, smed, ssig = sigma_clipped_stats(S[g], sigma=3.0, maxiters=5)
        z = (S - smed) / ssig

        # controls usable in this frame
        grow = ndimage.binary_dilation(anybad, iterations=6)
        cuse = ctrl[~grow[ctrl[:, 1], ctrl[:, 0]]]
        cv = A[cuse[:, 1], cuse[:, 0]]
        _, cmed, cstd = sigma_clipped_stats(cv, sigma=3.0, maxiters=5)

        # fakes at a subset of controls, flux = 20x control scatter
        fk = cuse[::max(1, len(cuse) // nfake)][:nfake]
        ffl = 20.0 * cstd
        Nf = N.copy()
        for x, y in fk:
            Nf[y - PSF_HALF:y + PSF_HALF + 1, x - PSF_HALF:x + PSF_HALF + 1] += ffl * pN
        Af = zogy(R, Nf, pR, pN, sR, sN)["alpha"]
        dA = Af - A
        ratio = dA[fk[:, 1], fk[:, 0]] / ffl
        offs = []
        for x, y in fk:
            box = dA[y - 2:y + 3, x - 2:x + 3]
            jy, jx = np.unravel_index(np.nanargmax(box), box.shape)
            offs.append((jx - 2, jy - 2))
        offs = np.array(offs)

        # SN: alpha at the centre pixel; score peak position nearby
        box = z[C - 3:C + 4, C - 3:C + 4]
        jy, jx = np.unravel_index(np.nanargmax(box), box.shape)
        pk = (np.nan, np.nan)
        if 1 <= jy <= 5 and 1 <= jx <= 5:
            def q(m, p, c):
                den = m - 2 * c + p
                return 0.5 * (m - p) / den if den != 0 else 0.0
            yy, xx = C - 3 + jy, C - 3 + jx
            pk = (xx - C + q(z[yy, xx - 1], z[yy, xx + 1], z[yy, xx]),
                  yy - C + q(z[yy - 1, xx], z[yy + 1, xx], z[yy, xx]))
        rr = np.hypot(*np.meshgrid(np.arange(-8, 9), np.arange(-8, 9)))
        snbad = bool(anybad[C - 8:C + 9, C - 8:C + 9][rr <= 6].any())

        hw = 100
        sl = np.s_[C - hw:C + hw + 1, C - hw:C + hw + 1]
        hh = fits.Header()
        hh["REF"] = os.path.basename(refpath)
        hh["MJD-OBS"] = nh["MJD-OBS"]
        fits.HDUList([fits.PrimaryHDU(np.asarray(D[sl], np.float32), hh),
                      fits.ImageHDU(np.asarray(z[sl], np.float32), name="SCORE"),
                      fits.ImageHDU(np.asarray(A[sl], np.float32), name="ALPHA"),
                      fits.ImageHDU(np.asarray(np.where(nbad, np.nan, N)[sl], np.float32), name="NEW"),
                      fits.ImageHDU(np.asarray(np.where(rbad, np.nan, R)[sl], np.float32), name="REF")]
                     ).writeto(f"{outdir}/{tag}.fits", overwrite=True)
        return dict(tag=tag, ok=True, mjd=float(nh["MJD-OBS"]), ref=os.path.basename(refpath)[:-5],
                    flux=float(A[C, C]), score=float(z[C, C]), alpha_std=float(out["alpha_std"][C, C]),
                    ctrl_bias=float(cmed), ctrl_std=float(cstd), n_ctrl=int(len(cuse)),
                    fake_ratio=float(np.median(ratio)), fake_ratio_std=float(np.std(ratio)),
                    fake_off=float(np.mean(np.hypot(offs[:, 0], offs[:, 1]))),
                    n_fake=int(len(fk)), pk_dx=float(pk[0]), pk_dy=float(pk[1]),
                    sn_bad=snbad, zp_nat=float(nh["ZP_NAT"]), zp_err=float(nh["ZP_ERR"]),
                    zp_hdr=float(nh["ZP_HDR"]), fwhm=float(nh["FWHM_PX"]) * SCALE,
                    sky=sN, ast_rms=float(nh["ASTRMS"]), chip=nh["CHIP"])
    except Exception as e:
        return dict(tag=tag, ok=False, err=f"{type(e).__name__}: {e}")


def cmd_diff(a):
    os.makedirs(f"{a.workdir}/diff", exist_ok=True)
    R, rbad, _, _ = read_grid(f"{a.workdir}/ref/ref.fits")
    ctrl = control_positions(np.where(rbad, 0.0, R), rbad)
    np.savetxt(f"{a.workdir}/diff/controls.txt", ctrl, fmt="%d")
    print(f"{len(ctrl)} control positions")
    jobs = []
    for p in sorted(glob.glob(f"{a.workdir}/prep/*.fits")):
        tag = os.path.basename(p)[:-5]
        loo = f"{a.workdir}/ref/loo_{tag}.fits"
        jobs.append((p, loo if os.path.exists(loo) else f"{a.workdir}/ref/ref.fits",
                     ctrl, f"{a.workdir}/diff", a.nfake))
    with ProcessPoolExecutor(max_workers=a.nproc) as ex:
        res = list(ex.map(diff_one, jobs))
    keys = ["tag", "ok", "chip", "mjd", "ref", "flux", "score", "alpha_std", "ctrl_bias",
            "ctrl_std", "n_ctrl", "fake_ratio", "fake_ratio_std", "fake_off", "n_fake",
            "pk_dx", "pk_dy", "sn_bad", "zp_nat", "zp_err", "zp_hdr", "fwhm", "sky",
            "ast_rms", "err"]
    with open(f"{a.workdir}/diff.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in res:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    for r in sorted(res, key=lambda r: r.get("mjd", 0)):
        if r["ok"]:
            print(f"  {r['tag']}  {'LOO ' if r['ref'].startswith('loo') else 'full'}  "
                  f"flux {r['flux']:+9.2f} uJy  ctrl {r['ctrl_bias']:+6.2f}+-{r['ctrl_std']:5.2f} "
                  f"(n={r['n_ctrl']})  S {r['score']:+7.1f}  fake {r['fake_ratio']:.3f}"
                  f"+-{r['fake_ratio_std']:.3f} off {r['fake_off']:.2f}  pk ({r['pk_dx']:+.2f},"
                  f"{r['pk_dy']:+.2f})  {'SNBAD' if r['sn_bad'] else ''}")
        else:
            print(f"  {r['tag']}  FAILED  {r['err']}")


# ------------------------------------------------------------------ lc

def solve_floor(f, e):
    """Floor s >= 0 with sum((f-mu)^2/(e^2+s^2)) = N-1 (weighted mean mu)."""
    def chi(s):
        w = 1.0 / (e ** 2 + s ** 2)
        mu = (w * f).sum() / w.sum()
        return ((f - mu) ** 2 * w).sum() - (len(f) - 1), mu
    if chi(0.0)[0] <= 0:
        return 0.0, chi(0.0)[1]
    lo, hi = 0.0, 10 * np.std(f) + 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if chi(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    return hi, chi(hi)[1]


def cmd_lc(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    rows = [r for r in csv.DictReader(open(f"{a.workdir}/diff.csv")) if r["ok"] == "True"]
    # src_err (optional, from sn_zogy_cutouts.py): photon noise of SN + host
    has_src = bool(rows) and "src_err" in rows[0]
    for r in rows:
        for k in ("mjd", "flux", "score", "ctrl_std", "ctrl_bias", "zp_err", "fake_ratio", "fwhm", "sky"):
            r[k] = float(r[k])
        r["src_err"] = float(r["src_err"]) if has_src else 0.0
        r["loo"] = r["ref"].startswith("loo")
        r["sn_bad"] = r["sn_bad"] == "True"
        r["phase"] = r["mjd"] - a.disc_mjd
    rows.sort(key=lambda r: r["mjd"])

    # null set: leave-one-out reference epochs, or (with --null-before, for a
    # reference sharing no epoch with the science frames) pre-explosion frames
    if a.null_before is not None:
        nul = [r for r in rows if r["mjd"] < a.null_before and not r["sn_bad"]]
        what = f"pre-explosion frames before MJD {a.null_before}"
    else:
        nul = [r for r in rows if r["loo"] and not r["sn_bad"]]
        what = "reference epochs, leave-one-out"
    f = np.array([r["flux"] for r in nul])
    e = np.array([np.hypot(r["ctrl_std"], r["src_err"]) for r in nul])
    floor, mu = solve_floor(f, e)
    chi0 = float((((f - np.average(f, weights=1 / e ** 2)) / e) ** 2).sum() / max(len(f) - 1, 1))
    print(f"null test ({len(nul)} {what}): weighted mean {mu:+.3f} uJy "
          f"(+-{1 / np.sqrt((1 / e ** 2).sum()):.3f}), chi2/dof with control"
          f"{' + source' if has_src else ''} noise only = {chi0:.2f}, floor = {floor:.3f} uJy")

    for r in rows:
        zp_term = abs(r["flux"]) * 0.4 * np.log(10) * r["zp_err"]
        r["err"] = float(np.sqrt(r["ctrl_std"] ** 2 + r["src_err"] ** 2 + floor ** 2 + zp_term ** 2))
        r["snr"] = r["flux"] / r["err"]
        if r["snr"] >= 3:
            r["mag"] = ZP_OUT - 2.5 * np.log10(r["flux"])
            r["magerr"] = 2.5 / np.log(10) * r["err"] / r["flux"]
        else:
            r["mag"] = r["magerr"] = np.nan
        r["lim3"] = ZP_OUT - 2.5 * np.log10(3 * r["err"])

    keys = ["tag", "chip", "mjd", "phase", "loo", "sn_bad", "flux", "err", "snr", "mag",
            "magerr", "lim3", "ctrl_std", "ctrl_bias", "zp_nat", "zp_err", "fwhm", "sky",
            "fake_ratio", "score"] + (["src_err"] if has_src else [])
    with open(f"{a.workdir}/lightcurve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) and np.isfinite(r[k])
                            else ("" if isinstance(r[k], float) else r[k])) for k in keys})

    print(f"\n{'frame':26s} {'phase':>7s} {'flux uJy':>16s} {'S/N':>7s}  {'g (AB)':>14s}  note")
    for r in rows:
        m = (f"{r['mag']:6.3f}+-{r['magerr']:.3f}" if np.isfinite(r["mag"])
             else f"   >{r['lim3']:6.2f}   ")
        note = ("ref(LOO)" if r["loo"] else "") + (" SN-PIXELS-FLAGGED" if r["sn_bad"] else "")
        print(f"{r['tag']:26s} {r['phase']:+7.2f} {r['flux']:+8.2f}+-{r['err']:6.2f} "
              f"{r['snr']:+7.1f}  {m}  {note}")

    # ---- plot: flux (top) and magnitude (bottom) vs days from discovery
    INK, MUTED, BLUE, GREY = "#1a1a1a", "#6b6b6b", "#0072B2", "#9a9a9a"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True,
                                   constrained_layout=True,
                                   gridspec_kw={"height_ratios": [1, 1.15]})
    # frames with 3-sigma limits brighter than 18.5 (thick cloud) stay in the
    # CSV but would only flatten the plot
    use = [r for r in rows if not r["sn_bad"] and r["lim3"] > 18.5]
    nskip = len(rows) - len(use)
    ref_ep = [r for r in use if r["loo"]]
    sci = [r for r in use if not r["loo"]]
    for grp, col, mk, lab in ((ref_ep, GREY, "s", "reference epochs (leave-one-out)"),
                              (sci, BLUE, "o", "science epochs")):
        if grp:
            ax1.errorbar([r["phase"] for r in grp], [r["flux"] for r in grp],
                         yerr=[r["err"] for r in grp], fmt=mk, color=col, ms=5,
                         capsize=0, elinewidth=1.2, label=lab, mec="white", mew=0.6)
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1.set_ylabel(f"{a.band} difference flux (µJy)", color=INK)
    det = [r for r in sci if np.isfinite(r["mag"])]
    lim = [r for r in use if not np.isfinite(r["mag"])]
    if det:
        ax2.errorbar([r["phase"] for r in det], [r["mag"] for r in det],
                     yerr=[r["magerr"] for r in det], fmt="o", color=BLUE, ms=5,
                     capsize=0, elinewidth=1.2, mec="white", mew=0.6, label="detection (S/N ≥ 3)")
    if lim:
        ax2.plot([r["phase"] for r in lim], [r["lim3"] for r in lim], "v", color=GREY,
                 ms=6, label="3σ upper limit")
    ax2.invert_yaxis()
    ax2.set_ylabel(f"LS4 {a.band} (AB)", color=INK)
    ax2.set_xlabel(f"days from discovery (MJD {a.disc_mjd})", color=INK)
    for ax in (ax1, ax2):
        ax.axvline(0, color=MUTED, ls="--", lw=1)
        ax.grid(alpha=0.18, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.text(0, 1.0, " discovery", transform=ax1.get_xaxis_transform(), color=MUTED,
             fontsize=8, va="top")
    fig.suptitle(f"{a.name} — LS4 {a.band}-band ZOGY difference photometry", color=INK,
                 fontsize=12.5)
    if nskip:
        ax2.text(1.0, -0.16, f"{nskip} frame(s) with 3σ limit brighter than 18.5 not shown "
                 "(in lightcurve.csv)", transform=ax2.transAxes, ha="right", fontsize=7.5,
                 color=MUTED)
    fig.savefig(f"{a.workdir}/lightcurve.png", dpi=145)

    # ---- montage: ref / new / diff around the SN for every epoch
    hw = 30
    n = len(rows)
    ncol = 3
    fig, axes = plt.subplots(n, ncol, figsize=(ncol * 1.55, n * 1.55), squeeze=False)
    zs = ZScaleInterval()
    for i, r in enumerate(rows):
        with fits.open(f"{a.workdir}/diff/{r['tag']}.fits") as h:
            c = h[0].data.shape[0] // 2
            cut = {k: h[k].data[c - hw:c + hw + 1, c - hw:c + hw + 1] for k in ("REF", "NEW")}
            cut["DIFF"] = h[0].data[c - hw:c + hw + 1, c - hw:c + hw + 1]
        lo, hi = zs.get_limits(cut["NEW"][np.isfinite(cut["NEW"])])
        dfin = cut["DIFF"][np.isfinite(cut["DIFF"])]
        dsig = sigma_clipped_stats(dfin, sigma=3.0)[2] if dfin.size else 1.0
        for j, k in enumerate(("REF", "NEW", "DIFF")):
            ax = axes[i][j]
            v = (lo, hi) if k != "DIFF" else (-5 * dsig, 5 * dsig)
            ax.imshow(cut[k], origin="lower", cmap="gray", vmin=v[0], vmax=v[1],
                      interpolation="nearest")
            ax.plot(hw, hw, "+", color="#D55E00", ms=10, mew=0.8, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(k.lower(), fontsize=9)
        axes[i][0].set_ylabel(f"{r['phase']:+.1f} d\n{r['tag'].split('_')[2]}", fontsize=6.5)
    fig.savefig(f"{a.workdir}/cutouts.png", dpi=110, bbox_inches="tight")
    print(f"\nwrote {a.workdir}/lightcurve.csv, lightcurve.png, cutouts.png")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--coverage", required=True)
    p.add_argument("--band", required=True, choices=["g", "i", "z"])
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    p.add_argument("--only", default=None, help="comma-separated expids (testing)")
    p = sp.add_parser("ref")
    p.add_argument("--ref-before", type=float, required=True, help="MJD")
    p = sp.add_parser("diff")
    p.add_argument("--nfake", type=int, default=40)
    p = sp.add_parser("lc")
    p.add_argument("--disc-mjd", type=float, required=True)
    p.add_argument("--name", default="SN")
    p.add_argument("--band", default="g")
    p.add_argument("--null-before", type=float, default=None,
                   help="MJD: null test on frames before this instead of the LOO epochs")
    for p in sp.choices.values():
        p.add_argument("--workdir", required=True)
        p.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    {"prep": cmd_prep, "ref": cmd_ref, "diff": cmd_diff, "lc": cmd_lc}[a.cmd](a)


if __name__ == "__main__":
    main()
