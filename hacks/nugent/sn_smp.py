#!/usr/bin/env python3
"""Scene-modelling photometry (SMP) of one SN from SeeChange cutouts.

A forward model of the native pixels of every epoch -- no resampling, no
subtraction (cf. Holtzman et al. 2008 for SDSS, Brout et al. 2019 for DES):

    d_e(x) = sum_k g_k PSF_e(x - x_ek - s_e) + F_e PSF_e(x - x_e,SN - s_e) + b_e(x)

  g_k     the galaxy: fluxes on a 1"/px sky grid (the TAN grid of
          sn_zogy_cutouts.py, so the DECam reference maps onto it exactly),
          shared by all epochs
  F_e     SN flux in epoch e (uJy); fixed at 0 before --null-before
  b_e     sky plane of epoch e (constant + x + y)
  s_e     registration shift of epoch e, refined by Gauss-Newton between
          linear solves with exact derivatives and a 0.05 px Gaussian prior
          about the Gaia registration (LS4 shifts detrended: mean 0)
  PSF_e   the epoch's empirical PSF in native pixels (azimuthal wings), the
          one its zero point is measured with
  x_ek    grid point k in epoch e's pixels, through its Gaia-registered WCS

Sub-pixel PSF placement is scenic's exact quartic B-spline evaluation (the
spline-prefiltered stamp + 5-tap weights; = scipy.ndimage.shift order 4),
which also gives the analytic position derivatives.  The sky plane, the
in-fit registration and the SN-free-epoch guard follow scenic too
(github.com/nugent68/scenic, scene_model/v2_frame.py).

d_e is the cutout in uJy (DN x 10^(0.4 (23.9 - ZP_e))), weighted by the
SeeChange inverse-variance map plus the photon noise of the model (gain from
the ZOGY diff stage).  Everything is linear: one weighted least-squares solve,
with a light ridge on g_k for the modes the data leave free (the galaxy is
deconvolved to 1" from 2.4-3" PSFs), 5-sigma pixel clipping and the photon
noise iterated.  Errors come from the covariance -- they include the
galaxy-model uncertainty, which is correlated between epochs -- plus the ZP
error.

Variants: "ls4" -- LS4 epochs only, classic SMP with no reference image at
all; "ref" -- the DECam coadd (ref/ref.fits, on the grid) is one more epoch
with F = 0, its noise inflated to the PSF-flux scatter at blank positions
(it is smoothed and resampled, so its pixels are correlated).  With
--free-all (ref only) F is free in every LS4 epoch and the coadd alone
anchors zero: the pre-explosion fluxes then measure any source at the SN
position relative to the coadd's epochs (2013-2025).

Checks:
  null  each pre-explosion epoch in turn gets a free F (leave-one-out)
  host  point sources at ~10 host-disk positions 6-14 px from the SN, free in the
        post-explosion epochs and 0 before: their fluxes should scatter about
        0 within their errors

Needs, in --workdir: the cutouts, gaia.ecsv, grid.json, diff.csv (gain),
ref/ref.fits (also picks the host controls) and diff/controls.txt --
i.e. a finished sn_zogy_cutouts.py prep/ref/diff.  Writes smp/.

Usage:
  sn_smp.py --workdir W --null-before 61183.4 --disc-mjd 61184.397 --name "SN 2026nmw"
"""
import argparse
import csv
import glob
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from scipy import linalg, ndimage, sparse
from scipy.ndimage import spline_filter

import sn_zogy_cutouts as sc

base = sc.base
C, ZP_OUT, PSF_HALF = sc.C, sc.ZP_OUT, base.PSF_HALF
OFF = np.arange(-PSF_HALF, PSF_HALF + 1)
MIN_FREE = (3, 0.15)                      # at least 3 and 15% of the LS4 epochs SN-free


# ------------------------------------------------------------------ epochs

def load_ls4(path, gaia, gw, ra0, dec0, rs, gsky, hsky, null_before):
    """Native cutout -> stamp data (uJy), variance, PSF and model geometry."""
    m = re.match(r"ls4_(\d{8})_(\d{6})_([A-Z]{2}_[A-Z])_", os.path.basename(path))
    tag = "_".join(m.groups())
    stem = path[:-len(".image.fits.fz")]
    with fits.open(path) as h:
        img = h[1].data.astype(np.float64)
        hdr = h[1].header.copy()
    wgt = fits.getdata(stem + ".weight.fits.fz", 1).astype(np.float64)
    flg = fits.getdata(stem + ".flags.fits.fz", 1)
    bad = np.ascontiguousarray((flg != 0) | ~(wgt > 0) | ~np.isfinite(img))
    win = sc.cutout_wcs(stem)
    mjd = float(Time(hdr["DATE-OBS"], scale="utc").mjd)
    sky, _, _ = sc.flat_sky(img, bad, 64)
    sub = np.where(bad, 0.0, img - sky)
    cxf, cyf, reg_rms, reg_n = sc.register(sub, bad, gaia, mjd, win, 0, 0, (13.5, 18.0), 8.0, 1.3)

    def affine(x, y):
        return x + cxf[0] + cxf[1] * x + cxf[2] * y, y + cyf[0] + cyf[1] * x + cyf[2] * y

    def to_pix(ra, dec):
        return affine(*win.all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0, quiet=True))

    xsn, ysn = (float(v[0]) for v in to_pix(ra0, dec0))
    gx, gy = affine(*base.gaia_xy(gaia, mjd, win, sip=True))
    ch = base.characterize(np.where(bad, np.nan, sub), bad, gaia, mjd, None, xy=(gx, gy),
                           center=(xsn, ysn), **sc.LS4_CHAR)
    s = 10 ** (0.4 * (ZP_OUT - ch["zp"]))
    yy, xx = np.indices(img.shape)
    py, px = np.where(np.hypot(xx - xsn, yy - ysn) <= rs)
    var0 = np.where(bad[py, px], np.inf, s ** 2 / np.where(wgt[py, px] > 0, wgt[py, px], 1.0))
    return dict(tag=tag, kind="ls4", mjd=mjd, pre=mjd < null_before, s=s, zp=ch["zp"],
                zp_err=ch["zp_err"], fwhm=ch["fwhm_px"], psf=ch["psf"], shape=img.shape,
                px=px, py=py, d=s * sub[py, px], var0=var0, sn=(xsn, ysn),
                gal=to_pix(*gsky), ctl=to_pix(*hsky), reg_rms=reg_rms)


def load_ref(workdir, rs, gxy, hxy):
    """The DECam coadd as an F = 0 epoch on the grid."""
    R, rbad, pR, rh = base.read_grid(f"{workdir}/ref/ref.fits")
    # white-noise sigma that reproduces the PSF-flux scatter at blank controls
    ctrl = np.loadtxt(f"{workdir}/diff/controls.txt", dtype=int)
    s2 = float((pR ** 2).sum())
    F = []
    for x, y in ctrl:
        st = R[y - PSF_HALF:y + PSF_HALF + 1, x - PSF_HALF:x + PSF_HALF + 1]
        if st.shape == pR.shape and np.isfinite(st).all():
            F.append(((st - np.median(st)) * pR).sum() / s2)
    F = np.array(F)
    sig = 1.4826 * np.median(np.abs(F - np.median(F))) * np.sqrt(s2)
    yy, xx = np.indices(R.shape)
    py, px = np.where(np.hypot(xx - C, yy - C) <= rs)
    var0 = np.where(rbad[py, px] | ~np.isfinite(R[py, px]), np.inf, sig ** 2)
    print(f"  ref: per-pixel RMS {rh['SKYSIG']:.4f} uJy, effective white sigma {sig:.4f} uJy "
          f"(from {len(F)} blank-control PSF fluxes)")
    return dict(tag="decam_ref", kind="ref", mjd=float(rh["MJD-OBS"]), pre=True, s=1.0, zp=ZP_OUT,
                zp_err=0.0, fwhm=float(rh["FWHM_PX"]), psf=pR, shape=R.shape, px=px, py=py,
                d=np.nan_to_num(R[py, px]), var0=var0, sn=(float(C), float(C)), gal=gxy, ctl=hxy,
                reg_rms=0.0)


# ------------------------------------------------------------------ design matrix

def b4_weights(f):
    """Quartic B-spline taps d = -2..2 for a sub-pixel shift f (|f| <= 0.5) and
    their derivative in f, vectorised over f -- numpy port of scenic's
    bspline4_weights (scene_model/v1_jax.py).  With the spline-prefiltered
    stamp this reproduces scipy.ndimage.shift(order=4) exactly, and the
    derivative gives exact position gradients for the registration step."""
    t = np.arange(-2.0, 3.0)[None, :] + np.asarray(f, float)[:, None]
    x, s = np.abs(t), np.sign(t)
    w = np.where(x <= 0.5, 115.0 / 192.0 - 0.625 * x ** 2 + x ** 4 / 4.0,
        np.where(x <= 1.5, (55.0 + 20.0 * x - 120.0 * x ** 2 + 80.0 * x ** 3 - 16.0 * x ** 4) / 96.0,
        np.where(x <= 2.5, (5.0 - 2.0 * x) ** 4 / 384.0, 0.0)))
    dw = s * np.where(x <= 0.5, -1.25 * x + x ** 3,
             np.where(x <= 1.5, (20.0 - 240.0 * x + 240.0 * x ** 2 - 64.0 * x ** 3) / 96.0,
             np.where(x <= 2.5, -(5.0 - 2.0 * x) ** 3 / 48.0, 0.0)))
    return w, dw


def psf_coef(psf):
    """Spline coefficients of an (S, S) stamp, padded to (S+4, S+4): scenic's
    recipe (edge-pad 12, order-4 grid-mirror prefilter)."""
    S = psf.shape[0]
    return spline_filter(np.pad(psf, 12, mode="edge"), order=4, mode="grid-mirror")[10:14 + S, 10:14 + S]


def stamps(coef, fy, fx, deriv=False, chunk=512):
    """(n, S, S) unit stamps shifted by (fy, fx) along (rows, cols), and with
    deriv their derivatives in x and y."""
    S = coef.shape[0] - 4
    rows = np.stack([coef[k:k + S, :] for k in range(5)])              # (5, S, S+4)
    outs = [[], [], []]
    for c0 in range(0, len(fx), chunk):
        wy, dwy = b4_weights(fy[c0:c0 + chunk])
        wx, dwx = b4_weights(fx[c0:c0 + chunk])
        tmp = np.einsum("nk,kij->nij", wy, rows)                       # (n, S, S+4)
        cols = np.stack([tmp[:, :, k:k + S] for k in range(5)])        # (5, n, S, S)
        outs[0].append(np.einsum("nk,knij->nij", wx, cols))
        if deriv:
            outs[1].append(np.einsum("nk,knij->nij", dwx, cols))
            tmpd = np.einsum("nk,kij->nij", dwy, rows)
            outs[2].append(np.einsum("nk,knij->nij", wx,
                                     np.stack([tmpd[:, :, k:k + S] for k in range(5)])))
    if not deriv:
        return np.concatenate(outs[0])
    return tuple(np.concatenate(o) for o in outs)


def point_block(e, X, Y):
    """Rows, stamp values and x/y derivatives of unit point sources at native
    positions (X, Y) in epoch e; returns (rows, comp index, v, dx, dy)."""
    X, Y = np.atleast_1d(X).astype(float), np.atleast_1d(Y).astype(float)
    ix, iy = np.floor(X + 0.5).astype(int), np.floor(Y + 0.5).astype(int)
    st, dsx, dsy = stamps(e["coef"], Y - iy, X - ix, deriv=True)
    PX = ix[:, None, None] + OFF[None, None, :]
    PY = iy[:, None, None] + OFF[None, :, None]
    PX, PY = np.broadcast_arrays(PX, PY)
    H, W = e["shape"]
    ok = (PX >= 0) & (PX < W) & (PY >= 0) & (PY < H)
    rr = np.full(PX.shape, -1)
    rr[ok] = e["rowmap"][PY[ok], PX[ok]]
    ok = rr >= 0
    k = np.broadcast_to(np.arange(len(X))[:, None, None], PX.shape)
    return rr[ok], k[ok], st[ok], dsx[ok], dsy[ok]


def build(epochs, K, nctl, shifts, free_all=False):
    """Sparse design matrix A and its position derivatives Dx, Dy (same
    pattern) over all epochs, every component of epoch e displaced by its
    registration shift.  Columns: galaxy (K); per epoch a sky plane
    (constant, x, y); F per LS4 epoch; nctl host controls per
    post-explosion LS4 epoch."""
    R, Cc, V, DX, DY = [], [], [], [], []

    def add(r, c, v, dx=None, dy=None):
        R.append(r); Cc.append(c); V.append(v)
        DX.append(np.zeros_like(v) if dx is None else dx)
        DY.append(np.zeros_like(v) if dy is None else dy)

    col = {"gal": np.arange(K), "b": {}, "F": {}, "ctl": {}}
    nb, row0 = K, 0
    for e, (sx, sy) in zip(epochs, shifts):
        e["row0"] = row0
        e["rowmap"] = np.full(e["shape"], -1)
        e["rowmap"][e["py"], e["px"]] = np.arange(len(e["px"]))
        e["coef"] = psf_coef(e["psf"])
        r, k, v, dx, dy = point_block(e, np.asarray(e["gal"][0]) + sx, np.asarray(e["gal"][1]) + sy)
        add(r + row0, k, v, dx, dy)
        n = len(e["px"])
        rr = np.arange(n) + row0
        col["b"][e["tag"]] = [nb, nb + 1, nb + 2]
        u = (e["px"] - e["sn"][0]) / max(PSF_HALF, 1)
        w_ = (e["py"] - e["sn"][1]) / max(PSF_HALF, 1)
        for j, val in enumerate((np.ones(n), u, w_)):
            add(rr, np.full(n, nb + j), val.astype(float))
        nb += 3
        row0 += n
    for e, (sx, sy) in zip(epochs, shifts):
        if e["kind"] != "ls4":
            continue
        r, _, v, dx, dy = point_block(e, e["sn"][0] + sx, e["sn"][1] + sy)
        add(r + e["row0"], np.full(len(r), nb), v, dx, dy)
        col["F"][e["tag"]] = nb
        nb += 1
        if free_all or not e["pre"]:
            r, k, v, dx, dy = point_block(e, np.asarray(e["ctl"][0]) + sx, np.asarray(e["ctl"][1]) + sy)
            add(r + e["row0"], k + nb, v, dx, dy)
            col["ctl"][e["tag"]] = list(range(nb, nb + nctl))
            nb += nctl
    rows, cols = np.concatenate(R), np.concatenate(Cc)
    mk = lambda vals: sparse.csr_matrix((np.concatenate(vals), (rows, cols)), shape=(row0, nb))
    return mk(V), mk(DX), mk(DY), col


# ------------------------------------------------------------------ fit

class Fit:
    def __init__(self, epochs, K, nctl, gain, ridge, free_all=False):
        self.epochs, self.K, self.nctl, self.gain, self.ridge = epochs, K, nctl, gain, ridge
        self.free_all = free_all
        self.shifts = np.zeros((len(epochs), 2))
        self.rebuild()
        self.d = np.concatenate([e["d"] for e in epochs])
        self.var0 = np.concatenate([e["var0"] for e in epochs])
        self.ep = np.concatenate([np.full(len(e["px"]), i) for i, e in enumerate(epochs)])
        self.s = np.array([e["s"] for e in epochs])[self.ep]
        self.isls4 = np.array([e["kind"] == "ls4" for e in epochs])[self.ep]
        self.var = self.var0.copy()
        self.clip = ~np.isfinite(self.var0)

    def rebuild(self):
        self.A, self.Dx, self.Dy, self.col = build(self.epochs, self.K, self.nctl, self.shifts,
                                                   self.free_all)

    def active(self, extra_F=()):
        act = np.zeros(self.A.shape[1], bool)
        act[self.col["gal"]] = True
        for v in self.col["b"].values():
            act[v] = True
        for e in self.epochs:
            if e["kind"] == "ls4" and (self.free_all or not e["pre"] or e["tag"] in extra_F):
                act[self.col["F"][e["tag"]]] = True
        for v in self.col["ctl"].values():
            act[v] = True
        return act

    def normal(self):
        w = np.where(self.clip, 0.0, 1.0 / self.var)
        self.N = (self.A.T @ (sparse.diags(w) @ self.A)).toarray()
        self.rhs = self.A.T @ (w * self.d)
        self.w = w

    def solve(self, act, ridge=None, want=()):
        idx = np.where(act)[0]
        Na = self.N[np.ix_(idx, idx)]
        g = np.isin(idx, self.col["gal"])
        lam = (self.ridge if ridge is None else ridge) * np.median(np.diag(Na)[g])
        Na[g, g] += lam
        # A uniform galaxy level (and a uniform gradient) is exactly degenerate
        # with the epochs' sky planes, and the ridge above -- scaled to the
        # median galaxy diagonal, which the barely-constrained outer grid
        # dominates -- is far too weak to fix it: the solve then parks tens of
        # uJy/px in the galaxy and the opposite in every sky, which inflates
        # the photon-noise term.  Pin sum_e b_e (and each slope sum) to zero.
        for j in range(3):
            bi = [int(np.searchsorted(idx, v[j])) for v in self.col["b"].values() if act[v[j]]]
            Na[np.ix_(bi, bi)] += 1e3 * np.median(np.diag(Na)[bi])
        cf = linalg.cho_factor(Na)
        x = np.zeros(self.A.shape[1])
        x[idx] = linalg.cho_solve(cf, self.rhs[idx])
        cov = {}
        if len(want):
            pos = np.searchsorted(idx, want)
            E = np.zeros((len(idx), len(want)))
            E[pos, np.arange(len(want))] = 1.0
            cv = linalg.cho_solve(cf, E)
            for j, c in enumerate(want):
                cov[c] = float(cv[pos[j], j])
            cov["matrix"] = cv[pos, :]
        return x, cov

    def linear(self, n=2, kclip=5.0, quiet=False):
        """Linear solve with the photon noise of the model and 5-sigma
        clipping iterated n times."""
        act = self.active()
        for it in range(n):
            self.normal()
            x, _ = self.solve(act)
            model = self.A @ x
            xs = np.zeros_like(x)
            for v in self.col["b"].values():
                xs[v] = x[v]
            sig = np.clip(model - self.A @ xs, 0.0, None)   # galaxy + SN light, no sky
            self.var = np.where(self.isls4, self.var0 + self.s * sig / self.gain, self.var0)
            z = (self.d - model) / np.sqrt(self.var)
            new = np.zeros_like(self.clip)
            for i, e in enumerate(self.epochs):
                m = self.ep == i
                im = np.zeros(e["shape"], bool)
                im[e["py"], e["px"]] = np.abs(z[m]) > kclip
                im = ndimage.binary_dilation(im, iterations=1)
                new[m] = im[e["py"], e["px"]]
            self.clip = new | ~np.isfinite(self.var0)
        self.normal()
        self.x, _ = self.solve(act)
        self.model = self.A @ self.x
        if not quiet:
            print(f"    linear solve: clipped {int((self.clip & np.isfinite(self.var0)).sum())} px", flush=True)

    def register_step(self, max_step=0.2, sigma=0.05):
        """Per-epoch registration shift by Gauss-Newton on the residuals, with
        the exact model derivatives (as scenic.v2_frame.register_update), and a
        Gaussian prior of `sigma` px about the Gaia registration (whose own
        residuals are 0.03-0.12 px): without it the shifts and the 1" galaxy
        co-adapt and the shifts random-walk (0.02 px per pass, not
        converging).  The LS4 shifts are detrended (mean removed) so the
        absolute astrometry stays on Gaia; the DECam epoch is the anchor."""
        r = self.d - self.model
        mx, my = self.Dx @ self.x, self.Dy @ self.x
        steps = np.zeros_like(self.shifts)
        for i, e in enumerate(self.epochs):
            if e["kind"] != "ls4":
                continue
            m = (self.ep == i)
            w = self.w[m]
            H = np.array([[np.sum(w * mx[m] ** 2), np.sum(w * mx[m] * my[m])],
                          [np.sum(w * mx[m] * my[m]), np.sum(w * my[m] ** 2)]])
            g = np.array([np.sum(w * r[m] * mx[m]), np.sum(w * r[m] * my[m])])
            g -= self.shifts[i] / sigma ** 2
            steps[i] = np.clip(np.linalg.solve(H + np.eye(2) / sigma ** 2, g), -max_step, max_step)
        ls4 = np.array([e["kind"] == "ls4" for e in self.epochs])
        new = self.shifts + steps
        new[ls4] -= new[ls4].mean(axis=0)
        step = np.abs(new - self.shifts).max()
        self.shifts = new
        self.rebuild()
        return step

    def fit(self, n_reg=4):
        self.linear()
        for k in range(n_reg):
            step = self.register_step()
            self.linear(quiet=True)
            print(f"    registration pass {k + 1}: largest shift change {step:.4f} px; "
                  f"shift RMS {np.sqrt((self.shifts ** 2).sum(1).mean()):.4f} px", flush=True)
        self.linear()


# ------------------------------------------------------------------ run

def pick_controls(workdir, rmin=6, rmax=14, sep=5, nmax=10, seed=3):
    """Host-disk positions like the SN's (sn_zogy_cutouts.host_controls, on
    the reference) rmin-rmax px from it, at least `sep` px apart."""
    R, rbad, _, _ = base.read_grid(f"{workdir}/ref/ref.fits")
    cand = sc.host_controls(R, rbad, rmin=rmin, rmax=rmax, step=1)
    pts = []
    for x, y in np.random.default_rng(seed).permutation(cand):
        if len(pts) < nmax and all(np.hypot(x - a, y - b) >= sep for a, b in pts):
            pts.append((x, y))
    return np.array(pts, float)


def run_variant(name, epochs, K, nctl, gain, a):
    print(f"\n=== SMP variant '{name}': {len(epochs)} epochs, {K} galaxy elements, "
          f"{nctl} host controls", flush=True)
    ls4 = [e for e in epochs if e["kind"] == "ls4"]
    free_all = a.free_all and any(e["kind"] == "ref" for e in epochs)
    if a.free_all and not free_all:
        print("    --free-all needs the reference epoch as the anchor: skipping this variant")
        return None
    nfree = 0 if free_all else sum(e["pre"] for e in ls4)
    if not free_all and nfree < max(MIN_FREE[0], MIN_FREE[1] * len(ls4)):
        # without SN-free epochs the host under the SN is degenerate with a
        # constant light-curve offset (scenic aborts on the same condition)
        raise SystemExit(f"only {nfree} of {len(ls4)} LS4 epochs are SN-free")
    f = Fit(epochs, K, nctl, gain, a.ridge, free_all=free_all)
    print(f"    {f.A.shape[0]} pixels, {f.A.shape[1]} parameters, {f.A.nnz} design entries; "
          + ("SN flux free in every LS4 epoch, anchored by the reference epoch alone"
             if free_all else f"{nfree} SN-free LS4 epochs"))
    f.fit(n_reg=a.reg_passes)
    act = f.active()
    Fcols = [f.col["F"][e["tag"]] for e in epochs
             if e["kind"] == "ls4" and (free_all or not e["pre"])]
    ccols = [c for v in f.col["ctl"].values() for c in v]
    x, cov = f.solve(act, want=Fcols + ccols)
    # ridge sensitivity
    x10, _ = f.solve(act, ridge=a.ridge * 10)
    x01, _ = f.solve(act, ridge=a.ridge / 10)
    dr = max(max(abs(x10[c] - x[c]), abs(x01[c] - x[c])) for c in Fcols) if Fcols else 0.0

    out = []
    for i, e in enumerate(epochs):
        if e["kind"] != "ls4":
            continue
        m = f.ep == i
        r = (f.d - f.model)[m]
        good = ~f.clip[m]
        chi2 = float((r[good] ** 2 / f.var[m][good]).sum() / max(good.sum(), 1))
        c = f.col["F"][e["tag"]]
        if e["pre"] and not free_all:
            xl, cl = f.solve(f.active(extra_F=(e["tag"],)), want=[c])
            flux, ecov = xl[c], np.sqrt(cl[c])
        else:
            flux, ecov = x[c], np.sqrt(cov[c])
        err = float(np.hypot(ecov, flux * 0.4 * np.log(10) * e["zp_err"]))
        rec = dict(tag=e["tag"], mjd=e["mjd"], phase=e["mjd"] - a.disc_mjd, pre=e["pre"],
                   flux=float(flux), err_cov=float(ecov), err=err, snr=float(flux / err),
                   chi2=chi2, n_pix=int(good.sum()), n_clip=int((~good).sum()),
                   zp=e["zp"], zp_err=e["zp_err"], fwhm=e["fwhm"],
                   sky_b=float(x[f.col["b"][e["tag"]][0]]),
                   reg_dx=float(f.shifts[i, 0]), reg_dy=float(f.shifts[i, 1]))
        if rec["snr"] >= 3:
            rec["mag"] = ZP_OUT - 2.5 * np.log10(flux)
            rec["magerr"] = 2.5 / np.log(10) * err / flux
        else:
            rec["mag"] = rec["magerr"] = np.nan
        rec["lim3"] = ZP_OUT - 2.5 * np.log10(3 * err)
        out.append(rec)

    nul = [r for r in out if r["pre"]]
    fn = np.array([r["flux"] for r in nul]); en = np.array([r["err"] for r in nul])
    if free_all:
        # every LS4 flux shares the reference anchor, so the pre-explosion
        # fluxes are correlated: weight their mean with the full covariance
        ip = [Fcols.index(f.col["F"][r["tag"]]) for r in nul]
        Cp = cov["matrix"][np.ix_(ip, ip)] + np.diag([(r["flux"] * 0.4 * np.log(10) * r["zp_err"]) ** 2
                                                       for r in nul])
        Ci = np.linalg.inv(Cp)
        one = np.ones(len(fn))
        wm, wmerr = float(one @ Ci @ fn / (one @ Ci @ one)), float(1 / np.sqrt(one @ Ci @ one))
        chi_c = float((fn - wm) @ Ci @ (fn - wm) / max(len(fn) - 1, 1))
        rho = float(np.mean(Cp[~np.eye(len(fn), dtype=bool)] /
                            np.sqrt(np.outer(np.diag(Cp), np.diag(Cp)))[~np.eye(len(fn), dtype=bool)]))
        print(f"    pre-explosion baseline vs the reference epoch ({len(nul)} LS4 epochs, free): "
              f"covariance-weighted mean {wm:+.2f} +- {wmerr:.2f} uJy; chi2/dof about that mean "
              f"{chi_c:.2f}; mean error correlation {rho:+.2f}")
    else:
        wm = float((fn / en ** 2).sum() / (1 / en ** 2).sum())
        print(f"    null test (leave-one-out, {len(nul)} pre-explosion epochs): weighted mean "
              f"{wm:+.2f} +- {1 / np.sqrt((1 / en ** 2).sum()):.2f} uJy, chi2/dof about 0 = "
              f"{float(((fn / en) ** 2).sum() / len(fn)):.2f}")
    if ccols:
        cf_ = np.array([x[c] for c in ccols]); ce = np.sqrt([cov[c] for c in ccols])
        print(f"    host controls ({nctl} positions x {len(f.col['ctl'])} "
              f"{'LS4' if free_all else 'post'} epochs): mean "
              f"{np.mean(cf_):+.2f} uJy, mean pull {np.mean(cf_ / ce):+.2f}, chi2/dof "
              f"{float(((cf_ / ce) ** 2).mean()):.2f}, median error {np.median(ce):.2f} uJy")
        if free_all:
            pre_tags = {r["tag"] for r in nul}
            cp = [c for t, v in f.col["ctl"].items() if t in pre_tags for c in v]
            vp = np.array([x[c] for c in cp]); ep_ = np.sqrt([cov[c] for c in cp])
            print(f"    host controls in the pre-explosion epochs only: mean {np.mean(vp):+.2f} uJy "
                  f"(the same anchor offset would show here), mean pull {np.mean(vp / ep_):+.2f}")
    good = ~f.clip
    chi_all = float(((f.d - f.model)[good] ** 2 / f.var[good]).sum())
    print(f"    chi2 / (pixels - parameters) = {chi_all:.0f} / {good.sum() - act.sum()} = "
          f"{chi_all / (good.sum() - act.sum()):.3f}  (per-epoch chi2/px below is lowered "
          f"where the shared galaxy absorbs noise)")
    print(f"    largest SN-flux change for ridge x10 / x0.1: {dr:.2f} uJy")
    print(f"    {'epoch':22s} {'phase':>7s} {'flux':>9s} {'err':>6s} {'(cov)':>6s} {'S/N':>6s} "
          f"{'g':>14s} {'chi2/px':>7s} {'clip':>4s} {'reg dx,dy (px)':>15s}")
    for r in out:
        mg = f"{r['mag']:6.3f}+-{r['magerr']:.3f}" if np.isfinite(r["mag"]) else f"   >{r['lim3']:6.2f}   "
        print(f"    {r['tag']:22s} {r['phase']:+7.2f} {r['flux']:+9.2f} {r['err']:6.2f} {r['err_cov']:6.2f} "
              f"{r['snr']:+6.1f} {mg:>14s} {r['chi2']:7.2f} {r['n_clip']:4d} "
              f"{r['reg_dx']:+7.3f},{r['reg_dy']:+7.3f}"
              f"{('  (pre-explosion, free)' if free_all else '  (null, LOO)') if r['pre'] else ''}")

    os.makedirs(f"{a.workdir}/smp", exist_ok=True)
    keys = ["tag", "mjd", "phase", "pre", "flux", "err", "err_cov", "snr", "mag", "magerr", "lim3",
            "chi2", "n_pix", "n_clip", "zp", "zp_err", "fwhm", "sky_b", "reg_dx", "reg_dy"]
    with open(f"{a.workdir}/smp/lightcurve_smp_{name}{a.suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and np.isfinite(v)
                            else ("" if isinstance(v, float) else v)) for k, v in r.items()})
    side = int(round(np.sqrt(K)))
    fits.writeto(f"{a.workdir}/smp/galaxy_{name}{a.suffix}.fits",
                 x[f.col["gal"]].reshape(side, side).astype(np.float32), overwrite=True)
    montage(f, x, name, a)
    return out


def montage(f, x, name, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval
    eps = [i for i, e in enumerate(f.epochs) if e["kind"] == "ls4"]
    xg = np.zeros_like(x); xg[f.col["gal"]] = x[f.col["gal"]]
    gal = f.A @ xg
    xs_ = np.zeros_like(x)
    for v in f.col["b"].values():
        xs_[v] = x[v]
    sky = f.A @ xs_
    fig, axes = plt.subplots(len(eps), 4, figsize=(4 * 1.6, len(eps) * 1.6), squeeze=False)
    zs = ZScaleInterval()
    for row, i in enumerate(eps):
        e = f.epochs[i]
        m = f.ep == i
        b = sky[m]
        ims = {}
        for k, v in (("data", f.d[m] - b), ("galaxy model", gal[m]),
                     ("data - galaxy", f.d[m] - b - gal[m]), ("residual", f.d[m] - f.model[m])):
            im = np.full(e["shape"], np.nan)
            im[e["py"], e["px"]] = v
            xs, ys = e["sn"]
            h = a.stamp
            ims[k] = im[int(ys) - h:int(ys) + h + 1, int(xs) - h:int(xs) + h + 1]
        lo, hi = zs.get_limits(ims["data"][np.isfinite(ims["data"])])
        sg = np.nanmedian(np.sqrt(f.var[m]))
        for j, k in enumerate(ims):
            ax = axes[row][j]
            v = (lo, hi) if j < 2 else (-5 * sg, 5 * sg)
            ax.imshow(ims[k], origin="lower", cmap="gray", vmin=v[0], vmax=v[1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(k, fontsize=8)
        axes[row][0].set_ylabel(f"{e['mjd'] - a.disc_mjd:+.1f} d", fontsize=7)
    fig.savefig(f"{a.workdir}/smp/montage_{name}{a.suffix}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def compare_plot(results, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED = "#1a1a1a", "#6b6b6b"
    sty = {"zogy": ("ZOGY vs DECam coadd", "#0072B2", "o"),
           "ls4": ("SMP, LS4 epochs only", "#D55E00", "s"),
           "ref": ("SMP, LS4 + DECam coadd", "#009E73", "D")}
    zr = {r["tag"]: r for r in csv.DictReader(open(f"{a.workdir}/lightcurve.csv"))}
    series = {"zogy": [dict(tag=t, phase=float(r["phase"]), flux=float(r["flux"]), err=float(r["err"]))
                       for t, r in zr.items()]}
    series.update(results)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, constrained_layout=True,
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    for j, (k, rows) in enumerate(series.items()):
        lab, colr, mk = sty[k]
        dx = (j - 1) * 0.25
        ax1.errorbar([r["phase"] + dx for r in rows], [r["flux"] for r in rows],
                     yerr=[r["err"] for r in rows], fmt=mk, color=colr, ms=4.5, capsize=0,
                     elinewidth=1.1, mec="white", mew=0.5, label=lab)
        if k != "zogy":
            ok = [r for r in rows if r["tag"] in zr]
            ax2.errorbar([r["phase"] + dx for r in ok],
                         [r["flux"] - float(zr[r["tag"]]["flux"]) for r in ok],
                         yerr=[r["err"] for r in ok], fmt=mk, color=colr, ms=4.5, capsize=0,
                         elinewidth=1.1, mec="white", mew=0.5, label=lab)
    ax1.set_ylabel("g flux (µJy)", color=INK)
    ax2.set_ylabel("SMP − ZOGY (µJy)", color=INK)
    ax2.set_xlabel(f"days from discovery (MJD {a.disc_mjd})", color=INK)
    for ax in (ax1, ax2):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, ls="--", lw=1)
        ax.grid(alpha=0.18, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{a.name} — LS4 g: scene-modelling vs ZOGY photometry", color=INK, fontsize=12.5)
    fig.savefig(f"{a.workdir}/smp/lightcurve_smp{a.suffix}.png", dpi=145)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cutouts", default="ls4/*_g_Sci_*.image.fits.fz")
    ap.add_argument("--null-before", type=float, required=True, help="MJD: F = 0 before this")
    ap.add_argument("--disc-mjd", type=float, required=True)
    ap.add_argument("--name", default="SN")
    ap.add_argument("--stamp", type=int, default=15, help="fit radius around the SN (native px)")
    ap.add_argument("--grid-half", type=int, default=25, help="galaxy grid half-size (1\" px)")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="ridge on galaxy elements, relative to the median diagonal")
    ap.add_argument("--variants", default="ls4,ref")
    ap.add_argument("--free-all", action="store_true",
                    help="SN flux free in every LS4 epoch (pre-explosion included), the "
                         "reference epoch alone anchoring zero; 'ref' variant only")
    ap.add_argument("--suffix", default="", help="appended to output file names")
    ap.add_argument("--reg-passes", type=int, default=6,
                    help="Gauss-Newton passes on the per-epoch registration shifts (0 = off)")
    a = ap.parse_args()

    w = a.workdir
    g = json.load(open(f"{w}/grid.json"))
    gw = base.grid_wcs(g["ra"], g["dec"])
    gaia = Table.read(f"{w}/gaia.ecsv", format="ascii.ecsv")
    gain = float(next(csv.DictReader(open(f"{w}/diff.csv")))["gain"])
    dg = np.arange(-a.grid_half, a.grid_half + 1, dtype=float)
    GX, GY = np.meshgrid(C + dg, C + dg)
    gxy = (GX.ravel(), GY.ravel())
    gsky = gw.wcs_pix2world(*gxy, 0)
    ctl = pick_controls(w)
    hxy = (ctl[:, 0], ctl[:, 1])
    hsky = gw.wcs_pix2world(*hxy, 0)
    rc = np.hypot(ctl[:, 0] - C, ctl[:, 1] - C)
    print(f"gain {gain:.3f} e-/DN; galaxy grid {len(dg)}x{len(dg)} at 1\"; stamp radius {a.stamp} px; "
          f"{len(ctl)} host controls at {rc.min():.0f}-{rc.max():.0f} px")

    ls4 = []
    for p in sorted(glob.glob(f"{w}/{a.cutouts}")):
        try:
            e = load_ls4(p, gaia, gw, g["ra"], g["dec"], a.stamp, gsky, hsky, a.null_before)
            ls4.append(e)
            print(f"  {e['tag']:22s} zp {e['zp']:.3f}+-{e['zp_err']:.3f}  fwhm {e['fwhm']:.2f} px  "
                  f"reg {e['reg_rms']:.3f} px  SN at ({e['sn'][0]:.2f},{e['sn'][1]:.2f})"
                  f"{'  pre' if e['pre'] else ''}", flush=True)
        except Exception as ex:
            print(f"  {os.path.basename(p)[:40]}  SKIPPED  {type(ex).__name__}: {ex}")
    results = {}
    for v in a.variants.split(","):
        eps = ls4 + ([load_ref(w, a.stamp, gxy, hxy)] if v == "ref" else [])
        res = run_variant(v, eps, len(dg) ** 2, len(ctl), gain, a)
        if res is not None:
            results[v] = res
    compare_plot(results, a)
    print(f"\nwrote {w}/smp/")


if __name__ == "__main__":
    main()
