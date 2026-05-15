#!/usr/bin/env python3
"""Diagnostic plot for the suspect NW_B chip vs. a known-good chip (NE_G).

Reads one raw bias file from /ls4/zeros, locates the NW_B and NE_G HDUs,
and produces a 4x2 panel figure:

    row 1 (raw):       full raw image | bias-strip + data layout overlay
    row 2 (trimmed):   bias-subtracted trimmed image | histogram of pixel values
    row 3 (column collapse): mean(row) for raw bias strip | mean(row) for trimmed image
    row 4 (row collapse):    mean(col) for raw bias strip | mean(col) for trimmed image

Output: /ls4/zeros_overscan/NW_B_vs_NE_G_diagnostic.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from models.instrument import Instrument


INFILE = "/ls4/zeros/20260422045838d_00000.fits.fz"
OUTPNG = "/ls4/zeros_overscan/NW_B_vs_NE_G_diagnostic.png"


def find_hdu(hdul, ccd_loc):
    for i in range(1, len(hdul)):
        if hdul[i].header.get("CCD_LOC", "").strip() == ccd_loc:
            return i
    raise KeyError(f"CCD_LOC={ccd_loc} not found in {INFILE}")


def chip_panels(fig, gs_col, ls4, hdul, ccd_loc, label):
    """Plot 4 rows of diagnostics for one chip into column gs_col of fig."""
    ihdu = find_hdu(hdul, ccd_loc)
    hdr = hdul[ihdu].header
    raw = hdul[ihdu].data.astype(np.float32)
    secs = ls4.overscan_and_data_sections(hdr)
    sec = secs[0]   # LS4Cam returns one section per chip in single-amp mode

    bs = sec["biassec"]
    ds = sec["datasec"]
    trimmed = ls4.overscan_and_trim(hdr, raw)

    bias_strip = raw[bs["y0"]:bs["y1"], bs["x0"]:bs["x1"]]

    # --- row 0: full raw with overlay ----------------------------------------
    ax = fig.add_subplot(gs_col[0])
    z = ZScaleInterval()
    vmin, vmax = z.get_limits(raw)
    ax.imshow(raw, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")
    # Bias section in red, data sections in cyan
    ax.add_patch(Rectangle((bs["x0"], bs["y0"]),
                           bs["x1"] - bs["x0"], bs["y1"] - bs["y0"],
                           ec="red", fc="none", lw=1.0, label="biassec"))
    ax.add_patch(Rectangle((ds["x0"], ds["y0"]),
                           ds["x1"] - ds["x0"], ds["y1"] - ds["y0"],
                           ec="cyan", fc="none", lw=1.0, label="datasec"))
    ax.set_title(f"{label} {ccd_loc}: raw {raw.shape} (zscale)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=7)

    # --- row 1: trimmed image + histogram ------------------------------------
    ax = fig.add_subplot(gs_col[1])
    vmin, vmax = z.get_limits(trimmed)
    ax.imshow(trimmed, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.set_title(f"trimmed {trimmed.shape}  median={np.median(trimmed):+.2f}  "
                 f"std={trimmed.std():.2f}")

    ax = fig.add_subplot(gs_col[2])
    flat = trimmed.ravel()
    # Clip extreme outliers for the hist so you can see the bulk
    lo, hi = np.percentile(flat, [0.5, 99.5])
    ax.hist(flat[(flat >= lo) & (flat <= hi)], bins=200, color="C0", alpha=0.85)
    ax.set_title(f"pixel histogram (0.5–99.5 pct: {lo:.0f}…{hi:.0f})")
    ax.set_xlabel("ADU after overscan subtraction")
    ax.set_yscale("log")

    # --- row 2: column collapse (mean of each row) ---------------------------
    ax = fig.add_subplot(gs_col[3])
    ax.plot(bias_strip.mean(axis=1), color="C3", lw=0.5, label="raw biassec mean(row)")
    ax.plot(trimmed.mean(axis=1), color="C0", lw=0.5, label="trimmed mean(row)")
    ax.set_xlabel("y (row)"); ax.set_ylabel("ADU")
    ax.set_title("row-direction profile")
    ax.legend(loc="upper right", fontsize=7)

    # --- row 3: row collapse (mean of each col) ------------------------------
    ax = fig.add_subplot(gs_col[4])
    ax.plot(trimmed.mean(axis=0), color="C0", lw=0.5)
    ax.set_xlabel("x (col)"); ax.set_ylabel("mean ADU")
    ax.set_title("column-direction profile of trimmed (look for bad columns)")


def main():
    if not os.path.isfile(INFILE):
        print(f"Missing {INFILE}", file=sys.stderr); sys.exit(1)

    ls4 = Instrument.get_instrument_instance("LS4Cam")

    with fits.open(INFILE) as hdul:
        fig = plt.figure(figsize=(16, 18), constrained_layout=True)
        outer = fig.add_gridspec(1, 2, wspace=0.05)
        col_left = outer[0].subgridspec(5, 1, height_ratios=[2.4, 2.4, 1.0, 1.0, 1.0])
        col_right = outer[1].subgridspec(5, 1, height_ratios=[2.4, 2.4, 1.0, 1.0, 1.0])
        chip_panels(fig, col_left, ls4, hdul, "NE_G", "GOOD")
        chip_panels(fig, col_right, ls4, hdul, "NW_B", "SUSPECT")
        fig.suptitle(f"Overscan diagnostic — {os.path.basename(INFILE)}", fontsize=14)
        fig.savefig(OUTPNG, dpi=110)
        print(f"Wrote {OUTPNG}")


if __name__ == "__main__":
    main()
