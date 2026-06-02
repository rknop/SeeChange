#!/usr/bin/env python3
"""Apply LS4Cam overscan subtraction + trim to every bias FITS in /ls4/zeros.

Each input file is a single multi-extension FITS with one image HDU per
controller/chip (32 HDUs total for LS4Cam).  For every image HDU we call
``LS4Cam.overscan_and_trim(header, raw_data)`` and write the trimmed,
bias-subtracted result into a new FITS at /ls4/zeros_overscan/.

Per-HDU summary stats (median bias level inferred from the overscan
region, plus mean/median/std of the trimmed image) are written to
/ls4/zeros_overscan/overscan_summary.csv.
"""

import csv
import glob
import os
import sys
import time
import traceback

import numpy as np
from astropy.io import fits

from models.instrument import Instrument


INDIR = "/ls4/zeros"
OUTDIR = "/ls4/zeros_overscan"
SUMMARY_CSV = os.path.join(OUTDIR, "overscan_summary.csv")


def trimmed_stats(arr):
    """Robust-ish per-image stats for the bias-subtracted, trimmed data."""
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan"), float("nan"), float("nan")
    vals = arr[finite]
    return float(vals.mean()), float(np.median(vals)), float(vals.std())


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ls4 = Instrument.get_instrument_instance("LS4Cam")

    infiles = sorted(glob.glob(os.path.join(INDIR, "*.fits.fz")))
    if not infiles:
        print(f"No *.fits.fz files in {INDIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(infiles)} files from {INDIR} -> {OUTDIR}")

    rows = []
    n_files_ok = 0
    n_hdu_ok = 0
    n_hdu_fail = 0
    t0 = time.time()

    for ifile, infile in enumerate(infiles, start=1):
        base = os.path.basename(infile).replace(".fits.fz", "")
        outpath = os.path.join(OUTDIR, f"{base}_oscan.fits")
        print(f"[{ifile}/{len(infiles)}] {base} ...", flush=True)

        try:
            with fits.open(infile) as hdul:
                # Carry the primary header forward; data goes in extensions.
                out_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header)])

                for ihdu in range(1, len(hdul)):
                    hdr = hdul[ihdu].header
                    sec_id = hdr.get("CCD_LOC", f"HDU{ihdu}").strip()
                    try:
                        # Median bias level (raw, pre-subtraction) for QA
                        secs = ls4.overscan_and_data_sections(hdr)
                        raw = hdul[ihdu].data
                        med_bias_per_sec = []
                        for s in secs:
                            b = raw[s["biassec"]["y0"]:s["biassec"]["y1"],
                                    s["biassec"]["x0"]:s["biassec"]["x1"]]
                            med_bias_per_sec.append(float(np.median(b)))
                        med_bias = float(np.mean(med_bias_per_sec))

                        # The actual overscan + trim
                        trimmed = ls4.overscan_and_trim(hdr, raw)
                        mean, med, std = trimmed_stats(trimmed)

                        # Strip per-amp section keywords now that we've trimmed
                        clean_hdr = hdr.copy()
                        for k in ls4.overscan_trim_keywords_to_strip():
                            if k in clean_hdr:
                                del clean_hdr[k]
                        out_hdul.append(
                            fits.ImageHDU(data=trimmed, header=clean_hdr, name=sec_id)
                        )
                        rows.append({
                            "file": os.path.basename(infile),
                            "hdu": ihdu,
                            "ccd_loc": sec_id,
                            "raw_bias_median": med_bias,
                            "trimmed_mean": mean,
                            "trimmed_median": med,
                            "trimmed_std": std,
                            "trimmed_shape": f"{trimmed.shape[0]}x{trimmed.shape[1]}",
                            "status": "ok",
                        })
                        n_hdu_ok += 1
                    except Exception as exc:
                        n_hdu_fail += 1
                        rows.append({
                            "file": os.path.basename(infile),
                            "hdu": ihdu,
                            "ccd_loc": sec_id,
                            "raw_bias_median": "",
                            "trimmed_mean": "",
                            "trimmed_median": "",
                            "trimmed_std": "",
                            "trimmed_shape": "",
                            "status": f"{type(exc).__name__}: {exc}",
                        })
                        print(f"    HDU {ihdu} ({sec_id}) FAILED: {type(exc).__name__}: {exc}",
                              flush=True)

                out_hdul.writeto(outpath, overwrite=True)
                n_files_ok += 1
                print(f"    -> {outpath} ({len(out_hdul) - 1} extensions)", flush=True)

        except Exception:
            print(f"    !! file-level failure on {infile}:", flush=True)
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. {n_files_ok}/{len(infiles)} files written, "
          f"{n_hdu_ok} HDUs ok, {n_hdu_fail} HDUs failed.")

    if rows:
        with open(SUMMARY_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
