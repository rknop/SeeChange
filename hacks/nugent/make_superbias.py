#!/usr/bin/env python3
"""Build per-chip superbias, NMAD, and bad-pixel-mask FITS for LS4Cam.

Two input layouts are supported via ``--input-mode``:

  * ``multi-ext`` (default) — each input file is a multi-extension FITS
    (one image HDU per chip, raw with bias section).  The script runs
    ``LS4Cam.overscan_and_trim`` on each chip extension before stacking.

  * ``per-chip`` — each input file holds a single chip's data, already
    overscan-subtracted and trimmed.  Files are grouped by the ``CCD_LOC``
    header keyword (which can live in the primary or any extension).
    Different chips may have different numbers of input frames.

For each chip the script writes:

  * ``superbias_<CCD>.fits.fz``       — per-pixel **median** across all bias
    frames.  32-bit float, Rice-compressed.
    Header ``MED_BIAS`` = median of this image.

  * ``superbias_nmad_<CCD>.fits.fz``  — per-pixel **NMAD** (= 1.4826 × MAD)
    across all bias frames.  32-bit float, Rice-compressed.
    Header ``MB_NMAD`` = median of this image.

  * ``superbias_mask_<CCD>.fits.fz``  — bad-pixel mask.  16-bit int.
    ``mask = 1`` where ``NMAD > 3 * 1.4826 * median(NMAD)`` else ``0``.

Each chip's stack is processed independently to keep peak memory bounded
(~1 GB per chip for ~30 frames × 4096 × 2048 × float32).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from astropy.io import fits


DEFAULT_INDIR = "/ls4/zeros"
DEFAULT_OUTDIR = "/ls4/superbias"
NMAD_K = 1.4826                  # MAD → Gaussian-σ equivalent
MASK_THRESHOLD_NSIGMA = 3.0      # mask pixels with NMAD above this × NMAD_K × median(NMAD)
QUANTIZE_LEVEL = 16              # Rice/quantize: σ/16 quantum (lossless to ~0.06σ)

INPUT_MODE_MULTI_EXT = "multi-ext"
INPUT_MODE_PER_CHIP = "per-chip"


# ---------------------------------------------------------------------------
# Generic FITS helpers
# ---------------------------------------------------------------------------

def find_ccd_loc_in_file(path):
    """Return ``CCD_LOC`` from the first HDU header that has it, or ``None``."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            v = hdu.header.get("CCD_LOC", "")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def find_2d_image_in_file(path):
    """Return the first 2-D ``numpy`` image array found in ``path``."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ValueError(f"No 2-D image data found in {path}")


# ---------------------------------------------------------------------------
# Chip-stack builders, one per input mode
# ---------------------------------------------------------------------------

def discover_chips_multi_ext(infile):
    """Return ``[(chip_loc, hdu_index), ...]`` from one multi-ext input file."""
    chips = []
    with fits.open(infile) as hdul:
        for i in range(1, len(hdul)):
            loc = hdul[i].header.get("CCD_LOC", "")
            if isinstance(loc, str) and loc.strip():
                chips.append((loc.strip(), i))
    return chips


def build_chip_stack_multi_ext(infiles, chip_loc, hdu_idx, ls4, log):
    """Load + overscan-and-trim one chip across all multi-ext input files."""
    frames = []
    for f in infiles:
        with fits.open(f) as hdul:
            hdr = hdul[hdu_idx].header
            actual = hdr.get("CCD_LOC", "").strip() if isinstance(hdr.get("CCD_LOC", ""), str) else ""
            if actual != chip_loc:
                # HDU layout drifted between files — search by header
                for j in range(1, len(hdul)):
                    if hdul[j].header.get("CCD_LOC", "").strip() == chip_loc:
                        hdr = hdul[j].header
                        raw = hdul[j].data
                        break
                else:
                    log(f"   WARN: {os.path.basename(f)} has no CCD_LOC={chip_loc}, skipping")
                    continue
            else:
                raw = hdul[hdu_idx].data
            frames.append(ls4.overscan_and_trim(hdr, raw))
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def discover_chips_per_chip(infiles, log):
    """Group flat per-chip files by their ``CCD_LOC`` header.

    Returns ``{chip_loc: [paths...]}``; files without ``CCD_LOC`` are skipped
    with a warning.
    """
    chip_to_files = {}
    skipped = 0
    for f in infiles:
        loc = find_ccd_loc_in_file(f)
        if loc is None:
            log(f"   WARN: no CCD_LOC in {os.path.basename(f)}, skipping")
            skipped += 1
            continue
        chip_to_files.setdefault(loc, []).append(f)
    if skipped:
        log(f"   ({skipped} input files were skipped for missing CCD_LOC)")
    return chip_to_files


def build_chip_stack_per_chip(files_for_chip, log):
    """Load + stack already-trimmed data arrays from per-chip input files."""
    frames = []
    shapes = set()
    for f in files_for_chip:
        arr = find_2d_image_in_file(f)
        shapes.add(arr.shape)
        frames.append(arr)
    if len(shapes) > 1:
        raise ValueError(f"Inconsistent chip shapes across input files: {shapes}")
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_float_fz(path, data, header_kw, header_val, extra_keys):
    """Write a float32 image to a Rice-compressed FITS (.fits.fz)."""
    primary = fits.PrimaryHDU()
    comp = fits.CompImageHDU(
        data=data.astype(np.float32, copy=False),
        compression_type="RICE_1",
        quantize_level=QUANTIZE_LEVEL,
    )
    comp.header[header_kw] = (header_val, "Median of this image")
    for k, (v, c) in extra_keys.items():
        comp.header[k] = (v, c)
    fits.HDUList([primary, comp]).writeto(path, overwrite=True)


def write_int16_fz(path, data, extra_keys):
    """Write an int16 mask to a Rice-compressed FITS (.fits.fz)."""
    primary = fits.PrimaryHDU()
    comp = fits.CompImageHDU(
        data=data.astype(np.int16, copy=False),
        compression_type="RICE_1",
    )
    for k, (v, c) in extra_keys.items():
        comp.header[k] = (v, c)
    fits.HDUList([primary, comp]).writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# Core: superbias / NMAD / mask from a single chip's stack
# ---------------------------------------------------------------------------

def process_chip_stack(stack, chip, outdir, n_input, log):
    """Compute superbias + NMAD + mask for one chip and write three FITS.

    Returns ``(med_bias, med_nmad, n_masked, pct_masked)``.
    """
    if stack.shape[0] < 2:
        raise ValueError(f"{chip}: need >=2 frames to compute NMAD, got {stack.shape[0]}")

    log(f"   stack shape={stack.shape} dtype={stack.dtype}")

    # Per-pixel median = superbias
    superbias = np.median(stack, axis=0).astype(np.float32, copy=False)
    med_bias = float(np.median(superbias))

    # Per-pixel NMAD = K * median(|x - median(x)|)
    mad = np.median(np.abs(stack - superbias[None, :, :]), axis=0)
    nmad = (NMAD_K * mad).astype(np.float32, copy=False)
    med_nmad = float(np.median(nmad))

    # Bad-pixel mask: NMAD above MASK_THRESHOLD_NSIGMA × NMAD_K × median(NMAD)
    threshold = MASK_THRESHOLD_NSIGMA * NMAD_K * med_nmad
    mask = (nmad > threshold).astype(np.int16)
    n_masked = int(mask.sum())
    pct_masked = 100.0 * n_masked / mask.size

    # Free the big stack before writing — no longer needed
    del stack

    common_meta = {
        "CCD_LOC":  (chip, "Sensor section identifier"),
        "NSTACK":   (n_input, "Number of bias frames stacked"),
        "MASKTHR":  (threshold, "NMAD mask threshold (ADU)"),
        "MASKNSIG": (MASK_THRESHOLD_NSIGMA, "n-sigma factor used for mask"),
    }

    path_bias = os.path.join(outdir, f"superbias_{chip}.fits.fz")
    write_float_fz(path_bias, superbias, "MED_BIAS", med_bias, common_meta)
    log(f"   wrote {os.path.basename(path_bias)}  MED_BIAS={med_bias:+.3f}")

    path_nmad = os.path.join(outdir, f"superbias_nmad_{chip}.fits.fz")
    write_float_fz(path_nmad, nmad, "MB_NMAD", med_nmad, common_meta)
    log(f"   wrote {os.path.basename(path_nmad)}  MB_NMAD={med_nmad:.3f}")

    path_mask = os.path.join(outdir, f"superbias_mask_{chip}.fits.fz")
    mask_meta = dict(common_meta,
                     NMASKED=(n_masked, "Number of masked pixels"),
                     PCTMASK=(pct_masked, "Fraction of pixels masked (%)"))
    write_int16_fz(path_mask, mask, mask_meta)
    log(f"   wrote {os.path.basename(path_mask)}  masked={n_masked} ({pct_masked:.3f}%)")

    return med_bias, med_nmad, n_masked, pct_masked


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=DEFAULT_INDIR,
                    help=f"Directory of input FITS files (default {DEFAULT_INDIR})")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help=f"Output directory (default {DEFAULT_OUTDIR})")
    ap.add_argument("--input-mode", choices=[INPUT_MODE_MULTI_EXT, INPUT_MODE_PER_CHIP],
                    default=INPUT_MODE_MULTI_EXT,
                    help=("multi-ext: each input is a multi-ext raw exposure with bias "
                          "section; script runs overscan_and_trim per chip.  "
                          "per-chip: each input file is one chip, already trimmed; "
                          "files are grouped by CCD_LOC header. (default: multi-ext)"))
    ap.add_argument("--chips", nargs="+", default=None,
                    help="Only process these CCD_LOC values (default: all)")
    args = ap.parse_args()

    log = lambda msg: print(msg, flush=True)

    os.makedirs(args.outdir, exist_ok=True)

    # *.fits and *.fits.fz both match *.fits*
    infiles = sorted(glob.glob(os.path.join(args.indir, "*.fits*")))
    if not infiles:
        log(f"No *.fits / *.fits.fz files in {args.indir}")
        sys.exit(1)
    log(f"Found {len(infiles)} input files in {args.indir} (mode={args.input_mode})")

    # ---------- discover chips and build the work plan ----------
    if args.input_mode == INPUT_MODE_MULTI_EXT:
        # Lazy-import the SeeChange instrument code only when we need it
        from models.instrument import Instrument
        ls4 = Instrument.get_instrument_instance("LS4Cam")

        chips = discover_chips_multi_ext(infiles[0])
        chip_locs = sorted({loc for loc, _ in chips})
        chip_hdu = {loc: idx for loc, idx in chips}
        log(f"Discovered {len(chip_locs)} chips from {os.path.basename(infiles[0])}: {chip_locs}")

        # Same N for every chip in this mode
        chip_to_n = {c: len(infiles) for c in chip_locs}

        def build_stack(chip):
            return build_chip_stack_multi_ext(infiles, chip, chip_hdu[chip], ls4, log)

    else:  # INPUT_MODE_PER_CHIP
        chip_to_files = discover_chips_per_chip(infiles, log)
        chip_locs = sorted(chip_to_files.keys())
        log(f"Discovered {len(chip_locs)} chips by CCD_LOC: {chip_locs}")
        for c in chip_locs:
            log(f"   {c}: {len(chip_to_files[c])} files")

        chip_to_n = {c: len(chip_to_files[c]) for c in chip_locs}

        def build_stack(chip):
            return build_chip_stack_per_chip(chip_to_files[chip], log)

    if args.chips:
        before = len(chip_locs)
        chip_locs = [c for c in chip_locs if c in args.chips]
        log(f"Filtered from {before} chips to {len(chip_locs)}: {chip_locs}")

    # ---------- run ----------
    t_total = time.time()
    for ichip, chip in enumerate(chip_locs, start=1):
        t0 = time.time()
        log(f"\n[{ichip}/{len(chip_locs)}] {chip}")
        try:
            stack = build_stack(chip)
            process_chip_stack(stack, chip, args.outdir, chip_to_n[chip], log)
        except Exception as exc:
            log(f"   FAILED: {type(exc).__name__}: {exc}")
            continue
        log(f"   [{time.time() - t0:.1f}s]")

    log(f"\nAll done in {time.time() - t_total:.1f}s.")


if __name__ == "__main__":
    main()
