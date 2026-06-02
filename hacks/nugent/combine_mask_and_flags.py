#!/usr/bin/env python3
"""Combine an LS4 per-chip bad-pixel mask with a per-image flags FITS.

Inputs
------
chip_mask
    Static per-chip bad-pixel mask (e.g. ``mask_SE_C.fits.fz``) produced
    from biases by ``hacks/make_superbias.py``.  Stored as int16 with
    1 = bad pixel, 0 = good pixel.

flags
    Per-image flags FITS (``*.flags.fits.fz``) produced upstream of the
    pipeline.  Stored as uint16 using the bit conventions in
    ``models/enums_and_bitflags.py:flag_image_bits``::

        bit 0 = bad pixel
        bit 1 = zero weight
        bit 2 = saturated
        bit 3 = out of bounds

Output
------
``<basename>.mask.fits.fz`` (or ``-o OUTPUT``).  uint16, Rice-compressed.
The chip mask is promoted to bit 0 ("bad pixel") before being bitwise
OR'd with the per-image flags.

Aborts with non-zero exit code if shapes do not match (we do not try
to project / align between mask grids).
"""

import argparse
import datetime
import pathlib
import sys

import numpy as np
from astropy.io import fits


CODE_VERSION = "combine_mask_and_flags 0.1"
COMPRESSION_TYPE = "RICE_1"
BIT_BAD_PIXEL = 1 << 0   # 1 — matches flag_image_bits 'bad pixel'

# Header keys that CompImageHDU manages itself; copying these from the
# source flags header would produce duplicates in the output.
_STRUCTURAL_KEYS = (
    "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
    "PCOUNT", "GCOUNT", "BSCALE", "BZERO", "ZIMAGE", "ZTILE1", "ZTILE2",
    "ZNAME1", "ZVAL1", "ZNAME2", "ZVAL2", "ZPCOUNT", "ZGCOUNT",
    "ZQUANTIZ", "ZDITHER0", "ZSIMPLE", "ZBITPIX", "ZNAXIS",
    "ZNAXIS1", "ZNAXIS2", "ZBLANK", "ZHECKSUM", "ZDATASUM",
    "TFIELDS", "EXTNAME",
)


def derive_output_path(flags_path: pathlib.Path) -> pathlib.Path:
    """Replace the trailing .flags.fits[.fz] of a path with .mask.fits.fz."""
    name = flags_path.name
    for suf in (".flags.fits.fz", ".flags.fits"):
        if name.endswith(suf):
            return flags_path.with_name(name[:-len(suf)] + ".mask.fits.fz")
    raise SystemExit(
        f"Cannot derive output path from {flags_path} (no .flags.fits[.fz] suffix). "
        f"Pass --output explicitly."
    )


def _read_image_hdu(path):
    """Return (data, header) from the first 2-D image HDU in path."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                return np.asarray(hdu.data), hdu.header.copy()
    raise SystemExit(f"No 2-D image found in {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("chip_mask", help="Per-chip bad-pixel mask (int16; 1=bad).")
    ap.add_argument("flags", help="Per-image flags FITS (uint16).")
    ap.add_argument(
        "-o", "--output", default=None,
        help="Output path. Default: replace .flags.fits[.fz] with .mask.fits.fz "
             "next to the input flags file.",
    )
    args = ap.parse_args()

    flags_path = pathlib.Path(args.flags)
    out_path = (pathlib.Path(args.output) if args.output is not None
                else derive_output_path(flags_path))

    mask_data, _ = _read_image_hdu(args.chip_mask)
    flag_data, flag_hdr = _read_image_hdu(args.flags)

    if mask_data.shape != flag_data.shape:
        raise SystemExit(
            f"Shape mismatch: mask {mask_data.shape} vs flags {flag_data.shape}. "
            f"This script does not align grids; aborting."
        )

    # Promote the chip mask (1 = bad) to uint16 with bit 0 set, then OR
    # with the per-image flags array.
    mask_uint16 = np.where(mask_data != 0, BIT_BAD_PIXEL, 0).astype(np.uint16)
    combined = (mask_uint16 | flag_data.astype(np.uint16))

    n_bad = int((mask_uint16 != 0).sum())
    n_flag = int((flag_data != 0).sum())
    n_comb = int((combined != 0).sum())

    # ---- write out ---------------------------------------------------------
    primary = fits.PrimaryHDU()
    comp = fits.CompImageHDU(data=combined, compression_type=COMPRESSION_TYPE)

    # Carry over interesting keys from the input flags header (skipping
    # structural keys CompImageHDU manages itself), then add provenance.
    for k in flag_hdr:
        if not k or k in _STRUCTURAL_KEYS or k.startswith(("TFORM", "TTYPE", "TUNIT", "ZNAME", "ZVAL")):
            continue
        try:
            comp.header[k] = (flag_hdr[k], flag_hdr.comments[k])
        except Exception:
            # Some HISTORY/COMMENT cards or unusual entries can't be re-set
            # via subscript; ignore them silently.
            pass

    comp.header["MASKSRC"] = (pathlib.Path(args.chip_mask).name,
                              "Per-chip bad-pixel mask source")
    comp.header["FLAGSRC"] = (flags_path.name,
                              "Per-image flags source")
    comp.header["MASKVER"] = (CODE_VERSION, "Mask combiner version")
    comp.header["MASKDATE"] = (datetime.datetime.now(datetime.UTC)
                                                  .isoformat(timespec="seconds"),
                               "UTC time mask was combined")
    comp.header["BITDEFN"] = ("0=bad,1=zw,2=sat,3=oob",
                              "flag_image_bits convention")
    comp.header["NBADPIX"] = (n_bad, "Pixels flagged by chip mask")
    comp.header["NFLAGPIX"] = (n_flag, "Pixels flagged by per-image flags")
    comp.header["NCOMBPIX"] = (n_comb, "Pixels flagged in combined mask")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, comp]).writeto(out_path, overwrite=True)

    pct = 100.0 * n_comb / combined.size
    print(
        f"wrote {out_path}",
        f"  shape={combined.shape}, dtype={combined.dtype}",
        f"  chip-mask flagged: {n_bad:,}",
        f"  per-image flagged: {n_flag:,}",
        f"  combined flagged:  {n_comb:,} ({pct:.4f}%)",
        sep="\n",
    )


if __name__ == "__main__":
    sys.exit(main())
