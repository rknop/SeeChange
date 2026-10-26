# LS4 i-band glint / median co-add — session handoff

> **For how to operate Perlmutter itself** (ssh/rsync flags, SLURM patterns,
> job-waiting, SWarp/astrometry.net/SCAMP/STIFF gotchas, safety rules) see
> **[SKILLS.md](SKILLS.md)** in this directory. This file covers project state.

Working brief for continuing this work in a Claude session **on stargazer**.
Last updated 2026-06-02 (work done from a machine that can reach Anthropic;
stargazer's normal Claude was blocked — see "Connectivity" below).

## What the project is
LS4 (La Silla Schmidt Southern Survey) i-band SWarp co-adds of the field
**center (158.3, -27.8), radius 1.35 deg**, built on the SeeChange pipeline.
Goal of recent work: remove a bright-star **serial glint** — a horizontal
streak along chip-X (the LEFT|RIGHT amp / serial-register axis), distinct from
the vertical saturation bleed already handled. In the izg false-color the glint
shows up as **green horizontal streaks** (i = green channel).

## Where things live
- **NERSC Perlmutter** (heavy compute): `/pscratch/sd/n/nugent/ls4/`
  - `proc/std/<night>/<exp>/ls4_*_<chip>_<band>_Sci_STDALN.{image.w,weight,mask,flags}.fits.fz`
    — WCS-solved per-chip frames. Chips 2048(X) x 4096(Y); mask in HDU1 (uint16, saturation bit 0x04).
  - `stack/<band>-band[-<TAG>]/coadd.fits` — SWarp outputs; `inp/` staging + `list.txt`.
  - `code/` — runtime copies of the scripts below; `code/logs/` — sbatch logs.
  - env: `source /pscratch/sd/n/nugent/zp_env/bin/activate`
  - swarp: `/global/common/software/m3623/exaepi/astro/bin/swarp`
  - **stiff** (built from source): `/pscratch/sd/n/nugent/ls4/build/stiff-master/src/stiff` (v2.7.1)
- **stargazer / local**: `/Users/nugent/claude/ls4/` mirrors the layout; scripts in
  `SeeChange/hacks/nugent/`. Products land in `stack/`.

## Connectivity (IMPORTANT)
- LBL blocks `api.anthropic.com` (intercepted as `blocked.lbl.gov`), which is why
  this session ran off a different machine and pushed results to stargazer. If
  Claude on stargazer can't reach Anthropic, that's the cause (ticket filed with LBL IT).
- NERSC access from the work machine: `ssh perlmutter.nersc.gov` using an sshproxy
  cert (`~/.ssh/nersc`, ~24h). **Use `ssh -x -a -T`** (disable X11/agent-fwd) or the
  session hangs after the login banner. Re-run `sshproxy` (e.g. `~/bin/sshproxy.sh`)
  when the cert expires. Verify stargazer has its own valid NERSC cert.
- Both ends are macOS **openrsync**: do NOT use `--info=`/`--stats` rsync flags
  (unsupported); use plain `-av`.

## Status / what's done
1. Characterized the glint (glint_diag.py): horizontal serial streak at a bright
   star's core row(s), amp-local (doesn't cross x=1024), length grows with brightness.
2. Per-sky-point offender finding (find_glint_frames.py): at a glint point, the
   contaminating frames are a small minority and (for the 2 cases checked) ALL in the
   SW quadrant — the NE quadrant gives clean redundant coverage (i-band is imaged by
   2 diagonally-opposite quadrants, NE + SW). Verified removing them cleans the point.
   Built `coadd_i_noglint2` by excluding 14 offender frames (glint_offenders_all.txt).
3. "Why only certain nights?" — NOT moon (illum 14-75% across offender nights). The
   source is a FIXED star; the glint is **pointing-dependent** (only tiles that drop the
   star onto a SW chip with its streak crossing that point, observed on those nights).
4. Full per-frame glint sweep (per_frame_glint.py over all 2234 in-field frames):
   **921/2234 (41%) have a glint** (score>=100), on BOTH quadrants, many nights/stars
   — glints are widespread, no clean glint/no-glint split. Scores in stack/glint_scores_all.csv.
5. **Decision: MEDIAN co-adds** (reject glints statistically as a per-pixel minority,
   ~25% noisier than weighted mean but robust; no exclusion lists needed).
6. **Built 3 median i co-adds** (run_swarp_nights.sbatch, COMBINE_TYPE MEDIAN), all
   verified glint-free in preview:
   - `i-band-apr10_17/` 229 frames (nights 20260410-17)
   - `i-band-apr21_24/` 997 frames (20260421-24) — deepest
   - `i-band-may19_22/` 595 frames (20260519-22) — contains the glint nights, clean after median
   Pulled local + pushed to stargazer as `stack/coadd_i_<tag>.fits` (+ `.weight.fits`).

## Scripts (in SeeChange/hacks/nugent/, also on NERSC code/)
- `stack_field.py` — stage frames for SWarp (effective weight = weight folded with
  mask==0 + grown Y-bleed). Flags: `--band g|i|z`, `--center RA DEC`, `--radius`,
  `--seeing-cut`, `--max-frames N` (0=all), `--nights START END`, `--exclude FILE`,
  `--band-dir DIR`. Built-in serial-glint mask is left `GLINT_MODE="off"` (its
  saturation-count row picker over-masked; superseded by the median approach).
- `run_swarp_nights.sbatch` — **the main tool now.** Median (default) co-add for one
  band over a night range into its own dir:
  `sbatch --export=ALL,BAND=i,N0=20260410,N1=20260417,TAG=apr10_17,COMBINE=MEDIAN run_swarp_nights.sbatch`
  (COMBINE=MEDIAN|WEIGHTED|CLIPPED; MAXFRAMES optional).
- `run_swarp_band.sbatch` — original full-field single co-add (WEIGHTED; supports EXCLUDE).
- `find_glint_frames.py --pos RA DEC` — list frames whose flux at a sky point is an
  outlier (glint offenders) -> writes a list usable with `--exclude`.
- `per_frame_glint.py` — per-frame glint score; `--all` sweep, `--dumpall CSV`, `--thresh`.
- `count_iband_vs_time.py` — per-night frame histogram (stack/iband_per_night.png).
- `find_glint_source.py`, `glint_diag.py`, `glint_streaks_in_coadd.py`,
  `verify_glint_exclusion.py`, `compare_coadds_at_P.py`, `make_glint_compare.py`,
  `make_i_preview.py` — diagnostics/QA (see headers).
- false-color: `stiff z-band/coadd.fits i-band/coadd.fits g-band/coadd.fits -OUTFILE_NAME out.tif`
  -> izg = **R=z, G=i, B=g**.

## Natural next steps (not yet done)
- Build matching **g- and z-band medians** for the same 3 night ranges, then make a
  per-epoch izg false-color (R=z,G=i,B=g) for each.
- The medians show faint **vertical banding** (per-chip background steps) — add sky/
  background matching (SWarp BACK_ or a per-frame sky subtraction) if it matters.
- Consider a single deep full-survey median, or weighted-mean-with-glint-mask if depth
  matters more than the median's noise penalty.
- Tune STIFF stretch (MIN_LEVEL/MAX_LEVEL/GAMMA/COLOUR_SAT) to match the older
  make_lupton look (defaults render dark).
