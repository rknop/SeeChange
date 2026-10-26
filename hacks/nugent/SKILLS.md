# Running on NERSC Perlmutter — operational notes

Hard-won specifics from the LS4 / SeeChange work. Point an agent at this file
before it touches Perlmutter. Companion to `HANDOFF.md` (project state) — this
file is about *how to operate the machine*, not what the science is.

Everything below was verified in practice; the "why" notes are failures we
actually hit.

---

## 1. Connecting

```bash
ssh perlmutter.nersc.gov '<command>'        # command mode: works fine as-is
ssh -x -a -T perlmutter.nersc.gov           # interactive: disable X11 + agent fwd
```

- **Command mode** (`ssh host 'cmd'`) is what an agent should use almost always,
  and needs no special flags.
- For an **interactive** shell add `-x -a -T`; without them the session can hang
  after the login banner.
- Auth is an **sshproxy certificate** (`~/.ssh/nersc`, ~24 h). When it expires
  every call fails at once — re-run `sshproxy` (e.g. `~/bin/sshproxy.sh`). If
  ssh suddenly stops working mid-session, check this first.

### File transfer

```bash
rsync -av host:/remote/path .               # plain -av only
scp file host:/pscratch/sd/n/nugent/ls4/code/
```

**Do not use `--info=` or `--stats`.** macOS ships **openrsync**, which does not
support them; the transfer dies with a usage error.

---

## 2. Where things live

```
/pscratch/sd/n/nugent/ls4/
    proc/<night>/            raw exposures (one MEF per exposure, 32 chip HDUs)
    proc/std/<night>/<exp>/  processed per-chip frames (*_Sci_STDALN.*)
    stack/                   SWarp coadds, one dir per band/tag
    code/                    runtime copies of scripts  <- scp here
    code/logs/               sbatch stdout/stderr (%x_%j.out)
    astrometry/              self-built astrometry.net + indices + calib
    biases/ flats/ masks/    calibration masters
```

Shared/reference data (**read-only — never write here**):

```
/global/cfs/cdirs/cosmo/work/users/dstn/index-5200/   astrometry.net indices (Gaia EDR3)
/global/cfs/projectdirs/cosmo/data/gaia/dr3/healpix/  Gaia DR3 by healpix (nside=32 NESTED)
/global/common/software/m3623/exaepi/astro/bin/       swarp, sex, scamp
```

Local tools:

```
source /pscratch/sd/n/nugent/zp_env/bin/activate      python venv (astropy, numpy, sep, ...)
/pscratch/sd/n/nugent/stiff_build/install/bin/stiff   STIFF 2.7.1 false-colour
/pscratch/sd/n/nugent/ls4/astrometry/bin/solve-field  astrometry.net 0.97 (self-built)
```

The venv is **not** active by default — `source` it in every sbatch script and
every ssh command that runs python.

---

## 3. Login node vs compute node

**Login node is fine for:** scp/rsync, small python analysis, building software,
inspecting FITS headers, a handful of `solve-field` runs, STIFF renders (~3 s),
Gaia catalogue queries.

**Use SLURM for:** anything touching hundreds of frames, SWarp coadds,
ZOGY on full tiles, multi-hundred-core pools.

Rule of thumb: if it will run more than a couple of minutes or use more than a
few cores, submit it.

> **Shifter containers do not work on login nodes.** The
> `docker:astrometrynet/solver:0.98` image fails there with a libcurl-gnutls /
> nghttp2 symbol-lookup error, but the *same image runs fine inside sbatch* on
> compute nodes. If a container mysteriously fails, try it in a batch job before
> concluding the image is broken.

---

## 4. Submitting jobs

Standard header used by all our scripts:

```bash
#SBATCH --account=m4616
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --license=cfs,scratch          # required to touch CFS + scratch
#SBATCH --output=/pscratch/sd/n/nugent/ls4/code/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/n/nugent/ls4/code/logs/%x_%j.err
```

**qos=debug limits (verified): 30 min wall, 5 jobs submitted, 2 running.**
Fine for most of our work; use `regular` for anything longer.

Parameterise with `--export` rather than editing scripts:

```bash
sbatch --export=ALL,BAND=i,TAG=f2,CRA=158.6979,CDEC=-28.4871,SIZE=3600 run_swarp_field.sbatch
```

and read them with defaults inside: `TAG="${TAG:?set TAG}"`, `SIZE="${SIZE:-3600}"`.

**Prefer one serial job over several small ones.** The debug qos only runs 2 at
a time, and a job that loops over 3 bands internally (each < 5 min) is easier to
track than 3 separate submissions. This is an explicit preference of the user's.

### Waiting for a job to finish

The pattern that works reliably — remote-side wait loop, launched in the
background so the agent isn't blocked:

```bash
ssh perlmutter.nersc.gov "while squeue -h -j $JOBID 2>/dev/null | grep -q .; do sleep 30; done; \
  echo DONE; sacct -j $JOBID -X --format=State,Elapsed --noheader | head -1"
```
(run with `run_in_background: true`; a task notification fires on completion)

For a quick status check without waiting:

```bash
ssh perlmutter.nersc.gov "sacct -j $JOBID -X --format=State,Elapsed --noheader"
```

**Harness gotchas around this:**
- Long polling loops run *locally* have intermittently hit `ENOSPC` on the
  agent's tmpfs (`/tmp/claude-502`) even with hundreds of GB free. Keeping the
  loop on the remote side (as above) avoids it.
- **Never delete a background task's `.output` file while it is live** — doing so
  makes the task fail with ENOENT.

---

## 5. Tool-specific gotchas

### SWarp
- `COMBINE_BUFSIZE 2048` — **essential.** The default made a coadd balloon to
  ~40 GB and die in a double-free; 2048 keeps it near 2 GB.
- `NOPENFILES_MAX 131072` **and** `ulimit -n 131072` in the script — many input
  frames otherwise exhaust file descriptors.
- `MEM_MAX 16384`, `VMEM_MAX 31000`, and point `RESAMPLE_DIR`/`VMEM_DIR` at a
  job-local `inp/` directory.
- `HEADER_SUFFIX .head` makes SWarp read `<input_basename>.head` **from the
  input image's own directory**, not the cwd. Copy the scamp `.head` next to the
  input (and clean it up with a `trap`).
- `BACK_SIZE` strongly affects background flatness. For LS4 chips: 1028 leaves
  visible banding, 256 is 4.4× flatter, 128 another ~24% better with no galaxy
  over-subtraction on ~arcmin-scale objects.
- `COMBINE_TYPE MEDIAN` is the glint/outlier-robust choice (~25 % noisier than a
  weighted mean, but needs no exclusion lists).

### astrometry.net (building it yourself)
```bash
AP=/pscratch/sd/n/nugent/ls4/astrometry
# 1. cfitsio first
./configure --prefix=$AP && make -j8 && make install
# 2. then astrometry.net, pointing at it
export PKG_CONFIG_PATH=$AP/lib/pkgconfig:$PKG_CONFIG_PATH
make && make install INSTALL_DIR=$AP
```
- Builds cleanly on the login node with the default `gcc` (14, Cray native).
- **netpbm is not needed** if you pass `--fits-image` to `solve-field`; without
  that flag it shells out to `pnmfile` and dies.
- Release **0.97 needs `autoindex` in the config**, not the newer `indexset`.
- **Index sizes are counter-intuitive:** the *small-scale* index files are the
  huge ones. For a ~30–60′ field, scales 5204–5206 total 4.8 GB and solve
  everything; 5206 alone (0.7 GB) also worked. The full 5200 set is 326 GB and
  the 5000 series is 62 GB of Gaia-DR2 small scales — don't copy either.

### SExtractor / SCAMP
- SExtractor must emit `FITS_LDAC` for SCAMP to read it.
- SCAMP against a reference catalogue: `-ASTREF_CATALOG FILE
  -ASTREFCAT_NAME ref.ldac -PROJECTION_TYPE TPV -DISTORT_DEGREES 3`.
- Read residuals from the XML (`-WRITE_XML Y`), field
  `AstromSigma_Reference` — sub-0.1″ is a good solve.

### STIFF
- Channel order is **R G B**, so `stiff z.fits i.fits g.fits` gives an izg image.
- Set the black point with `-MIN_TYPE QUANTILE -MIN_LEVEL <q>`. Putting it
  *below* the sky makes the background take on a colour cast (we saw purple);
  putting it at the per-channel sky level renders the sky neutral, then lift the
  faint end with `-GAMMA`.

### LS4 frame and coadd headers: the SEEING / LMT_MG scale
- `zp_gaia_ls4cam.py` and `zp_coadd.py` converted sep's second-moment axes to
  FWHM with 2√(ln 2)·σ instead of 2√(2 ln 2)·σ, so SEEING came out **low by √2**
  and LMT_MG **0.376 mag too faint**. Fixed in `zp_gaia_ls4cam 0.2` /
  `zp_coadd 0.2`. ZPs were never affected — FWHM doesn't enter star selection
  or the fluxes.
- **The existing processed batch was deliberately left uncorrected.** Every
  proc/std frame from nights 2026-04-10 .. 06-07 carries
  `ZP_VER = 'zp_gaia_ls4cam 0.1'`, and coadds made before the fix have no
  `ZP_VER` at all. To put an old value on the true scale:
  **SEEING × √2, LMT_MG − 0.376**.
- The seeing/depth cuts already in the sbatch scripts (2.1 / 2.4 / 2.5″,
  lmt 20.5 …) were tuned on the old scale, so they stay consistent with the old
  headers. **Check `ZP_VER` before mixing old and new frames in one cut** —
  new-scale frames have √2-larger SEEING and would fail the old cuts far more
  often.
- SeeChange proper (`improc/`, `pipeline/`, `models/`) uses 2.35482 correctly,
  so Rob's `fwhm_estimate` in his catalogues is on the true scale (≈ 1.40× our
  old SEEING across shared exposures).

---

## 6. Python / astropy gotchas

- `Table.write("x.cat")` fails — astropy cannot infer that format. Use
  `.ecsv` with `format="ascii.ecsv"` (and read it back the same way).
- unWISE / many archives serve `.gz` FITS **without** `Content-Encoding: gzip`,
  so `requests` hands back raw gzip bytes — `gzip.decompress()` before
  `fits.open`.
- `WCS.all_world2pix` returns 0-d arrays; wrap in `float()` before using them in
  arithmetic that expects scalars (`round()` will raise otherwise).
- Raw LS4 headers carry **TELE-RA in hours** (multiply by 15); the processed
  frames use degrees. Guard with `if ra < 24: ra *= 15`.
- Frame-table columns are lowercase (`mjd`, `use`, `included`) even though FITS
  record access is case-insensitive — don't test membership with uppercase.

---

## 7. Writing scripts remotely

**Write files locally with the Write tool, then `scp`.** Do *not* pipe python
into ssh with a heredoc — quoting gets mangled (backslashes and nested quotes
are eaten) and you get syntax errors that look like logic bugs.

Keep local `SeeChange/hacks/nugent/` and remote `ls4/code/` in sync; scp after
every edit, since the sbatch scripts run the *remote* copy.

---

## 8. Safety rules

- **Never write to shared CFS project directories** (`/global/cfs/cdirs/cosmo/...`)
  without explicit consent. Read from them; write to `/pscratch/sd/n/nugent/`.
- Check `squeue -u nugent` before assuming a job is yours — the user runs
  unrelated jobs (`pros_*`, `zfix_*`) from other projects on the same account.
- Don't delete `*.output` / log files that a running process owns.
- Scratch is purged periodically; anything precious belongs on CFS or pulled
  local.
