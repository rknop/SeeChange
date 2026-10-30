#!/bin/bash
#SBATCH --account=m4616
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --license=cfs,scratch
#SBATCH --job-name=sn_zogy_cut
#SBATCH --output=/pscratch/sd/n/nugent/ls4/code/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/n/nugent/ls4/code/logs/%x_%j.err
# ZOGY light curve of one SN from SeeChange cutouts vs a DECam reference:
# sn_zogy_cutouts.py prep -> ref -> diff, then sn_zogy_lc.py lc; STAGES=smp adds
# scene-modelling photometry (sn_smp.py, needs the diff outputs).
# WORK holds ls4/ (cutouts) and decam/*/ (CP ooi/oow/ood).  NAME uses _ for spaces.
#   sbatch --export=ALL,WORK=/pscratch/sd/n/nugent/ls4/sne/sn2026nmw_new,BAND=g,\
#          RA=191.681309,DEC=-40.025125,DISC=61184.397,NULLBEFORE=61183.4,NAME=SN_2026nmw \
#          run_sn_zogy_cutouts.sbatch
set -euo pipefail
WORK="${WORK:?set WORK}"; BAND="${BAND:-g}"
RA="${RA:?set RA}"; DEC="${DEC:?set DEC}"
DISC="${DISC:?set DISC (MJD)}"; NULLBEFORE="${NULLBEFORE:?set NULLBEFORE (MJD)}"
NAME="${NAME:-SN}"; NAME="${NAME//_/ }"
STAGES="${STAGES:-prep ref diff lc}"
CUTOUTS="${CUTOUTS:-ls4/*_${BAND}_Sci_*.image.fits.fz}"
DECAM="${DECAM:-decam/*/*_ooi_${BAND}_*.fits.fz}"

source /pscratch/sd/n/nugent/zp_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=/pscratch/sd/n/nugent/ls4/code
cd "$WORK"
for st in $STAGES; do
  echo "=== $st $(date +%T)"
  case $st in
    prep) python "$D/sn_zogy_cutouts.py" prep --workdir . --cutouts "$CUTOUTS" --decam "$DECAM" \
                 --band "$BAND" --ra "$RA" --dec "$DEC" --nproc 32 ;;
    ref)  python "$D/sn_zogy_cutouts.py" ref  --workdir . ;;
    diff) python "$D/sn_zogy_cutouts.py" diff --workdir . --nproc 32 ;;
    lc)   python "$D/sn_zogy_lc.py" lc --workdir . --disc-mjd "$DISC" --null-before "$NULLBEFORE" \
                 --name "$NAME" --band "$BAND" ;;
    smp)  python "$D/sn_smp.py" --workdir . --cutouts "$CUTOUTS" --null-before "$NULLBEFORE" \
                 --disc-mjd "$DISC" --name "$NAME" ;;
  esac
done
echo "=== done $(date +%T)"
