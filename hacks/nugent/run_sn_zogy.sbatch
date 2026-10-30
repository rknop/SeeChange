#!/bin/bash
#SBATCH --account=m4616
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --license=cfs,scratch
#SBATCH --job-name=sn_zogy
#SBATCH --output=/pscratch/sd/n/nugent/ls4/code/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/n/nugent/ls4/code/logs/%x_%j.err
# ZOGY light curve of one SN with sn_zogy_lc.py: prep -> ref -> diff -> lc.
# WORK must hold coverage.csv (dump_coverage.py).  NAME uses _ for spaces.
#   sbatch --export=ALL,WORK=/pscratch/sd/n/nugent/ls4/sne/sn2026nmw_g,BAND=g,\
#          RA=191.6814,DEC=-40.025261,REFBEFORE=61160,DISC=61184.397,NAME=SN_2026nmw \
#          run_sn_zogy.sbatch
set -euo pipefail
WORK="${WORK:?set WORK}"; BAND="${BAND:-g}"
RA="${RA:?set RA}"; DEC="${DEC:?set DEC}"
REFBEFORE="${REFBEFORE:?set REFBEFORE (MJD)}"; DISC="${DISC:?set DISC (MJD)}"
NAME="${NAME:-SN}"; NAME="${NAME//_/ }"
STAGES="${STAGES:-prep ref diff lc}"

source /pscratch/sd/n/nugent/zp_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
S=/pscratch/sd/n/nugent/ls4/code/sn_zogy_lc.py
cd "$WORK"
for st in $STAGES; do
  echo "=== $st $(date +%T)"
  case $st in
    prep) python "$S" prep --workdir . --coverage coverage.csv --band "$BAND" \
                 --ra "$RA" --dec "$DEC" --nproc 32 ;;
    ref)  python "$S" ref  --workdir . --ref-before "$REFBEFORE" ;;
    diff) python "$S" diff --workdir . --nproc 32 ;;
    lc)   python "$S" lc   --workdir . --disc-mjd "$DISC" --name "$NAME" --band "$BAND" ;;
  esac
done
echo "=== done $(date +%T)"
