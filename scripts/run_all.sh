#!/usr/bin/env bash
# Full experiment tuned for: Intel i5-10400F (6C/12T), 16 GB RAM, Linux.
# ONE FRESH JVM PER CONFIGURATION, so peak RSS (VmHWM, monotone per process) belongs to one configuration.
# Appends to results/raw_measurements.csv. Expect roughly 1-2 hours; run it unattended:
#     nohup bash scripts/run_all.sh > results_log.txt 2>&1 &
#
# BEFORE RUNNING (maximum, stable performance):
#   sudo cpupower frequency-set -g performance      # fixed high clock, no ramp-up noise
#   plug in the power; close browser/IDE/other heavy programs; do not use the PC while it runs.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- JVM: fixed heap (no resizing), low-noise GC. No AlwaysPreTouch: it would make peak RSS ~= heap size for every run. Part of the experiment: report them.
JVM="-Xms10g -Xmx10g -XX:+UseParallelGC"
# Optional: pin the single-threaded kernels to one core to avoid migrations, e.g. PIN="taskset -c 2"
PIN=""
CP="-cp out"

REPS=5; WARMUP=2; SEED=42
MAXSEC=240          # budget for ONE multiplication (s)
MEMMB=8192          # estimated memory budget per configuration (MB); machine has 16 GB
IJK_MAX=2500        # ijk_baseline is skipped above this n (it would exceed the time budget; reported as a limit)

DENSE_SIZES="100 200 400 800 1200 1600 2000 2500 3000 4000"
BLOCKS="32 64 128"
SPARSE_SIZES="1000 2000 4000"
SPARSITIES="90 95 99 99.9"

COMMON="--reps $REPS --warmup $WARMUP --seed $SEED --max-seconds $MAXSEC --mem-budget-mb $MEMMB --out-dir results"
mkdir -p results
run() { $PIN java $JVM $CP bigdata.matmul.Main "$@" $COMMON; }

# Part A: one JVM per (size, method[, block size]).
for n in $DENSE_SIZES; do
  if [ "$n" -le "$IJK_MAX" ]; then
    run dense --dense-sizes $n --dense-methods ijk_baseline
  fi
  run dense --dense-sizes $n --dense-methods ikj_reorder
  for bs in $BLOCKS; do
    run dense --dense-sizes $n --dense-methods blocked_ikj --block-sizes $bs
  done
done

# Part B: CSR kernel in its own JVM; dense comparators in separate JVMs (same logical inputs: same seeds).
for n in $SPARSE_SIZES; do
  for sp in $SPARSITIES; do
    run sparse --sparse-sizes $n --sparsity $sp --sparse-dense-methods none
    run sparse --sparse-sizes $n --sparsity $sp --no-csr --sparse-dense-methods ikj_reorder
    if [ "$n" -le "$IJK_MAX" ]; then
      run sparse --sparse-sizes $n --sparsity $sp --no-csr --sparse-dense-methods ijk_baseline
    fi
  done
done
echo "Done. Now run: python3 scripts/analyze.py results"
