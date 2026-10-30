#!/usr/bin/env bash
# Small smoke run (about a minute). Output: results_quick/
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf results_quick
java -Xms2g -Xmx2g -cp out bigdata.matmul.Main all \
  --dense-sizes 100,200,400,800 --block-sizes 32,64,128 \
  --sparse-sizes 500,1000 --sparsity 90,99 \
  --reps 5 --warmup 2 --out-dir results_quick
