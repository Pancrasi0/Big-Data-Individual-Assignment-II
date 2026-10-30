# Assignment 2 — Optimised Matrix Multiplication and Sparse Matrices (Java)

BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027
Author: Pablo Martín Perdomo — Assignment 2

Plain Java 17+ (no external libraries) plus Python scripts for tables and plots. The study was run with JDK 21.0.12 and Python 3.14.

## What is implemented

**Part A — dense, sequential (n × n, `double`, flat row-major `double[]`)**

| Method | Change vs. baseline | Expected effect |
|---|---|---|
| `ijk_baseline` | textbook triple loop i-j-k | inner loop strides through B by `n` → many cache misses |
| `ikj_reorder` | loop order i-k-j, `a[i][k]` hoisted | inner loop is stride-1 over B and C → cache/prefetch friendly (same arithmetic) |
| `blocked_ikj` | tiling (block size `bs`) on top of i-k-j | tiles of B and C are reused while in cache; matters for large n |

**Part B — sparse**

* `csr_gustavson`: own CSR implementation (`CsrMatrix.java`) and Gustavson row-wise sparse×sparse product with a dense accumulator; output rows sorted. No library code is used.
* Compared against dense methods on the *same logical matrices* (same seeds).
* Phases measured separately: `multiply` (kernel only), `convert_dense_to_csr`, and `convert_plus_multiply` (end-to-end from dense inputs).
* Sparsity = % zero entries; density = % nonzeros. Realised `nnz(A)`, `nnz(B)`, `nnz(C)` are recorded.

## Layout

```
src/main/java/bigdata/matmul/
  DenseMultiply.java    dense kernels
  CsrMatrix.java        CSR storage + sparse product (own code)
  MatrixGenerator.java  seeded input generation
  Validator.java        self-tests, tolerance, Freivalds checks
  Experiments.java      Part A and Part B experiment drivers
  Harness.java          timer, memory probes, environment capture, CSV
  Config.java, Main.java
scripts/  build.sh  run_quick.sh  run_all.sh  analyze.py
results/  raw_measurements.csv  environment.txt  summary_*.csv  fig_*.png  (+ results_log.txt)
report/   report.tex, report.pdf, make_assets.py, figs/, tables/
```

## Build and run

Requirements: JDK 17+ (tested with 21.0.12), Python 3 with `pandas` and `matplotlib` (`pip install -r requirements.txt`).

```bash
bash scripts/build.sh                       # compiles into ./out
java -cp out bigdata.matmul.Main test       # correctness self-tests only (266 tests)
bash scripts/run_quick.sh                   # ~1 min smoke run -> results_quick/
bash scripts/run_all.sh                     # full study, one fresh JVM per configuration -> results/ (about 80 min on the study machine)
python3 scripts/analyze.py results          # summaries (CSV) + figures (PNG) in results/
```

`scripts/run_all.sh` contains the exact parameters of the study (JVM flags `-Xms10g -Xmx10g -XX:+UseParallelGC`, dense sizes 100–4000, block sizes 32/64/128, sparse sizes 1000/2000/4000, sparsity 90/95/99/99.9 %, 2 warm-ups, 5 repetitions, seed 42, 240 s per multiplication, 8192 MB estimated memory). A single invocation looks like this (all options: `java -cp out bigdata.matmul.Main --help`):

```bash
java -Xms10g -Xmx10g -XX:+UseParallelGC -cp out bigdata.matmul.Main dense \
  --dense-sizes 1600 --dense-methods blocked_ikj --block-sizes 64 \
  --reps 5 --warmup 2 --seed 42 --max-seconds 240 --mem-budget-mb 8192 --out-dir results
```

Self-tests always run first and abort the benchmark on failure. If you rerun the full study, delete `results/` first: the CSV is appended to.

## Inputs (reproducible)

* `java.util.Random` (algorithm fixed by the Java spec → identical matrices on any JVM). A uses `seed`, B uses `seed+1` (default 42/43).
* Nonzeros are uniform in [1, 2); positions are independent Bernoulli(density) → uniform random nonzero distribution. Dense inputs (Part A) have density 100 %.
* Nothing needs downloading: rerunning with the same seed regenerates identical inputs.

## Timing and memory protocol

* Timer: `System.nanoTime()` (monotonic). Inputs and the output array are created **outside** the timed region; zero-initialisation of C is **inside** the dense kernels (all three), allocation of CSR output arrays is inside the CSR kernel.
* Warm-up runs (default 2, at the actual size, for JIT) are written to the CSV with `phase=warmup` and excluded from summaries. Default 5 measured repetitions. All raw samples are kept.
* Results are consumed (checksum) outside the timed region so the JIT cannot eliminate the work.
* Summary statistics: median and IQR (in `analyze.py`).
* Memory columns: `est_storage_bytes` (array storage: 3·8·n² for dense; CSR arrays for sparse), `heap_used_bytes` (live JVM heap after GC with objects alive; `memory_csr_delta` = heap growth caused by the CSR objects), `peak_rss_kb` (`VmHWM` from `/proc/self/status`, Linux only). **These are different quantities and must not be compared as equals.** `VmHWM` is the process-wide maximum, so `run_all.sh` starts one JVM per configuration; the sparse-kernel JVM still allocates the dense inputs (they are the logical input), so its RSS includes them — rely on `est_storage_bytes`/`memory_csr_delta` to isolate CSR cost.

## Correctness

* `Main test`: 2×2 hand-computed product, identity and zero matrices, random matrices (n = 1…100, including sizes that are not multiples of the block size, block sizes 1/7/16/64) against an independent `double[][]` reference, sparse vs dense reference at densities 1 %–100 %, CSR round trip, sorted output rows, nnz equality.
* Tolerance: `|x−y| ≤ 1e-12 + 1e-9·max(|x|,|y|)`. Different summation orders change rounding only by a few ulps; 1e-9 relative leaves a wide margin yet catches real indexing errors (O(1) errors).
* Benchmark sizes: each result is verified with a Freivalds check (`C·x = A·(B·x)`, two random vectors, O(n²)); column `verified` = PASS/FAIL. In the study all 80 verifications are PASS.

## Budgets and limits

* `--max-seconds` bounds **one** multiplication. A running multiplication cannot be interrupted, so a run that exceeds it is recorded with its true time and `status=TIME_BUDGET_EXCEEDED`; larger sizes of that method are then `SKIPPED_TIME_BUDGET`.
* `--mem-budget-mb` bounds the *estimated* memory of a configuration (also capped to 85 % of `-Xmx`); otherwise `SKIPPED_MEMORY_BUDGET`. `OutOfMemoryError` is recorded as `OOM`. No time is ever invented for these cases; see `summary_limits.csv`.
* In the study no configuration exceeded the budgets (no timeouts, no memory failures). The largest tested and attempted size was n = 4000 (slowest multiplication 51.9 s). `ijk_baseline` was run only up to n = 2500 (`IJK_MAX` in `run_all.sh`), a protocol decision to limit total runtime, not a budget failure.

## Raw data format (`raw_measurements.csv`)

`experiment, method, n, density_pct, sparsity_pct, seed, block_size, numeric_type, rep, phase, time_ns, nnz_a, nnz_b, nnz_c, est_storage_bytes, heap_used_bytes, peak_rss_kb, checksum, verified, status`

Phases: `warmup`, `multiply`, `convert_dense_to_csr`, `convert_plus_multiply`, `verify`, `memory_live_heap`, `memory_csr_delta`.

`environment.txt` records CPU, cores, RAM, OS, JVM, flags and parameters. It is written by the first JVM only, so its "run configuration" block describes that first launch (dense, n = 100, ijk), not the whole study; the study parameters are those of `scripts/run_all.sh`. Study machine: Intel Core i5-10400F, 6 cores / 12 threads, 16 GB RAM, Linux (CachyOS), CPU governor `performance`, only a terminal running.

## Regenerating the report's figures and tables

```bash
python3 report/make_assets.py results report     # writes report/figs/*.pdf and report/tables/*.tex
cd report && latexmk -pdf report.tex
```

## Generative AI disclosure

Generative AI (Claude, Anthropic) was used to accelerate the creation of the benchmarks and of all the measurement and analysis scripts, in order to build them as carefully as possible, and for a first draft of the report. I had already prepared my own design of the matrix generation and multiplication components, with a very similar structure; I adopted the AI-generated version because it added improvements to some of them. I reviewed the code, ran all experiments on my own machine, and am responsible for the measurements, the interpretation and the conclusions.
