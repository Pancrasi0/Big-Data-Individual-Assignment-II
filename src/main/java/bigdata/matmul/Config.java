package bigdata.matmul;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Command-line configuration. See usage() for all options. */
final class Config {

    String mode = "";
    int[] denseSizes = {100, 200, 400, 800, 1200, 1600};
    int[] sparseSizes = {500, 1000, 2000};
    double[] sparsityPct = {90, 95, 99, 99.9};
    int[] blockSizes = {64};
    List<String> denseMethods = List.of(DenseMultiply.BASELINE, DenseMultiply.REORDER, DenseMultiply.BLOCKED);
    List<String> sparseDenseMethods = List.of(DenseMultiply.BASELINE, DenseMultiply.REORDER);
    boolean runCsr = true;
    int reps = 5;
    int warmup = 2;
    long seed = 42;
    double maxSeconds = 120;
    long memBudgetMb = 3072;
    Path outDir = Path.of("results");
    String outFile = "raw_measurements.csv";

    static String usage() {
        return """
                Usage: java -cp out bigdata.matmul.Main <mode> [options]

                Modes:
                  test      run correctness self-tests only
                  dense     Part A: dense optimisation experiment
                  sparse    Part B: sparse (CSR) vs dense experiment
                  all       self-tests, then dense, then sparse

                Options (comma-separated lists have no spaces):
                  --dense-sizes 100,200,...        sizes n for Part A              (default 100,200,400,800,1200,1600)
                  --dense-methods a,b              ijk_baseline,ikj_reorder,blocked_ikj (default all three)
                  --block-sizes 32,64,128          tile sizes for blocked_ikj      (default 64)
                  --sparse-sizes 500,1000,2000     sizes n for Part B              (default 500,1000,2000)
                  --sparsity 90,95,99,99.9         percentage of ZERO entries      (default 90,95,99,99.9)
                  --sparse-dense-methods a,b|none  dense comparators in Part B     (default ijk_baseline,ikj_reorder)
                  --no-csr                         in Part B, skip the CSR kernel (comparators only)
                  --reps 5                         measured repetitions per configuration (default 5)
                  --warmup 2                       warm-up runs per configuration, excluded from samples (default 2)
                  --seed 42                        base seed (A uses seed, B uses seed+1)
                  --max-seconds 120                time budget for ONE multiplication (default 120)
                  --mem-budget-mb 3072             estimated memory budget per configuration (default 3072)
                  --out-dir results                output directory
                  --out-file raw_measurements.csv  CSV file name (appended to if it exists)
                """;
    }

    private static int[] ints(String s) {
        return Arrays.stream(s.split(",")).map(String::trim).filter(x -> !x.isEmpty()).mapToInt(Integer::parseInt).toArray();
    }

    private static double[] doubles(String s) {
        return Arrays.stream(s.split(",")).map(String::trim).filter(x -> !x.isEmpty()).mapToDouble(Double::parseDouble).toArray();
    }

    private static List<String> methods(String s) {
        List<String> out = new ArrayList<>();
        if (s.equals("none")) {
            return out;
        }
        for (String m : s.split(",")) {
            m = m.trim();
            if (!DenseMultiply.isKnown(m)) {
                throw new IllegalArgumentException("Unknown dense method: " + m);
            }
            out.add(m);
        }
        return out;
    }

    static Config parse(String[] args) {
        Config c = new Config();
        c.mode = args[0];
        for (int i = 1; i < args.length; i++) {
            String key = args[i];
            if (key.equals("--no-csr")) {
                c.runCsr = false;
                continue;
            }
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("Missing value for " + key);
            }
            String v = args[++i];
            switch (key) {
                case "--dense-sizes" -> c.denseSizes = ints(v);
                case "--dense-methods" -> c.denseMethods = methods(v);
                case "--block-sizes" -> c.blockSizes = ints(v);
                case "--sparse-sizes" -> c.sparseSizes = ints(v);
                case "--sparsity" -> c.sparsityPct = doubles(v);
                case "--sparse-dense-methods" -> c.sparseDenseMethods = methods(v);
                case "--reps" -> c.reps = Integer.parseInt(v);
                case "--warmup" -> c.warmup = Integer.parseInt(v);
                case "--seed" -> c.seed = Long.parseLong(v);
                case "--max-seconds" -> c.maxSeconds = Double.parseDouble(v);
                case "--mem-budget-mb" -> c.memBudgetMb = Long.parseLong(v);
                case "--out-dir" -> c.outDir = Path.of(v);
                case "--out-file" -> c.outFile = v;
                default -> throw new IllegalArgumentException("Unknown option: " + key);
            }
        }
        if (c.reps < 1 || c.warmup < 0) {
            throw new IllegalArgumentException("--reps must be >= 1 and --warmup >= 0");
        }
        for (double s : c.sparsityPct) {
            if (s < 0 || s >= 100) {
                throw new IllegalArgumentException("--sparsity values must be in [0, 100)");
            }
        }
        return c;
    }

    List<String> describe() {
        List<String> l = new ArrayList<>();
        l.add("mode = " + mode);
        l.add("dense_sizes = " + Arrays.toString(denseSizes));
        l.add("dense_methods = " + denseMethods);
        l.add("block_sizes = " + Arrays.toString(blockSizes));
        l.add("sparse_sizes = " + Arrays.toString(sparseSizes));
        l.add("sparsity_pct = " + Arrays.toString(sparsityPct));
        l.add("sparse_dense_comparators = " + sparseDenseMethods);
        l.add("run_csr_kernel = " + runCsr);
        l.add("measured_repetitions = " + reps);
        l.add("warmup_runs = " + warmup + " (recorded in CSV with phase=warmup, excluded from summaries)");
        l.add("seed_A = " + seed + ", seed_B = " + (seed + 1) + " (java.util.Random)");
        l.add("value_range_nonzeros = [1, 2)");
        l.add("max_seconds_per_multiplication = " + maxSeconds);
        l.add("memory_budget_mb_estimated = " + memBudgetMb);
        l.add("out = " + outDir.resolve(outFile));
        return l;
    }
}
