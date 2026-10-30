package bigdata.matmul;

import java.lang.ref.Reference;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Part A (dense optimisation) and Part B (sparse CSR vs dense) experiments.
 *
 * Timing boundary (all phases): inputs are generated outside the timed region and the output array
 * is allocated outside it. For dense kernels, the zero-initialisation of C is inside the timed kernel.
 * For the CSR kernel, allocation of the (growing) output arrays is inside the kernel.
 * CSR conversion is measured separately (phase convert_dense_to_csr) and also together with the
 * multiplication (phase convert_plus_multiply). Phase "multiply" is the kernel only.
 *
 * Budgets: a configuration whose estimated memory exceeds the budget (or the JVM heap) is recorded as
 * SKIPPED_MEMORY_BUDGET. A run whose duration exceeds --max-seconds is recorded with its real measured time and
 * status TIME_BUDGET_EXCEEDED (a running multiplication cannot be interrupted); further repetitions and larger
 * sizes of that method are then skipped (SKIPPED_TIME_BUDGET). OutOfMemoryError is recorded as status OOM.
 */
final class Experiments {

    private final Config cfg;
    private final Csv csv;

    Experiments(Config cfg, Csv csv) {
        this.cfg = cfg;
        this.csv = csv;
    }

    private long usableBytes() {
        long budget = cfg.memBudgetMb * 1024L * 1024L;
        long heap = (long) (Runtime.getRuntime().maxMemory() * 0.85);
        return Math.min(budget, heap);
    }

    private Row make(Supplier<Row> mk, String method, int bs, String phase) {
        Row r = mk.get();
        r.method = method;
        r.blockSize = bs;
        r.phase = phase;
        return r;
    }

    private void skip(Supplier<Row> mk, String method, int bs, String status) {
        Row r = make(mk, method, bs, "multiply");
        r.status = status;
        csv.write(r);
    }


    void runDense() {
        System.out.println("== Part A: dense optimisation ==");
        Set<String> stopped = new HashSet<>();
        for (int n : cfg.denseSizes) {
            Supplier<Row> mk = () -> {
                Row r = new Row();
                r.experiment = "dense_opt";
                r.n = n;
                r.seed = cfg.seed;
                r.densityPct = 100.0;
                r.sparsityPct = 0.0;
                r.nnzA = (long) n * n;
                r.nnzB = (long) n * n;
                return r;
            };
            long est = 3L * 8L * n * n;
            boolean overBudget = est > usableBytes();
            double[] a = null;
            double[] b = null;
            if (!overBudget) {
                a = MatrixGenerator.dense(n, 1.0, cfg.seed);
                b = MatrixGenerator.dense(n, 1.0, cfg.seed + 1);
            }
            for (String method : cfg.denseMethods) {
                int[] blocks = method.equals(DenseMultiply.BLOCKED) ? cfg.blockSizes : new int[]{0};
                for (int bs : blocks) {
                    String key = method + "|" + bs;
                    if (overBudget) {
                        skip(mk, method, bs, "SKIPPED_MEMORY_BUDGET");
                    } else if (stopped.contains(key)) {
                        skip(mk, method, bs, "SKIPPED_TIME_BUDGET");
                    } else if (!denseConfig(a, b, n, method, bs, mk)) {
                        stopped.add(key);
                    }
                }
            }
        }
    }

    /** Runs warm-up, measured repetitions, verification and memory probe for one dense configuration. */
    private boolean denseConfig(double[] a, double[] b, int n, String method, int bs, Supplier<Row> mk) {
        final long maxNs = (long) (cfg.maxSeconds * 1e9);
        try {
            final double[] c = new double[n * n];
            final Runnable task = () -> DenseMultiply.run(method, bs, a, b, c, n);
            boolean exceeded = false;

            for (int w = 1; w <= cfg.warmup && !exceeded; w++) {
                long t = Harness.timeNs(task);
                Row r = make(mk, method, bs, "warmup");
                r.rep = w;
                r.timeNs = t;
                if (t > maxNs) {
                    r.status = "TIME_BUDGET_EXCEEDED";
                    exceeded = true;
                }
                csv.write(r);
            }
            if (!exceeded) {
                for (int rep = 1; rep <= cfg.reps; rep++) {
                    long t = Harness.timeNs(task);
                    Row r = make(mk, method, bs, "multiply");
                    r.rep = rep;
                    r.timeNs = t;
                    r.checksum = Harness.checksum(c);
                    if (t > maxNs) {
                        r.status = "TIME_BUDGET_EXCEEDED";
                        exceeded = true;
                    }
                    csv.write(r);
                    if (exceeded) {
                        break;
                    }
                }
            }

            Row v = make(mk, method, bs, "verify");
            boolean ok = Validator.freivalds(a, b, c, n, cfg.seed + 7);
            v.verified = ok ? "PASS" : "FAIL";
            v.nnzC = MatrixGenerator.countNonZeros(c);
            v.checksum = Harness.checksum(c);
            csv.write(v);
            if (!ok) {
                System.err.println("WARNING: verification FAILED for " + method + " n=" + n);
            }

            Row m = make(mk, method, bs, "memory_live_heap");
            m.heapUsedBytes = Harness.usedHeapAfterGc();
            m.estStorageBytes = 3L * 8L * n * n;
            m.peakRssKb = Harness.peakRssKb();
            csv.write(m);
            Reference.reachabilityFence(a);
            Reference.reachabilityFence(b);
            Reference.reachabilityFence(c);

            System.out.printf("  dense n=%d %s%s done%s%n", n, method, bs > 0 ? "(bs=" + bs + ")" : "",
                    exceeded ? " [time budget exceeded]" : "");
            return !exceeded;
        } catch (OutOfMemoryError e) {
            Row r = make(mk, method, bs, "multiply");
            r.status = "OOM";
            csv.write(r);
            System.out.printf("  dense n=%d %s OOM%n", n, method);
            return false;
        }
    }


    void runSparse() {
        System.out.println("== Part B: sparse CSR vs dense ==");
        Set<String> stopped = new HashSet<>();
        List<String> comparators = cfg.sparseDenseMethods;
        final String csrName = "csr_gustavson";
        for (int n : cfg.sparseSizes) {
            for (double sp : cfg.sparsityPct) {
                final double densityFrac = 1.0 - sp / 100.0;
                final long[] nnz = new long[2];
                Supplier<Row> mk = () -> {
                    Row r = new Row();
                    r.experiment = "sparse_vs_dense";
                    r.n = n;
                    r.seed = cfg.seed;
                    r.densityPct = 100.0 - sp;
                    r.sparsityPct = sp;
                    r.nnzA = nnz[0];
                    r.nnzB = nnz[1];
                    return r;
                };

                double nnzIn = densityFrac * n * n;
                double nnzCEst = Math.min((double) n * n, densityFrac * densityFrac * (double) n * n * n);
                long est = (long) (24.0 * n * n + 12.0 * (2 * nnzIn + nnzCEst));
                boolean overBudget = est > usableBytes();
                if (overBudget) {
                    if (cfg.runCsr) {
                        skip(mk, csrName, 0, "SKIPPED_MEMORY_BUDGET");
                    }
                    for (String dm : comparators) {
                        skip(mk, dm, 0, "SKIPPED_MEMORY_BUDGET");
                    }
                    continue;
                }

                double[] a = MatrixGenerator.dense(n, densityFrac, cfg.seed);
                double[] b = MatrixGenerator.dense(n, densityFrac, cfg.seed + 1);
                nnz[0] = MatrixGenerator.countNonZeros(a);
                nnz[1] = MatrixGenerator.countNonZeros(b);

                if (cfg.runCsr) {
                    String key = csrName + "|" + sp;
                    if (stopped.contains(key)) {
                        skip(mk, csrName, 0, "SKIPPED_TIME_BUDGET");
                    } else if (!sparseCsr(a, b, n, csrName, mk)) {
                        stopped.add(key);
                    }
                }
                for (String dm : comparators) {
                    String key = "dense:" + dm;
                    if (stopped.contains(key)) {
                        skip(mk, dm, 0, "SKIPPED_TIME_BUDGET");
                    } else if (!denseConfig(a, b, n, dm, 0, mk)) {
                        stopped.add(key);
                    }
                }
            }
        }
    }

    /**
     * Measures, for one sparse input pair: conversion alone, multiplication kernel alone, and conversion +
     * multiplication together. Verifies the product and probes memory. Returns false if the time budget was exceeded.
     */
    private boolean sparseCsr(double[] a, double[] b, int n, String name, Supplier<Row> mk) {
        final long h0 = Harness.usedHeapAfterGc();
        try {
            final CsrMatrix ca = CsrMatrix.fromDense(a, n);
            final CsrMatrix cb = CsrMatrix.fromDense(b, n);
            final CsrMatrix[] res = new CsrMatrix[1];

            final Runnable mul = () -> res[0] = CsrMatrix.multiply(ca, cb);
            final Runnable conv = () -> Harness.SINK += CsrMatrix.fromDense(a, n).nnz() + CsrMatrix.fromDense(b, n).nnz();
            final Runnable e2e = () -> {
                CsrMatrix x = CsrMatrix.fromDense(a, n);
                CsrMatrix y = CsrMatrix.fromDense(b, n);
                res[0] = CsrMatrix.multiply(x, y);
            };

            boolean ok = measurePhase(name, "multiply", mul, mk, res);
            if (ok) {
                measurePhase(name, "convert_dense_to_csr", conv, mk, null);
                ok = measurePhase(name, "convert_plus_multiply", e2e, mk, res);
            }

            CsrMatrix c = res[0];
            if (c != null) {
                Row v = make(mk, name, 0, "verify");
                boolean pass = Validator.freivaldsSparse(a, b, c, n, cfg.seed + 7);
                v.verified = pass ? "PASS" : "FAIL";
                v.nnzC = c.nnz();
                v.checksum = c.sumOfValues();
                csv.write(v);
                if (!pass) {
                    System.err.println("WARNING: verification FAILED for " + name + " n=" + n);
                }

                long h1 = Harness.usedHeapAfterGc();
                long csrBytes = ca.storageBytes() + cb.storageBytes() + c.storageBytes();
                Row m1 = make(mk, name, 0, "memory_live_heap");
                m1.heapUsedBytes = h1;
                m1.estStorageBytes = csrBytes;
                m1.nnzC = c.nnz();
                m1.peakRssKb = Harness.peakRssKb();
                csv.write(m1);
                Row m2 = make(mk, name, 0, "memory_csr_delta");
                m2.heapUsedBytes = Math.max(0, h1 - h0);
                m2.estStorageBytes = csrBytes;
                m2.nnzC = c.nnz();
                csv.write(m2);
            }
            Reference.reachabilityFence(ca);
            Reference.reachabilityFence(cb);
            Reference.reachabilityFence(res);
            System.out.printf("  sparse n=%d sparsity=%s%% %s done%s%n", n, mk.get().sparsityPct, name,
                    ok ? "" : " [time budget exceeded]");
            return ok;
        } catch (OutOfMemoryError e) {
            Row r = make(mk, name, 0, "multiply");
            r.status = "OOM";
            csv.write(r);
            System.out.printf("  sparse n=%d %s OOM%n", n, name);
            return false;
        }
    }

    /** Warm-up + measured repetitions of one sparse phase. Returns false if the time budget was exceeded. */
    private boolean measurePhase(String method, String phase, Runnable task, Supplier<Row> mk, CsrMatrix[] res) {
        final long maxNs = (long) (cfg.maxSeconds * 1e9);
        boolean exceeded = false;
        for (int w = 1; w <= cfg.warmup && !exceeded; w++) {
            long t = Harness.timeNs(task);
            Row r = make(mk, method, 0, "warmup");
            r.rep = w;
            r.timeNs = t;
            r.status = t > maxNs ? "TIME_BUDGET_EXCEEDED" : "OK";
            exceeded = t > maxNs;
            csv.write(r);
        }
        if (exceeded) {
            return false;
        }
        for (int rep = 1; rep <= cfg.reps; rep++) {
            long t = Harness.timeNs(task);
            Row r = make(mk, method, 0, phase);
            r.rep = rep;
            r.timeNs = t;
            if (res != null && res[0] != null) {
                r.nnzC = res[0].nnz();
                r.checksum = res[0].sumOfValues();
            }
            if (t > maxNs) {
                r.status = "TIME_BUDGET_EXCEEDED";
                exceeded = true;
            }
            csv.write(r);
            if (exceeded) {
                break;
            }
        }
        return !exceeded;
    }
}
