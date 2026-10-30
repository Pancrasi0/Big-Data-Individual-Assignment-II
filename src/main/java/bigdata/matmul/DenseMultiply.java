package bigdata.matmul;

import java.util.Arrays;

/**
 * Dense square matrix multiplication kernels (sequential).
 *
 * All matrices are n x n, stored row-major in a flat double[] of length n*n,
 * so element (i, j) is at index i*n + j. All kernels compute C = A * B and
 * overwrite C completely (the output initialisation is part of the timed kernel
 * for every method, so the timing boundary is identical across methods).
 *
 * Every kernel performs exactly n^3 multiply-add operations.
 */
public final class DenseMultiply {

    public static final String BASELINE = "ijk_baseline";
    public static final String REORDER = "ikj_reorder";
    public static final String BLOCKED = "blocked_ikj";

    private DenseMultiply() {
    }

    /**
     * Baseline: textbook triple loop, order i-j-k.
     * The inner loop walks B down a column (stride n): cache-unfriendly.
     */
    public static void ijk(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            final int iN = i * n;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[iN + k] * b[k * n + j];
                }
                c[iN + j] = sum;
            }
        }
    }

    /**
     * Optimisation 1: loop reordering to i-k-j.
     * The inner loop walks rows of B and C contiguously (stride 1),
     * and a[i][k] is hoisted into a register. Same arithmetic, different access pattern.
     */
    public static void ikj(double[] a, double[] b, double[] c, int n) {
        Arrays.fill(c, 0, n * n, 0.0);
        for (int i = 0; i < n; i++) {
            final int iN = i * n;
            for (int k = 0; k < n; k++) {
                final double aik = a[iN + k];
                final int kN = k * n;
                for (int j = 0; j < n; j++) {
                    c[iN + j] += aik * b[kN + j];
                }
            }
        }
    }

    /**
     * Optimisation 2: cache blocking (tiling) on top of the i-k-j order.
     * Works on bs x bs tiles so that the tiles of B and C being reused stay in cache.
     * Handles n that is not a multiple of bs.
     */
    public static void blocked(double[] a, double[] b, double[] c, int n, int bs) {
        if (bs < 1) {
            throw new IllegalArgumentException("block size must be >= 1");
        }
        Arrays.fill(c, 0, n * n, 0.0);
        for (int ii = 0; ii < n; ii += bs) {
            final int iMax = Math.min(ii + bs, n);
            for (int kk = 0; kk < n; kk += bs) {
                final int kMax = Math.min(kk + bs, n);
                for (int jj = 0; jj < n; jj += bs) {
                    final int jMax = Math.min(jj + bs, n);
                    for (int i = ii; i < iMax; i++) {
                        final int iN = i * n;
                        for (int k = kk; k < kMax; k++) {
                            final double aik = a[iN + k];
                            final int kN = k * n;
                            for (int j = jj; j < jMax; j++) {
                                c[iN + j] += aik * b[kN + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /** Dispatch by method name. {@code bs} is only used by the blocked method. */
    public static void run(String method, int bs, double[] a, double[] b, double[] c, int n) {
        switch (method) {
            case BASELINE -> ijk(a, b, c, n);
            case REORDER -> ikj(a, b, c, n);
            case BLOCKED -> blocked(a, b, c, n, bs);
            default -> throw new IllegalArgumentException("Unknown dense method: " + method);
        }
    }

    public static boolean isKnown(String method) {
        return BASELINE.equals(method) || REORDER.equals(method) || BLOCKED.equals(method);
    }
}
