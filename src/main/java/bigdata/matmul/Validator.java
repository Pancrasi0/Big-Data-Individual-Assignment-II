package bigdata.matmul;

import java.io.PrintStream;
import java.util.Random;

/**
 * Correctness checks.
 *
 * Tolerance: two values x, y are considered equal when
 *     |x - y| <= ABS_TOL + REL_TOL * max(|x|, |y|)
 * with ABS_TOL = 1e-12 and REL_TOL = 1e-9. Different summation orders (ijk vs ikj vs blocked vs
 * Gustavson) change rounding only at the level of a few ulps (~1e-16 relative) times a small
 * factor growing with n, so 1e-9 relative leaves a wide safety margin while still catching
 * any real indexing mistake (which produces O(1) relative errors).
 *
 * Two kinds of checks:
 *  - selfTest(): small matrices with known answers (identity, zero, 2x2 hand computed) and
 *    comparison of every method against an independent double[][] reference.
 *  - freivalds(): O(n^2) probabilistic check C*x == A*(B*x) used for large benchmark sizes where
 *    a full reference product would cost as much as the benchmark itself.
 */
public final class Validator {

    public static final double ABS_TOL = 1e-12;
    public static final double REL_TOL = 1e-9;

    private static int total;
    private static int failed;

    private Validator() {
    }

    public static boolean close(double x, double y) {
        return Math.abs(x - y) <= ABS_TOL + REL_TOL * Math.max(Math.abs(x), Math.abs(y));
    }

    public static boolean close(double[] x, double[] y) {
        if (x.length != y.length) {
            return false;
        }
        for (int i = 0; i < x.length; i++) {
            if (!close(x[i], y[i])) {
                return false;
            }
        }
        return true;
    }

    public static double maxAbsDiff(double[] x, double[] y) {
        double m = 0;
        for (int i = 0; i < x.length; i++) {
            m = Math.max(m, Math.abs(x[i] - y[i]));
        }
        return m;
    }

    private static double[] matVec(double[] m, double[] x, int n) {
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            final int iN = i * n;
            for (int j = 0; j < n; j++) {
                s += m[iN + j] * x[j];
            }
            y[i] = s;
        }
        return y;
    }

    /** Freivalds-style check for a dense product: two random vectors. */
    public static boolean freivalds(double[] a, double[] b, double[] c, int n, long seed) {
        Random r = new Random(seed);
        for (int round = 0; round < 2; round++) {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = r.nextDouble();
            }
            double[] abx = matVec(a, matVec(b, x, n), n);
            double[] cx = matVec(c, x, n);
            if (!close(cx, abx)) {
                return false;
            }
        }
        return true;
    }

    /** Freivalds-style check where C is a CSR matrix and A, B are the dense logical inputs. */
    public static boolean freivaldsSparse(double[] a, double[] b, CsrMatrix c, int n, long seed) {
        Random r = new Random(seed);
        for (int round = 0; round < 2; round++) {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = r.nextDouble();
            }
            double[] abx = matVec(a, matVec(b, x, n), n);
            double[] cx = c.multiplyVector(x);
            if (!close(cx, abx)) {
                return false;
            }
        }
        return true;
    }

    /** Independent reference: plain double[][] triple loop, deliberately not sharing code with the kernels. */
    public static double[] reference(double[] a, double[] b, int n) {
        double[][] ra = new double[n][n];
        double[][] rb = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ra[i][j] = a[i * n + j];
                rb[i][j] = b[i * n + j];
            }
        }
        double[] c = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += ra[i][k] * rb[k][j];
                }
                c[i * n + j] = s;
            }
        }
        return c;
    }

    private static void check(PrintStream out, String name, boolean ok) {
        total++;
        if (!ok) {
            failed++;
            out.println("  FAIL: " + name);
        }
    }

    private static final String[][] DENSE_VARIANTS = {
            {DenseMultiply.BASELINE, "0"},
            {DenseMultiply.REORDER, "0"},
            {DenseMultiply.BLOCKED, "1"},
            {DenseMultiply.BLOCKED, "7"},
            {DenseMultiply.BLOCKED, "16"},
            {DenseMultiply.BLOCKED, "64"},
    };

    private static double[] runDense(String[] v, double[] a, double[] b, int n) {
        double[] c = new double[n * n];
        java.util.Arrays.fill(c, 123.456);
        DenseMultiply.run(v[0], Integer.parseInt(v[1]), a, b, c, n);
        return c;
    }

    /** Runs all correctness tests. Returns true when everything passed. */
    public static boolean selfTest(PrintStream out) {
        total = 0;
        failed = 0;
        out.println("Running self-tests (tolerance: abs " + ABS_TOL + ", rel " + REL_TOL + ")...");

        double[] a2 = {1, 2, 3, 4};
        double[] b2 = {5, 6, 7, 8};
        double[] expected2 = {19, 22, 43, 50};
        for (String[] v : DENSE_VARIANTS) {
            check(out, "2x2 known answer, dense " + v[0] + " bs=" + v[1], close(runDense(v, a2, b2, 2), expected2));
        }
        check(out, "2x2 known answer, sparse",
                close(CsrMatrix.multiply(CsrMatrix.fromDense(a2, 2), CsrMatrix.fromDense(b2, 2)).toDense(), expected2));

        int[] sizes = {1, 2, 5, 17, 64, 70};
        for (int n : sizes) {
            double[] a = MatrixGenerator.dense(n, 1.0, 1000 + n);
            double[] id = MatrixGenerator.identity(n);
            double[] zero = new double[n * n];
            for (String[] v : DENSE_VARIANTS) {
                String tag = " n=" + n + " " + v[0] + " bs=" + v[1];
                check(out, "A*I == A" + tag, close(runDense(v, a, id, n), a));
                check(out, "I*A == A" + tag, close(runDense(v, id, a, n), a));
                check(out, "A*0 == 0" + tag, close(runDense(v, a, zero, n), zero));
            }
            CsrMatrix ca = CsrMatrix.fromDense(a, n);
            CsrMatrix cid = CsrMatrix.fromDense(id, n);
            CsrMatrix cz = CsrMatrix.fromDense(zero, n);
            check(out, "sparse A*I == A n=" + n, close(CsrMatrix.multiply(ca, cid).toDense(), a));
            check(out, "sparse I*A == A n=" + n, close(CsrMatrix.multiply(cid, ca).toDense(), a));
            CsrMatrix p = CsrMatrix.multiply(ca, cz);
            check(out, "sparse A*0 has 0 nnz n=" + n, p.nnz() == 0);
            check(out, "CSR round trip n=" + n, close(ca.toDense(), a));
        }

        for (int n : sizes) {
            double[] a = MatrixGenerator.dense(n, 1.0, 11L * n);
            double[] b = MatrixGenerator.dense(n, 1.0, 11L * n + 1);
            double[] ref = reference(a, b, n);
            for (String[] v : DENSE_VARIANTS) {
                check(out, "random dense vs reference n=" + n + " " + v[0] + " bs=" + v[1],
                        close(runDense(v, a, b, n), ref));
            }
            check(out, "freivalds accepts correct product n=" + n, freivalds(a, b, ref, n, 5));
            if (n >= 2) {
                double[] wrong = ref.clone();
                wrong[n + 1 - 1] += 1.0;
                check(out, "freivalds rejects wrong product n=" + n, !freivalds(a, b, wrong, n, 5));
            }
        }

        double[] densities = {0.01, 0.05, 0.1, 0.5, 1.0};
        int[] sparseSizes = {1, 17, 64, 100};
        for (int n : sparseSizes) {
            for (double d : densities) {
                double[] a = MatrixGenerator.dense(n, d, 31L * n);
                double[] b = MatrixGenerator.dense(n, d, 31L * n + 1);
                double[] ref = reference(a, b, n);
                CsrMatrix c = CsrMatrix.multiply(CsrMatrix.fromDense(a, n), CsrMatrix.fromDense(b, n));
                String tag = " n=" + n + " density=" + d;
                check(out, "sparse vs reference" + tag, close(c.toDense(), ref));
                check(out, "sparse nnz matches reference" + tag, c.nnz() == MatrixGenerator.countNonZeros(ref));
                check(out, "sparse freivalds" + tag, freivaldsSparse(a, b, c, n, 9));
                boolean sorted = true;
                for (int i = 0; i < n && sorted; i++) {
                    for (int p = c.rowPtr[i] + 1; p < c.rowPtr[i + 1]; p++) {
                        if (c.colIdx[p - 1] >= c.colIdx[p]) {
                            sorted = false;
                            break;
                        }
                    }
                }
                check(out, "sparse output rows sorted" + tag, sorted);
            }
        }

        out.println("Self-tests: " + (total - failed) + "/" + total + " passed.");
        return failed == 0;
    }
}
