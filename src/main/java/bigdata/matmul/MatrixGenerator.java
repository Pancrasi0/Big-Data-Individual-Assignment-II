package bigdata.matmul;

import java.util.Random;

/**
 * Deterministic input generation. Uses java.util.Random, whose algorithm (a 48-bit LCG)
 * is fixed by the Java specification, so the same seed yields the same matrices on any JVM.
 *
 * Nonzero entries are uniform in [1, 2) (never zero). Positions are chosen independently
 * with probability = density (Bernoulli), i.e. a uniform random nonzero distribution.
 * The realised number of nonzeros is therefore close to, but not exactly, density * n^2;
 * it is always counted and recorded.
 */
public final class MatrixGenerator {

    private MatrixGenerator() {
    }

    /** n x n matrix, row-major, with the given density in (0, 1]. */
    public static double[] dense(int n, double density, long seed) {
        Random rng = new Random(seed);
        double[] m = new double[n * n];
        for (int idx = 0; idx < m.length; idx++) {
            if (density >= 1.0 || rng.nextDouble() < density) {
                m[idx] = 1.0 + rng.nextDouble();
            }
        }
        return m;
    }

    public static double[] identity(int n) {
        double[] m = new double[n * n];
        for (int i = 0; i < n; i++) {
            m[i * n + i] = 1.0;
        }
        return m;
    }

    public static long countNonZeros(double[] m) {
        long c = 0;
        for (double v : m) {
            if (v != 0.0) {
                c++;
            }
        }
        return c;
    }
}
