package bigdata.matmul;

import java.util.Arrays;

/**
 * Compressed Sparse Row (CSR) matrix, written from scratch for this assignment
 * (no external library). Square or rectangular, double values.
 *
 *  rowPtr[i] .. rowPtr[i+1]-1 index the entries of row i in colIdx / values.
 *  Column indices within a row are sorted ascending.
 */
public final class CsrMatrix {

    public final int rows;
    public final int cols;
    public final int[] rowPtr;
    public final int[] colIdx;
    public final double[] values;

    public CsrMatrix(int rows, int cols, int[] rowPtr, int[] colIdx, double[] values) {
        this.rows = rows;
        this.cols = cols;
        this.rowPtr = rowPtr;
        this.colIdx = colIdx;
        this.values = values;
    }

    public int nnz() {
        return rowPtr[rows];
    }

    /** Approximate storage of the three arrays in bytes (int = 4 B, double = 8 B). */
    public long storageBytes() {
        return 4L * (rows + 1) + 4L * nnz() + 8L * nnz();
    }

    /** Convert a flat row-major dense matrix (rows x cols) to CSR, dropping exact zeros. */
    public static CsrMatrix fromDense(double[] d, int rows, int cols) {
        int nnz = 0;
        for (int i = 0; i < rows * cols; i++) {
            if (d[i] != 0.0) {
                nnz++;
            }
        }
        int[] rowPtr = new int[rows + 1];
        int[] colIdx = new int[nnz];
        double[] values = new double[nnz];
        int p = 0;
        for (int i = 0; i < rows; i++) {
            final int base = i * cols;
            for (int j = 0; j < cols; j++) {
                double v = d[base + j];
                if (v != 0.0) {
                    colIdx[p] = j;
                    values[p] = v;
                    p++;
                }
            }
            rowPtr[i + 1] = p;
        }
        return new CsrMatrix(rows, cols, rowPtr, colIdx, values);
    }

    public static CsrMatrix fromDense(double[] d, int n) {
        return fromDense(d, n, n);
    }

    /** Convert back to a flat row-major dense array (used for validation only). */
    public double[] toDense() {
        double[] d = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++) {
                d[i * cols + colIdx[p]] = values[p];
            }
        }
        return d;
    }

    /** y = this * x. */
    public double[] multiplyVector(double[] x) {
        double[] y = new double[rows];
        for (int i = 0; i < rows; i++) {
            double s = 0.0;
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++) {
                s += values[p] * x[colIdx[p]];
            }
            y[i] = s;
        }
        return y;
    }

    public double sumOfValues() {
        double s = 0.0;
        for (int p = 0; p < nnz(); p++) {
            s += values[p];
        }
        return s;
    }

    /**
     * Sparse x sparse product C = A * B using Gustavson's row-wise algorithm with a
     * dense sparse accumulator (SPA). Output rows are sorted by column index.
     * The output size is not known in advance, so arrays grow geometrically and are
     * trimmed at the end; this allocation cost is part of the kernel.
     */
    public static CsrMatrix multiply(CsrMatrix a, CsrMatrix b) {
        if (a.cols != b.rows) {
            throw new IllegalArgumentException("dimension mismatch");
        }
        final int m = a.rows;
        final int q = b.cols;

        int[] cRowPtr = new int[m + 1];
        int cap = Math.max(16, a.nnz() + b.nnz());
        int[] cCol = new int[cap];
        double[] cVal = new double[cap];
        int nnz = 0;

        final double[] acc = new double[q];
        final int[] mark = new int[q];
        Arrays.fill(mark, -1);
        final int[] touched = new int[q];

        for (int i = 0; i < m; i++) {
            int cnt = 0;
            for (int pa = a.rowPtr[i]; pa < a.rowPtr[i + 1]; pa++) {
                final int k = a.colIdx[pa];
                final double aik = a.values[pa];
                for (int pb = b.rowPtr[k]; pb < b.rowPtr[k + 1]; pb++) {
                    final int j = b.colIdx[pb];
                    final double prod = aik * b.values[pb];
                    if (mark[j] != i) {
                        mark[j] = i;
                        acc[j] = prod;
                        touched[cnt++] = j;
                    } else {
                        acc[j] += prod;
                    }
                }
            }
            Arrays.sort(touched, 0, cnt);
            if (nnz + cnt > cCol.length) {
                long newCap = Math.max(2L * cCol.length, (long) nnz + cnt);
                newCap = Math.min(newCap, Integer.MAX_VALUE - 8L);
                cCol = Arrays.copyOf(cCol, (int) newCap);
                cVal = Arrays.copyOf(cVal, (int) newCap);
            }
            for (int t = 0; t < cnt; t++) {
                final int j = touched[t];
                cCol[nnz] = j;
                cVal[nnz] = acc[j];
                nnz++;
            }
            cRowPtr[i + 1] = nnz;
        }
        return new CsrMatrix(m, q, cRowPtr, Arrays.copyOf(cCol, nnz), Arrays.copyOf(cVal, nnz));
    }
}
