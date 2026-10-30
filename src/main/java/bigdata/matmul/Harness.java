package bigdata.matmul;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/** Timing, memory probes, environment capture and CSV output helpers. */
final class Harness {

    /** Sinks so the JIT cannot discard computations whose results are otherwise unused. */
    static volatile long SINK;
    static volatile double DSINK;

    private Harness() {
    }

    /** Monotonic wall-clock time of one task, in nanoseconds. */
    static long timeNs(Runnable r) {
        long s = System.nanoTime();
        r.run();
        return System.nanoTime() - s;
    }

    /** Consumes the result (outside the timed region) and returns a checksum. */
    static double checksum(double[] c) {
        double s = 0.0;
        for (double v : c) {
            s += v;
        }
        DSINK += s;
        return s;
    }

    /** Live heap (bytes) after several GC requests. Only meaningful as an approximation. */
    static long usedHeapAfterGc() {
        Runtime rt = Runtime.getRuntime();
        for (int i = 0; i < 3; i++) {
            System.gc();
            try {
                Thread.sleep(30);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        return rt.totalMemory() - rt.freeMemory();
    }

    /** Peak resident set size of the whole JVM process (VmHWM) in kB; -1 when unavailable (non-Linux). */
    static long peakRssKb() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", ""));
                }
            }
        } catch (Exception e) {
        }
        return -1;
    }

    private static String firstMatch(Path file, String prefix) {
        try {
            for (String line : Files.readAllLines(file)) {
                if (line.startsWith(prefix)) {
                    int idx = line.indexOf(':');
                    return idx >= 0 ? line.substring(idx + 1).trim() : line.trim();
                }
            }
        } catch (Exception e) {
        }
        return null;
    }

    static void writeEnvironment(Path file, Config cfg, long idleHeapBytes) throws IOException {
        List<String> l = new ArrayList<>();
        l.add("timestamp_utc = " + Instant.now());
        l.add("java.version = " + System.getProperty("java.version"));
        l.add("java.vm.name = " + System.getProperty("java.vm.name"));
        l.add("java.vendor = " + System.getProperty("java.vendor"));
        l.add("jvm_input_arguments = " + ManagementFactory.getRuntimeMXBean().getInputArguments());
        l.add("os.name = " + System.getProperty("os.name"));
        l.add("os.version = " + System.getProperty("os.version"));
        l.add("os.arch = " + System.getProperty("os.arch"));
        l.add("logical_processors (availableProcessors) = " + Runtime.getRuntime().availableProcessors());
        Path cpuinfo = Path.of("/proc/cpuinfo");
        String model = firstMatch(cpuinfo, "model name");
        String cores = firstMatch(cpuinfo, "cpu cores");
        l.add("cpu_model = " + (model != null ? model : "UNKNOWN (fill in manually)"));
        l.add("physical_cores_per_socket = " + (cores != null ? cores : "UNKNOWN (fill in manually)"));
        long ram = -1;
        try {
            var os = (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
            ram = os.getTotalMemorySize();
        } catch (Throwable t) {
        }
        l.add("total_ram_bytes = " + (ram >= 0 ? ram : "UNKNOWN (fill in manually)"));
        l.add("jvm_max_heap_bytes = " + Runtime.getRuntime().maxMemory());
        l.add("idle_live_heap_bytes_at_start = " + idleHeapBytes);
        l.add("numeric_type = double (64-bit)");
        l.add("timer = System.nanoTime (monotonic)");
        l.add("timed_region = multiplication kernel only; inputs generated outside; output array allocated outside;"
                + " output initialisation (zero fill) INCLUDED in kernel; CSR conversion reported as its own phase");
        l.add("tolerance = abs " + Validator.ABS_TOL + ", rel " + Validator.REL_TOL);
        l.add("--- run configuration ---");
        l.addAll(cfg.describe());
        l.add("--- manual fields (fill in) ---");
        l.add("compiler_flags = javac -d out (see scripts/build.sh)");
        l.add("cpu_governor_or_power_mode = FILL IN");
        l.add("other_load_on_machine = FILL IN");
        Files.write(file, l);
    }
}

/** One line of the raw measurement CSV. Fields that do not apply stay at their "not available" default. */
final class Row {
    static final String HEADER = "experiment,method,n,density_pct,sparsity_pct,seed,block_size,numeric_type,rep,phase,"
            + "time_ns,nnz_a,nnz_b,nnz_c,est_storage_bytes,heap_used_bytes,peak_rss_kb,checksum,verified,status";

    String experiment = "";
    String method = "";
    String phase = "";
    String status = "OK";
    String verified = "NA";
    int n;
    int blockSize = 0;
    int rep = 0;
    double densityPct = Double.NaN;
    double sparsityPct = Double.NaN;
    long seed;
    long timeNs = -1;
    long nnzA = -1;
    long nnzB = -1;
    long nnzC = -1;
    long estStorageBytes = -1;
    long heapUsedBytes = -1;
    long peakRssKb = -1;
    double checksum = Double.NaN;

    private static String d(double v) {
        return Double.isNaN(v) ? "" : Double.toString(v);
    }

    private static String l(long v) {
        return v < 0 ? "" : Long.toString(v);
    }

    String toCsv() {
        return String.join(",",
                experiment, method, Integer.toString(n), d(densityPct), d(sparsityPct), Long.toString(seed),
                blockSize > 0 ? Integer.toString(blockSize) : "", "double", rep > 0 ? Integer.toString(rep) : "",
                phase, l(timeNs), l(nnzA), l(nnzB), l(nnzC), l(estStorageBytes), l(heapUsedBytes), l(peakRssKb),
                d(checksum), verified, status);
    }
}

/** Append-only CSV writer; flushes every row so partial results survive a crash or kill. */
final class Csv implements Closeable {
    private final BufferedWriter w;

    Csv(Path file) throws IOException {
        boolean needHeader = !Files.exists(file) || Files.size(file) == 0;
        w = Files.newBufferedWriter(file, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        if (needHeader) {
            w.write(Row.HEADER);
            w.newLine();
            w.flush();
        }
    }

    synchronized void write(Row r) {
        try {
            w.write(r.toCsv());
            w.newLine();
            w.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() throws IOException {
        w.close();
    }
}
