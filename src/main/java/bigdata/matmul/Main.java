package bigdata.matmul;

import java.nio.file.Files;
import java.nio.file.Path;

/** Entry point: java -cp out bigdata.matmul.Main <test|dense|sparse|all> [options] */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args[0].equals("--help") || args[0].equals("-h")) {
            System.out.print(Config.usage());
            return;
        }
        Config cfg;
        try {
            cfg = Config.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(Config.usage());
            System.exit(2);
            return;
        }

        switch (cfg.mode) {
            case "test" -> System.exit(Validator.selfTest(System.out) ? 0 : 1);
            case "dense", "sparse", "all" -> runExperiments(cfg);
            default -> {
                System.err.println("Unknown mode: " + cfg.mode);
                System.err.print(Config.usage());
                System.exit(2);
            }
        }
    }

    private static void runExperiments(Config cfg) throws Exception {
        Files.createDirectories(cfg.outDir);
        if (!Validator.selfTest(System.out)) {
            System.err.println("Self-tests failed: refusing to benchmark.");
            System.exit(1);
        }

        long idleHeap = Harness.usedHeapAfterGc();
        Path env = cfg.outDir.resolve("environment.txt");
        if (!Files.exists(env)) {
            Harness.writeEnvironment(env, cfg, idleHeap);
            System.out.println("Environment written to " + env);
        }

        Path csvPath = cfg.outDir.resolve(cfg.outFile);
        try (Csv csv = new Csv(csvPath)) {
            Experiments exp = new Experiments(cfg, csv);
            if (cfg.mode.equals("dense") || cfg.mode.equals("all")) {
                exp.runDense();
            }
            if (cfg.mode.equals("sparse") || cfg.mode.equals("all")) {
                exp.runSparse();
            }
        }
        System.out.println("Raw measurements: " + csvPath + " (sink=" + Harness.SINK + "," + Harness.DSINK + ")");
    }
}
