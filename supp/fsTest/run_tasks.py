from .tasks import run_benchmark, run_benchmark_topvar

if __name__ == '__main__':
    samples = [100, 300, 500, 1000, 2000]
    features = [100, 500, 1000, 5000, 10000, 20000]
    random_states = range(10)
    root = "/home/daniel/corr/supp/"

    for sample in samples:
        for feature in features:
            if sample < 1000 or feature < 5000:
                # -------------------------------------------------------------
                # RUN BENCHMARK EXPERIMENT 1
            
                # important features: 10% of all
                informative = int(feature * .1)
                result = run_benchmark.delay(sample, feature, informative,
                                             random_states, root)

                # important features: 5% of all
                informative = int(feature * .05)
                result = run_benchmark.delay(sample, feature, informative,
                                             random_states, root)

                # -------------------------------------------------------------
                # RUN BENCHMARK EXPERIMENT 2

                # important features: 10% of all
                informative = int(feature * .01)
                result = run_benchmark_topvar.delay(sample, feature,
                                                    informative, random_states,
                                                    root)

                # important features: 5% of all
                informative = int(feature * .05)
                result = run_benchmark_topvar.delay(sample, feature,
                                                    informative, random_states,
                                                    root)
