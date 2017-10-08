from .tasks import run_cm_test, run_cm_mixomics_test

if __name__ == '__main__':
    samples = [100, 300, 500]
    features = [1000, 2000, 5000]
    random_states = range(3)

    # samples = [100]
    # features = [500]
    # random_states = range(1)
    root = "/home/daniel/corr/supp/"

    # EXPERIMENT 1
    for sample in samples:
        for feature in features:
            informative = int(feature * .1)
            stars = .1
            result = run_cm_test.delay(sample, feature, informative, stars,
                                       random_states, root)
            stars = .05
            result = run_cm_test.delay(sample, feature, informative, stars,
                                       random_states, root)

            informative = int(feature * .05)
            stars = .1
            result = run_cm_test.delay(sample, feature, informative, stars,
                                       random_states, root)
            stars = .05
            result = run_cm_test.delay(sample, feature, informative, stars,
                                       random_states, root)
    
    # EXPERIMENT 2
    samples = [300, 500]
    features = [1000, 2000]
    random_states = range(3)
    
    for sample in samples:
        for feature in features:
            results = run_cm_mixomics_test.delay(sample, feature, 
                                                 random_states, root)