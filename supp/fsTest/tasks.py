from __future__ import absolute_import
from fsTest.celery import app

@app.task
def run_benchmark(sample, feature, informative, random_states, root):
    """
    Compare several feature selection methods on varying simulated datasets.
    """

    import numpy as np
    import pandas as pd
    import os
    from fsTest import test_utils as tu
    from sklearn.datasets import make_classification

    for random_state in random_states:
        # ----------------------------------------------------------------------
        # SIMULATE DATASET

        s = sample
        f = feature
        i = informative
        # redundant feature number
        r = int(i * .25)
        # number of sub-clusters within both clusters
        c = 2
        X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                                   n_redundant=r, n_clusters_per_class=c,
                                   random_state=random_state, shuffle=False)

        file_id = ("samp_%d_feat_%d_inf_%d_rel_%d_rand_%d" % (s, f, i, i+r,
                                                              random_state))
        results_folder = root + "fsTest/results/"

        # this ensure that we only run analysis that ins't already finished
        result_file = results_folder + file_id + '.txt'
        file_exist = os.path.isfile(result_file)
        if not file_exist or (file_exist and os.stat(result_file).st_size == 0):
            o = open(result_file, 'w')

            # by default informative features are the first columns, so let's
            # reshuffle the columns to make sure none of the methods can cheat
            shuffle_ind = np.arange(X.shape[1])
            np.random.shuffle(shuffle_ind)
            X = X[:, shuffle_ind]

            # perform FS
            all_methods = tu.do_fs(X, y)

            # ------------------------------------------------------------------
            # WRITE RESULTS

            all_method_names = ['uni_perc', 'uni_fdr', 'rfecv',
                                'lsvc', 'ss', 'boruta', 'jmi']
            o.write('Method\tSens\tPrec\tSelected\n')
            for mi, m in enumerate(all_methods):
                o.write(all_method_names[mi] + '\t')
                sens, prec = tu.check_selection(m, shuffle_ind, i, r)
                if np.isnan(sens) or np.isnan(prec):
                    sel = ''
                else:
                    sel = ','.join(map(str, sorted(shuffle_ind[m])))
                o.write(str(sens) + '\t' + str(prec) + '\t' + sel + '\n')
            o.close()
            

@app.task
def run_benchmark_topvar(sample, feature, informative, random_states, root):
    """
    Compare several feature selection methods on varying simulated datasets.
    Unlike the run_benchmark, this function follows CorrMapper and ensures that
    the n/p ratio is never lower than 0.5 by keeping only the top 2n features
    according to their variance.
    """

    import numpy as np
    import pandas as pd
    import os
    from fsTest import test_utils as tu
    from sklearn.datasets import make_classification

    # n to p ratio - as used in CorrMapper
    top_var_ratio = 2

    for random_state in random_states:
        # ----------------------------------------------------------------------
        # SIMULATE DATASET

        s = sample
        f = feature
        i = informative
        # redundant feature number
        r = int(i * .25)
        # number of sub-clusters within both clusters
        c = 2
        X, y = make_classification(n_samples=s, n_features=f, n_informative=i,
                                   n_redundant=r, n_clusters_per_class=c,
                                   random_state=random_state, shuffle=False)

        file_id = ("samp_%d_feat_%d_inf_%d_rel_%d_rand_%d" % (s, f, i, i + r,
                                                              random_state))
        results_folder = root + "fsTest/results_topvar/"

        # this ensure that we only run analysis that ins't already finished
        result_file = results_folder + file_id + '.txt'
        file_exist = os.path.isfile(result_file)
        if not file_exist or (file_exist and os.stat(result_file).st_size == 0):

            # make sure that we only test datasets with n/p < 0.5
            if int(top_var_ratio * X.shape[0]) <= X.shape[1]:
                # only keep the top 2n features with the highest variance
                X = pd.DataFrame(X)
                two_n = int(2 * X.shape[0])
                top_var_ix = np.array(sorted(np.argsort(X.var())[-two_n:]))
                X = X[top_var_ix]
                X = X.values

                o = open(result_file, 'w')

                shuffle_ind = np.arange(X.shape[1])
                np.random.shuffle(shuffle_ind)
                X = X[:, shuffle_ind]

                # perform FS
                all_methods = tu.do_fs(X, y)

                # --------------------------------------------------------------
                # WRITE RESULTS

                all_method_names = ['uni_perc', 'uni_fdr', 'rfecv',
                                    'lsvc', 'ss', 'boruta', 'jmi']
                o.write('Method\tSens\tPrec\tSelected\n')
                for mi, m in enumerate(all_methods):
                    o.write(all_method_names[mi] + '\t')
                    sens, prec = tu.check_selection(m, shuffle_ind, i, r,
                                                    top_var_ix)
                    if np.isnan(sens) or np.isnan(prec):
                        sel = ''
                    else:
                        sel_feats = sorted(top_var_ix[shuffle_ind[m]])
                        sel = ','.join(map(str, sel_feats))
                    o.write(str(sens) + '\t' + str(prec) + '\t' + sel + '\n')
                o.close()
