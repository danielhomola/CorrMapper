from __future__ import absolute_import
from cmTest.celery import app

@app.task
def run_cm_test(sample, feature, informative, stars, random_states, root):
    """
    Measure CorrMapper's performance on varying simulated datasets.
    """

    import numpy as np
    import copy
    import pandas as pd
    import scipy as sp
    from sklearn.datasets import make_sparse_spd_matrix
    from sklearn.preprocessing import StandardScaler

    import os
    from cmTest import test_utils as tu
    from cmTest import hugeR

    for random_state in random_states:

        # ----------------------------------------------------------------------
        # SIMULATE DATASET
        # ----------------------------------------------------------------------

        random_seed = random_state
        n_features = feature
        n_informative = informative
        n_redundant = 0
        n_relevant = n_informative + n_redundant
        s = sample
        f = feature
        i = informative
        r = n_redundant
        st = stars
        prec_real = make_sparse_spd_matrix(n_informative, smallest_coef=.4,
                                           alpha=.98, largest_coef=.8,
                                           random_state=random_seed)

        cov_real = sp.linalg.inv(prec_real)
        d = np.sqrt(np.diag(cov_real))
        # divide through cols
        cov_real /= d
        # divide through rows
        cov_real /= d[:, np.newaxis]
        prec_real *= d
        prec_real *= d[:, np.newaxis]

        covs = [cov_real, cov_real]
        X, y = tu.make_classification(n_samples=s, n_features=f,
                                      n_clusters_per_class=1, n_informative=i,
                                      n_redundant=r, shuffle=False,
                                      class_sep=.25, cov=covs,
                                      random_state=random_seed)

        # ----------------------------------------------------------------------
        # SPLIT DATASET INTO TWO
        # ----------------------------------------------------------------------

        # split relevant features equally and randomly
        X1_rel_feats = set(np.random.choice(n_relevant, n_relevant / 2, replace=False))
        X2_rel_feats = set(range(n_relevant)) - X1_rel_feats

        # now split all other features randomly and equally
        other_n = n_features - n_relevant
        X1_other_feats = set(np.random.choice(range(n_relevant, n_features), other_n / 2, replace=False))
        X2_other_feats = set(range(n_relevant, n_features)) - X1_other_feats

        # merge relevant and irrelevant features
        X1_feats = np.array(sorted(list(X1_other_feats.union(X1_rel_feats))))
        X2_feats = np.array(sorted(list(X2_other_feats.union(X2_rel_feats))))

        # check we have each feature only once
        sorted(list(set(X1_feats).union(X2_feats))) == range(n_features)

        # define X1 and X2
        X1 = X[:, X1_feats]
        X2 = X[:, X2_feats]
        datasets_original = [X1, X2]

        # these two lists will keep track of the features we are left with from
        # the original data matrix. feature nums < n_informative are informative.
        X12_feats_original = [X1_feats, X2_feats]

        # ----------------------------------------------------------------------
        # START PIPELINE
        # ----------------------------------------------------------------------
        if st == 0.05:
            stars = "005"
        else:
            stars = "01"
        file_id = ("samp_%d_feat_%d_inf_%d_star_%s_rand_%d" % (s, f, i, stars,
                                                               random_state))
        results_folder = root + "cmTest/results/"

        # this ensure that we only run analysis that ins't already finished
        result_file = results_folder + file_id + '.txt'
        file_exist = os.path.isfile(result_file)
        fs_methods = ["fdr", "l1svc", "boruta", "jmi"]
        ss = StandardScaler()
        if not file_exist or (file_exist and os.stat(result_file).st_size == 0):
            o = open(result_file, 'w')
            o.write('FS Method, Matrix, Prec, Recall\n')
            for fs_method in fs_methods:
                no_feat = False
                X12_feats = [copy.deepcopy(X12_feats_original[0]),
                             copy.deepcopy(X12_feats_original[1])]
                datasets = [copy.deepcopy(datasets_original[0]),
                            copy.deepcopy(datasets_original[1])]
                for i, dataset in enumerate(datasets):
                    # variance filtering
                    dataset = pd.DataFrame(dataset)
                    two_n = int(2 * dataset.shape[0])
                    top_var_ix = np.array(sorted(np.argsort(dataset.var())[-two_n:]))
                    dataset = dataset[top_var_ix].values
                    # update features of the datasets
                    X12_feats[i] = X12_feats[i][top_var_ix]

                    # FS
                    try:

                        sel = tu.do_fs(dataset, y, fs_method)
                        if len(sel) == 0:
                            no_feat = True
                        else:
                            # update features of the datasets
                            X12_feats[i] = X12_feats[i][sel]
                            datasets[i] = dataset[:, sel]
                    except:
                        no_feat = True

                if not no_feat:
                    # concatenate datasets
                    dataset1 = pd.DataFrame(datasets[0])
                    dataset2 = pd.DataFrame(datasets[1])
                    merged_datasets_df = dataset1.join(dataset2, how='inner',
                                                       lsuffix='_data1',
                                                       rsuffix='_data2')
                    X = merged_datasets_df.values
                    # standardise
                    ss = StandardScaler()
                    X = ss.fit_transform(X)

                    # run hugeR's glasso and StARS
                    cov, prec = hugeR.hugeR(X, st)

                    # match features to original informative ones, check docstring
                    # of translate_estimated_matrix_into_original for explanation

                    if prec.shape[0] > 1:
                        prec = tu.translate_estimated_m_into_original(prec, X12_feats,
                                                                      informative)
                        p, r = tu.quality_of_graph(prec_real, prec)
                    else:
                        p, r = np.nan, np.nan
                    o.write(','.join(map(str, [fs_method, "P", p, r])) + '\n')
            o.close()

@app.task
def run_cm_mixomics_test(sample, feature, random_states, root):
    """
    Compare corrmapper with mixomics and marginal corr networks
    """
    
    import os
    import numpy as np
    import copy
    import pandas as pd
    import scipy as sp
    import bottleneck as bn
    from sklearn.datasets import make_sparse_spd_matrix
    from sklearn.preprocessing import StandardScaler
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    
    from cmTest import test_utils as tu
    from cmTest import hugeR

    for random_state in random_states:

        # ----------------------------------------------------------------------
        # SIMULATE DATASET
        # ----------------------------------------------------------------------
        
        informative = int(feature * .05)
        
        n_features = feature
        n_informative = informative
        n_redundant = 0
        n_relevant = n_informative + n_redundant
        s = sample
        f = feature
        i = informative
        i_half = n_informative/2
        r = n_redundant
        prec_real = make_sparse_spd_matrix(n_informative, smallest_coef=.4,
                                           alpha=.98, largest_coef=.8,
                                           random_state=random_state)

        cov_real = sp.linalg.inv(prec_real)
        d = np.sqrt(np.diag(cov_real))
        # divide through cols
        cov_real /= d
        # divide through rows
        cov_real /= d[:, np.newaxis]
        prec_real *= d
        prec_real *= d[:, np.newaxis]

        covs = [cov_real, cov_real]
        X, y = tu.make_classification(n_samples=s, n_features=f,
                                      n_clusters_per_class=1, n_informative=i,
                                      n_redundant=r, shuffle=False,
                                      class_sep=.25, cov=covs,
                                      random_state=random_state)

        # ----------------------------------------------------------------------
        # SPLIT DATASET INTO TWO
        # ----------------------------------------------------------------------

        # split relevant features equally and randomly
        X1_rel_feats = set(np.random.choice(n_relevant, n_relevant / 2, replace=False))
        X2_rel_feats = set(range(n_relevant)) - X1_rel_feats

        # now split all other features randomly and equally
        other_n = n_features - n_relevant
        X1_other_feats = set(np.random.choice(range(n_relevant, n_features), other_n / 2, replace=False))
        X2_other_feats = set(range(n_relevant, n_features)) - X1_other_feats

        # merge relevant and irrelevant features
        X1_feats = np.array(sorted(list(X1_other_feats.union(X1_rel_feats))))
        X2_feats = np.array(sorted(list(X2_other_feats.union(X2_rel_feats))))

        # check we have each feature only once
        sorted(list(set(X1_feats).union(X2_feats))) == range(n_features)

        # define X1 and X2
        X1 = X[:, X1_feats]
        X2 = X[:, X2_feats]
        datasets_original = [X1, X2]

        # these two lists will keep track of the features we are left with from
        # the original data matrix. feature nums < n_informative are informative.
        X12_feats_original = [X1_feats, X2_feats]

        # ----------------------------------------------------------------------
        # START PIPELINE
        # ----------------------------------------------------------------------
        
        file_id = ("samp_%d_feat_%d_rand_%d" % (s, f, random_state))
        results_folder = root + "cmTest/results2/"

        # this ensure that we only run analysis that ins't already finished
        result_file = results_folder + file_id + '.txt'
        file_exist = os.path.isfile(result_file)
        fs_methods = ["fdr", "l1svc", "boruta", "jmi"]
        ss = StandardScaler()
        if not file_exist or (file_exist and os.stat(result_file).st_size == 0):
            o = open(result_file, 'w')
            
            # -----------------------------------------------------------------            
            # CORRMAPPER
            o.write('Method, Prec, Recall\n')
            for fs_method in fs_methods:
                no_feat = False
                X12_feats = [copy.deepcopy(X12_feats_original[0]),
                             copy.deepcopy(X12_feats_original[1])]
                datasets = [copy.deepcopy(datasets_original[0]),
                            copy.deepcopy(datasets_original[1])]
                for i, dataset in enumerate(datasets):
                    # variance filtering
                    dataset = pd.DataFrame(dataset)
                    two_n = int(2 * dataset.shape[0])
                    top_var_ix = np.array(sorted(np.argsort(dataset.var())[-two_n:]))
                    dataset = dataset[top_var_ix].values
                    # update features of the datasets
                    X12_feats[i] = X12_feats[i][top_var_ix]

                    # FS
                    try:
                        sel = tu.do_fs(dataset, y, fs_method)
                        if len(sel) == 0:
                            no_feat = True
                        else:
                            # update features of the datasets
                            X12_feats[i] = X12_feats[i][sel]
                            datasets[i] = dataset[:, sel]
                    except:
                        no_feat = True

                if not no_feat:
                    # concatenate datasets
                    dataset1 = pd.DataFrame(datasets[0])
                    dataset2 = pd.DataFrame(datasets[1])
                    merged_datasets_df = dataset1.join(dataset2, how='inner',
                                                       lsuffix='_data1',
                                                       rsuffix='_data2')
                    X_fs = merged_datasets_df.values
                    # standardise
                    X_fs = ss.fit_transform(X_fs)
                    
                    # run hugeR's glasso and StARS
                    cov, prec = hugeR.hugeR(X_fs, 0.05)

                    # match features to original informative ones, check docstring
                    # of translate_estimated_matrix_into_original for explanation
                    if prec.shape[0] > 1:
                        prec = tu.translate_estimated_m_into_original(prec, X12_feats,
                                                                      informative)
                        # we only compare the N12 network to make it fair for mixomics
                                                                      
                        p, r = tu.quality_of_graph(prec_real[i_half:,:i_half], 
                                                   prec[i_half:,:i_half], sym=False)
                    else:
                        p, r = np.nan, np.nan
                    o.write(','.join(map(str, [fs_method, p, r])) + '\n')
                    
            # reorder real precision for other two methods
            all_rel_feats = list(X1_rel_feats) + list(X2_rel_feats)
            prec_real2 = prec_real
            prec_real2 = prec_real2[all_rel_feats,:]
            prec_real2 = prec_real2[:,all_rel_feats]
            prec_real2 = prec_real2[i_half:,:i_half]

            # -----------------------------------------------------------------
            # GRAPH LASSO
            try:
                dataset = pd.DataFrame(X)
                two_n = int(2 * dataset.shape[0])
                top_var_ix = np.array(sorted(np.argsort(dataset.var())[-two_n:]))
                X_gl = dataset[top_var_ix].values                                
                X_gl = X[:, top_var_ix]
                X_gl = ss.fit_transform(X_gl)
                cov, prec = hugeR.hugeR(X_gl, 0.05)
                p, r = tu.quality_of_graph(prec_real[i_half:,:i_half], prec[X1.shape[1]:X1.shape[1]+i_half,:i_half], sym=False)
                o.write(','.join(map(str, ["glasso", p, r])) + '\n')
            except:
                o.write(','.join(map(str, ["glasso", np.nan, np.nan])) + '\n')
            
            # -----------------------------------------------------------------
            # MARGINAL CORR NETWORK
            cov_thresholds = [.05, .1, .2, .3, .5, .7, .8]
            
            rX = bn.rankdata(X, axis=0)
            marg_cov = np.corrcoef(rX, rowvar=0)
            marg_cov = np.abs(marg_cov[X1.shape[1]:X1.shape[1]+i_half,:i_half])
            for thresh in cov_thresholds:
                tmp_cov = (marg_cov > thresh).astype(int)
                p, r = tu.quality_of_graph(prec_real2, tmp_cov, sym=False)
                method = "marginal %f" % thresh
                o.write(','.join(map(str, [method, p, r])) + '\n')                                
            
            # -----------------------------------------------------------------            
            # MIXOMICS
            base = importr('base')
            # this allows us to send numpy to R directly, neat
            numpy2ri.activate()
            mo = importr('mixOmics')
            
            mo_spls_model = mo.spls(X1, X2, ncomp = 3, keepX=i_half, keepY=i_half)
            mo_network = mo.network(mo_spls_model)
            mo_cov = np.array(base.as_data_frame(mo_network.rx('M')))
            numpy2ri.deactivate()
            mo_cov = np.abs(mo_cov[:i_half,:i_half])
            for thresh in cov_thresholds:
                tmp_cov = (mo_cov > thresh).astype(int)
                p, r = tu.quality_of_graph(prec_real2, tmp_cov, sym=False)
                method = "mixomics %f" % thresh
                o.write(','.join(map(str, [method, p, r])) + '\n')                                
            
            o.close()