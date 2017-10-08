import os

import pandas as pd
import numpy as np
import bottleneck as bn
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder

import hugeR
import permutation as cp
import utils as cu
import network as cn
import write_js_vars as wjs
from pairplots import generate_pair_plots
from backend.utils.check_uploaded_files import open_file


def corr_main(params):
    """
    This is the main function which performs the following steps:
    - open dataset(s), load selected features, merge datasets
    - perform GLASSO with huge R package
    - calculated permuted p-values with GPD approximation in parallel
    - correct for multiple testing
    - save r and p value matrices for users
    - save networks from r values for users
    - write variables and datasets for visualisation in JS
    """

    # --------------------------------------------------------------------------
    # CALCULATE GRAPHLASSO AND PERMUTED P-VALS
    # --------------------------------------------------------------------------

    # open first dataset
    path = os.path.join(params['output_folder'], params['dataset1'])
    dataset1, sep = open_file(path)
    n, p = dataset1.shape
    # if there's a 2nd dataset, merge them
    if not params['autocorr']:
        path2 = os.path.join(params['output_folder'], params['dataset2'])
        dataset2, sep2 = open_file(path2)
        # if two featres has the same name we need prefixes
        merged_datasets_df = dataset1.join(dataset2, how='inner',
                                           lsuffix='_data1', rsuffix='_data2')
        X = merged_datasets_df.values
    else:
        merged_datasets_df = dataset1
        X = merged_datasets_df.values

    # standardise X
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # perform GLASSO with huge in R
    lambda_threshold = params['lambda_val']
    cov, prec = hugeR.hugeR(X, lambda_threshold)

    # create column ranked X for corr_permutation
    rX = bn.rankdata(X, axis=0)

    # get GPD approximated p-values
    perm_num = 10000
    rs, p_vals, p_mask = cp.gpd_spearman(rX, perm_num=perm_num, prec=prec,
                                         mc_method=params['multi_corr_method'],
                                         mc_alpha=params['alpha_val'])

    # delete correlations that did not pass the multi test correction
    rs[~p_mask] = 0
    p_vals[~p_mask] = 1

    # --------------------------------------------------------------------------
    # CHECK IF GENOMIC FILTERING IS NEEDED
    # --------------------------------------------------------------------------

    # if fs, load metadata column for fold_change calculation later
    if params['fs']:
        path = os.path.join(params['study_folder'], params['metadata_file'])
        y, _ = open_file(path)
        y = y[params['fs_cols']].iloc[1:].dropna()
    else:
        y = None

    # if genomic, check if filtering overlapping and distant corrs needed
    discard_or_constrain = params['discard_overlap'] or params['constrain_corr']
    if params['annotation'] and discard_or_constrain:
        genomic = True
    else:
        genomic = False

    # --------------------------------------------------------------------------
    # GENERATE PAIRWISE PLOTS FOR DATA1, DATA2, DATA1-2
    # --------------------------------------------------------------------------

    generate_pair_plots(params, rs, p_vals, merged_datasets_df, p)

    # --------------------------------------------------------------------------
    # WRITE RESULTS FOR DATA1, DATA2, DATA1-2
    # --------------------------------------------------------------------------

    params = write_results(params, rs[:p, :p], p_vals[:p, :p], genomic,
                           (dataset1, dataset1), 'dataset1', y, True)
    if not params['autocorr']:
        params = write_results(params, rs[p:, p:], p_vals[p:, p:], genomic,
                               (dataset2, dataset2), 'dataset2', y, True)
        params = write_results(params, rs[:p, p:], p_vals[:p, p:], genomic,
                               (dataset1, dataset2), 'dataset1_2', y)

    # if corr_done in params is False one of the writing steps failed
    if 'corr_done' not in params:
        params['corr_done'] = True
    return params


def write_results(params, r, p, genomic, datasets, name, y, sym=False):
    """
    Generates and saves all result files for user and visualisations.
    """
    # create r and p DataFrames
    r = pd.DataFrame(r, columns=datasets[1].columns, index=datasets[0].columns)
    p = pd.DataFrame(p, columns=datasets[1].columns, index=datasets[0].columns)

    if genomic:
        r, p = cu.filter_genomic(r, p, params, name, sym)

    # if sym, set diagonal to 1, because line 60 set it 0
    if sym:
        np.fill_diagonal(r.values, 0)
        np.fill_diagonal(p.values, 1)

    # filter rows and columns which are all 0, if sym=True then diag can be 1
    to_keep_row = ((r == 0).sum(axis=1) != r.shape[1]).values
    to_keep_col = ((r == 0).sum(axis=0) != r.shape[0]).values
    rf = r.iloc[to_keep_row, to_keep_col]
    rf_nrow, rf_ncol = rf.shape
    pf = p.iloc[to_keep_row, to_keep_col]

    # check size of the filtered data, abort if empty dim encountered
    if len(rf.shape) == 1 or rf.shape == (0, 0):
        params['corr_done'] = False
        return params

    # save r and p matrices
    params['r_' + name] = 'r_' + name + '.csv'
    path = os.path.join(params['output_folder'], params['r_' + name])
    rf.to_csv(path)
    params['p_' + name] = 'p_' + name + '.csv'
    path = os.path.join(params['output_folder'], params['p_' + name])
    pf.to_csv(path)

    # cannot cluster if either of the dimensions is one
    if rf.shape[0] == 1:
        mod_clusters_row = data_clusters_row = r_clusters_row = [0]
        mod_clusters_col = data_clusters_col = r_clusters_col = range(rf_ncol)
        modules = []
        # reorder by modules, this is the default in vis
        rf = rf.iloc[mod_clusters_row, mod_clusters_col]
        pf = pf.iloc[mod_clusters_row, mod_clusters_col]
    elif rf.shape[1] == 1:
        mod_clusters_row = data_clusters_row = r_clusters_row = range(rf_ncol)
        mod_clusters_col = data_clusters_col = r_clusters_col = [0]
        modules = []
        # reorder by modules, this is the default in vis
        rf = rf.iloc[mod_clusters_row, mod_clusters_col]
        pf = pf.iloc[mod_clusters_row, mod_clusters_col]
    else:
        # find modules based on the network topology
        mod_clusters_row, mod_clusters_col, modules = cu.find_modules(rf, sym)
        # reorder by modules, this has to before other orderings are calculated
        rf = rf.iloc[mod_clusters_row, mod_clusters_col]
        pf = pf.iloc[mod_clusters_row, mod_clusters_col]

        # calculate cluster ordering by values of the underlying data
        _, data_clusters_row = cu.order_by_hc(datasets[0][rf.index],
                                              get_ind=True)
        _, data_clusters_col = cu.order_by_hc(datasets[1][rf.columns],
                                              get_ind=True)
        # calculate cluster ordering by r values
        _, r_clusters_row = cu.order_by_hc(rf.T, get_ind=True)
        _, r_clusters_col = cu.order_by_hc(rf, get_ind=True)

    # calculate order by names
    _, name_order_col, name_order_row = cu.order_by_name(rf, True)

    # rename cols and rows, get short and long names.
    rf, names = cu.recode_rowcol_names(rf, sym)
    pf, _ = cu.recode_rowcol_names(pf, sym)

    # generate network, for symmetric matrices we don't need bipartite graph
    network = cn.create_network(rf, pf, sym)
    # save it for user as XML
    path = os.path.join(params['output_folder'], name + '_network.xml')
    nx.write_graphml(network, path)

    # calculate fold changes if y is a binary variable
    fold_change = {}
    if y is not None:
        classes = LabelEncoder().fit(y).classes_
        if classes.shape[0] == 2:
            fold_change = cu.fold_change(datasets[0], y)
            fold_change = cu.fold_change(datasets[1], y, fold_change)

    # write matrix for d3
    path = os.path.join(params['vis_folder'], name + '.csv')
    wjs.matrix_for_d3(rf, pf, sym, path)

    # write vars for d3
    path = os.path.join(params['vis_folder'], name + '_vars.js')
    wjs.write_vars_for_d3(rf, names, path, fold_change, sym, modules,
                          data_clusters_col, data_clusters_row,
                          r_clusters_col, r_clusters_row,
                          name_order_col, name_order_row)

    # write network for d3
    path = os.path.join(params['vis_folder'], name + '_network.js')
    wjs.write_network_for_d3(network, rf, names, fold_change, path)

    # save size of corr matrix to work out if we need two col layout in HTML
    params['col_num_' + name] = rf.shape[1]

    # save number of mudules of the dataset to params so we can pass it to jinja
    params['modules_n_' + name] = len(modules)
    return params
