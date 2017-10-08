import numpy as np
import os
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

import boruta
import lsvc_cv
import mifs
from backend.utils.check_uploaded_files import open_file


def fs_main(params):
    # get top variance features
    params = top_variance(params)
    # carry out feature selection if necessary
    if params['fs']:
        params = do_fs(params)
    else:
        params['fs_done'] = True
    return params


def top_variance(params):
    datasets = ['dataset1']
    if not params['autocorr']:
        datasets.append('dataset2')
    # this determines how many top variance features we include. 1 means as many
    # as many samples we have.
    top_var_ratio = 2
    for dataset in datasets:
        path = os.path.join(params['study_folder'], params[dataset])
        X, sep = open_file(path)
        if int(top_var_ratio * X.shape[0]) < X.shape[1]:
            # keep only the top N = top_var_ratio * X.shape[0] var features
            X = X[np.argsort(X.var())[-int(top_var_ratio * X.shape[0]):]]
            filename, ext = os.path.splitext(params[dataset])
            params[dataset] = filename + '_topvar' + ext
            path = os.path.join(params['output_folder'], params[dataset])
            X.to_csv(path, sep=sep)
    return params


def do_fs(params):
    # setup basic vars for feature selection
    datasets = ['dataset1']
    if not params['autocorr']:
        datasets.append('dataset2')
    dataset_names = []
    dataset_num = []
    metadata_names = []
    all_selected = []
    le = LabelEncoder()

    # open metadata file, define scaler, fs method
    path = os.path.join(params['study_folder'], params['metadata_file'])
    y, _ = open_file(path)
    y = y[params['fs_cols']]
    ss = StandardScaler()
    method = params['fs_method']

    # perform fs on dataset(s)
    for dataset in datasets:
        path = os.path.join(params['output_folder'], params[dataset])
        X, sep = open_file(path)

        # drop NaNs in metadata column
        y_ = y.iloc[1:].dropna()
        meta_type = y.iloc[0]

        # find intersecting samples and filter down X and y and call fs_()
        ind = X.index.intersection(y_.index)
        X_ = X.loc[ind]
        y_ = y_.loc[ind].values.reshape(-1, 1)

        # scale X to 0 mean and unit variance
        X_ = ss.fit_transform(X_.values)

        if meta_type == 'cat':
            # encode categorical values as numbers
            y_ = le.fit_transform(y_)
            selected = fs_categorical(X_, y_, method)
        else:
            # scale y to 0 mean and unit variance
            y_ = ss.fit_transform(y_)
            # override Boruta and JMI with L1 if continuous
            if method in ['Boruta', 'JMI']:
                selected = fs_continuous(X_, y_, 'L1')
            else:
                selected = fs_continuous(X_, y_, method)

        # we need at least 5 selected features per dataset
        if selected is None:
            selected = []
        if len(selected) <= 5:
            params['fs_done'] = False
            return params

        # saving filtered X into output folder
        filename, ext = os.path.splitext(params[dataset])
        params[dataset] = filename.replace('topvar', 'fs') + ext
        X_sel = X.iloc[:, selected]
        X_sel.to_csv(os.path.join(params['output_folder'], params[dataset]),
                     sep=sep)

        # saving results for selected_features.csv
        dataset_names.append(params[dataset])
        dataset_num.append(dataset)
        metadata_names.append(params['fs_cols'])
        all_selected.append('|'.join(map(str,np.array(selected))))

    # writing selected_features.csv
    results = zip(dataset_names, dataset_num, metadata_names, all_selected)
    cols = ['dataset_name', 'dataset_num', 'metadata_name', 'selected']
    selected_file = os.path.join(params['output_folder'], 'selected_features.csv')
    pd.DataFrame(results, columns=cols).to_csv(selected_file)
    params['fs_done'] = True
    return params


def fs_categorical(X, y, method):
    n, p = X.shape
    selected = []
    if method == 'Boruta':
        rf = RandomForestClassifier(n_jobs=-1)
        Boruta = boruta.BorutaPy(rf, n_estimators='auto')
        Boruta.fit(X, y)
        selected = np.where(Boruta.support_)[0]
    elif method == 'JMI':
        MIFS = mifs.MutualInformationFeatureSelector(method='JMI')
        MIFS.fit(X, y)
        selected = np.where(MIFS.support_)[0]
    elif method == 'L1':
        selected = lsvc_cv.recursive_lsvc_cv(X, y, -3, 3, 7)
    elif method == 'FDR':
        FDR = fs.SelectFdr(fs.f_classif, .05)
        FDR.fit(X, y)
        selected = FDR.transform(np.arange(p).reshape(1,-1))[0]
    return selected


def fs_continuous(X, y, method):
    """
    All 4 methods are implemented, but for Boruta and MIFS the method is over-
    riden and set to L1.
    """
    n, p = X.shape
    if method == 'Boruta':
        rf = RandomForestRegressor(n_jobs=-1)
        Boruta = boruta.BorutaPy(rf, n_estimators='auto')
        Boruta.fit(X, y)
        selected = np.where(Boruta.support_)[0]
    elif method == 'JMI':
        MIFS = mifs.MutualInformationFeatureSelector(method='JMI', categorical=False)
        MIFS.fit(X, y)
        selected = np.where(MIFS.support_)[0]
    elif method == 'L1':
        lasso = LassoCV(n_jobs=-1, normalize=False)
        sfm = SelectFromModel(lasso)
        sfm.fit(X, y)
        selected = sfm.transform(np.arange(p).reshape(1,-1))[0]
    elif method == 'FDR':
        FDR = fs.SelectFdr(fs.f_regression, .05)
        FDR.fit(X, y)
        selected = FDR.transform(np.arange(p).reshape(1,-1))[0]
    return selected
