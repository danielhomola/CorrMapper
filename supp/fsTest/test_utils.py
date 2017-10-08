import numpy as np
import boruta
from fsTest import lsvc_cv
from fsTest import mifs

import sklearn.feature_selection as fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression

def check_selection(selected, shuffle_ind, i, r, top_var_ix=None):
    """
    Calculate precision and recall given the selected and real features.
    """
    # reorder selected features
    try:
        if top_var_ix is not None:
            # If we selected top var features, we need to reorder by that too
            selected = set(top_var_ix[shuffle_ind[selected]])
        else:
            selected = set(shuffle_ind[selected])
        all_good = set(range(i + r))
        TP = len(selected.intersection(all_good))
        FP = len(selected - all_good)
        FN = len(all_good - selected)
        if (TP + FN) > 0:
            sens = TP / float(TP + FN)
        else:
            sens = np.nan
        if (TP + FP) > 0:
            prec = TP / float(TP + FP)
        else:
            prec = np.nan
    except:
        sens = np.nan
        prec = np.nan
    return sens, prec


def do_fs(X, y):
    s, f = X.shape
    y_test = np.arange(f).reshape(1, -1)

    # --------------------------------------------------------------
    # UNIVARIATE FEATURE SELECTION
    # percentile - take the top10% of features
    sel_uni_perc = fs.SelectPercentile(fs.f_classif, 10).fit(X, y).transform(y_test)[0]

    # fdr - minimize false discovery rate at alpha = .05
    sel_uni_fdr = fs.SelectFdr(fs.f_classif, .05).fit(X, y).transform(y_test)[0]

    # --------------------------------------------------------------
    # RFECV
    # do a cross-validated grid search for the optimal C
    gridC = {'C': np.logspace(-6, 3, 10)}
    svc = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=1e-4)
    grid_cv = GridSearchCV(svc, gridC, scoring='accuracy', n_jobs=-1)
    grid_cv.fit(X, y)

    # set the optimal C
    # adjust for the smaller training sample size, due to cross validation
    # http://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html
    cv_num = 3
    train_size = 1 - 1/float(cv_num)
    adjust_c = float(s * train_size)
    svc.set_params(C=grid_cv.best_params_['C'] * adjust_c)
    # do a stratified 3 fold cross-validated recursive feature elimination,
    # with 1% of the worst feautres removed each round

    rfecv = fs.RFECV(estimator=svc, step=.01, cv=cv_num, scoring='accuracy')
    rfecv.fit(X, y)
    sel_rfecv = rfecv.transform(y_test)[0]

    # --------------------------------------------------------------
    # L1 SVC
    sel_lsvc = lsvc_cv.recursive_lsvc_cv(X, y, -3, 3, 7)

    # --------------------------------------------------------------
    # STABILITY SELECTION
    rlr = RandomizedLogisticRegression(n_resampling=1000,
                                       C=np.logspace(-2, 2, 5),
                                       selection_threshold=0.7,
                                       sample_fraction=0.5)
    sel_rlr = rlr.fit(X, y).transform(y_test)[0]

    # --------------------------------------------------------------
    # BORUTA
    rf = RandomForestClassifier(n_jobs=-1)
    b = boruta.BorutaPy(rf, n_estimators='auto')
    b.fit(X, y)
    sel_b_rf = np.where(b.support_)[0]

    # --------------------------------------------------------------
    # JMI
    MIFS = mifs.MutualInformationFeatureSelector(method='JMI')
    MIFS.fit(X, y)
    sel_jmi = np.where(MIFS.support_)[0]

    return (sel_uni_perc, sel_uni_fdr, sel_rfecv, sel_lsvc, sel_rlr, sel_b_rf,
            sel_jmi)