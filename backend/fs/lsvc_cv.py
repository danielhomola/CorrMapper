import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score


def lsvc_cv(X, y, cs):
    """
    Does feature selection using LinearSVC with 3-fold stratified CV. Within
    each fold the selected features are validated on unseen test data with
    another 3-fold strafied CV.
    """
    skf = StratifiedKFold(n_splits=3)
    n, p = X.shape
    results = np.zeros((len(cs),3))
    outer_cv = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        for i, c in enumerate(cs):
            LSVC = LinearSVC(C=c, penalty="l1", dual=False)
            SFM = SelectFromModel(LSVC)
            SFM.fit(X_train, y_train)
            try:
                selected = SFM.transform(np.arange(p).reshape(1,-1))[0]
                if len(selected) > 0:
                    val = np.mean(cross_val_score(LSVC, X_test[:, selected],
                                                  y_test, cv=3, n_jobs=-1))
                results[i, outer_cv] = val
            except:
                pass
        outer_cv += 1
    return results


def recursive_lsvc_cv(X, y, start, end, steps):
    """
    Given X, y it chooses the best C (runs recursively till it finds it), then
    fits LinearSVC with that C, and returns the selected features.
    """
    cs = np.logspace(start, end, steps)
    n, p = X.shape
    results = lsvc_cv(X, y, cs)
    good = np.argmax(results.mean(axis=1))
    if good == 0:
        start -= 1
        end = start + 2
        steps = 3
        recursive_lsvc_cv(X, y, start, end, steps)
    elif good == steps-1:
        end += 1
        start = end - 2
        steps = 3
        recursive_lsvc_cv(X, y, start, end, steps)
    else:
        c = cs[good]
        LSVC = LinearSVC(C=c, penalty="l1", dual=False)
        SFM = SelectFromModel(LSVC)
        SFM.fit(X, y.ravel())
        selected = SFM.transform(np.arange(p).reshape(1,-1))[0]
        return selected