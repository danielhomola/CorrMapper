from cmTest import lsvc_cv
from cmTest import mifs

import boruta
import numpy as np
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.fixes import astype
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle


def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions
    """
    if dimensions > 30:
        return np.hstack([_generate_hypercube(samples, dimensions - 30, rng),
                          _generate_hypercube(samples, 30, rng)])
    out = astype(sample_without_replacement(2 ** dimensions, samples,
                                            random_state=rng),
                 dtype='>u4', copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out


def make_classification(n_samples=100, n_features=20, cov=None, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    if 2 ** n_informative < n_classes * n_clusters_per_class:
        raise ValueError("n_classes * n_clusters_per_class must"
                         " be smaller or equal 2 ** n_informative")
    if weights and len(weights) not in [n_classes, n_classes - 1]:
        raise ValueError("Weights specified but incompatible with number "
                         "of classes.")

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    if weights and len(weights) == (n_classes - 1):
        weights.append(1.0 - sum(weights))

    if weights is None:
        weights = [1.0 / n_classes] * n_classes
        weights[-1] = 1.0 - sum(weights[:-1])

    # Distribute samples among clusters by weight
    n_samples_per_cluster = []
    for k in range(n_clusters):
        n_samples_per_cluster.append(int(n_samples * weights[k % n_classes]
                                         / n_clusters_per_class))
    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative,
                                    generator).astype(float)
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.rand(n_clusters, 1)
        centroids *= generator.rand(1, n_informative)

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.randn(n_samples, n_informative)

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster
        if cov == None:
            # introduce random covariance
            A = 2 * generator.rand(n_informative, n_informative) - 1
            X_k[...] = np.dot(X_k, A)
        else:
            # use the user-specified covariance matrix
            A = np.linalg.cholesky(cov[k])
            X_k[...] = np.dot(A, X_k.T).T

        X_k += centroid  # shift the cluster to a vertex

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_useless > 0:
        X[:, -n_useless:] = generator.randn(n_samples, n_useless)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.rand(n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.rand(n_features)
    X *= scale

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y


def do_fs(X, y, method):
    s, f = X.shape
    y_test = np.arange(f).reshape(1, -1)
    if method == "fdr":
        sel = fs.SelectFdr(fs.f_classif, .05).fit(X, y).transform(y_test)[0]
    elif method == "l1svc":
        sel = lsvc_cv.recursive_lsvc_cv(X, y, -3, 3, 7)
    elif method == "boruta":
        rf = RandomForestClassifier(n_jobs=-1)
        b = boruta.BorutaPy(rf, n_estimators='auto')
        b.fit(X, y)
        sel = np.where(b.support_)[0]
    elif method == "jmi":
        MIFS = mifs.MutualInformationFeatureSelector(method='JMI')
        MIFS.fit(X, y)
        sel = np.where(MIFS.support_)[0]
    return sel


def mirror_lower_triangle(m):
    """
    Copies the lower triangle's values to the upper triangle by mirroring
    the matrix over the diagonal.
    """
    n = m.shape[0]
    if n != m.shape[1]:
        raise ValueError('This function expects square a matrix.')
    for r in range(n):
        for c in range(n):
            m[r, c] = m[c, r]
    return m


def translate_estimated_m_into_original(m, X12_feats, n_info=0):
    """
    Takes in the estimated covariance or precision matrix. The
    problem is that not all informative feautres are were selected
    and instead a lot of non-informative are selected. So m
    needs to be filtered so it can be compared to the original
    covariance and precision matrix of the informative
    features. For this we need X12feats, which is has two elements.
    The first one tells us the indices original features that
    went into the feature selected dataset1, and its second
    element does the same for the 2nd feature selected dataset.
    Any feature whose index is < n_informative is a good, ie.
    informative feature and we can keep it. The rest we need to
    get rid off. Finally, those informatives that were not selected
    need to be padded with zeros so we have the same
    n_informative x n_informative matrix dimensions for the estimated
    and real precision and cov matrices.
    """

    # merge selected feature lists
    all_feats = list(X12_feats[0]) + list(X12_feats[1])
    # reorder matrix so columns are in ascending order
    reorder_ix = np.argsort(all_feats)
    m = m[reorder_ix, :]
    m = m[:, reorder_ix]
    # reorder all_feats too
    all_feats = np.array(all_feats)[reorder_ix]

    n = m.shape[0]
    d = {}
    for r in range(n):
        for c in range(r):
            orig_r = all_feats[r]
            orig_c = all_feats[c]
            # if cell is between two informative features save its location
            if orig_r < n_info and orig_c < n_info:
                d[(orig_r, orig_c)] = m[r, c]

    zero_m = np.zeros((n_info, n_info))
    for r in range(n_info):
        for c in range(r):
            if (r, c) in d:
                zero_m[r, c] = d[(r, c)]
    zero_m = mirror_lower_triangle(zero_m)
    zero_m[np.diag_indices(n_info)] = 1
    return zero_m


def quality_of_graph(prec, prec_, sym=True, detailed=False):
    """
    Given the true precision matrix and an estimated one,
    this function returns the precision and recall for the 
    network reconstruction. 
    
    It calculates more is needed for these metrics so you 
    set detailed=True it will return much more. 
    """
    # get the matrix of edges, diagonals are not interesting
    n_features = prec.shape[0]
    real_edges = (prec != 0).astype(np.int) 
    estimated_edges = (prec_ != 0).astype(np.int)
    if sym:
        real_edges = real_edges - np.identity(n_features)
        estimated_edges = estimated_edges - np.identity(n_features)
    
    if sym:
        divider = 2.
    else:
        divider = 1.
    true_edges_n  = np.sum(real_edges)/divider
    tp = np.sum((real_edges * estimated_edges))/divider
    tn = np.sum((real_edges + estimated_edges) == 0)/divider
    fp = np.sum((real_edges < estimated_edges))/divider
    fn = np.sum((real_edges > estimated_edges))/divider
    
    to_return = dict()
    if detailed:
        to_return["true_edge_n"] = true_edges_n
        to_return["tp"] = tp
        to_return["tn"] = tn
        to_return["fp"] = fp
        to_return["fn"] = fn
        
    p = tp / float(tp + fp)
    r = tp / float(tp + fn)
    return p, r