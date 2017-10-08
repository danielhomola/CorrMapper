import os
import numpy as np
import pandas as pd
import community
import scipy.cluster.hierarchy as hclust
from scipy.spatial.distance import pdist
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects import numpy2ri
from backend.utils.check_uploaded_files import open_file
from backend.corr.network import create_network
from frontend import app

# We need to load and parse an R script for get_cliques(). We do it once instead
# of at every function call
r_path = os.path.join(app.config["BACKEND_FOLDER"], "corr", 'bivar_modules.R')
r_source = open(r_path )
string = ''.join(r_source.readlines())
r_code = SignatureTranslatedAnonymousPackage(string, "R")

def filter_genomic(r, p, params, name, sym=False):
    """
    Filters overlapping and distant correlations in genomic datasets.
    """
    # --------------------------------------------------------------------------
    # OPEN AND FILTER ANNOTATION FILES
    # --------------------------------------------------------------------------

    if name == 'dataset1':
        annot1 = annot2 = params['annotation1']
    elif name == 'dataset2':
        annot1 = annot2 = params['annotation2']
    else:
        annot1 = params['annotation1']
        annot2 = params['annotation2']

    path = os.path.join(params['study_folder'], annot1)
    annot1, _ = open_file(path)
    path = os.path.join(params['study_folder'], annot2)
    annot2, _ = open_file(path)

    annot1 = annot1.loc[r.index]
    annot2 = annot2.loc[r.columns]

    # --------------------------------------------------------------------------
    # OPEN BIN AND CHROMO FILE, BUILD DICTS FROM THEM
    # --------------------------------------------------------------------------

    bin_file = params['bin_file']
    chromo_file = ''.join(bin_file.split('__')[:-1]) + '__chromosomes.txt'
    bins = pd.read_table(bin_file)
    chromo = pd.read_table(chromo_file)
    chr_starts = {}
    chr2num = {c:ci for ci, c in enumerate(chromo.Chromosome)}
    for i in bins.index:
        row = bins.loc[i]
        if row.Chromosome not in chr_starts:
            chr_starts[row.Chromosome] = int(row.ChromoStart)

    # --------------------------------------------------------------------------
    # HELPER FUNCTIONS FOR DISCARD AND CONSTRAIN
    # --------------------------------------------------------------------------

    def check_edge(row, col):
        # get location of source and target GE
        c1, s1, e1 = get_location(annot1, row)
        c2, s2, e2 = get_location(annot2, col)

        # do we need to discard overlapping GEs? True means we're good to go.
        if params['discard_overlap']:
            overlap = overlap_test(c1, c2, s1, s2, e1, e2)
        else:
            overlap = True
        # do we need to restrict corrs to a certain distance? True is good.
        if params['constrain_corr']:
            dist = params['constrain_dist']
            constrain = constrain_test(c1, c2, s1, s2, e1, e2, dist)
        else:
            constrain = True

        if overlap and constrain:
            return True
        else:
            return False

    def get_location(annot, row):
        row = annot.loc[row]
        return row.Chromosome, row.Start, row.End

    def overlap_test(c1, c2, s1, s2, e1, e2):
        """
        Check if two genomic elements overlap. If not return True.
        """
        # if they're on different chromosomes they cannot overlap
        if c1 != c2:
            return True
        else:
            # we will assume that s1 is smaller, otherwise swap them
            if s1 > s2:
                s1, s2 = s2, s1
                e1, e2 = e2, e1
            if e1 > s2:
                return False
            else:
                return True

    def constrain_test(c1, c2, s1, s2, e1, e2, dist):
        """
        Check if two genomic elements are within a distance specified by dist in
        Mbps. If yes, return True.
        """
        dist = int(dist)
        # if dist == 0 GEs need to be on the same chromosome
        if dist == 0:
            if c1 == c2:
                return True
            else:
                return False
        # otherwise dist is provided in Mbps but bin files use raw bp nums
        elif dist > 0:
            dist = int(dist * 10000000)
            # they are on the same chromosome
            if c1 == c2:
                # we will assume s1 is closer to start of c1, if not swap them
                if s1 > s2:
                    s1, s2 = s2, s1
                    e1, e2 = e2, e1
                if e1 + dist > s2:
                    return True
                else:
                    return False

            # they are on different chromosomes (we need their ranked pos)
            elif chr2num[c1] > chr2num[c2]:
                # we will assume the c1 is closer to chr1, if not swap them
                c1, c2 = c2, c1
                s1, s2 = s2, s1
                e1, e2 = e2, e1

            # get absolute location of 1st GE's end and 2nd GE's start
            abs_e1 = chr_starts[c1] + e1
            abs_s2 = chr_starts[c2] + s2
            if abs_e1 + dist > abs_s2:
                return True
            else:
                return False

    # --------------------------------------------------------------------------
    # CHECK EDGES IN R FOR OVERLAPPING AND DISTANT CORRS
    # --------------------------------------------------------------------------

    for ri, row in enumerate(r.index):
        for ci, col in enumerate(r.columns):
            # if sym, only use the upper triangle of the r matrix
            if not sym or (sym and ci > ri):
                cell = r[col].loc[row]
                if cell != 0:
                    # if didn't pass the check, set it cell to 0
                    if not check_edge(row, col):
                        r[col].loc[row] = 0
                        p[col].loc[row] = 1
                        if sym:
                            r[row].loc[col] = 0
                            p[row].loc[col] = 1
    return r, p


def int_or_float(l):
    for i,li in enumerate(l):
        if type(li) == str:
            try:
                l[i] = int(li)
            except:
                try:
                    l[i] = float(li)
                except:
                    l[i] = li
    return l


def order_by_name(data, get_ind=False):
    """
    Simply order the columns and indices by their value.

    If the supported DataFrame has col and row names that could be coerced
    to float or int it orders them by that.

    Returns
    ------
    - data: DataFrame with ordered columns and indices.
    - if get_ind=True, the index for reordering is returned as well
    """

    cols = list(data.columns.values)
    rows = list(data.index.values)

    col_ind = np.argsort(int_or_float(cols))
    cols = sorted(int_or_float(cols))
    row_ind = np.argsort(int_or_float(rows))
    rows = sorted(int_or_float(rows))

    data = data.loc[map(str,rows)]
    data = data.T.loc[map(str,cols)].T
    if get_ind:
        return data, col_ind, row_ind
    else:
        return data


def order_by_hc(data, method='average', metric='correlation', get_ind=False):
    """
    Orders the DataFrame cols by hierarchical clustering.

    Correlation is used a distance metric, and the UPGMA for joining clusters
    by default. Correlation as a distance measure cares more about the shape
    of the data then its magnitude.

    Returns
    ------
    - The reordered DataFrame.
    - if get_ind=True, the index for reordering is returned as well
    """
    dist = pdist(data.T.values, metric=metric)
    dist = np.clip(dist, 0, dist.max())
    d = hclust.linkage(dist, method=method)
    # sometimes correlation based distance doesn't work, so then use euc dist
    if np.any(d < 0):
        dist = pdist(data.T.values, metric='euclidean')
        dist = np.clip(dist, 0, dist.max())
        d = hclust.linkage(dist, method=method)
    d = hclust.dendrogram(d, labels=data.columns, no_plot=True,
                          count_sort='descending')
    data = data.T.loc[d['ivl']].T
    if get_ind:
         return data, d['leaves']
    else:
        return data


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    :param arrays : list of array-like
        1-D arrays to form the cartesian product of.
    :param out : ndarray
        Array to place the cartesian product in.
    :return out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def get_mod_strings(rl, cl, sym=False):
    """
    Takes in the row and column labels that indicate the modules in the
    adjecancy matrix of a network. It turns each module into a string for the
    selected variable of the vis.js JavaScript of the frontend. This way the
    user can select modules individually and filter down the network to them.

    :param rl: row labels
    :param cl: column labels
    :param sym: whether the adjacency matrix is symmetric, i.e. the network is
                uniparite or bipartite
    :return modules: a list of strings, each describing the cells of a module
    """
    modules = []
    for i in np.unique(rl):
        # find rows and cols of the module
        mod_rix = np.where(rl == i)[0]
        mod_cix = np.where(cl == i)[0]

        # generate string of module that will be passed to JS
        cells = cartesian([mod_rix, mod_cix])
        if sym:
            mod_str = ["r" + str(ix[0]) + "|r" + str(ix[1]) for ix in cells]
        else:
            mod_str = ["r" + str(ix[0]) + "|c" + str(ix[1]) for ix in cells]

        # discard modules with only one cell
        if len(mod_str) > 1:
            mod_str = ','.join(mod_str)
            modules.append(mod_str)
    return modules


def find_modules(data, sym=False):
    """
    Take an adjacency matrix, and return the modules that are found and the
    reordering of the matrix and the corresponding strings for vis.js.
    """
    # If it's a bipartite graph we need to use a different modularity finding
    # algorithm: http://rsos.royalsocietypublishing.org/content/3/1/140536
    if not sym:
        numpy2ri.activate()
        res = r_code.run_bivar_modules(np.abs(data.values))
        rix = np.array(res[0]) - 1
        cix = np.array(res[1]) - 1
        rl = np.array(res[2])
        cl = np.array(res[3])
        numpy2ri.deactivate()

    # Else we can use the highly popular: https://arxiv.org/abs/0803.0476
    else:
        graph = create_network(r=data, p=None, sym=True, modules=True)
        part = community.best_partition(graph)

        # not all indices make it into part. those that don't, will be assigned
        # to a dummy module with k=max(modules)+1.
        part_complete = []
        part_non_complete = []
        max_module_n = max(part.values())
        for i in data.index:
            if i in part:
                part_complete.append(part[i])
                part_non_complete.append(part[i])
            else:
                part_complete.append(max_module_n + 1)

        # reorder data matrix so modules end up next to each other
        cix = rix = np.argsort(part_complete)
        # we only want to keep the true module labels not the dummy ones
        part_non_complete = np.array(part_non_complete)
        cl = rl = part_non_complete[np.argsort(part_non_complete)]

    # transform the selected modules into strings that are understood by vis.js
    modules = get_mod_strings(rl, cl, sym)
    return list(rix), list(cix), modules


def recode_rowcol_names(data, sym=False):
    """
    Return a dict of dicts for each row and col. Keys are c1, c2.. and r1, r2..
    for cols and rows respectively. Values are dicts with short_name and
    long_name as keys.
    """
    n, p = data.shape
    rows = [x + str(i) for i, x in enumerate(['r'] * n)]
    cols = [x + str(i) for i, x in enumerate(['c'] * p)]
    d = {}
    threshold = 12

    # row names
    for i, x in enumerate(data.index):
        short_name = x[:threshold]
        d[rows[i]] = {'short_name': short_name, 'long_name': x}
    data.index = rows
    if sym:
        data.columns = rows
    else:
        for i, x in enumerate(data.columns):
            short_name = x[:threshold]
            d[cols[i]] = {'short_name': short_name, 'long_name': x}
        data.columns = cols
    return data, d


def fold_change(data, meta, d=None):
    """
    Calculates how much each feature changed/differ between two classes.
    This is only called if params['fs_cols'] is a binary categorical feature.
    Data has to be a pandas DataFrame, and meta a vector of two classes.

    Returns
    -------
    Dictionary of all features (column in DataFrame) and their log2 median
    fold-change
    """
    # make sure we have the same dimensions
    shared_ind = data.index.intersection(meta.index)
    data = data.loc[shared_ind]
    meta = meta.loc[shared_ind]
    # sort both meta and data so the log ratio has the (alphabetically)
    # 'smaller' value in the numerator.
    ind = np.argsort(meta.values)
    meta = meta.iloc[ind]
    data = data.iloc[ind, :]
    # convert meta to binary label 0, 1.
    from sklearn.preprocessing import LabelEncoder
    meta = LabelEncoder().fit_transform(meta)
    if d is None:
        d = {}
    for c in data.columns:
        class0 = data[c][meta == 0]
        class1 = data[c][meta == 1]
        # we discard nans
        class0 = class0[~pd.isnull(class0)]
        class1 = class1[~pd.isnull(class1)]
        # we go for the median instead of the mean
        class0_median = np.median(class0.values)
        class1_median = np.median(class1.values)
        if class1_median == 0:
            ratio = np.nan
        else:
            ratio = class0_median / float(class1_median)
        d[c] = ratio
    return d


def plot_tree(P, pos=None):
    """
    With this we can plot a dendrogram, and disect what's going on.
    It would be relatively easy to save this as well.
    """
    from matplotlib import pyplot as plt
    icoord = np.array( P['icoord'] )
    dcoord = np.array( P['dcoord'] )
    color_list = np.array( P['color_list'] )
    xmin, xmax = icoord.min(), icoord.max()
    ymin, ymax = dcoord.min(), dcoord.max()
    if pos:
        icoord = icoord[pos]
        dcoord = dcoord[pos]
        color_list = color_list[pos]
    for xs, ys, color in zip(icoord, dcoord, color_list):
        plt.plot(xs, ys,  color)
    plt.xlim( xmin-10, xmax + 0.1*abs(xmax) )
    plt.ylim( ymin, ymax + 0.1*abs(ymax) )
    plt.show()
