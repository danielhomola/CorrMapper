import numpy as np
from joblib import Parallel, delayed
from statsmodels.sandbox.stats.multicomp import multipletests as multicor
from gpd import gpdPerm as gpd

"""
This module calculates empirical Spearman correlations from X with permuted
p-values in parallel using Generalized Paretho Distribution Approximation and
corrects for multiple testing using FDR.
"""


def gpd_spearman(rX, perm_num=10000, prec=None, mc_method='fdr_bh', mc_alpha=0.05):
    """
    This is the main function of the module. 
    
    The correlation values are calculated in matrix form in parallel, and the 
    p-values are approximated using Monte Carlo sampling (1e4 by default), 
    followed by Generalized Paretho Distribution Approximation to make them 
    more precise.
    
    Then the p values are corrected for multiple testing using the Benjamini
    Hochberg FDR with .05 alpha (by default). For this, only the lower triangle
    of the correlation matrix is used, to avoid including each value twice.

    Parameters
    ----------
    
    - rX : array of shape [n sample, p features]
        Column ranked X matrix of observations and features. Used to calculate
        Spearman correlations and their p-values. 
    - perm_num : int, default = 10000
        Number of permutations to use to estimate each correlation's p-value.
    - prec : array of shape [p features, p features], default = None
        Estimated precision matrix by some Graph Lasso algorithm. If provided,
        only those correlations will be considered whose precision is not 0.
    - mc_method : string, default = 'fdr_bh'
        Method for correction for multiple testing.
    - mc_alpha : float, default = 0.05
        Threshold for q-values after FDR correction.
    
    Returns
    -------
    
    Tuple with three matrices:
    - rs : array of shape [p features, p features]
        Holds empirical Spearman correlation values.
    - p_mask : array of shape [p features, p features]
        Mask for those p-values which passed the FDR correction with alpha=0.05
    - p_vals : array of shape [p features, p features]
        FDR corrected p-values
    """
    n, p = rX.shape
    # little trick to avoid having many ifs everywhere in code
    prec_mask = np.ones((p, p), dtype=np.bool)
    if prec is not None:
        prec_mask = prec != 0

    # calculate empirical Spearman from data
    rs = np.corrcoef(rX, rowvar=0)
    # get permuted Spearman values, in parallel
    r_perms = np.array(Parallel(n_jobs=-1,backend='threading')
                       (delayed(perm_r)(rX, i) for i in xrange(perm_num)))
    
    # GPD APPROXIMATION OF P-VALUES IN PARALLEL
    pGPDs = np.ones((p,p))
    # we should only each value once, i.e. the cells below the diagonal    
    lower_tri_ind = np.tril_indices(p, k=-1)
    lower_tri_mask = np.array(np.tril(np.ones((p, p)), k=-1), dtype=np.bool)
    pGPDs[lower_tri_ind] = np.array(Parallel(n_jobs=-1,backend='threading')(delayed(perm_gpd)
                           (prec_mask[lower_tri_ind[0][i], lower_tri_ind[1][i]],
                            rs[lower_tri_ind[0][i], lower_tri_ind[1][i]],
                            r_perms[:, lower_tri_ind[0][i], lower_tri_ind[1][i]]) 
                            for i in xrange(lower_tri_ind[0].shape[0])))    
    
    # correction for multiple testing, make sure that we only include each 
    # p-value once not twice (symmetric rs), by combining two numpy mask arrays: 
    #   - the lower triangular rs matrix
    #   - prec_mask, i.e. p-vals whose precision is not 0
    ps_to_check = lower_tri_mask * prec_mask
    ps = multicor(pGPDs[ps_to_check], method=mc_method, alpha=float(mc_alpha))
    # make Boolean and corrected p-val matrices
    p_mask = np.zeros((p, p), dtype=np.bool)
    p_mask[ps_to_check] = ps[0]
    p_mask = mirror_lower_triangle(p_mask)        
    p_vals = np.ones((p, p))
    p_vals[ps_to_check] = ps[1]
    p_vals = mirror_lower_triangle(p_vals)
    return rs, p_vals, p_mask


def get_shuffle(seq):
    np.random.shuffle(seq)
    return seq


def perm_r(rX, p):
    """
    Helper function to make the permuted Spearman calculation parallel
    """
    # we need to seed all permutation because of joblib/unix implementation
    np.random.seed(p)
    rX_local = np.copy(rX)
    np.apply_along_axis(get_shuffle, 0, rX_local)
    return np.corrcoef(rX_local, rowvar=0)


def perm_gpd(prec_mask, r_bench, r_perm):
    """
    Helper function to gets GPD approximated p-values in parallel. 
    """
    # if this is a real edge in the graphical model, calculate p-value, 
    # if GL = None, all prec_mask is True so we test every lower tri values
    if prec_mask:
        if r_bench < 0:
            r_bench = -r_bench
            r_perm = -r_perm
        pGPD = gpd.est(r_bench, r_perm)
        # ocasionally the gdp.est returns a NaN for some reason
        if np.isnan(pGPD):
            pGPD = 1
    else:
        pGPD = 1
    return pGPD


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
            m[r,c] = m[c,r]
    return m

