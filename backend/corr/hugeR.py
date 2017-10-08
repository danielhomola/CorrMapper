import os
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

def hugeR(X, lambda_threshold):
    """
    This function computes the covariance matrix and the corresponding sparse 
    inverse covariance matrix of the numpy input matrix X, using the huge R 
    package by Liu et al.
    
    It transforms the variables in X into the nonparanormal family which allows
    then us to use the glasso algorithm to estimate the sparse inverse cov
    matrix. The lossy pre-screening isn't used here to speed up the
    calculations because we prefilter X so n ~ p. We test 30 lambda values for
    regularisation. The best model is selected using the 'stars' stability 
    approach with default threshold.
    
    For more details check docs and vignette: 
    https://cran.r-project.org/web/packages/huge/huge.pdf 
    http://r.meteo.uni.wroc.pl/web/packages/huge/vignettes/vignette.pdf
    """
    base = importr('base')
    # this allows us to send numpy to R directly, neat
    numpy2ri.activate()

    huge = importr('huge')
    X_npn = huge.huge_npn(X, npn_func="shrinkage")
    model = huge.huge(X_npn, nlambda=30, method='glasso', scr=False,
                      cov_output=True)
    model_stars = huge.huge_select(model, criterion="stars",
                                   stars_thresh=lambda_threshold)
    cov = np.array(base.as_matrix(model_stars.rx('opt.cov')[0]))
    prec = np.array(base.as_matrix(model_stars.rx('opt.icov')[0]))
    # network = np.array(base.as_matrix(model_stars.rx('refit')[0]))
    # we need to turn this off once we're done
    numpy2ri.deactivate()
    return cov, prec
