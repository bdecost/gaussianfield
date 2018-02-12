""" gaussianfield.py

Python implementation of the Gaussian Field harmonic classifier from Zhu 2003 (ICML)
"""

import numpy as np

def combinatorial_laplacian(W):
    """ compute the combinatorial laplacian given edge weight matrix W """
    # the degree matrix -- sum up edge weights for each node
    D = np.diag(np.sum(W, axis=1))

    # the combinatorial laplacian
    laplacian = D - W
    return laplacian


def solve(W, labels, labeled):
    """  Compute solution to the harmonic function from Zhu 2003 (ICML) 

    Careful: the workshop paper has an error in equation 3
    Follow equation 5 from the ICML conference paper instead

    Args:
        W: nxn matrix of edge weights
        labels: class label array (one-hot encoding)
        labeled: n-element indicator vector for the labeled datapoints

    Returns:
        field: Gaussian Field values at unlabeled points
        delta_uu_inv: inverted Laplacian submatrix for unlabeled points
    """
    
    laplacian = combinatorial_laplacian(W)

    # partition the laplacian into labeled and unlabeled blocks...
    laplacian_uu = laplacian[np.ix_(~labeled, ~labeled)]
    W_ul = W[np.ix_(~labeled, labeled)]
    
    # Naive solution to the gaussian field
    laplacian_uu_inv = np.linalg.inv(laplacian_uu)
    field = laplacian_uu_inv.dot(W_ul).dot(labels)
    
    return field, laplacian_uu_inv


def expected_risk(field, Linv):
    """ Compute the expected risk of the classifier f+(k), i.e. after adding each potential query k

    Args:
        field: Gaussian Field values at unlabeled points
        Linv: inverted Laplacian matrix for unlabeled points

    Returns: 
        risk: vector with one entry for each potential query
    """

    # translate Zhu's vectorized matlab code from active_learning.m here...

    U, nclasses = field.shape
    
    # divide kth column-wise by diagonal elements
    nG = Linv / np.diag(Linv)[:,np.newaxis]

    # for each possible label
    risk = np.zeros(U)
    for yk in range(nclasses):
        maxfplus = np.zeros((U,U))
        # for each potential prediction
        for c in range(nclasses):
            yk_c = yk == c

            # compute fplus for all k
            fplus = field[:,c] + (yk_c - field[:,c])[:,np.newaxis] * nG
            np.maximum(fplus, maxfplus, out=maxfplus)
        risk = risk + np.sum(1+maxfplus) * field[:,yk].T
    return risk
