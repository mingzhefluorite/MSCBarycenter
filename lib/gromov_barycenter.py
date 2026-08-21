import numpy as np 
import matplotlib.pyplot as plt
import ot 
import os

import numba as nb
#os.chdir(parent_path)

# import our gw, (bounded) pgw, mpgw 

from .gromov import (
    gromov_wasserstein,
    fused_gromov_wasserstein,
    cost_matrix_d,
    tensor_dot_param,
    tensor_dot_func,
    gwgrad_partial1,
    partial_gromov_wasserstein,
    partial_gromov_ver1,
    partial_fused_gromov_ver1,
    GW_dist,
    MPGW_dist,
    PGW_dist_with_penalty,
    PFGW_dist_with_penalty,
)

from pathlib import Path

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA

from sklearn.datasets import load_digits
from sklearn.manifold import MDS


from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg, line_search_armijo, solve_1d_linesearch_quad
from ot.utils import check_random_state, unif
from ot.backend import get_backend, NumpyBackend






# It is referenced from PythonOT (https://pythonot.github.io/)
def im2mat(img):
    """Converts and image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

# It is adapted from PythonOT (https://pythonot.github.io/)
def smacof_mds(C, dim, max_iter=3000, eps=1e-9,seed=3):
    """
    It is imported from PythonOT
    Returns an interpolated point cloud following the dissimilarity matrix C
    using SMACOF multidimensional scaling (MDS) in specific dimensioned
    target space

    Parameters
    ----------
    C : ndarray, shape (ns, ns)
        dissimilarity matrix
    dim : int
          dimension of the targeted space
    max_iter :  int
        Maximum number of iterations of the SMACOF algorithm for a single run
    eps : float
        relative tolerance w.r.t stress to declare converge

    Returns
    -------
    npos : ndarray, shape (R, dim)
           Embedded coordinates of the interpolated point cloud (defined with
           one isometry)
    """

    rng = np.random.RandomState(seed=seed)

    mds = manifold.MDS(
        dim,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity='precomputed',
        n_init=1,
        normalized_stress='auto')
    pos = mds.fit(C).embedding_

    nmds = manifold.MDS(
        2,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity="precomputed",
        random_state=rng,
        normalized_stress='auto',
        n_init=1)
    npos = nmds.fit_transform(C, init=pos)


    return npos

from sklearn.manifold import MDS


def MDS_test(dist_mat, d,seed=0):
    mds = MDS(n_components=d, dissimilarity='precomputed',random_state=seed, normalized_stress='auto')
    lower_dimensional_points = mds.fit_transform(dist_mat)
    return lower_dimensional_points

def rotation_2d(theta=0):
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
    ])

clf=PCA(n_components=2)






# It is referenced from PythonOT (https://pythonot.github.io/)
def gromov_barycenters(
        N, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        symmetric=True, armijo=False, max_iter=1000, tol=1e-9,
        stop_criterion='barycenter', warmstartT=False, verbose=False,
        log=False, init_C=None, random_state=None, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if stop_criterion not in ['barycenter', 'loss']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

    Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1e15  # either the error on 'barycenter' or 'loss'

    if warmstartT:
        T = [None] * S

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []

    while (err > tol and cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev = C
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [gromov_wasserstein(
                C, Cs[s], p, ps[s], loss_fun, symmetric=symmetric, armijo=armijo, G0=T[s],
                max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
        else:
            res = [gromov_wasserstein(
                C, Cs[s], p, ps[s], loss_fun, symmetric=symmetric, armijo=armijo, G0=None,
                max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]

        T = res

        # update barycenters
        if loss_fun == 'square_loss':
            C = pgw_update_square_loss_nb(p, lambdas, T, Cs)        

        # update convergence criterion
        #if stop_criterion == 'barycenter':
        err = nx.norm(C - Cprev)
        if log:
            log_['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p

        return C, log_
    else:
        return C


# This function is referenced from PythonOT
def update_square_loss(p, lambdas, T, Cs, nx=None):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        nx = get_backend(p, *T, *Cs)
    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s], Cs[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return tmpsum / ppt


def pgw_update_square_loss(p, lambdas, T, Cs):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    # Correct order mistake in Equation 14 in [12]
    tmpsum = sum([
        lambdas[s] * np.dot(
            np.dot(T[s], Cs[s]),
            T[s].T
        ) for s in range(len(T))
    ])
    
    ppt =sum([
        lambdas[s] * np.outer(T[s].sum(1), T[s].sum(1))
        for s in range(len(T))
    ])
    
    return tmpsum / ppt

@nb.njit(cache=True)
def pgw_update_square_loss_nb(p, lambdas, T, Cs):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of the GW
    barycenter problem in :ref:`[12]`:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (N, ns)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    # Correct order mistake in Equation 14 in [12]
    # Caller must pass T with full-zero rows/columns perturbed (see _perturb_zero_rows_columns).
    N, n = len(T), p.shape[0]
    tmpsum = np.zeros((n, n))
    ppt = np.zeros((n, n))
    for s in range(N):
        T1 = T[s].sum(1)
        tmpsum += lambdas[s] * np.dot(np.dot(T[s], Cs[s]), T[s].T)
        ppt += lambdas[s] * np.outer(T1, T1)
    return tmpsum / ppt


def pfgw_update_square_loss_nb(p, lambdas, T, Cs):
    """PFGW barycenter structure update. Same formula as PGW (structure-only part)."""
    return pgw_update_square_loss_nb(p, lambdas, T, Cs)


def _perturb_zero_rows_columns(T, eps=1e-15):
    """
    Add eps to full-zero rows and columns in each T[s] so the modified T remains a valid
    coupling. Returns a list of modified copies (does not mutate input).
    """
    T_mod = []
    for s in range(len(T)):
        T_s = T[s].copy()
        n, ns = T_s.shape
        T1 = T_s.sum(1)
        for i in range(n):
            if T1[i] == 0.0:
                for j in range(ns):
                    T_s[i, j] += eps / ns
        T_mod.append(T_s)
    return T_mod


def pgw_barycenters(
        N, Cs, ps=None, p=None, Lambda_list=None, lambdas=None, loss_fun='square_loss', max_iter=1000, tol=1e-9, verbose=False,
        log=False, init_C=None, random_state=None, stop_criterion='barycenter',
        gw_restarts=1, gw_seed_start=0, debug=False, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if stop_criterion not in ['barycenter']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

    Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = cost_matrix_d(xalea,xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1e15  # either the error on 'barycenter' or 'loss'

        
    if Lambda_list is None:
        c_max=[np.max(C) for C in Cs]
        Lambda_list=np.array(c_max)
        

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if debug:
            log_['debug'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []

    if debug:
        print("barycenter starts...")
        
    while (err > tol and cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev=C.copy()

        # get transport plans
        gw_kwargs = dict(kwargs)
        # Guard against accidentally forwarding barycenter-only args to the OT solver
        gw_kwargs.pop("gw_restarts", None)
        gw_kwargs.pop("gw_seed_start", None)
        gw_kwargs.pop("debug", None)
        res = []
        best_costs = []
        for s in range(S):
            if gw_restarts <= 1:
                T_best = partial_gromov_ver1(
                    C, Cs[s], p, ps[s], Lambda_list[s], G0=None,
                    numItermax_gw=max_iter, tol=1e-5, log=inner_log, verbose=verbose,
                    seed=gw_seed_start, **gw_kwargs)
            else:
                T_best, best_cost = None, None
                for r in range(gw_restarts):
                    seed = gw_seed_start + r
                    T = partial_gromov_ver1(
                        C, Cs[s], p, ps[s], Lambda_list[s], G0=None,
                        numItermax_gw=max_iter, tol=1e-5, log=inner_log, verbose=verbose,
                        seed=seed, **gw_kwargs)
                    dist, penalty = PGW_dist_with_penalty(C, Cs[s], T, p, ps[s], Lambda_list[s])
                    cost = dist + penalty
                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost
                        T_best = T
                best_costs.append(best_cost * lambdas[s])
            res.append(T_best)

        if debug:
            print('iter is', cpt)
            print('best_costs sum is', np.sum(best_costs))
       
        T = res
        T = _perturb_zero_rows_columns(T, eps=1e-15)

        # update barycenters
        if loss_fun == 'square_loss':
            C = pgw_update_square_loss_nb(p, lambdas, T, Cs)

        # elif loss_fun == 'kl_loss':
        #     C = update_kl_loss(p, lambdas, T, Cs, nx)
        
        if stop_criterion == 'barycenter':
            diff = C - Cprev
            err = nx.norm(diff)
        if log:
            log_['err'].append(err)
        
        if debug:
            print('err is', err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p

        return C, log_
    else:
        return C


def _pfgw_update_features(p, lambdas, T, Ys, eps=1e-15):
    """
    Update barycenter features from couplings. T[s] is (N, ns), Ys[s] is (ns, d).

    For partial matching, normalizes by total matched mass per node instead of p,
    so partially matched sources are not under-weighted. Uses eps when
    total_matched is negligible to avoid division by zero.
    """
    S = len(T)
    total_matched = sum(lambdas[s] * T[s].sum(1) for s in range(S))
    denom = np.maximum(total_matched, eps)
    inv_denom = np.nan_to_num(1.0 / denom, nan=1.0, posinf=1.0, neginf=1.0)
    list_features = [lambdas[s] * np.dot(T[s], Ys[s]) for s in range(S)]
    return sum(list_features) * inv_denom[:, np.newaxis]


def pfgw_barycenters(
        N, Ys, Cs, ps=None, p=None, Lambda_list=None, lambdas=None, alpha=0.5,
        loss_fun='square_loss', max_iter=1000, tol=1e-9, verbose=False,
        log=False, init_C=None, init_X=None, random_state=None, stop_criterion='barycenter',
        gw_restarts=1, gw_seed_start=0, debug=False, accumulate_fails_limit=10, **kwargs):
    r"""
    Returns the Partial Fused Gromov-Wasserstein barycenters of S structured objects
    with features (Ys[s], Cs[s]) and weights ps[s].

    Returns
    -------
    C : array-like, shape (N, N)
        Structure matrix of the barycenter
    X : array-like, shape (N, d)
        Feature matrix of the barycenter
    log : dict
        Only returned when log=True. Contains 'T', 'p', 'err', 'X'.
    """
    if stop_criterion not in ['barycenter']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter'}.")

    Cs = list_to_array(*Cs)
    Ys = list_to_array(*Ys)
    arr = [*Cs, *Ys]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    generator = check_random_state(random_state)
    d = Ys[0].shape[1]

    if init_C is None:
        xalea = generator.randn(N, 2)
        C = cost_matrix_d(xalea, xalea)
        C = np.asarray(C, dtype=float)
        C /= C.max()
    else:
        C = init_C

    if init_X is None:
        X = np.tile(np.mean(np.vstack(Ys), axis=0), (N, 1)) + 0.01 * generator.randn(N, d)
    else:
        X = init_X

    if Lambda_list is None:
        Lambda_list = np.array([np.max(Cs[s]) for s in range(S)])

    cpt = 0
    err = 1e15

    if log:
        log_ = {'err': []}

    if debug:
        print("pfgw barycenter starts...")
        
    fails = 0

    while err > tol and cpt < max_iter:
        Cprev = C.copy()
        Xprev = X.copy()

        gw_kwargs = dict(kwargs)
        gw_kwargs.pop("gw_restarts", None)
        gw_kwargs.pop("gw_seed_start", None)
        gw_kwargs.pop("debug", None)

        res = []
        for s in range(S):
            M_s = dist(X, Ys[s])
            if gw_restarts <= 1:
                T_best = partial_fused_gromov_ver1(
                    M_s, C, Cs[s], p, ps[s], Lambda_list[s], alpha=alpha, G0=None,
                    numItermax_gw=max_iter, tol=1e-5, log=False, verbose=verbose,
                    seed=gw_seed_start, **gw_kwargs)
                
                T_best_diff = abs(np.max(T_best) - np.min(T_best))
                if T_best_diff < np.sum(ps[s]) / (T_best.shape[0] * T_best.shape[1]):
                    fails += 1
            else:
                T_best, best_cost = None, None
                for r in range(gw_restarts):
                    seed = gw_seed_start + r
                    T = partial_fused_gromov_ver1(
                        M_s, C, Cs[s], p, ps[s], Lambda_list[s], alpha=alpha, G0=None,
                        numItermax_gw=max_iter, tol=1e-5, log=False, verbose=verbose,
                        seed=seed, **gw_kwargs)
                    dist_val, penalty = PFGW_dist_with_penalty(
                        M_s, C, Cs[s], T, p, ps[s], Lambda_list[s], alpha=alpha)
                    cost = dist_val + penalty
                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost
                        T_best = T
                
                T_best_diff = abs(np.max(T_best) - np.min(T_best))
                if T_best_diff < np.sum(ps[s]) / (T_best.shape[0] * T_best.shape[1]):
                    fails += 1
            res.append(T_best)

        T = res
        T = _perturb_zero_rows_columns(T, eps=1e-15)

        C = pfgw_update_square_loss_nb(p, lambdas, T, Cs)
        X = _pfgw_update_features(p, lambdas, T, Ys)
        
        if fails > accumulate_fails_limit:
            print("Reached accumulated fails to find a good transport plan, returning current best result")
            break

        err = np.linalg.norm(C - Cprev) + np.linalg.norm(X - Xprev)
        if log:
            log_['err'].append(err)

        if debug:
            print('iter', cpt, 'err', err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p
        log_['X'] = X
        return C, X, log_
    else:
        return C, X


def mpgw_barycenters(
        N, Cs, ps=None, p=None, lambdas=None,mass_list=None, loss_fun='square_loss', max_iter=1000, tol=1e-9, verbose=False,
        log=False, init_C=None, stop_criterion = 'barycenter', random_state=None, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if stop_criterion not in ['barycenter']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter'}.")

    Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = cost_matrix_d(xalea,xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1e15  # either the error on 'barycenter' or 'loss'

    # if warmstartT:
    #     T = [None] * S
        
    if mass_list is None:
        mass_list=np.zeros(K)
        for id_K in range(K):
            mass_list=np.min(p.sum(),ps[id_K].sum())
            

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []

    while (err > tol and cpt < max_iter):
        #print('cpt is',cpt)
        Cprev=C.copy()
        # get transport plans
        res = [partial_gromov_wasserstein(
                C, Cs[s], p, ps[s], mass_list[s], G0=None,
                numItermax_gw=max_iter, tol=1e-5, log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
        
       
        T = res
        T = _perturb_zero_rows_columns(T, eps=1e-15)
        # else:
        #     T = [output[0] for output in res]
        #     curr_loss = np.sum([output[1]['gw_dist'] for output in res])

        # update barycenters
        if loss_fun == 'square_loss':
            C = pgw_update_square_loss_nb(p, lambdas, T, Cs)

        # elif loss_fun == 'kl_loss':
        #     C = update_kl_loss(p, lambdas, T, Cs, nx)
        if stop_criterion == 'barycenter':
            err = nx.norm(C - Cprev)
            if log:
                log_['err'].append(err)


        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p

        return C, log_
    else:
        return C