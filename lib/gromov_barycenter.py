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
        log=False, init_C=None, init_X=None, init_T=None, random_state=None, stop_criterion='barycenter',
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

    init_T : list of array-like, optional
        Warm-start couplings, one ``(N, n_s)`` plan per input. Used as ``G0``
        on the first inner PFGW solve (and as the first ``gw_restarts``
        candidate). Later barycenter iterations reuse the latest plans.
        Default ``None`` keeps the original random ``G0`` initialization.
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

    T_warm = None
    if init_T is not None:
        T_warm = [np.asarray(Ts, dtype=float).copy() for Ts in init_T]
        if len(T_warm) != S:
            raise ValueError(f"init_T length {len(T_warm)} != n_inputs {S}")
        for s, T0 in enumerate(T_warm):
            if T0.ndim != 2 or T0.shape[0] != N or T0.shape[1] != Cs[s].shape[0]:
                raise ValueError(
                    f"init_T[{s}] shape {T0.shape} incompatible with "
                    f"N={N}, n_s={Cs[s].shape[0]}"
                )

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
            G0_s = T_warm[s] if T_warm is not None else None
            if gw_restarts <= 1:
                T_best = partial_fused_gromov_ver1(
                    M_s, C, Cs[s], p, ps[s], Lambda_list[s], alpha=alpha, G0=G0_s,
                    numItermax_gw=max_iter, tol=1e-5, log=False, verbose=verbose,
                    seed=gw_seed_start, **gw_kwargs)
                
                T_best_diff = abs(np.max(T_best) - np.min(T_best))
                if T_best_diff < np.sum(ps[s]) / (T_best.shape[0] * T_best.shape[1]):
                    fails += 1
            else:
                T_best, best_cost = None, None
                for r in range(gw_restarts):
                    seed = gw_seed_start + r
                    G0_r = G0_s if (r == 0 and G0_s is not None) else None
                    T = partial_fused_gromov_ver1(
                        M_s, C, Cs[s], p, ps[s], Lambda_list[s], alpha=alpha, G0=G0_r,
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
        if init_T is not None:
            T_warm = [np.asarray(Ts, dtype=float).copy() for Ts in T]

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


def lambda_decay_schedule(lambda_init=1.0, lambda_target=0.01, decay_steps=5):
    """Geometric λ sequence from ``lambda_init`` down to ``lambda_target``.

    The uniform quotient is ``q = (lambda_init / lambda_target) ** (1 / decay_steps)``.
    Starting from ``lambda_current = lambda_init``, each subsequent value is
    ``max(lambda_target, lambda_current / q)``. This yields ``decay_steps + 1``
    values when ``decay_steps >= 1``. ``decay_steps == 0`` returns
    ``[lambda_target]`` only.
    """
    lam0 = float(lambda_init)
    lamT = float(lambda_target)
    steps = int(decay_steps)
    if lam0 <= 0.0 or lamT <= 0.0:
        raise ValueError("lambda_init and lambda_target must be positive")
    if lamT > lam0:
        raise ValueError("lambda_target must be <= lambda_init")
    if steps < 0:
        raise ValueError("decay_steps must be >= 0")
    if steps == 0 or np.isclose(lam0, lamT):
        return [lamT]
    q = (lam0 / lamT) ** (1.0 / steps)
    schedule = []
    lam = lam0
    for _ in range(steps + 1):
        schedule.append(float(lam))
        if lam <= lamT or np.isclose(lam, lamT):
            break
        lam = max(lamT, lam / q)
    schedule[-1] = lamT
    return schedule


def _pfgw_loss_and_masses(C, X, T, Cs, Ys, p, ps, Lambda_list, lambdas, alpha):
    """Weighted PFGW loss sum and per-input transported mass."""
    total = 0.0
    masses = []
    p = np.asarray(p, dtype=float)
    for s in range(len(Cs)):
        M_s = dist(X, Ys[s])
        d, penalty = PFGW_dist_with_penalty(
            M_s, C, Cs[s], T[s], p, ps[s], Lambda_list[s], alpha=alpha
        )
        total += float(lambdas[s]) * (float(d) + float(penalty))
        masses.append(float(np.asarray(T[s]).sum()))
    return float(total), masses


def pfgw_barycenters_lambda_converge(
        N, Ys, Cs, ps=None, p=None, lambdas=None, alpha=0.5,
        lambda_init=1.0, lambda_target=0.01, decay_steps=5,
        loss_fun='square_loss', max_iter=1000, tol=1e-9, verbose=False,
        init_C=None, init_X=None, init_T=None, random_state=None,
        stop_criterion='barycenter', gw_restarts=1, gw_seed_start=0,
        debug=False, accumulate_fails_limit=10, progress=True, **kwargs):
    r"""
    PFGW barycenter with geometric decay of the partial-matching penalty λ.

    For each λ on the schedule from :func:`lambda_decay_schedule`, a fully
    converged PFGW barycenter is computed. The encodings ``(C, X)`` and
    couplings ``log['T']`` warm-start the next (smaller) λ. The returned
    barycenter is the one obtained at ``lambda_target``.

    Returns
    -------
    C_center_final : ndarray, shape (N, N)
    X_center_final : ndarray, shape (N, d)
    log_final : dict
        Same keys as :func:`pfgw_barycenters` (``T``, ``p``, ``err``, ``X``),
        plus ``lambda_schedule``, ``lambda_converge_loss``,
        ``lambda_converge_transported_mass``, ``lambda_init``,
        ``lambda_target``, and ``decay_steps``.
    """
    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    schedule = lambda_decay_schedule(lambda_init, lambda_target, decay_steps)
    C_cur = None if init_C is None else np.asarray(init_C, dtype=float).copy()
    X_cur = None if init_X is None else np.asarray(init_X, dtype=float).copy()
    T_cur = None if init_T is None else [np.asarray(Ts, dtype=float).copy() for Ts in init_T]

    losses = []
    masses_hist = []
    log_final = None
    C_center_final = None
    X_center_final = None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, **_kwargs):
            return iterable

    pbar = tqdm(schedule, desc="λ convergence", unit="λ", disable=not progress)
    for step, lam in enumerate(pbar):
        Lambda_list = [float(lam)] * S
        C_center, X_center, log = pfgw_barycenters(
            N=N,
            Ys=Ys,
            Cs=Cs,
            ps=ps,
            p=p,
            Lambda_list=Lambda_list,
            lambdas=lambdas,
            alpha=alpha,
            loss_fun=loss_fun,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            log=True,
            init_C=C_cur,
            init_X=X_cur,
            init_T=T_cur,
            random_state=random_state,
            stop_criterion=stop_criterion,
            gw_restarts=gw_restarts,
            gw_seed_start=gw_seed_start,
            debug=debug,
            accumulate_fails_limit=accumulate_fails_limit,
            **kwargs,
        )
        p_used = log.get("p", p)
        loss_sum, masses = _pfgw_loss_and_masses(
            C_center, X_center, log["T"], Cs, Ys, p_used, ps, Lambda_list, lambdas, alpha
        )
        losses.append(loss_sum)
        masses_hist.append(masses)

        if progress:
            try:
                pbar.set_postfix_str(
                    f"λ={lam:.4g}  ΣFPGW={loss_sum:.6g}  "
                    f"masses=[{', '.join(f'{m:.4g}' for m in masses)}]"
                )
            except AttributeError:
                pass
            print(
                f"[λ converge {step + 1}/{len(schedule)}] λ={lam:.6g}  "
                f"FPGW loss sum={loss_sum:.6g}  "
                f"transported mass={masses}"
            )

        C_cur = np.asarray(C_center, dtype=float).copy()
        X_cur = np.asarray(X_center, dtype=float).copy()
        T_cur = [np.asarray(Ts, dtype=float).copy() for Ts in log["T"]]
        C_center_final = C_cur
        X_center_final = X_cur
        log_final = log

    log_final["lambda_schedule"] = list(schedule)
    log_final["lambda_converge_loss"] = list(losses)
    log_final["lambda_converge_transported_mass"] = [list(m) for m in masses_hist]
    log_final["lambda_init"] = float(lambda_init)
    log_final["lambda_target"] = float(lambda_target)
    log_final["decay_steps"] = int(decay_steps)
    return C_center_final, X_center_final, log_final


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