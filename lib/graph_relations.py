"""
Pairwise relation matrices and scaling for MSC / graph barycenter pipelines.
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Tuple


def relation_matrix(
    G: nx.Graph,
    kind: str = "adjacency",
    *,
    nodelist: Optional[List] = None,
    laplacian_base: str = "adjacency",
    laplacian_normalization: str = "none",
    tau: float = 1.0,
    sp_weighted: bool = True,
    sp_affinity: str = "exp",
    sp_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, List]:
    """
    Build relation matrices for MSC / graph barycenter pipelines.

    kind:
      - ``adjacency``: 0/1 adjacency
      - ``shortestpath``: all-pairs shortest paths (raw; rescale with
        ``apply_relation_scaling`` before FGW)
      - ``laplacian``: heat kernel ``exp(-tau * L)``

    For ``shortestpath`` on MSC graphs, use ``sp_weighted=True`` (default) so
    Floyd–Warshall sums **3D Euclidean edge lengths** already stored in
    ``G.edges[].weight``. With ``sp_weighted=False``, paths are **hop counts**
    (integers) and are not comparable to physical Euclidean distances.

    For the combined adjacency + shortest-path relation, build raw shortest-path
    matrices, rescale them with ``apply_relation_scaling``, then call
    ``apply_combined_relation``.
    """
    if nodelist is None:
        nodelist = list(G.nodes())

    kind = kind.lower()
    laplacian_base = laplacian_base.lower()
    laplacian_normalization = laplacian_normalization.lower()
    sp_affinity = sp_affinity.lower()
    sp_weight = "weight" if sp_weighted else None

    if kind == "adjacency":
        A = nx.to_numpy_array(G, nodelist=nodelist, dtype=float, weight="weight")
        A = (A != 0).astype(float)
        return A, nodelist

    if kind == "shortestpath":
        D = nx.floyd_warshall_numpy(G, nodelist=nodelist, weight=sp_weight)
        return D, nodelist

    if kind != "laplacian":
        raise ValueError("kind must be one of: adjacency, shortestpath, laplacian")

    if laplacian_base == "adjacency":
        A = nx.to_numpy_array(G, nodelist=nodelist, dtype=float, weight="weight")
    elif laplacian_base == "shortestpath":
        D = nx.floyd_warshall_numpy(G, nodelist=nodelist, weight=sp_weight)
        finite = np.isfinite(D) & (D > 0)
        if sp_affinity == "exp":
            if sp_sigma is None:
                sp_sigma = np.median(D[finite]) if np.any(finite) else 1.0
            A = np.zeros_like(D, dtype=float)
            A[finite] = np.exp(-D[finite] / max(sp_sigma, 1e-12))
        elif sp_affinity == "inverse":
            A = np.zeros_like(D, dtype=float)
            A[finite] = 1.0 / (D[finite] + 1e-12)
        else:
            raise ValueError("sp_affinity must be 'exp' or 'inverse'")
        np.fill_diagonal(A, 0.0)
    else:
        raise ValueError("laplacian_base must be 'adjacency' or 'shortestpath'")

    if laplacian_normalization == "none":
        deg = np.sum(A, axis=1)
        L = np.diag(deg) - A
    elif laplacian_normalization == "sym":
        deg = np.sum(A, axis=1)
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        Dm = np.diag(inv_sqrt)
        L = np.eye(A.shape[0]) - Dm @ A @ Dm
    elif laplacian_normalization == "rw":
        deg = np.sum(A, axis=1)
        with np.errstate(divide="ignore"):
            inv = np.where(deg > 0, 1.0 / deg, 0.0)
        Dm = np.diag(inv)
        L = np.eye(A.shape[0]) - Dm @ A
    else:
        raise ValueError("laplacian_normalization must be 'none', 'sym', or 'rw'")

    try:
        import scipy.linalg as sla

        H = sla.expm((-tau) * L)
    except Exception:
        w, V = np.linalg.eigh(L)
        H = (V * np.exp(-tau * w)) @ V.T
    return H, nodelist


def compute_relation_rescale(
    graphs: List[nx.Graph],
    relation_kind: str = "shortestpath",
    *,
    sp_weighted: bool = True,
    relation_offset: float = 0.0,
) -> Tuple[float, float, dict]:
    """
    Compute ``(relation_rescale, relation_offset)`` matching contour-tree FGW.

    For ``shortestpath``: ``rescale = 1 / max(all-pairs shortest path)`` over
    input graphs; ``offset = 0``. Each matrix is then multiplied by ``rescale``.

    Returns diagnostics (max raw distance per graph, hop vs weighted if applicable).
    """
    relation_kind = relation_kind.lower()
    diag: dict = {"relation_kind": relation_kind, "sp_weighted": sp_weighted}

    if relation_kind in ("shortestpath", "combined"):
        max_d = 0.0
        per_graph = []
        for i, G in enumerate(graphs):
            D, _ = relation_matrix(G, kind="shortestpath", sp_weighted=sp_weighted)
            finite = np.isfinite(D) & (D > 0)
            gmax = float(np.max(D[finite])) if np.any(finite) else 1.0
            per_graph.append(gmax)
            max_d = max(max_d, gmax)
        diag["max_dist_per_graph"] = per_graph
        diag["max_dist_global"] = max_d
        rescale = 1.0 / max_d if max_d > 0 else 1.0
        return rescale, relation_offset, diag

    if relation_kind == "weighted_adjacency":
        max_nonzeros, min_nonzeros = [], []
        for G in graphs:
            WA, _ = relation_matrix(G, kind="adjacency")
            nz = WA[WA > 0]
            max_nonzeros.append(float(np.max(nz)) if nz.size else 1.0)
            min_nonzeros.append(float(np.min(nz)) if nz.size else 0.0)
        span = max(max_nonzeros) - min(min_nonzeros)
        diag["max_nonzeros"] = max_nonzeros
        diag["min_nonzeros"] = min_nonzeros
        rescale = 1.0 / span if span > 0 else 1.0
        offset = min(min_nonzeros)
        return rescale, offset, diag

    diag["note"] = "identity scaling"
    return 1.0, relation_offset, diag


def apply_relation_scaling(
    C: np.ndarray,
    relation_kind: str,
    relation_rescale: float,
    relation_offset: float = 0.0,
) -> np.ndarray:
    """Apply the same scaling used before FGW (in-place on a copy)."""
    C = np.asarray(C, dtype=float).copy()
    kind = relation_kind.lower()
    if kind == "weighted_adjacency":
        edge_mask = C > 0
        C[edge_mask] = (C[edge_mask] - relation_offset) * relation_rescale
        C[edge_mask] += 1.0
    elif kind == "shortestpath":
        finite = np.isfinite(C) & (C > 0)
        if np.any(finite):
            C[finite] = C[finite] * relation_rescale
    return C


def apply_combined_relation(
    D_scaled: np.ndarray,
    A: np.ndarray,
    alpha_fgw: float,
) -> np.ndarray:
    """
    Combine rescaled shortest-path distances with binary adjacency.

    ``C = (1 - A) / alpha_fgw + D_scaled``

    Call only **after** ``apply_relation_scaling`` so every graph uses the same
    global shortest-path extent. Direct pairs lie in ``[0, 1]``; indirect pairs
    lie in ``[1/alpha_fgw, 1/alpha_fgw + 1]``.
    """
    if alpha_fgw is None or alpha_fgw <= 0:
        raise ValueError("alpha_fgw must be a positive float")
    D_scaled = np.asarray(D_scaled, dtype=float)
    A = np.asarray(A, dtype=float)
    if D_scaled.shape != A.shape:
        raise ValueError("D_scaled and A must have the same shape")
    indirect_offset = 1.0 / alpha_fgw
    C = (1.0 - A) * indirect_offset + D_scaled
    np.fill_diagonal(C, 0.0)
    return C


def combined_relation_target_distances(
    C_center: np.ndarray,
    *,
    alpha_fgw: float,
    direct_edge_max: float = 1.0,
) -> np.ndarray:
    """
    Decode a combined-relation matrix into pure normalized distances.

    Direct pairs (``C <= direct_edge_max``) are unchanged. Indirect pairs
    (``C > direct_edge_max``) have the ``1/alpha_fgw`` offset removed.
    """
    if alpha_fgw is None or alpha_fgw <= 0:
        raise ValueError("alpha_fgw must be a positive float")
    C = np.asarray(C_center, dtype=float)
    C_target = np.nan_to_num(C, nan=np.inf, posinf=np.inf).copy()
    indirect_offset = 1.0 / alpha_fgw
    indirect_mask = C_target > direct_edge_max
    C_target[indirect_mask] -= indirect_offset
    np.fill_diagonal(C_target, 0.0)
    return C_target
