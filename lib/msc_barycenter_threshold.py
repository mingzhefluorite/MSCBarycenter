"""
Threshold / complete-graph construction for MSC FGW barycenters.

Shared by visualization, evaluation, and I/O so edge selection is identical
across notebooks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from lib.msc_barycenter_reconstruction import _inverse_relation_weight

EdgeCandidate = Tuple[float, int, int]  # (unscaled_weight, i, j)


def thresholded_barycenter_edge_candidates(
    C_center: np.ndarray,
    *,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
) -> List[EdgeCandidate]:
    """
    Finite positive off-diagonal pairs from ``C_center``, sorted by increasing
    unscaled weight (same ordering used by thresholded plots and evaluation).
    """
    C = np.asarray(C_center, dtype=float)
    if relation_kind == "shortestpath":
        C = np.nan_to_num(C, nan=np.inf, posinf=np.inf)
    else:
        C = np.nan_to_num(C, nan=0.0)

    n = C.shape[0]
    candidates: List[EdgeCandidate] = []
    for i in range(n):
        for j in range(i + 1, n):
            w_scaled = C[i, j]
            if not np.isfinite(w_scaled) or w_scaled <= 0:
                continue
            w = _inverse_relation_weight(
                float(w_scaled),
                relation_kind=relation_kind,
                relation_rescale=relation_rescale,
                relation_offset=relation_offset,
            )
            candidates.append((w, i, j))
    candidates.sort(key=lambda e: e[0])
    return candidates


def select_thresholded_edges(
    candidates: Sequence[EdgeCandidate],
    edge_fraction_pct: float,
) -> List[EdgeCandidate]:
    """Keep the smallest-weight ``edge_fraction_pct``% of candidates."""
    if not candidates:
        return []
    k = max(1, int(np.ceil(len(candidates) * float(edge_fraction_pct) / 100.0)))
    return list(candidates[:k])


def _decode_barycenter_nodes(
    G: nx.Graph,
    X_center: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    scalar_min: Optional[float] = None,
    scalar_range: Optional[float] = None,
    msc_type_order: Optional[Sequence[int]] = None,
) -> None:
    """Add barycenter nodes 0..n-1 with decoded attributes (mutates ``G``)."""
    X = np.asarray(X_center, dtype=float)
    n = X.shape[0]
    decode_scalar = (
        scalar_min is not None
        and scalar_range is not None
        and X.shape[1] >= 4
    )
    type_order = list(msc_type_order) if msc_type_order is not None else []
    type_col_start = 4 if decode_scalar else 3
    decode_type = bool(type_order) and X.shape[1] >= type_col_start + len(type_order)

    for i in range(n):
        attrs = {
            "x": float(min_x + X[i, 0] * domain_size),
            "y": float(min_y + X[i, 1] * domain_size),
            "z": float(min_z + X[i, 2] * domain_size),
        }
        if decode_scalar:
            s = float(scalar_min + X[i, 3] * scalar_range)
            attrs["height"] = s
            attrs["scalar_value"] = s
        if decode_type:
            type_idx = int(np.argmax(X[i, type_col_start : type_col_start + len(type_order)]))
            attrs["type"] = int(type_order[type_idx]) if 0 <= type_idx < len(type_order) else -1
        G.add_node(i, **attrs)


def build_thresholded_barycenter_graph(
    C_center: np.ndarray,
    X_center: np.ndarray,
    edge_fraction_pct: float,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    scalar_min: Optional[float] = None,
    scalar_range: Optional[float] = None,
    msc_type_order: Optional[Sequence[int]] = None,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
) -> Tuple[nx.Graph, List[EdgeCandidate], List[EdgeCandidate]]:
    """
    Build a sparse ``nx.Graph`` by keeping the smallest-weight
    ``edge_fraction_pct``% of candidate pairs from ``C_center``.

    Edge ``weight`` attributes are unscaled (physical) lengths so
    ``relation_matrix(..., sp_weighted=True)`` matches the input pipeline.

    Returns
    -------
    G, kept, candidates
    """
    candidates = thresholded_barycenter_edge_candidates(
        C_center,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
    )
    if not candidates:
        raise ValueError("No finite positive edges in C_center.")

    kept = select_thresholded_edges(candidates, edge_fraction_pct)

    G = nx.Graph()
    _decode_barycenter_nodes(
        G,
        X_center,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        domain_size=domain_size,
        scalar_min=scalar_min,
        scalar_range=scalar_range,
        msc_type_order=msc_type_order,
    )
    for w, u, v in kept:
        G.add_edge(u, v, weight=float(w))

    G.graph.update(
        {
            "relation_kind": relation_kind,
            "relation_rescale": float(relation_rescale),
            "relation_offset": float(relation_offset),
            "sp_weighted": bool(sp_weighted),
            "edge_fraction_pct": float(edge_fraction_pct),
            "n_candidates": len(candidates),
            "n_kept_edges": len(kept),
            "graph_kind": "thresholded",
        }
    )
    return G, kept, candidates


def build_complete_barycenter_graph(
    C_center: np.ndarray,
    X_center: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    scalar_min: Optional[float] = None,
    scalar_range: Optional[float] = None,
    msc_type_order: Optional[Sequence[int]] = None,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
) -> Tuple[nx.Graph, List[EdgeCandidate]]:
    """
    Build the *complete* (unthresholded) barycenter graph: all finite positive
    candidate edges from ``C_center``, with unscaled weights.

    This is the graph that thresholding later subsets.
    """
    G, kept, candidates = build_thresholded_barycenter_graph(
        C_center,
        X_center,
        edge_fraction_pct=100.0,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        domain_size=domain_size,
        scalar_min=scalar_min,
        scalar_range=scalar_range,
        msc_type_order=msc_type_order,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        sp_weighted=sp_weighted,
    )
    G.graph["edge_fraction_pct"] = 100.0
    G.graph["graph_kind"] = "complete"
    G.graph["n_kept_edges"] = len(kept)
    return G, candidates
