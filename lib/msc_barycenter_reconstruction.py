"""
Reconstruct MSC barycenter graphs from FGW outputs and related diagnostics.
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Sequence, Tuple

from lib.graph_relations import combined_relation_target_distances


class _UnionFind:
    """Disjoint-set union for Kruskal-style MST construction."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def _msc_barycenter_edge_candidates(
    C: np.ndarray,
    *,
    relation_kind: str = "shortestpath",
    edge_threshold: float = 0.5,
    direct_edge_max: float = 1.0,
) -> List[Tuple[float, int, int]]:
    """Return finite candidate edges (weight, i, j) sorted by increasing weight."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    relation_kind = relation_kind.lower()
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = C[i, j]
            if relation_kind == "shortestpath":
                if np.isfinite(w) and w > 0:
                    edges.append((float(w), i, j))
            elif relation_kind == "combined":
                if np.isfinite(w) and 0.0 < w <= direct_edge_max:
                    edges.append((float(w), i, j))
            elif w >= edge_threshold:
                edges.append((float(w), i, j))
    edges.sort(key=lambda e: e[0])
    return edges


def reconstruct_msc_barycenter_mst_graph(
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
    type_col_start: Optional[int] = None,
    relation_kind: str = "shortestpath",
    edge_threshold: float = 0.5,
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
) -> nx.Graph:
    """
    Reconstruct an MSC barycenter graph from FGW outputs via a Kruskal MST.

    Edge structure depends only on ``C_center``. ``X_center`` supplies node
    coordinates in columns 0–2; optional scalar and Morse-type blocks are
    decoded when requested and only affect node attributes on the output graph.

    Parameters
    ----------
    C_center : ndarray (n, n)
        Barycenter relation / shortest-path matrix.
    X_center : ndarray (n, d)
        Barycenter features. Columns 0–2 are normalized (x, y, z). Additional
        columns may hold a normalized scalar and/or MSC type features.
    scalar_min, scalar_range : float, optional
        When both are provided and ``X_center`` has at least 4 columns, column 3
        is decoded as ``(scalar - scalar_min) / scalar_range``.
    msc_type_order : sequence of int, optional
        Morse types for one-hot decoding (e.g. ``[0, 1, 2, -1]`` for
        minimum, saddle, maximum, regular). Decoded only when this is provided
        and ``X_center`` has enough columns for the type block.
    type_col_start : int, optional
        First column index of the type block. Defaults to 4 when scalar
        decoding is enabled, otherwise 3.
    relation_rescale, relation_offset : float
        Scaling applied to input relation matrices before FGW (stored on ``G.graph``
        for debug unscaling). For ``shortestpath``: ``C_scaled = C_raw * relation_rescale``.
    """
    C = np.asarray(C_center, dtype=float)
    X = np.asarray(X_center, dtype=float)
    relation_kind = relation_kind.lower()

    if relation_kind == "shortestpath":
        C = np.nan_to_num(C, nan=np.inf, posinf=np.inf)
    else:
        C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)

    n = C.shape[0]
    decode_scalar = (
        scalar_min is not None
        and scalar_range is not None
        and X.shape[1] >= 4
    )
    type_order = list(msc_type_order) if msc_type_order is not None else []
    decode_type = bool(type_order)
    if decode_type and type_col_start is None:
        type_col_start = 4 if decode_scalar else 3
    type_col_end = (
        type_col_start + len(type_order)
        if decode_type and type_col_start is not None
        else None
    )
    if decode_type and type_col_end is not None and X.shape[1] < type_col_end:
        decode_type = False

    G = nx.Graph()
    for i in range(n):
        node_attrs = {
            "x": float(min_x + X[i, 0] * domain_size),
            "y": float(min_y + X[i, 1] * domain_size),
            "z": float(min_z + X[i, 2] * domain_size),
        }
        if decode_scalar:
            scalar_value = float(scalar_min + X[i, 3] * scalar_range)
            node_attrs["height"] = scalar_value
            node_attrs["scalar_value"] = scalar_value
        if decode_type:
            type_idx = int(np.argmax(X[i, type_col_start:type_col_end]))
            node_type = (
                int(type_order[type_idx])
                if 0 <= type_idx < len(type_order)
                else -1
            )
            node_attrs["type"] = node_type
        G.add_node(i, **node_attrs)

    candidates = _msc_barycenter_edge_candidates(
        C, relation_kind=relation_kind, edge_threshold=edge_threshold
    )

    uf = _UnionFind(n)
    for w, u, v in candidates:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            G.add_edge(u, v, weight=w, edge_pass="mst", artifact_alarm=False)

    G.graph["relation_kind"] = relation_kind
    G.graph["relation_rescale"] = float(relation_rescale)
    G.graph["relation_offset"] = float(relation_offset)
    G.graph["sp_weighted"] = bool(sp_weighted)
    return G


# Backward-compatible alias (deprecated).
reconstruct_msc_degree_constrained_graph = reconstruct_msc_barycenter_mst_graph


def verify_msc_barycenter_reconstruction(
    G: nx.Graph,
    C_center: np.ndarray,
    *,
    alpha_fgw: float,
    relation_kind: str = "combined",
    direct_edge_max: float = 1.0,
) -> np.ndarray:
    """
    Compare reconstructed shortest-path distances to decoded ``C_center``.

    Returns ``reconstruct_quality_center`` where off-diagonal entries are
    ``sp_dist / C_target`` (values closer to 1 indicate better agreement).
    ``C_center`` entries are already normalized distances, so no extra
    rescaling is applied to the shortest-path matrix.
    """
    relation_kind = relation_kind.lower()
    if relation_kind != "combined":
        raise NotImplementedError(
            f"verify_msc_barycenter_reconstruction supports relation_kind='combined' only; got {relation_kind!r}"
        )
    if alpha_fgw is None or alpha_fgw <= 0:
        raise ValueError("alpha_fgw must be a positive float")

    n = int(np.asarray(C_center).shape[0])
    sp_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    sp_dist = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        row = sp_lengths.get(i, {i: 0.0})
        for j, d in row.items():
            sp_dist[i, j] = float(d)

    C_target = combined_relation_target_distances(
        C_center, alpha_fgw=alpha_fgw, direct_edge_max=direct_edge_max
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        reconstruct_quality_center = sp_dist / C_target
    reconstruct_quality_center[~np.isfinite(C_target) | (C_target <= 0)] = np.nan
    np.fill_diagonal(reconstruct_quality_center, 1.0)
    return reconstruct_quality_center


def reconstruct_msc_connectivity_based_graph(
    C_center: np.ndarray,
    X_center: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    height_min: float,
    height_range: float,
    max_degree: float,
    relation_kind: str = "combined",
    direct_edge_max: float = 1.0,
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
    alpha_fgw: Optional[float] = None,
) -> nx.Graph:
    """
    Reconstruct an MSC barycenter graph from a combined-relation ``C_center``.

    ``C_center`` encodes edge structure for the barycenter nodes:

    - ``C[i, j] < direct_edge_max`` (default 1): direct pair; value is the
      normalized Euclidean / shortest-path distance.
    - ``C[i, j] > direct_edge_max``: indirect pair; value is at least
      ``1/alpha_fgw`` plus the underlying normalized distance.

    Reconstruction keeps **all** direct edges (``0 < C[i, j] < direct_edge_max``).
    Indirect entries are not turned into edges here.

    Each node carries ``height`` (from the FGW feature block) and a uniform
    ``target_degree`` cap (``max_degree`` from input graphs) for downstream use.
    """
    C = np.asarray(C_center, dtype=float)
    X = np.asarray(X_center, dtype=float)
    relation_kind = relation_kind.lower()
    if relation_kind != "combined":
        raise NotImplementedError(
            f"reconstruct_msc_connectivity_based_graph supports relation_kind='combined' only; got {relation_kind!r}"
        )

    C = np.nan_to_num(C, nan=np.inf, posinf=np.inf)
    np.fill_diagonal(C, 0.0)

    n = C.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(
            i,
            x=float(min_x + X[i, 0] * domain_size),
            y=float(min_y + X[i, 1] * domain_size),
            z=float(min_z + X[i, 2] * domain_size),
            height=float(height_min + X[i, 3] * height_range),
            target_degree=float(max_degree),
        )

    for i in range(n):
        for j in range(i + 1, n):
            w = C[i, j]
            if np.isfinite(w) and 0.0 < w < direct_edge_max:
                G.add_edge(
                    i,
                    j,
                    weight=float(w),
                    edge_pass="direct",
                    artifact_alarm=False,
                )

    G.graph["relation_kind"] = relation_kind
    G.graph["relation_rescale"] = float(relation_rescale)
    G.graph["relation_offset"] = float(relation_offset)
    G.graph["sp_weighted"] = bool(sp_weighted)
    if alpha_fgw is not None:
        G.graph["alpha_fgw"] = float(alpha_fgw)
    G.graph["direct_edge_max"] = float(direct_edge_max)
    return G


def _inverse_relation_weight(
    weight_scaled: float,
    *,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
) -> float:
    """
    Undo relation-matrix scaling used before FGW so weights match physical units.

    ``shortestpath``: ``w_raw = w_scaled / relation_rescale``
    ``weighted_adjacency``: ``w_raw = (w_scaled - 1) / relation_rescale + relation_offset``
    """
    if not np.isfinite(weight_scaled):
        return float("nan")
    scale = relation_rescale if relation_rescale != 0 else 1.0
    kind = relation_kind.lower()
    if kind in ("shortestpath", "combined"):
        return float(weight_scaled / scale)
    if kind == "weighted_adjacency":
        return float((weight_scaled - 1.0) / scale + relation_offset)
    return float(weight_scaled)


def _relation_scaling_from_graph(
    G: nx.Graph,
    *,
    relation_rescale: Optional[float] = None,
    relation_offset: Optional[float] = None,
    relation_kind: Optional[str] = None,
) -> Tuple[str, float, float]:
    kind = relation_kind if relation_kind is not None else G.graph.get("relation_kind", "shortestpath")
    rescale = relation_rescale if relation_rescale is not None else G.graph.get("relation_rescale", 1.0)
    offset = relation_offset if relation_offset is not None else G.graph.get("relation_offset", 0.0)
    return str(kind).lower(), float(rescale), float(offset)


def _node_xyz_array(G: nx.Graph, n) -> np.ndarray:
    return np.array([G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]], dtype=float)


def msc_barycenter_weight_euclidean_differences(
    G: nx.Graph,
    *,
    relation_rescale: Optional[float] = None,
    relation_offset: Optional[float] = None,
    relation_kind: Optional[str] = None,
) -> np.ndarray:
    """
    For each edge, return ``weight_raw - euclidean_distance`` (3D Euclidean from node x,y,z).

    Edge ``weight`` attributes are inverse-scaled using ``relation_rescale`` / ``relation_offset``
    (from arguments or ``G.graph``) before comparison.
    """
    info = msc_barycenter_weight_euclidean_debug(
        G,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        relation_kind=relation_kind,
    )
    return info["differences"]


def msc_barycenter_weight_euclidean_debug(
    G: nx.Graph,
    *,
    relation_rescale: Optional[float] = None,
    relation_offset: Optional[float] = None,
    relation_kind: Optional[str] = None,
) -> dict:
    """
    Per-edge comparison of **unscaled** relation weight vs 3D Euclidean distance.

    Returns
    -------
    dict with keys:
        differences : ndarray (n_edges,) — weight_raw - euclidean
        weights_scaled : ndarray — stored edge weights from FGW / reconstruction
        weights : ndarray — unscaled (raw) weights
        euclidean : ndarray
        records : list of dict per edge
    """
    kind, rescale, offset = _relation_scaling_from_graph(
        G,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        relation_kind=relation_kind,
    )

    records = []
    for u, v, data in G.edges(data=True):
        pu = _node_xyz_array(G, u)
        pv = _node_xyz_array(G, v)
        eucl = float(np.linalg.norm(pu - pv))
        weight_scaled = float(data.get("weight", np.nan))
        weight_raw = _inverse_relation_weight(
            weight_scaled,
            relation_kind=kind,
            relation_rescale=rescale,
            relation_offset=offset,
        )
        diff = weight_raw - eucl
        records.append(
            {
                "u": u,
                "v": v,
                "weight_scaled": weight_scaled,
                "weight": weight_raw,
                "euclidean": eucl,
                "difference": diff,
                "abs_difference": abs(diff) if np.isfinite(diff) else np.nan,
                "edge_pass": data.get("edge_pass"),
                "artifact_alarm": data.get("artifact_alarm", False),
            }
        )
    differences = np.array([r["difference"] for r in records], dtype=float)
    weights_scaled = np.array([r["weight_scaled"] for r in records], dtype=float)
    weights = np.array([r["weight"] for r in records], dtype=float)
    euclidean = np.array([r["euclidean"] for r in records], dtype=float)
    return {
        "differences": differences,
        "weights_scaled": weights_scaled,
        "weights": weights,
        "euclidean": euclidean,
        "records": records,
        "relation_kind": kind,
        "relation_rescale": rescale,
        "relation_offset": offset,
    }
