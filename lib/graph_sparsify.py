"""
Anisotropic edge-density sparsification for geometric graphs.

Given a dense / thresholded graph with node coordinates, score edges with an
anisotropic kernel over (midpoint, direction, log-length), keep local maxima,
enforce adaptive degree caps, then repair connectivity. Nodes are preserved
exactly; only the edge set is sparsified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx


EdgeKey = Tuple[Any, Any]
CrossingFn = Callable[
    [EdgeKey, Sequence[EdgeKey], Mapping[Any, np.ndarray]],
    int,
]


@dataclass
class SparsifyConfig:
    """Tunable parameters for ``sparsify_graph``."""

    h_mid: Optional[float] = None
    h_angle: float = 0.25
    h_len: float = 0.35
    beta: float = 1.0
    degree_cap_min: int = 2
    degree_cap_max: int = 4
    crossing_penalty: float = 0.0
    similarity_threshold: float = 0.05
    target_edge_ratio: Optional[float] = None
    target_edge_count: Optional[int] = None
    repair_budget: Optional[int] = None
    length_eps: float = 1e-12
    auto_bandwidth_quantile: float = 0.35
    conf_attr: str = "confidence"
    weight_attr: str = "weight"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeFeatureBundle:
    """Edge geometry extracted from a graph."""

    edges: List[EdgeKey]
    midpoints: np.ndarray  # (m, d)
    directions: np.ndarray  # (m, d)
    lengths: np.ndarray  # (m,)
    confidence: np.ndarray  # (m,)
    edge_index: Dict[EdgeKey, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_index:
            self.edge_index = {e: i for i, e in enumerate(self.edges)}


@dataclass
class SparsifyResult:
    """Output of ``sparsify_graph``."""

    graph: nx.Graph
    removed_edges: List[EdgeKey]
    edge_scores: Dict[EdgeKey, float]
    diagnostics: Dict[str, Any]
    kept_edges: List[EdgeKey]
    reconnect_edges: List[EdgeKey]
    features: EdgeFeatureBundle


# ---------------------------------------------------------------------------
# Positions / helpers
# ---------------------------------------------------------------------------


def node_positions_from_graph(
    G: nx.Graph,
    *,
    dims: Optional[Sequence[str]] = None,
) -> Dict[Any, np.ndarray]:
    """
    Read node coordinates from attributes.

    Tries ``(x, y, z)`` by default; falls back to ``(x, y)`` if ``z`` is absent.
    """
    if dims is None:
        sample = next(iter(G.nodes(data=True)))[1] if G.number_of_nodes() else {}
        if "z" in sample:
            dims = ("x", "y", "z")
        else:
            dims = ("x", "y")
    dims = tuple(dims)
    pos: Dict[Any, np.ndarray] = {}
    for n, data in G.nodes(data=True):
        try:
            pos[n] = np.asarray([float(data[d]) for d in dims], dtype=float)
        except KeyError as exc:
            raise KeyError(
                f"Node {n!r} missing coordinate attribute(s) {dims}: {exc}"
            ) from exc
    return pos


def normalize_undirected_edge(u: Any, v: Any) -> EdgeKey:
    """Canonical undirected edge key (order-independent for hashable nodes)."""
    try:
        return (u, v) if u <= v else (v, u)
    except TypeError:
        return (u, v) if str(u) <= str(v) else (v, u)


def _safe_unit(vec: np.ndarray, length_eps: float) -> Tuple[np.ndarray, float]:
    """Return (unit direction, length); zero-length edges get a zero direction."""
    length = float(np.linalg.norm(vec))
    if length <= length_eps:
        return np.zeros_like(vec, dtype=float), 0.0
    return (vec / length).astype(float), length


def _median_positive(values: np.ndarray, default: float = 1.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float(default)
    return float(np.median(vals))


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------


def compute_edge_features(
    G: nx.Graph,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
    *,
    conf_attr: str = "confidence",
    length_eps: float = 1e-12,
) -> EdgeFeatureBundle:
    """
    Extract midpoint, unit direction, length, and confidence for every edge.

    Zero-length edges get length ``length_eps`` for scoring stability and a
    zero direction vector (they contribute little under the angle kernel).
    """
    if pos is None:
        pos = node_positions_from_graph(G)

    edges: List[EdgeKey] = []
    midpoints: List[np.ndarray] = []
    directions: List[np.ndarray] = []
    lengths: List[float] = []
    confidence: List[float] = []

    for u, v, data in G.edges(data=True):
        key = normalize_undirected_edge(u, v)
        p_u = np.asarray(pos[u], dtype=float)
        p_v = np.asarray(pos[v], dtype=float)
        delta = p_v - p_u
        unit, length = _safe_unit(delta, length_eps)
        safe_len = max(length, length_eps)
        conf = float(data.get(conf_attr, 1.0))
        if not np.isfinite(conf) or conf < 0:
            conf = 1.0

        edges.append(key)
        midpoints.append(0.5 * (p_u + p_v))
        directions.append(unit)
        lengths.append(safe_len)
        confidence.append(conf)

    mid = (
        np.vstack(midpoints)
        if midpoints
        else np.zeros((0, next(iter(pos.values())).shape[0] if pos else 2), dtype=float)
    )
    direc = np.vstack(directions) if directions else np.zeros_like(mid)
    return EdgeFeatureBundle(
        edges=edges,
        midpoints=mid,
        directions=direc,
        lengths=np.asarray(lengths, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
    )


# ---------------------------------------------------------------------------
# Kernel similarity / density / score
# ---------------------------------------------------------------------------


def estimate_bandwidths(
    features: EdgeFeatureBundle,
    *,
    h_mid: Optional[float] = None,
    h_angle: float = 0.25,
    h_len: float = 0.35,
    quantile: float = 0.35,
) -> Tuple[float, float, float]:
    """
    Resolve kernel bandwidths.

    If ``h_mid`` is None, use a quantile of pairwise midpoint distances
    among a random sample of edges (stable default for MSC-scale graphs).
    """
    m = len(features.edges)
    if h_mid is None:
        if m <= 1:
            h_mid = 1.0
        else:
            # Sample O(min(m, 256)^2) pairs for a cheap scale estimate.
            rng = np.random.default_rng(0)
            idx = np.arange(m)
            if m > 256:
                idx = rng.choice(m, size=256, replace=False)
            pts = features.midpoints[idx]
            # Pairwise distances without forming a full m x m matrix over all edges.
            diff = pts[:, None, :] - pts[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            tri = d2[np.triu_indices(pts.shape[0], k=1)]
            tri = tri[np.isfinite(tri) & (tri > 0)]
            if tri.size == 0:
                h_mid = 1.0
            else:
                h_mid = float(np.sqrt(np.quantile(tri, np.clip(quantile, 0.05, 0.9))))
                h_mid = max(h_mid, 1e-6)
    return float(h_mid), float(h_angle), float(h_len)


def edge_kernel_similarity(
    features: EdgeFeatureBundle,
    *,
    h_mid: float,
    h_angle: float,
    h_len: float,
    length_eps: float = 1e-12,
) -> np.ndarray:
    """
    Anisotropic Gaussian-style edge-to-edge similarity matrix K (m x m).

    K(e, e') = exp(-||m-m'||^2 / (2 h_mid^2))
             * exp(-(1 - |u·u'|)^2 / (2 h_angle^2))
             * exp(-(log l - log l')^2 / (2 h_len^2))
    """
    m = len(features.edges)
    if m == 0:
        return np.zeros((0, 0), dtype=float)

    mid = features.midpoints
    direc = features.directions
    lengths = np.maximum(features.lengths, length_eps)

    h_mid2 = max(2.0 * float(h_mid) ** 2, 1e-18)
    h_ang2 = max(2.0 * float(h_angle) ** 2, 1e-18)
    h_len2 = max(2.0 * float(h_len) ** 2, 1e-18)

    # Midpoint term
    diff = mid[:, None, :] - mid[None, :, :]
    mid_term = np.exp(-np.sum(diff * diff, axis=-1) / h_mid2)

    # Direction term: |u·u'|; zero-direction edges get |dot|=0.
    dots = np.abs(direc @ direc.T)
    dots = np.clip(dots, 0.0, 1.0)
    ang_term = np.exp(-((1.0 - dots) ** 2) / h_ang2)

    # Log-length term
    log_l = np.log(lengths)
    dlog = log_l[:, None] - log_l[None, :]
    len_term = np.exp(-(dlog * dlog) / h_len2)

    return mid_term * ang_term * len_term


def compute_edge_density(
    K: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Local density rho(e) = sum_{e' != e} K(e,e') * c_{e'}."""
    conf = np.asarray(confidence, dtype=float)
    if K.size == 0:
        return np.zeros(0, dtype=float)
    # Zero diagonal so an edge does not boost itself.
    K_off = K.copy()
    np.fill_diagonal(K_off, 0.0)
    return K_off @ conf


def count_edge_crossings_placeholder(
    edge: EdgeKey,
    other_edges: Sequence[EdgeKey],
    pos: Mapping[Any, np.ndarray],
) -> int:
    """
    Placeholder crossing hook (always returns 0).

    Swap in ``count_edge_crossings_2d`` (or a custom callable) when geometric
    crossings should influence scoring / repair.
    """
    return 0


def _segments_properly_cross_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tol: float = 1e-12,
) -> bool:
    """True if open segments ab and cd properly intersect in 2D."""

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if abs(o1) < tol or abs(o2) < tol or abs(o3) < tol or abs(o4) < tol:
        return False  # treat endpoint/collinear touches as non-crossing
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def count_edge_crossings_2d(
    edge: EdgeKey,
    other_edges: Sequence[EdgeKey],
    pos: Mapping[Any, np.ndarray],
) -> int:
    """Count proper 2D geometric crossings between ``edge`` and ``other_edges``."""
    u, v = edge
    a = np.asarray(pos[u][:2], dtype=float)
    b = np.asarray(pos[v][:2], dtype=float)
    count = 0
    for e in other_edges:
        if e == edge:
            continue
        p, q = e
        if u in e or v in e:
            continue  # share a vertex → not a proper crossing
        c = np.asarray(pos[p][:2], dtype=float)
        d = np.asarray(pos[q][:2], dtype=float)
        if _segments_properly_cross_2d(a, b, c, d):
            count += 1
    return count


def _node_xy(graph: nx.Graph, node: Any) -> Optional[np.ndarray]:
    data = graph.nodes[node]
    x = data.get("x")
    y = data.get("y")
    if x is None or y is None:
        return None
    xy = np.asarray([x, y], dtype=float).reshape(-1)[:2]
    if xy.size < 2 or not np.all(np.isfinite(xy)):
        return None
    return xy


def graph_xy_positions(graph: nx.Graph) -> Dict[Any, np.ndarray]:
    """Finite ``(x, y)`` positions from node attributes, omitting incomplete nodes."""
    positions: Dict[Any, np.ndarray] = {}
    for node in graph.nodes():
        xy = _node_xy(graph, node)
        if xy is not None:
            positions[node] = xy
    return positions


def edge_has_proper_crossing_2d(
    edge: EdgeKey,
    other_edges: Sequence[EdgeKey],
    pos: Mapping[Any, np.ndarray],
) -> bool:
    """True if the open segment ``edge`` properly intersects any of ``other_edges``.

    Edges that share a vertex are not crossings. Incomplete coordinates are skipped.
    """
    u, v = edge
    if u not in pos or v not in pos:
        return False
    a = np.asarray(pos[u][:2], dtype=float)
    b = np.asarray(pos[v][:2], dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return False
    for other in other_edges:
        if other == edge:
            continue
        p, q = other
        if p == u or p == v or q == u or q == v:
            continue
        if p not in pos or q not in pos:
            continue
        c = np.asarray(pos[p][:2], dtype=float)
        d = np.asarray(pos[q][:2], dtype=float)
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)):
            continue
        if _segments_properly_cross_2d(a, b, c, d):
            return True
    return False


def geometric_planarity_allows_edge(
    graph: nx.Graph,
    u: Any,
    v: Any,
    *,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
) -> bool:
    """Whether straight-line ``uv`` avoids proper crossings with existing edges.

    Shared endpoints are allowed. Missing coordinates cannot prove a crossing,
    so the edge is allowed. Does not mutate ``graph``.
    """
    if pos is not None:
        return not edge_has_proper_crossing_2d((u, v), list(graph.edges()), pos)
    a = _node_xy(graph, u)
    b = _node_xy(graph, v)
    if a is None or b is None:
        return True
    for p, q in graph.edges():
        if p == u or p == v or q == u or q == v:
            continue
        c = _node_xy(graph, p)
        d = _node_xy(graph, q)
        if c is None or d is None:
            continue
        if _segments_properly_cross_2d(a, b, c, d):
            return False
    return True


def is_geometrically_planar_2d(
    graph: nx.Graph,
    *,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
) -> bool:
    """True iff no two edges properly intersect in the fixed ``(x, y)`` drawing."""
    positions = graph_xy_positions(graph) if pos is None else pos
    edges = list(graph.edges())
    for index, edge in enumerate(edges):
        if edge_has_proper_crossing_2d(edge, edges[index + 1 :], positions):
            return False
    return True


def score_edges(
    features: EdgeFeatureBundle,
    rho: np.ndarray,
    *,
    beta: float = 1.0,
    crossing_penalty: float = 0.0,
    crossing_fn: Optional[CrossingFn] = None,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
    length_eps: float = 1e-12,
) -> np.ndarray:
    """
    Saliency score(e) = rho(e) * c_e / l_e^beta  [- crossing_penalty * crossings].

    Crossing counts use ``crossing_fn`` when provided; otherwise no penalty.
    """
    lengths = np.maximum(features.lengths, length_eps)
    scores = rho * features.confidence / np.power(lengths, float(beta))

    if crossing_penalty and crossing_fn is not None and pos is not None:
        for i, e in enumerate(features.edges):
            n_cross = int(crossing_fn(e, features.edges, pos))
            scores[i] -= float(crossing_penalty) * n_cross
    return scores


# ---------------------------------------------------------------------------
# Selection stages
# ---------------------------------------------------------------------------


def non_max_suppression_edges(
    features: EdgeFeatureBundle,
    K: np.ndarray,
    scores: np.ndarray,
    *,
    similarity_threshold: float = 0.05,
) -> List[EdgeKey]:
    """
    Keep edges that are local maxima in their similarity neighborhood.

    N(e) = {e' : K(e,e') >= similarity_threshold}; keep e if score(e) is
    maximal in N(e) (ties broken by higher confidence, then edge index).
    """
    m = len(features.edges)
    if m == 0:
        return []

    kept: List[EdgeKey] = []
    conf = features.confidence
    thr = float(similarity_threshold)

    for i in range(m):
        nbrs = np.where(K[i] >= thr)[0]
        if nbrs.size == 0:
            nbrs = np.array([i], dtype=int)
        # Local maximum: compare (score, confidence, -index) lexicographically.
        best = nbrs[0]
        best_key = (scores[best], conf[best], -best)
        for j in nbrs[1:]:
            key = (scores[j], conf[j], -j)
            if key > best_key:
                best = j
                best_key = key
        if best == i:
            kept.append(features.edges[i])
    return kept


def enforce_degree_caps(
    G: nx.Graph,
    kept_edges: Sequence[EdgeKey],
    scores: Mapping[EdgeKey, float],
    *,
    degree_cap_min: int = 2,
    degree_cap_max: int = 4,
    node_density: Optional[Mapping[Any, float]] = None,
) -> List[EdgeKey]:
    """
    Per-node adaptive degree cap: keep strongest incident edges up to the cap.

    Cap interpolates between ``degree_cap_min`` and ``degree_cap_max`` using
    normalized node-local density (default: sum of incident edge scores).
    """
    if degree_cap_min < 1 or degree_cap_max < degree_cap_min:
        raise ValueError("Require 1 <= degree_cap_min <= degree_cap_max")

    # Build incident lists on the candidate edge set.
    incident: Dict[Any, List[EdgeKey]] = {n: [] for n in G.nodes()}
    for e in kept_edges:
        u, v = e
        if u in incident:
            incident[u].append(e)
        if v in incident:
            incident[v].append(e)

    if node_density is None:
        node_density = {
            n: float(sum(scores.get(e, 0.0) for e in incident[n])) for n in G.nodes()
        }

    dens_vals = np.asarray(list(node_density.values()), dtype=float)
    d_min = float(np.min(dens_vals)) if dens_vals.size else 0.0
    d_max = float(np.max(dens_vals)) if dens_vals.size else 1.0
    span = d_max - d_min

    def cap_for(n: Any) -> int:
        if span <= 1e-12:
            t = 0.5
        else:
            t = (float(node_density[n]) - d_min) / span
        return int(round(degree_cap_min + t * (degree_cap_max - degree_cap_min)))

    # Sort each node's edges by score descending and keep up to cap.
    allowed: Dict[EdgeKey, int] = {}  # how many endpoints still accept this edge
    for n in G.nodes():
        edges_n = sorted(
            incident[n],
            key=lambda e: (scores.get(e, float("-inf")), e),
            reverse=True,
        )
        for e in edges_n[: cap_for(n)]:
            allowed[e] = allowed.get(e, 0) + 1

    # An edge survives only if both endpoints kept it (or it is a loop).
    survivors: List[EdgeKey] = []
    for e in kept_edges:
        u, v = e
        need = 1 if u == v else 2
        if allowed.get(e, 0) >= need:
            survivors.append(e)
    return survivors


def _apply_target_budget(
    edges: Sequence[EdgeKey],
    scores: Mapping[EdgeKey, float],
    *,
    target_edge_count: Optional[int] = None,
    target_edge_ratio: Optional[float] = None,
    n_original: int,
) -> List[EdgeKey]:
    """Optionally truncate to a global target count / ratio by score."""
    if target_edge_count is None and target_edge_ratio is None:
        return list(edges)

    target = n_original
    if target_edge_ratio is not None:
        target = min(target, max(0, int(round(float(target_edge_ratio) * n_original))))
    if target_edge_count is not None:
        target = min(target, max(0, int(target_edge_count)))

    ranked = sorted(edges, key=lambda e: (scores.get(e, float("-inf")), e), reverse=True)
    return ranked[:target]


def repair_connectivity(
    G_full: nx.Graph,
    kept_edges: Sequence[EdgeKey],
    scores: Mapping[EdgeKey, float],
    *,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
    crossing_penalty: float = 0.0,
    crossing_fn: Optional[CrossingFn] = None,
    degree_cap_max: int = 4,
    repair_budget: Optional[int] = None,
) -> Tuple[List[EdgeKey], List[EdgeKey]]:
    """
    Add back high-score edges that bridge distinct components.

    Candidate pool = edges present in ``G_full`` but not currently kept.
    Preference: high score, low crossing penalty, soft penalty for degree growth.

    Returns
    -------
    final_edges, reconnect_edges
    """
    nodes = list(G_full.nodes())
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_edges_from(kept_edges)

    if nx.is_connected(H) or H.number_of_nodes() <= 1:
        return list(kept_edges), []

    kept_set = {normalize_undirected_edge(u, v) for u, v in kept_edges}
    candidates: List[EdgeKey] = []
    for u, v in G_full.edges():
        key = normalize_undirected_edge(u, v)
        if key not in kept_set:
            candidates.append(key)

    if not candidates:
        return list(kept_edges), []

    # Soft degree headroom relative to current degrees in H.
    def bridge_priority(e: EdgeKey) -> float:
        u, v = e
        base = float(scores.get(e, 0.0))
        deg_pen = 0.0
        if H.degree(u) >= degree_cap_max:
            deg_pen += 1.0 + (H.degree(u) - degree_cap_max)
        if H.degree(v) >= degree_cap_max:
            deg_pen += 1.0 + (H.degree(v) - degree_cap_max)
        cross_pen = 0.0
        if crossing_penalty and crossing_fn is not None and pos is not None:
            cross_pen = float(crossing_penalty) * float(
                crossing_fn(e, list(H.edges()), pos)
            )
        return base - deg_pen - cross_pen

    reconnect: List[EdgeKey] = []
    budget = len(candidates) if repair_budget is None else max(0, int(repair_budget))
    # Greedy: repeatedly add best inter-component edge.
    remaining = list(candidates)
    while budget > 0 and remaining and not nx.is_connected(H):
        # Components for quick reject.
        comp_id = {}
        for cid, comp in enumerate(nx.connected_components(H)):
            for n in comp:
                comp_id[n] = cid

        best_e = None
        best_pri = float("-inf")
        for e in remaining:
            u, v = e
            if comp_id.get(u) == comp_id.get(v):
                continue
            pri = bridge_priority(e)
            if pri > best_pri:
                best_pri = pri
                best_e = e

        if best_e is None:
            break
        H.add_edge(*best_e)
        reconnect.append(best_e)
        remaining.remove(best_e)
        budget -= 1

    final = list(kept_set | set(reconnect))
    return final, reconnect


# ---------------------------------------------------------------------------
# Diagnostics / build helpers
# ---------------------------------------------------------------------------


def _degree_stats(G: nx.Graph) -> Dict[str, float]:
    if G.number_of_nodes() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    degs = np.asarray([d for _, d in G.degree()], dtype=float)
    return {
        "mean": float(degs.mean()),
        "std": float(degs.std()),
        "min": float(degs.min()),
        "max": float(degs.max()),
        "median": float(np.median(degs)),
    }


def _count_crossings_graph(
    G: nx.Graph,
    pos: Mapping[Any, np.ndarray],
    crossing_fn: CrossingFn,
) -> int:
    edges = [normalize_undirected_edge(u, v) for u, v in G.edges()]
    total = 0
    for i, e in enumerate(edges):
        total += int(crossing_fn(e, edges[i + 1 :], pos))
    return total


def build_sparse_graph(
    G_source: nx.Graph,
    kept_edges: Sequence[EdgeKey],
    *,
    edge_scores: Optional[Mapping[EdgeKey, float]] = None,
    weight_attr: str = "weight",
) -> nx.Graph:
    """
    Build a new graph with the same nodes as ``G_source`` and only ``kept_edges``.

    Copies node attributes and, when present, original edge attributes.
    """
    H = nx.Graph()
    H.add_nodes_from((n, dict(data)) for n, data in G_source.nodes(data=True))
    for key in kept_edges:
        u, v = key
        if G_source.has_edge(u, v):
            data = dict(G_source.edges[u, v])
        else:
            data = {}
        if edge_scores is not None and "sparsify_score" not in data:
            data["sparsify_score"] = float(edge_scores.get(key, 0.0))
        H.add_edge(u, v, **data)
    H.graph.update(dict(G_source.graph))
    H.graph["sparsified"] = True
    return H


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sparsify_graph(
    G: nx.Graph,
    config: Optional[SparsifyConfig] = None,
    *,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
    crossing_fn: Optional[CrossingFn] = None,
    use_2d_crossings: bool = False,
) -> SparsifyResult:
    """
    Sparsify a geometric graph while preserving all nodes.

    Pipeline: edge features → anisotropic kernel density → saliency scores →
    non-max suppression → degree caps → optional target budget → connectivity
    repair → cleanup.

    Parameters
    ----------
    G :
        Input graph (typically a thresholded barycenter). Nodes must carry
        coordinate attributes or ``pos`` must be provided.
    config :
        ``SparsifyConfig`` instance; defaults are used when ``None``.
    pos :
        Optional explicit positions ``{node: ndarray}``.
    crossing_fn :
        Optional ``(edge, other_edges, pos) -> int`` crossing counter.
    use_2d_crossings :
        If True and ``crossing_fn`` is None, use ``count_edge_crossings_2d``.

    Returns
    -------
    SparsifyResult
        Sparse graph (same nodes), removed edges, scores, and diagnostics.
    """
    cfg = config or SparsifyConfig()
    if pos is None:
        pos = node_positions_from_graph(G)

    if crossing_fn is None and use_2d_crossings:
        crossing_fn = count_edge_crossings_2d
    elif crossing_fn is None and cfg.crossing_penalty:
        crossing_fn = count_edge_crossings_placeholder

    features = compute_edge_features(
        G, pos, conf_attr=cfg.conf_attr, length_eps=cfg.length_eps
    )
    n_orig = len(features.edges)
    if n_orig == 0:
        empty = build_sparse_graph(G, [])
        return SparsifyResult(
            graph=empty,
            removed_edges=[],
            edge_scores={},
            diagnostics={
                "n_components": nx.number_connected_components(empty),
                "edge_retention_ratio": 0.0,
                "degree_stats": _degree_stats(empty),
                "crossing_count": 0,
                "reconnect_edges": [],
                "n_original_edges": 0,
                "n_kept_edges": 0,
                "config": cfg.to_dict(),
            },
            kept_edges=[],
            reconnect_edges=[],
            features=features,
        )

    h_mid, h_angle, h_len = estimate_bandwidths(
        features,
        h_mid=cfg.h_mid,
        h_angle=cfg.h_angle,
        h_len=cfg.h_len,
        quantile=cfg.auto_bandwidth_quantile,
    )
    K = edge_kernel_similarity(
        features,
        h_mid=h_mid,
        h_angle=h_angle,
        h_len=h_len,
        length_eps=cfg.length_eps,
    )
    rho = compute_edge_density(K, features.confidence)
    scores_arr = score_edges(
        features,
        rho,
        beta=cfg.beta,
        crossing_penalty=cfg.crossing_penalty,
        crossing_fn=crossing_fn,
        pos=pos,
        length_eps=cfg.length_eps,
    )
    score_map: Dict[EdgeKey, float] = {
        e: float(scores_arr[i]) for i, e in enumerate(features.edges)
    }

    # Stage 1: NMS in edge space
    after_nms = non_max_suppression_edges(
        features, K, scores_arr, similarity_threshold=cfg.similarity_threshold
    )

    # Node density for adaptive caps: sum of densities of incident original edges.
    node_rho: Dict[Any, float] = {n: 0.0 for n in G.nodes()}
    for e, r in zip(features.edges, rho):
        u, v = e
        node_rho[u] = node_rho.get(u, 0.0) + float(r)
        node_rho[v] = node_rho.get(v, 0.0) + float(r)

    # Stage 2: degree caps
    after_deg = enforce_degree_caps(
        G,
        after_nms,
        score_map,
        degree_cap_min=cfg.degree_cap_min,
        degree_cap_max=cfg.degree_cap_max,
        node_density=node_rho,
    )

    # Stage 3: optional global target
    after_budget = _apply_target_budget(
        after_deg,
        score_map,
        target_edge_count=cfg.target_edge_count,
        target_edge_ratio=cfg.target_edge_ratio,
        n_original=n_orig,
    )

    # Stage 4: connectivity repair (from original edge pool)
    final_edges, reconnect = repair_connectivity(
        G,
        after_budget,
        score_map,
        pos=pos,
        crossing_penalty=cfg.crossing_penalty,
        crossing_fn=crossing_fn,
        degree_cap_max=cfg.degree_cap_max,
        repair_budget=cfg.repair_budget,
    )

    # Cleanup: unique undirected edges, sorted by score descending.
    uniq = {normalize_undirected_edge(u, v): (u, v) for u, v in final_edges}
    kept_sorted = sorted(
        uniq.values(),
        key=lambda e: (score_map.get(normalize_undirected_edge(*e), float("-inf")), e),
        reverse=True,
    )
    kept_set = {normalize_undirected_edge(u, v) for u, v in kept_sorted}
    removed = [
        e for e in features.edges if normalize_undirected_edge(*e) not in kept_set
    ]
    removed_sorted = sorted(
        removed,
        key=lambda e: (score_map.get(e, float("-inf")), e),
        reverse=True,
    )

    G_sparse = build_sparse_graph(
        G, kept_sorted, edge_scores=score_map, weight_attr=cfg.weight_attr
    )

    crossing_count = None
    if crossing_fn is not None and crossing_fn is not count_edge_crossings_placeholder:
        crossing_count = _count_crossings_graph(G_sparse, pos, crossing_fn)

    diagnostics = {
        "n_components": int(nx.number_connected_components(G_sparse)),
        "edge_retention_ratio": float(len(kept_sorted) / n_orig) if n_orig else 0.0,
        "degree_stats": _degree_stats(G_sparse),
        "crossing_count": crossing_count,
        "reconnect_edges": [
            normalize_undirected_edge(u, v) for u, v in reconnect
        ],
        "n_original_edges": int(n_orig),
        "n_kept_edges": int(len(kept_sorted)),
        "n_removed_edges": int(len(removed_sorted)),
        "n_after_nms": int(len(after_nms)),
        "n_after_degree_caps": int(len(after_deg)),
        "n_after_budget": int(len(after_budget)),
        "n_reconnect": int(len(reconnect)),
        "bandwidths": {"h_mid": h_mid, "h_angle": h_angle, "h_len": h_len},
        "config": cfg.to_dict(),
        "is_connected": bool(nx.is_connected(G_sparse))
        if G_sparse.number_of_nodes() > 0
        else True,
    }

    return SparsifyResult(
        graph=G_sparse,
        removed_edges=removed_sorted,
        edge_scores=score_map,
        diagnostics=diagnostics,
        kept_edges=kept_sorted,
        reconnect_edges=[normalize_undirected_edge(u, v) for u, v in reconnect],
        features=features,
    )


def draw_sparsify_comparison(
    G_source: nx.Graph,
    result: SparsifyResult,
    *,
    pos: Optional[Mapping[Any, np.ndarray]] = None,
    figsize: Tuple[float, float] = (14.0, 6.0),
    kept_color: str = "#1f77b4",
    removed_color: str = "#bbbbbb",
    reconnect_color: str = "#d62728",
    node_size: float = 8.0,
    ax=None,
):
    """
    Visualize kept (blue), removed (gray), and reconnect (red) edges in 2D.

    Uses the first two coordinates of each node position.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if pos is None:
        pos = node_positions_from_graph(G_source)

    xy = {n: np.asarray(p[:2], dtype=float) for n, p in pos.items()}
    reconnect_set = set(result.reconnect_edges)
    kept_draw = [e for e in result.kept_edges if e not in reconnect_set]
    removed_draw = list(result.removed_edges)

    def segs(edges):
        out = []
        for u, v in edges:
            if u in xy and v in xy:
                out.append([xy[u], xy[v]])
        return out

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if removed_draw:
        ax.add_collection(
            LineCollection(
                segs(removed_draw),
                colors=removed_color,
                linewidths=0.6,
                alpha=0.45,
                zorder=1,
            )
        )
    if kept_draw:
        ax.add_collection(
            LineCollection(
                segs(kept_draw),
                colors=kept_color,
                linewidths=1.4,
                alpha=0.9,
                zorder=2,
            )
        )
    if result.reconnect_edges:
        ax.add_collection(
            LineCollection(
                segs(result.reconnect_edges),
                colors=reconnect_color,
                linewidths=1.8,
                alpha=1.0,
                zorder=3,
            )
        )

    xs = np.array([p[0] for p in xy.values()])
    ys = np.array([p[1] for p in xy.values()])
    ax.scatter(xs, ys, s=node_size, c="#222222", zorder=4)

    d = result.diagnostics
    ax.set_title(
        "Sparsified barycenter edges\n"
        f"kept {d['n_kept_edges']}/{d['n_original_edges']} "
        f"(retention {d['edge_retention_ratio']:.1%}), "
        f"components={d['n_components']}, "
        f"reconnect={d['n_reconnect']}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    if created:
        fig.tight_layout()
    return fig, ax
