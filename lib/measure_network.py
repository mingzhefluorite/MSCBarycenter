"""
Split and duplicate nodes in a discrete measure metric space / measure network.

A measure network is the triple ``(C, X, p)``:

- ``C``: pairwise relation matrix, shape ``(n, n)`` (shortest-path, adjacency,
  or a combination)
- ``X``: node attribute matrix, shape ``(n, d)`` (also written ``Y`` for inputs)
- ``p``: probability / mass vector, shape ``(n,)``
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


MeasureSpace = Tuple[np.ndarray, np.ndarray, np.ndarray]
SplitSpec = Sequence[Tuple[int, float]]


def _as_float_matrix(C: ArrayLike, name: str = "C") -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array; got shape {C.shape}.")
    return C


def _as_feature_matrix(X: ArrayLike, n: int, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        if X.shape[0] != n:
            raise ValueError(f"{name} must have length {n}; got shape {X.shape}.")
        return X
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError(
            f"{name} must have shape ({n}, d) or ({n},); got shape {X.shape}."
        )
    return X


def _as_mass(p: ArrayLike, n: int, name: str = "p") -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},); got shape {p.shape}.")
    if np.any(p < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return p


def _validate_measure_space(
    C: ArrayLike,
    X: ArrayLike,
    p: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    C = _as_float_matrix(C, "C")
    n = int(C.shape[0])
    X = _as_feature_matrix(X, n, "X")
    p = _as_mass(p, n, "p")
    return C, X, p, n


def _validate_old_indices(indices: np.ndarray, n: int, name: str = "indices") -> None:
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of old node indices.")
    if indices.size == 0:
        return
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError(
            f"{name} must contain old indices in [0, {n}); "
            f"got min={int(indices.min())}, max={int(indices.max())}."
        )


def _gather_from_sources(
    C: np.ndarray,
    X: np.ndarray,
    source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Copy attributes and pairwise relations from original nodes ``source``.

    Sibling copies of the same original node (including a node and its
    duplicates) get pairwise relation 0, matching both adjacency (no edge)
    and shortest-path (zero distance) conventions.
    """
    C_new = C[np.ix_(source, source)]
    sibling = source[:, None] == source[None, :]
    C_new[sibling] = 0.0
    X_new = X[source].copy()
    return C_new, X_new


def _uniform_mass(n: int, balanced_distribution: bool) -> np.ndarray:
    """
    Rebuild a uniform mass vector in the pipeline convention.

    ``balanced_distribution=True``  -> each node has mass ``1/n`` (sum 1).
    ``balanced_distribution=False`` -> each node has mass ``1`` (sum ``n``).
    """
    if n == 0:
        return np.zeros(0, dtype=float)
    if balanced_distribution:
        return np.full(n, 1.0 / n, dtype=float)
    return np.ones(n, dtype=float)


def _node_xyz_from_attrs(attrs: dict) -> np.ndarray:
    return np.asarray([attrs["x"], attrs["y"], attrs["z"]], dtype=float)


def _edge_length_for_measure(G, u, v, data: Optional[dict] = None) -> float:
    """
    Length of ``(u, v)`` for the discretized arc-length measure.

    Prefers the polyline ``arc_length`` recorded during Morse-path sampling
    so that subdividing a sampled shortcut does not change the total length.
    Falls back to ``weight``, then to the Euclidean chord.
    """
    if data is None:
        data = G.get_edge_data(u, v) or {}
    if "arc_length" in data and data["arc_length"] is not None:
        return float(data["arc_length"])
    path = data.get("path_nodes")
    if path is not None and len(path) >= 2:
        try:
            total = 0.0
            for a, b in zip(path[:-1], path[1:]):
                total += float(
                    np.linalg.norm(
                        _node_xyz_from_attrs(G.nodes[a])
                        - _node_xyz_from_attrs(G.nodes[b])
                    )
                )
            return total
        except KeyError:
            pass
    if "weight" in data and data["weight"] is not None:
        return float(data["weight"])
    return float(
        np.linalg.norm(
            _node_xyz_from_attrs(G.nodes[u]) - _node_xyz_from_attrs(G.nodes[v])
        )
    )


def arc_length_node_measure(
    G,
    nodelist: Optional[Sequence] = None,
    rescale_factor: float = 1.0,
) -> np.ndarray:
    """
    Sampling-invariant discretized arc-length probability on a graph.

    For each node ``v``

        ell(v) = (1/2) * sum_{e ni v} length(e)
        mu(v)  = ell(v) / sum_{e in E} length(e)

    The identity ``sum_v ell(v) = sum_e length(e)`` means every edge
    contributes half its length to each endpoint. Subdividing one edge
    into additional regular points therefore leaves the measure on all
    nodes not incident to that edge unchanged, and the total mass stays 1.

    Isolated nodes receive mass 0. If the graph has no positive-length
    edges, the measure falls back to uniform ``1/n``.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose edges carry ``arc_length`` (preferred), ``weight``, or
        node coordinates ``x, y, z``.
    nodelist : sequence, optional
        Node order of the returned vector. Defaults to ``list(G.nodes())``.

    Returns
    -------
    p : ndarray, shape (n,)
        Probability vector (nonnegative, sums to 1 when ``n > 0``).
    """
    if nx is None:
        raise ImportError("arc_length_node_measure requires networkx")
    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)
    if n == 0:
        return np.zeros(0, dtype=float)

    index = {node: i for i, node in enumerate(nodelist)}
    ell = np.zeros(n, dtype=float)
    total = 0.0
    for u, v, data in G.edges(data=True):
        if u not in index or v not in index:
            continue
        length = _edge_length_for_measure(G, u, v, data)
        if length < 0.0:
            raise ValueError(
                f"Edge {(u, v)} has negative length {length}."
            )
        if length == 0.0:
            continue
        total += length
        ell[index[u]] += 0.5 * length
        ell[index[v]] += 0.5 * length

    if total <= 0.0:
        return np.full(n, 1.0 / n, dtype=float)
    return ell / total * rescale_factor


def duplicate_measure_nodes(
    C: ArrayLike,
    X: ArrayLike,
    p: ArrayLike,
    indices: ArrayLike,
    *,
    balanced_distribution: bool,
) -> MeasureSpace:
    """
    Duplicate selected nodes of a measure network.

    Each entry of ``indices`` is an **old** node index. That node is copied
    once, so the number of nodes grows by ``len(indices)``. Repeating an
    index duplicates that node multiple times.

    New nodes are emitted in original-index order. Copies of a node are
    placed immediately after it (not concatenated at the end), matching
    ``split_measure_nodes``.

    Attributes of each copy match the original node. Relations from a copy to
    every other distinct original node match the original node's row/column.
    Relations among all copies of the same original node (and with that
    original) are 0.

    After duplication, ``p`` is **recomputed as a uniform distribution** on the
    new node set, using the same convention as the barycenter pipeline:

    - ``balanced_distribution=True``: ``p_new = 1 / n_new`` (total mass 1)
    - ``balanced_distribution=False``: ``p_new = 1`` per node (total mass
      ``n_new``)

    Input ``p`` is validated for shape but not reused; the flag fully
    determines the new masses.

    Parameters
    ----------
    C : array-like, shape (n, n)
        Pairwise relation matrix.
    X : array-like, shape (n, d) or (n,)
        Node attributes.
    p : array-like, shape (n,)
        Mass / probability vector of the input space (shape-checked only).
    indices : array-like of int, shape (k,)
        Old indices of nodes to duplicate. ``[i, j]`` duplicates node ``i``
        once and node ``j`` once.
    balanced_distribution : bool
        If True, renormalize total mass to 1. If False, give every node
        mass 1.

    Returns
    -------
    C_new : ndarray, shape (n + k, n + k)
    X_new : ndarray, shape (n + k, d) or (n + k,)
    p_new : ndarray, shape (n + k,)
        Uniform masses under ``balanced_distribution``.
    """
    C, X, p, n = _validate_measure_space(C, X, p)
    dup = np.asarray(indices, dtype=int).reshape(-1)
    _validate_old_indices(dup, n, "indices")

    extra = np.bincount(dup, minlength=n)
    source = np.repeat(np.arange(n, dtype=int), extra + 1)
    C_new, X_new = _gather_from_sources(C, X, source)
    p_new = _uniform_mass(int(source.shape[0]), balanced_distribution)
    return C_new, X_new, p_new


def split_measure_nodes(
    C: ArrayLike,
    X: ArrayLike,
    p: ArrayLike,
    splits: Union[SplitSpec, ArrayLike],
    *,
    atol: float = 1e-8,
) -> MeasureSpace:
    """
    Split selected nodes of a measure network, dividing their mass.

    Each entry of ``splits`` is ``(old_index, proportion)``. All pieces that
    share the same ``old_index`` replace that original node. Nodes that do not
    appear in ``splits`` are kept unchanged. The new size is therefore

        n - (# distinct split indices) + len(splits)

    rather than ``n + len(splits)``.

    For every original index that appears, the listed proportions **must sum
    to 1**. Partial-OT leftover mass should be handled before calling this
    function. Mass is not reinitialized: a piece ``(i, a)`` receives
    ``a * p[i]``.

    Attributes of each piece match the original node. Relations from a piece
    to every other distinct original node match the original node's
    row/column. Relations among pieces of the same original node are 0.

    New nodes are emitted in original-index order. Pieces of one node follow
    the order they appear in ``splits``.

    Parameters
    ----------
    C : array-like, shape (n, n)
        Pairwise relation matrix.
    X : array-like, shape (n, d) or (n,)
        Node attributes.
    p : array-like, shape (n,)
        Mass / probability vector; split in place, not recomputed.
    splits : sequence of (int, float) or array of shape (k, 2)
        Old indices with mass fractions. Splitting node ``x`` at 0.7 + 0.3
        is ``[(x, 0.7), (x, 0.3)]``.
    atol : float, optional
        Absolute tolerance for the per-node proportion sum.

    Returns
    -------
    C_new : ndarray
    X_new : ndarray
    p_new : ndarray
    """
    C, X, p, n = _validate_measure_space(C, X, p)

    if splits is None or (isinstance(splits, (list, tuple)) and len(splits) == 0):
        return C.copy(), X.copy(), p.copy()

    pairs = np.asarray(splits, dtype=float)
    if pairs.size == 0:
        return C.copy(), X.copy(), p.copy()
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            "splits must be a sequence of (index, proportion) pairs "
            f"or an array of shape (k, 2); got shape {pairs.shape}."
        )

    split_idx = np.rint(pairs[:, 0]).astype(int)
    if not np.allclose(pairs[:, 0], split_idx, atol=1e-12, rtol=0.0):
        raise ValueError("Split indices must be integers (old node indices).")
    _validate_old_indices(split_idx, n, "split indices")

    proportions = pairs[:, 1]
    if np.any(proportions <= 0) or np.any(proportions > 1.0 + atol):
        raise ValueError(
            "Each proportion_of_split must lie in (0, 1]; "
            f"got min={float(proportions.min())}, max={float(proportions.max())}."
        )

    groups: Dict[int, List[float]] = defaultdict(list)
    for idx, prop in zip(split_idx.tolist(), proportions.tolist()):
        groups[idx].append(float(prop))

    for idx, props in groups.items():
        total_prop = float(np.sum(props))
        if not np.isclose(total_prop, 1.0, atol=atol, rtol=0.0):
            raise ValueError(
                f"Proportions for old index {idx} must sum to 1.0; "
                f"got {total_prop}."
            )

    source_parts = []
    mass_parts = []
    for i in range(n):
        if i in groups:
            for prop in groups[i]:
                source_parts.append(i)
                mass_parts.append(prop * p[i])
        else:
            source_parts.append(i)
            mass_parts.append(p[i])

    source = np.asarray(source_parts, dtype=int)
    C_new, X_new = _gather_from_sources(C, X, source)
    p_new = np.asarray(mass_parts, dtype=float)
    return C_new, X_new, p_new
