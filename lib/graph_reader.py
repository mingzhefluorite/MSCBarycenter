"""
Functions for reading and processing Morse graphs from files.
"""

import heapq
import itertools
import os
import warnings
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

SampleMode = Literal["step", "geo"]

if TYPE_CHECKING:
    from gwmc.core.morse_graph import MorseGraph

COORD_EPS = 1e-8

# Morse-Smale node file constants (7-column format).
SEPARATRIX_CRITICAL = 0
SEPARATRIX_REGULAR = 1
REGULAR_CRITICAL_TYPE = 65535
REGULAR_NODE_TYPE = -1  # visualization alias when criticalType == 65535


def _is_critical_row(separatrix_type: int, critical_type: int) -> bool:
    """True for Morse critical points (separatrixType == 0)."""
    return int(separatrix_type) == SEPARATRIX_CRITICAL


def _visual_type(separatrix_type: int, critical_type: int) -> int:
    """Map file fields to node ``type`` (0=min, 1=saddle, 2=max, -1=regular)."""
    if _is_critical_row(separatrix_type, critical_type):
        return int(critical_type)
    return REGULAR_NODE_TYPE


def _is_critical_node(attrs: dict) -> bool:
    """True if canonical node attributes describe a critical point."""
    if "separatrixType" in attrs:
        return int(attrs["separatrixType"]) == SEPARATRIX_CRITICAL
    # Legacy 4-column files: type == 0 marked any critical point.
    return int(attrs.get("type", REGULAR_NODE_TYPE)) == 0


def _parse_node_row(node: np.ndarray, row_idx: int) -> dict:
    """
    Parse one node row into attribute fields.

    Supported layouts
    -----------------
    * 7 columns (current): x, y, z, scalar_value, separatrixType,
      separatrixID, criticalType
    * 5 columns (legacy): x, y, z, type, cell_id  (type 0 = critical)
    * 4 columns (legacy): x, y, z, type
    * 3 columns: x, y, z only (regular node)
    """
    n = len(node)
    if n not in {3, 4, 5, 7}:
        raise ValueError(f"Invalid node dimension: {n} (expected 3, 4, 5, or 7)")

    attrs = {
        "id": row_idx,
        "x": float(node[0]),
        "y": float(node[1]),
        "z": float(node[2]),
        "color": "r",
    }

    if n == 7:
        scalar_value = float(node[3])
        separatrix_type = int(node[4])
        separatrix_id = int(node[5])
        critical_type = int(node[6])
        attrs.update(
            scalar_value=scalar_value,
            height=scalar_value,
            separatrixType=separatrix_type,
            separatrixID=separatrix_id,
            criticalType=critical_type,
            type=_visual_type(separatrix_type, critical_type),
        )
        return attrs

    # Legacy formats without explicit Morse fields.
    separatrix_type = SEPARATRIX_REGULAR
    critical_type = REGULAR_CRITICAL_TYPE
    if n >= 4:
        legacy_type = int(node[3])
        if legacy_type == 0:
            separatrix_type = SEPARATRIX_CRITICAL
            critical_type = 0
        attrs["type"] = legacy_type
    else:
        attrs["type"] = REGULAR_NODE_TYPE
    if n >= 5:
        attrs["cell_id"] = int(node[4])
    attrs["separatrixType"] = separatrix_type
    attrs["criticalType"] = critical_type
    return attrs


def _merge_node_attrs(attrs: Optional[dict], row_attrs: dict) -> dict:
    """
    Merge two node attribute dicts for the same (x, y, z).

    If any row is critical (separatrixType == 0), the canonical node is
    critical. Scalar / separatrix metadata prefer the critical row.
    """
    if attrs is None:
        return dict(row_attrs)

    row_critical = _is_critical_node(row_attrs)
    attrs_critical = _is_critical_node(attrs)

    if row_critical and not attrs_critical:
        merged = dict(attrs)
        merged.update(row_attrs)
        merged["type"] = row_attrs["type"]
        return merged
    if attrs_critical and not row_critical:
        merged = dict(row_attrs)
        merged.update(attrs)
        return merged

    merged = dict(attrs)
    merged.update(row_attrs)

    if row_critical and attrs_critical:
        # Same location listed as critical in multiple separatrix sheets.
        if row_attrs["criticalType"] != REGULAR_CRITICAL_TYPE:
            merged["criticalType"] = row_attrs["criticalType"]
            merged["type"] = row_attrs["type"]
        if row_attrs.get("separatrixID") is not None:
            merged["separatrixID"] = row_attrs["separatrixID"]
        if "scalar_value" in row_attrs:
            merged["scalar_value"] = row_attrs["scalar_value"]
            merged["height"] = row_attrs["scalar_value"]

    return merged


def merge_duplicate_nodes(
    nodes_data: np.ndarray,
    eps: float = COORD_EPS,
) -> Tuple[List[int], List[dict]]:
    """
    Merge nodes that share the same (x, y, z) within `eps`.

    Parameters
    ----------
    nodes_data : ndarray, shape (n, >= 3)
        Raw node rows. Current format (7 columns):
        x, y, z, scalar_value, separatrixType, separatrixID, criticalType.
        Legacy formats with 3--5 columns are still accepted.
    eps : float
        Maximum coordinate distance to treat two nodes as duplicates.

    Returns
    -------
    orig_to_canon : list of int
        Maps each original row index to a canonical node id in [0, n_canon).
    canon_attrs : list of dict
        Node attributes for each canonical node. A node is critical
        (separatrixType == 0) if any duplicate row was critical.
    """
    if nodes_data.ndim == 1:
        nodes_data = nodes_data.reshape(1, -1)

    n_rows = len(nodes_data)
    if n_rows == 0:
        return [], []

    order = np.lexsort((nodes_data[:, 2], nodes_data[:, 1], nodes_data[:, 0]))
    orig_to_canon = [-1] * n_rows
    canon_attrs: List[dict] = []

    def _merge_row_into(attrs: Optional[dict], row_idx: int) -> dict:
        row_attrs = _parse_node_row(nodes_data[row_idx], row_idx)
        return _merge_node_attrs(attrs, row_attrs)

    def _finalize_group(group: List[int]) -> None:
        canon_id = len(canon_attrs)
        attrs = None
        for orig in group:
            attrs = _merge_row_into(attrs, orig)
        canon_attrs.append(attrs)
        for orig in group:
            orig_to_canon[orig] = canon_id

    group = [int(order[0])]
    for k in range(1, n_rows):
        i = int(order[k])
        j = int(order[k - 1])
        if np.linalg.norm(nodes_data[i, :3] - nodes_data[j, :3]) <= eps:
            group.append(i)
        else:
            _finalize_group(group)
            group = [i]
    _finalize_group(group)

    return orig_to_canon, canon_attrs


def read_morse_graph_txt(
    node_filename: str,
    edge_filename: str
) -> Tuple[nx.Graph, List[int]]:
    """
    Read a Morse graph from text files.

    Edge files may list the same undirected edge more than once with different
    separatrix IDs. Those IDs are merged into edge attribute ``separatrixID`` as
    a sorted comma-separated string (e.g. ``"0,1,6,7"``).
    
    Parameters
    ----------
    node_filename : str
        Path to node file
    edge_filename : str
        Path to edge file
        
    Returns
    -------
    graph : nx.Graph
        NetworkX graph representing the Morse-Smale complex
    critical_nodes : list of int
        List of critical node IDs
    """
    nodes_data = np.loadtxt(node_filename)
    edges_data = np.loadtxt(edge_filename)

    if nodes_data.ndim == 1:
        nodes_data = nodes_data.reshape(1, -1)

    orig_to_canon, canon_attrs = merge_duplicate_nodes(nodes_data, eps=COORD_EPS)

    T = nx.Graph()
    critical_nodes: List[int] = []

    for canon_id, node_data in enumerate(canon_attrs):
        T.add_node(canon_id, **node_data)
        if _is_critical_node(node_data):
            critical_nodes.append(canon_id)

    if edges_data.ndim == 1:
        edges_data = edges_data.reshape(1, -1)

    # Accumulate undirected edges; duplicate geometric edges may list several
    # separatrix IDs that we merge into one comma-separated property.
    edge_accum: dict = {}
    for edge in edges_data:
        u_orig = int(edge[0])
        v_orig = int(edge[1])
        edge_type = int(edge[2]) if len(edge) > 2 else 0
        edge_separatrix_id = int(edge[3]) if len(edge) > 3 else 0

        u = orig_to_canon[u_orig]
        v = orig_to_canon[v_orig]
        if u == v:
            continue

        edge_key = (min(u, v), max(u, v))
        if edge_key not in edge_accum:
            edge_accum[edge_key] = {"type": edge_type, "ids": {edge_separatrix_id}}
        else:
            edge_accum[edge_key]["ids"].add(edge_separatrix_id)

    for (u, v), info in edge_accum.items():
        dist = float(
            np.linalg.norm(
                np.asarray([T.nodes[u]["x"], T.nodes[u]["y"], T.nodes[u]["z"]])
                - np.asarray([T.nodes[v]["x"], T.nodes[v]["y"], T.nodes[v]["z"]])
            )
        )
        T.add_edge(
            u,
            v,
            weight=dist,
            type=info["type"],
            separatrixID=format_edge_separatrix_ids(info["ids"]),
        )

    return T, critical_nodes


def format_edge_separatrix_ids(ids) -> str:
    """Encode edge separatrix IDs as a sorted comma-separated string."""
    return ",".join(str(i) for i in sorted({int(x) for x in ids}))


def edge_separatrix_id_key(data: Optional[dict]) -> str:
    """
    Canonical ``separatrixID`` string for an edge attribute dict.

    Accepts the stored comma-separated form, a single int, or an iterable of ids.
    Missing values default to ``"-1"``.
    """
    if not data or "separatrixID" not in data:
        return "-1"
    raw = data["separatrixID"]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        if not parts:
            return "-1"
        return format_edge_separatrix_ids(parts)
    if isinstance(raw, (int, np.integer)):
        return str(int(raw))
    try:
        return format_edge_separatrix_ids(raw)
    except TypeError:
        return str(raw)


def _node_xyz(G: nx.Graph, n) -> np.ndarray:
    return np.asarray([G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]], dtype=float)


def _edge_euclid(G: nx.Graph, a, b) -> float:
    return float(np.linalg.norm(_node_xyz(G, a) - _node_xyz(G, b)))


def _polyline_arc_length(G: nx.Graph, path: Sequence) -> float:
    """Sum of 3D Euclidean hops along ``path`` using coordinates in ``G``."""
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += _edge_euclid(G, a, b)
    return total


def _oriented_path_nodes(u, v, path: Optional[Sequence]) -> List:
    """Return ``path`` oriented from ``u`` to ``v`` when possible."""
    if path is None:
        return [u, v]
    nodes = list(path)
    if len(nodes) < 2:
        return [u, v]
    if nodes[0] == u and nodes[-1] == v:
        return nodes
    if nodes[0] == v and nodes[-1] == u:
        nodes.reverse()
        return nodes
    return nodes


def _edge_matches_type(G: nx.Graph, u, v, edge_type: int) -> bool:
    data = G.get_edge_data(u, v)
    if data is None:
        return False
    if "type" in data:
        return int(data["type"]) == edge_type
    return edge_type == 0


def sample_morse_graph(
    G: nx.Graph,
    critical_nodes: List[int],
    rate: float,
    mode: SampleMode = "step",
    preserve_min_degree: int = 3,
) -> Tuple[nx.Graph, List[int]]:
    """
    Subsample a Morse-Smale graph by walking along manifold paths (per edge type)
    and keeping nodes spaced by ``rate``, while always retaining critical nodes
    and high-degree junctions.

    ``rate`` (``sample_rate`` in :func:`prepare_graph_for_barycenter`) is the
    minimum spacing between kept regular nodes along a path: in ``step`` mode it
    counts edges; in ``geo`` mode it is 3D Euclidean arc length. Larger values
    yield sparser graphs (fewer remaining nodes).

    Adapted from path-based MSC simplification (Sample2D): DFS from each critical
    along edges of type 0 and 1 separately. Critical nodes and junctions (degree
    >= ``preserve_min_degree``) are invariant anchors always kept; regular nodes
    are kept only when path spacing exceeds ``rate``. Edges connect consecutive
    kept nodes along each path (no self-loops).

    Within one walk, only hops that share the same edge ``separatrixID`` string
    (merged set of IDs) are traversed, so sampling never concatenates segments
    from different separatrices. Nodes incident to more than one distinct
    ``separatrixID`` are treated as junctions (always kept). Shortcut edges
    inherit the path ``separatrixID``.

    Each sampled edge stores the original walk as ``path_nodes`` (endpoint
    ids included) and the polyline length as ``arc_length``. ``weight`` remains
    the 3D Euclidean chord between the kept endpoints (used by shortest-path
    ``C``). Dropped interior nodes can later be reinserted by
    :func:`expand_sampled_graph_to_size`.

    Returns
    -------
    G_sample : nx.Graph
        Subsampled graph (node ids match the input graph).
    critical_nodes : list of int
        Critical node ids present in ``G_sample``.
    """
    if mode not in ("step", "geo"):
        raise ValueError(f"Unrecognized mode {mode!r}; use 'step' or 'geo'.")

    critical_set = {n for n in critical_nodes if n in G}
    junction_set = {
        n for n in G.nodes() if G.degree(n) >= preserve_min_degree
    }
    # Nodes where incident edges disagree on separatrixID are natural segment
    # seams: always keep them so sampling cannot skip across SID changes.
    for n in G.nodes():
        sids = {
            edge_separatrix_id_key(G.get_edge_data(n, m))
            for m in G.neighbors(n)
        }
        if len(sids) > 1:
            junction_set.add(n)
    anchor_set = critical_set | junction_set
    G_sample = nx.Graph()
    visited = set()

    def _euclid(a, b) -> float:
        return _edge_euclid(G, a, b)

    def _add_node(n):
        if not G_sample.has_node(n):
            G_sample.add_node(n, **G.nodes[n])

    def _add_edge(a, b, edge_type: int, separatrix_id: str, path_nodes: Sequence):
        if a == b:
            return
        path_nodes = list(path_nodes)
        if len(path_nodes) < 2:
            path_nodes = [a, b]
        _add_node(a)
        _add_node(b)
        arc_length = _polyline_arc_length(G, path_nodes)
        if G_sample.has_edge(a, b):
            old_path = G_sample[a][b].get("path_nodes") or []
            if len(path_nodes) <= len(old_path):
                return
        G_sample.add_edge(
            a,
            b,
            weight=_euclid(a, b),
            arc_length=arc_length,
            path_nodes=path_nodes,
            type=edge_type,
            separatrixID=separatrix_id,
        )

    def dfs(start, edge_type: int = 0):
        if start in visited and start not in critical_set:
            return

        # (path_start, node, length, path_separatrix_id or None, path_nodes)
        stack = [(start, start, 0.0, None, [start])]

        while stack:
            start, node, length, path_sid, path = stack.pop()

            if node in visited and node not in critical_set:
                continue

            visited.add(node)

            for n in G.neighbors(node):
                if n in visited:
                    continue
                if not _edge_matches_type(G, node, n, edge_type):
                    continue

                hop_sid = edge_separatrix_id_key(G.get_edge_data(node, n))
                if path_sid is not None and hop_sid != path_sid:
                    continue
                next_sid = hop_sid if path_sid is None else path_sid
                new_path = path + [n]

                if n in anchor_set:
                    _add_edge(start, n, edge_type, next_sid, new_path)
                    # Junctions (non-critical) must fan out into their other branches.
                    if n in junction_set and n not in critical_set:
                        stack.append((n, n, 0.0, None, [n]))
                    continue

                if mode == "step":
                    new_length = length + 1.0
                else:
                    new_length = length + _euclid(node, n)

                if new_length > rate:
                    _add_edge(start, n, edge_type, next_sid, new_path)
                    stack.append((n, n, 0.0, None, [n]))
                else:
                    stack.append((start, n, new_length, next_sid, new_path))

    for crit in critical_set:
        _add_node(crit)
        dfs(crit, 0)
        dfs(crit, 1)

    if critical_set:
        missing = [n for n in critical_set if not G_sample.has_node(n)]
        if missing:
            raise RuntimeError(f"Sampled graph missing critical nodes: {missing}")

    if G_sample.number_of_nodes() > 0 and not nx.is_connected(G_sample):
        warnings.warn(
            "sample_morse_graph produced a disconnected graph; "
            "input may span several components.",
            stacklevel=2,
        )

    critical_out = [n for n in critical_set if G_sample.has_node(n)]
    return G_sample, critical_out


def _recover_dropped_path(
    source_graph: nx.Graph,
    G_sample: nx.Graph,
    u,
    v,
) -> List:
    """
    Reconstruct interior nodes dropped between ``u`` and ``v``.

    Restricts to dropped nodes plus the two endpoints so the recovered walk
    does not pass through other kept sample nodes.
    """
    present = set(G_sample.nodes())
    allowed = (set(source_graph.nodes()) - present) | {u, v}
    H = source_graph.subgraph(allowed)
    try:
        return list(nx.shortest_path(H, u, v))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [u, v]


def _edge_path_nodes(
    G: nx.Graph,
    source_graph: nx.Graph,
    u,
    v,
    data: Optional[dict] = None,
) -> List:
    """Path of original nodes along sampled edge ``(u, v)``."""
    if data is None:
        data = G.get_edge_data(u, v) or {}
    stored = data.get("path_nodes")
    path = _oriented_path_nodes(u, v, stored)
    if len(path) >= 2 and path[0] == u and path[-1] == v:
        return path
    return _recover_dropped_path(source_graph, G, u, v)


def _choose_split_node(
    source_graph: nx.Graph,
    path: Sequence,
    present: set,
):
    """
    Dropped node on ``path`` closest to the arc-length midpoint.

    Returns ``None`` when the edge has no remaining insertable node.
    """
    if len(path) < 3:
        return None
    interior = [n for n in path[1:-1] if n not in present and n in source_graph]
    if not interior:
        return None

    cum = [0.0]
    for a, b in zip(path[:-1], path[1:]):
        cum.append(cum[-1] + _edge_euclid(source_graph, a, b))
    total = cum[-1]
    index_of = {n: i for i, n in enumerate(path)}
    if total <= 0.0:
        target_idx = 0.5 * (len(path) - 1)
        return min(interior, key=lambda n: abs(index_of[n] - target_idx))

    target = 0.5 * total
    best = None
    best_dist = None
    for n in interior:
        dist = abs(cum[index_of[n]] - target)
        if best is None or dist < best_dist:
            best = n
            best_dist = dist
    return best


def expand_sampled_graph_to_size(
    G_sample: nx.Graph,
    n_target: int,
    source_graph: nx.Graph,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Grow a subsampled MSC graph by reinserting dropped original nodes.

    This does **not** modify ``G_sample``. The input barycenter graphs should
    stay at their subsampled size; only the returned copy is expanded (used
    as the barycenter initialization / seed network).

    Algorithm
    ---------
    1. Push every current edge's polyline arc length onto a max-heap.
    2. Pop the longest edge. Among original nodes that were dropped on that
       edge, insert the one closest to cutting the polyline in half.
    3. If the edge has no remaining dropped node, skip it (do not re-push).
    4. Replace the split edge by the two sub-edges and push their lengths.
    5. Repeat until the graph has ``n_target`` nodes, or no splittable edge
       remains.

    Parameters
    ----------
    G_sample : nx.Graph
        Subsampled Morse graph (typically from :func:`sample_morse_graph`).
    n_target : int
        Desired node count (``N_max`` of the barycenter pipeline).
    source_graph : nx.Graph
        Graph before subsampling, with the original node attributes and the
        nodes that were dropped. Node ids must match ``G_sample``.

    Returns
    -------
    G_expanded : nx.Graph
        Copy of ``G_sample`` with dropped nodes reinserted.
    info : dict
        Diagnostics: ``n_inserted``, ``n_skipped_edges``, ``n_target``,
        ``n_start``, ``n_end``, ``inserted_nodes``.
    """
    G = G_sample.copy()
    info: Dict[str, Any] = {
        "n_start": int(G.number_of_nodes()),
        "n_target": int(n_target),
        "n_inserted": 0,
        "n_skipped_edges": 0,
        "inserted_nodes": [],
    }
    if n_target <= G.number_of_nodes():
        info["n_end"] = int(G.number_of_nodes())
        return G, info

    counter = itertools.count()
    heap: List[Tuple[float, int, Tuple]] = []

    def _push_edge(path: Sequence) -> None:
        if len(path) < 2:
            return
        length = _polyline_arc_length(source_graph, path)
        heapq.heappush(heap, (-float(length), next(counter), tuple(path)))

    for u, v, data in G.edges(data=True):
        _push_edge(_edge_path_nodes(G, source_graph, u, v, data))

    while G.number_of_nodes() < n_target and heap:
        _neg_len, _tick, path_t = heapq.heappop(heap)
        path = list(path_t)
        u, v = path[0], path[-1]
        data = G.get_edge_data(u, v)
        if data is None:
            continue
        current_path = _edge_path_nodes(G, source_graph, u, v, data)
        if tuple(current_path) != tuple(path) and tuple(reversed(current_path)) != tuple(path):
            continue
        if tuple(reversed(current_path)) == tuple(path):
            path = current_path

        present = set(G.nodes())
        w = _choose_split_node(source_graph, path, present)
        if w is None:
            info["n_skipped_edges"] += 1
            continue
        if w not in source_graph:
            info["n_skipped_edges"] += 1
            continue

        try:
            split_at = path.index(w)
        except ValueError:
            info["n_skipped_edges"] += 1
            continue
        left = path[: split_at + 1]
        right = path[split_at:]
        if len(left) < 2 or len(right) < 2:
            info["n_skipped_edges"] += 1
            continue

        G.add_node(w, **dict(source_graph.nodes[w]))
        edge_type = data.get("type", 0)
        separatrix_id = data.get("separatrixID")
        G.remove_edge(u, v)

        def _add_split_edge(a, b, subpath: List) -> None:
            attrs = {
                "weight": _edge_euclid(G, a, b),
                "arc_length": _polyline_arc_length(source_graph, subpath),
                "path_nodes": list(subpath),
            }
            if edge_type is not None:
                attrs["type"] = edge_type
            if separatrix_id is not None:
                attrs["separatrixID"] = separatrix_id
            G.add_edge(a, b, **attrs)

        _add_split_edge(u, w, left)
        _add_split_edge(w, v, right)
        _push_edge(left)
        _push_edge(right)
        info["n_inserted"] += 1
        info["inserted_nodes"].append(w)

    info["n_end"] = int(G.number_of_nodes())
    if G.number_of_nodes() < n_target:
        warnings.warn(
            "expand_sampled_graph_to_size reached "
            f"{G.number_of_nodes()} nodes (target {n_target}); "
            "no remaining dropped node could be inserted on a seed edge.",
            stacklevel=2,
        )
    return G, info


def summarize_graph_connectivity(
    G: nx.Graph,
    label: str = "graph",
    critical_nodes: Optional[List[int]] = None,
    top_k: int = 5,
) -> dict:
    """
    Print and return connectivity statistics for an MSC graph.

    Morse-Smale files are often stored as many short path components; a single
    connected graph is not guaranteed after loading.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        info = {
            "label": label,
            "n_nodes": 0,
            "n_edges": 0,
            "n_components": 0,
            "is_connected": True,
            "largest_component_size": 0,
            "critical_in_largest": 0,
            "critical_total": 0,
        }
        print(f"[{label}] empty graph")
        return info

    components = list(nx.connected_components(G))
    n_comp = len(components)
    sizes = sorted((len(c) for c in components), reverse=True)
    largest = max(components, key=len)
    is_connected = n_comp == 1

    crit_by_list = [n for n in (critical_nodes or []) if n in G]
    crit_by_type = [n for n in G.nodes() if _is_critical_node(G.nodes[n])]
    crit_in_largest = sum(1 for n in crit_by_list if n in largest)

    info = {
        "label": label,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": n_comp,
        "is_connected": is_connected,
        "largest_component_size": sizes[0],
        "component_sizes_top": sizes[:top_k],
        "critical_in_largest": crit_in_largest,
        "critical_total": len(crit_by_list),
        "critical_by_type": len(crit_by_type),
    }

    status = "CONNECTED" if is_connected else f"DISCONNECTED ({n_comp} components)"
    print(f"[{label}] {status}")
    print(f"  nodes={n_nodes}, edges={n_edges}, largest component={sizes[0]}")
    if not is_connected:
        print(f"  top component sizes: {sizes[:top_k]}")
    if critical_nodes is not None:
        print(
            f"  critical nodes (id list): {len(crit_by_list)} in graph, "
            f"{crit_in_largest} in largest component"
        )
        print(f"  critical nodes (separatrixType=0): {len(crit_by_type)}")
        if len(crit_by_list) != len(crit_by_type):
            warnings.warn(
                f"[{label}] critical id list ({len(crit_by_list)}) does not match "
                f"separatrixType=0 count ({len(crit_by_type)}); "
                "node ids may have been renumbered.",
                stacklevel=2,
            )
    return info


def prepare_graph_for_barycenter(
    G: nx.Graph,
    critical_nodes: List[int],
    sample_rate: Optional[float] = None,
    rate: float = 10.0,
    mode: SampleMode = "step",
    component: str = "largest",
    preserve_min_degree: int = 3,
) -> Tuple[nx.Graph, List[int]]:
    """
    Reduce an MSC graph for GW/PGW using path sampling between critical points.

    Parameters
    ----------
    G : nx.Graph
        Full Morse-Smale complex.
    critical_nodes : list of int
        Critical node ids (always kept by the sampler).
    sample_rate : float, optional
        Target spacing between kept regular nodes along each Morse path. In
        ``step`` mode this is an edge-hop count; in ``geo`` mode it is 3D
        Euclidean arc length. **Higher** ``sample_rate`` keeps **fewer** nodes
        (larger gap between consecutive kept nodes). If given, overrides
        ``rate``.
    rate : float
        Alias for ``sample_rate`` when ``sample_rate`` is omitted.
    mode : {'step', 'geo'}
        'step': spacing measured in edge hops; 'geo': spacing in 3D length.
    component : str
        'largest' (default): largest connected component — subsample is connected
        if the input component is connected (matches Sample2D assumptions).
        'largest_critical': component with the most critical nodes.
        'full': entire graph (often disconnected for MSC file data).
    preserve_min_degree : int
        Nodes with degree >= this value in the input graph are always kept
        (junctions where multiple Morse paths meet).
    """
    if sample_rate is not None:
        rate = sample_rate
    if G.number_of_nodes() == 0:
        return G.copy()

    crit_in_graph = [n for n in critical_nodes if n in G]
    if component == "full":
        Gc = G
    elif component == "largest":
        nodes_comp = max(nx.connected_components(G), key=len)
        Gc = G.subgraph(nodes_comp).copy()
    elif component == "largest_critical":
        crit_set = set(crit_in_graph)
        if not crit_set:
            nodes_comp = max(nx.connected_components(G), key=len)
            Gc = G.subgraph(nodes_comp).copy()
        else:
            candidates = [
                G.subgraph(comp).copy()
                for comp in nx.connected_components(G)
                if comp & crit_set
            ]
            Gc = max(candidates, key=lambda h: len(set(h.nodes()) & crit_set))
    else:
        raise ValueError(
            "component must be 'largest_critical', 'largest', or 'full'"
        )

    crit_sub = [n for n in critical_nodes if n in Gc]
    return sample_morse_graph(
        Gc,
        crit_sub,
        rate=rate,
        mode=mode,
        preserve_min_degree=preserve_min_degree,
    )


def load_morse_graphs(*filenames: str) -> Tuple[List["MorseGraph"], List[List[int]]]:
    """
    Load multiple Morse graphs from files.
    
    Parameters
    ----------
    *filenames : str
        Node filenames (edge files are auto-detected by replacing "Node" with "Edge")
        
    Returns
    -------
    graphs : list of MorseGraph
        List of loaded Morse graphs
    critical_nodes_list : list of list of int
        List of critical node IDs for each graph
    """
    try:
        from gwmc.core.morse_graph import MorseGraph
    except ImportError as exc:
        raise ImportError(
            "load_morse_graphs requires the gwmc package. "
            "Use read_morse_graph_txt for NetworkX graphs without gwmc."
        ) from exc

    graphs = []
    critical_nodes_list = []

    for filename in filenames:
        if not filename.endswith(".txt"):
            continue
        # Support both old naming (treeNodes) and new naming (graphNodes)
        if "Node" not in filename and "graphNode" not in filename:
            continue
        
        # Handle both naming conventions
        if "graphNode" in filename:
            edge_filename = filename.replace("graphNodes", "graphEdges")
        elif "treeNode" in filename:
            edge_filename = filename.replace("treeNodes", "treeEdges")
        else:
            edge_filename = filename.replace("Node", "Edge")
        if not os.path.exists(edge_filename):
            print(f"Warning: Edge file not found: {edge_filename}")
            continue
        
        graph, critical_nodes = read_morse_graph_txt(filename, edge_filename)
        graphs.append(MorseGraph(graph, critical_nodes))
        critical_nodes_list.append(critical_nodes)
    
    return graphs, critical_nodes_list


# Backward-compatible re-exports (prefer importing from the dedicated modules).
from lib.graph_relations import (  # noqa: E402
    apply_combined_relation,
    apply_relation_scaling,
    combined_relation_target_distances,
    compute_relation_rescale,
    relation_matrix,
)
from lib.msc_barycenter_reconstruction import (  # noqa: E402
    _inverse_relation_weight,
    msc_barycenter_weight_euclidean_debug,
    msc_barycenter_weight_euclidean_differences,
    reconstruct_msc_connectivity_based_graph,
    reconstruct_msc_barycenter_mst_graph,
    verify_msc_barycenter_reconstruction,
)
from lib.msc_plotting import (  # noqa: E402
    MSC_TYPE_COLORS,
    barycenter_xy_positions,
    draw_barycenter_colormap,
    draw_barycenter_thresholded_complete,
    draw_msc_barycenter_graph,
    draw_msc_graphs_overlay,
    draw_msc_nodes,
    draw_processed_msc_graphs,
    msc_node_visual_type,
    msc_type_color,
    msc_xy_positions,
    node_height,
    plot_msc_barycenter_weight_euclidean_debug,
    set_barycenter_axes_limits,
)
from lib.msc_barycenter_threshold import (  # noqa: E402
    build_complete_barycenter_graph,
    build_thresholded_barycenter_graph,
    thresholded_barycenter_edge_candidates,
)

__all__ = [
    "COORD_EPS",
    "REGULAR_NODE_TYPE",
    "SampleMode",
    "MSC_TYPE_COLORS",
    "apply_combined_relation",
    "apply_relation_scaling",
    "barycenter_xy_positions",
    "build_complete_barycenter_graph",
    "build_thresholded_barycenter_graph",
    "combined_relation_target_distances",
    "compute_relation_rescale",
    "draw_barycenter_colormap",
    "draw_barycenter_thresholded_complete",
    "draw_msc_barycenter_graph",
    "draw_msc_graphs_overlay",
    "draw_msc_nodes",
    "draw_processed_msc_graphs",
    "edge_separatrix_id_key",
    "expand_sampled_graph_to_size",
    "format_edge_separatrix_ids",
    "load_morse_graphs",
    "merge_duplicate_nodes",
    "msc_barycenter_weight_euclidean_debug",
    "msc_barycenter_weight_euclidean_differences",
    "msc_node_visual_type",
    "msc_type_color",
    "msc_xy_positions",
    "node_height",
    "plot_msc_barycenter_weight_euclidean_debug",
    "prepare_graph_for_barycenter",
    "read_morse_graph_txt",
    "reconstruct_msc_connectivity_based_graph",
    "reconstruct_msc_barycenter_mst_graph",
    "relation_matrix",
    "sample_morse_graph",
    "set_barycenter_axes_limits",
    "summarize_graph_connectivity",
    "thresholded_barycenter_edge_candidates",
    "verify_msc_barycenter_reconstruction",
]
