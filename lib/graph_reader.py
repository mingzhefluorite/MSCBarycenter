"""
Functions for reading and processing Morse graphs from files.
"""

import os
import warnings
import numpy as np
import networkx as nx
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING

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

    seen_edges = set()
    for edge in edges_data:
        u_orig = int(edge[0])
        v_orig = int(edge[1])
        edge_type = int(edge[2]) if len(edge) > 2 else 0

        u = orig_to_canon[u_orig]
        v = orig_to_canon[v_orig]
        if u == v:
            continue

        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        dist = float(
            np.linalg.norm(
                np.asarray([T.nodes[u]["x"], T.nodes[u]["y"], T.nodes[u]["z"]])
                - np.asarray([T.nodes[v]["x"], T.nodes[v]["y"], T.nodes[v]["z"]])
            )
        )
        T.add_edge(u, v, weight=dist, type=edge_type)

    return T, critical_nodes


def _node_xyz(G: nx.Graph, n) -> np.ndarray:
    return np.asarray([G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]], dtype=float)


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
    anchor_set = critical_set | junction_set
    G_sample = nx.Graph()
    visited = set()

    def _euclid(a, b) -> float:
        return float(np.linalg.norm(_node_xyz(G, a) - _node_xyz(G, b)))

    def _add_node(n):
        if not G_sample.has_node(n):
            G_sample.add_node(n, **G.nodes[n])

    def _add_edge(a, b, edge_type: int):
        if a == b:
            return
        _add_node(a)
        _add_node(b)
        G_sample.add_edge(a, b, weight=_euclid(a, b), type=edge_type)

    def dfs(start, edge_type: int = 0):
        if start in visited and start not in critical_set:
            return

        stack = [(start, start, 0.0)]

        while stack:
            start, node, length = stack.pop()

            if node in visited and node not in critical_set:
                continue

            visited.add(node)

            for n in G.neighbors(node):
                if n in visited:
                    continue
                if not _edge_matches_type(G, node, n, edge_type):
                    continue

                if n in anchor_set:
                    _add_edge(start, n, edge_type)
                    # Junctions (non-critical) must fan out into their other branches.
                    if n in junction_set and n not in critical_set:
                        stack.append((n, n, 0.0))
                    continue

                if mode == "step":
                    new_length = length + 1.0
                else:
                    new_length = length + _euclid(node, n)

                if new_length > rate:
                    _add_edge(start, n, edge_type)
                    stack.append((n, n, 0.0))
                else:
                    stack.append((start, n, new_length))

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
    draw_msc_nodes,
    msc_node_visual_type,
    msc_type_color,
    node_height,
    plot_msc_barycenter_weight_euclidean_debug,
)

__all__ = [
    "COORD_EPS",
    "REGULAR_NODE_TYPE",
    "SampleMode",
    "MSC_TYPE_COLORS",
    "apply_combined_relation",
    "apply_relation_scaling",
    "combined_relation_target_distances",
    "compute_relation_rescale",
    "draw_msc_nodes",
    "load_morse_graphs",
    "merge_duplicate_nodes",
    "msc_barycenter_weight_euclidean_debug",
    "msc_barycenter_weight_euclidean_differences",
    "msc_node_visual_type",
    "msc_type_color",
    "node_height",
    "plot_msc_barycenter_weight_euclidean_debug",
    "prepare_graph_for_barycenter",
    "read_morse_graph_txt",
    "reconstruct_msc_connectivity_based_graph",
    "reconstruct_msc_barycenter_mst_graph",
    "relation_matrix",
    "sample_morse_graph",
    "summarize_graph_connectivity",
    "verify_msc_barycenter_reconstruction",
]
