"""
Color transfer visualization for FGW barycenter.

Transfers colors from source contour trees to barycenter (and vice versa) via
the optimal transport coupling. Supports categorical and sequential colormaps.
For nodes with multiple significant sources (>1% of row/column sum), draws pie charts.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
from matplotlib import cm

# Optional: import coupling solver
try:
    from .gromov import partial_fused_gromov_ver1, PFGW_dist_with_penalty
    from .gromov_barycenter import pfgw_barycenters
    from ot.utils import dist
    _HAS_GROMOV = True
except ImportError:
    _HAS_GROMOV = False


def _get_height(G: nx.Graph, node) -> float:
    return G.nodes[node].get("height", 0.0)


def _parse_order_spec(order_spec: Optional[str], G: nx.Graph) -> Callable:
    """
    Parse order_spec into an order_key(node) callable for sequential color mode.

    Supported formats:
    - "height" : sort by (height, node_id)
    - "id" : sort by node id
    - "(x, y, z)" : sort by tuple of attributes (lexicographic)
    - "x+y+z" : sort by sum of attributes
    - "attr" : single attribute, sort by (attr, node_id)
    - None : default (height, node_id)
    """
    if order_spec is None or order_spec.strip() == "":
        return lambda n: (_get_height(G, n), n)
    spec = order_spec.strip()
    if spec == "height":
        return lambda n: (_get_height(G, n), n)
    if spec == "id":
        return lambda n: n
    # (x, y, z) - tuple of attributes
    if spec.startswith("(") and spec.endswith(")"):
        inner = spec[1:-1].strip()
        attrs = [a.strip() for a in inner.split(",") if a.strip()]
        if not attrs:
            return lambda n: (_get_height(G, n), n)
        return lambda n: tuple(G.nodes[n].get(a, 0) for a in attrs)
    # x+y+z - sum of attributes
    if "+" in spec:
        attrs = [a.strip() for a in spec.split("+") if a.strip()]
        if not attrs:
            return lambda n: (_get_height(G, n), n)
        return lambda n: sum(G.nodes[n].get(a, 0) for a in attrs)
    # single attribute
    return lambda n: (G.nodes[n].get(spec, 0), n)


def compute_coupling_source_barycenter(
    idx_source: int,
    C_center: np.ndarray,
    X_center: np.ndarray,
    C_source: np.ndarray,
    Y_source: np.ndarray,
    p_center: np.ndarray,
    p_source: np.ndarray,
    Lambda: Optional[float] = None,
    alpha: float = 0.5,
    numItermax: int = 1000,
    tol: float = 1e-5,
    seed: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Compute coupling T between barycenter and source.
    T has shape (N_center, n_source). T[i,j] = mass from barycenter node i to source node j.

    Parameters
    ----------
    idx_source : int
        Source index (for logging only).
    C_center, X_center : barycenter structure and features
    C_source, Y_source : source structure and features
    p_center, p_source : distributions
    Lambda : penalty for partial GW (default: max of C_source)
    alpha : FGW trade-off
    """
    if not _HAS_GROMOV:
        raise ImportError("gromov module required for compute_coupling_source_barycenter")
    if Lambda is None:
        Lambda = float(np.max(C_source))
    M = dist(X_center, Y_source)
    # Solver returns (N_center, n_source): G0[i,j] = mass from barycenter i to source j
    G0 = partial_fused_gromov_ver1(
        M, C_center, C_source, p_center, p_source, Lambda,
        alpha=alpha, G0=None, numItermax_gw=numItermax, tol=tol, seed=seed, **kwargs
    )
    return np.asarray(G0, dtype=float)


def assign_source_colors(
    G: nx.Graph,
    nodes_order: List,
    mode: str = "categorical",
    attr: str = "type",
    cmap: Optional[str] = None,
    order_key: Optional[Callable] = None,
) -> Tuple[Dict[Any, str], List[str]]:
    """
    Assign colors to source graph nodes.

    Parameters
    ----------
    G : nx.Graph
        Source contour tree.
    nodes_order : list
        Node order for matrix indices (nodes_order[i] = node at index i).
    mode : str
        "categorical" = each node gets its own unique color (one color per node).
        "sequential" = color by position in sorted order (order_key).
    attr : str
        Unused when mode="categorical". For legacy compatibility.
    cmap : str
        Matplotlib colormap name. For categorical: tab10 or similar (cycles for many nodes).
        For sequential: Spectral (default, rainbow-style divergent), coolwarm, viridis, etc.
    order_key : callable, optional
        For sequential mode: key(node) to sort nodes. Default: height then node id.

    Returns
    -------
    node_to_color : dict
        Maps node -> hex color string.
    color_list : list
        Colors in nodes_order for use with coupling.
    """
    node_to_color = {}
    if mode == "categorical":
        # Each node gets its own unique color (by index in nodes_order)
        if cmap is None:
            cmap = "tab10"
        cmap_obj = plt.get_cmap(cmap)
        n_nodes = len(nodes_order)
        for i, n in enumerate(nodes_order):
            node_to_color[n] = mcolors.to_hex(cmap_obj(i % len(cmap_obj.colors)))
    else:
        # sequential mode (accept "serial" as legacy alias)
        if order_key is None:
            order_key = lambda n: (_get_height(G, n), n)
        sorted_nodes = sorted(nodes_order, key=order_key)
        if cmap is None:
            cmap = "Spectral"
        cmap_obj = plt.get_cmap(cmap)
        n_nodes = len(sorted_nodes)
        for i, n in enumerate(sorted_nodes):
            node_to_color[n] = mcolors.to_hex(cmap_obj(i / max(n_nodes - 1, 1)))
    color_list = [node_to_color[n] for n in nodes_order]
    return node_to_color, color_list


def transfer_colors_via_coupling(
    source_colors: List[str],
    p_target: np.ndarray,
    T: np.ndarray,
    direction: str,
    threshold: float = 0.01,
    match_threshold: float = 0.01,
) -> Tuple[List[Union[str, List[Tuple[str, float]]]], List[bool]]:
    """
    Transfer colors from source to target using coupling T.

    T has shape (N_center, n_source). T[i,j] = mass from source j to barycenter i.

    Parameters
    ----------
    source_colors : list of str
        Colors for source nodes (index j in source_to_target; index i in target_to_source).
    p_target : ndarray
        Expected mass per target row (same row order as the effective matrix ``W``).
        - source_to_target: rows correspond to barycenter nodes -> use p_center
        - target_to_source: rows correspond to source/input nodes -> use p_source[s]
    T : ndarray
        Coupling (N_center, n_source). T[i,j] = mass from source j to barycenter i.
    direction : str
        "source_to_target" : target = barycenter rows, get color from source columns.
        "target_to_source" : target = source rows, get color from barycenter columns.
    threshold : float
        For pie chart: include source if its weight >= threshold * row_sum.
    match_threshold : float
        Row sum must be >= match_threshold * expected_row_sum to count as a valid match.
        Expected row sum = total_mass / n_target. Otherwise treat as noise (black).

    Returns
    -------
    target_colors : list
        For each target node: either a single hex color, or a list of (color, frac) for pie.
    use_pie : list of bool
        True if node should use pie chart.
    """
    if direction == "source_to_target":
        # rows = target (barycenter), cols = source. Target i gets color from source j.
        W = T  # (N_center, n_source)
    else:
        # target = source, get color from barycenter. W[j,i] = T[i,j]
        W = T.T  # (n_source, N_center)
    n_target, n_source = W.shape
    expected_row_sum = np.asarray(p_target, dtype=float).reshape(-1)
    if expected_row_sum.shape[0] != n_target:
        raise ValueError(
            f"p_target length {expected_row_sum.shape[0]} does not match target rows {n_target} "
            f"(direction={direction})."
        )
    min_valid_row_sum = match_threshold * expected_row_sum

    source_colors = list(source_colors)
    if len(source_colors) != n_source:
        source_colors = source_colors[:n_source] + ["#808080"] * (n_source - len(source_colors))
    target_colors = []
    use_pie = []
    for i in range(n_target):
        row = W[i, :]
        row_sum = row.sum()
        if row_sum <= 0 or row_sum < min_valid_row_sum[i]:
            target_colors.append("black")
            use_pie.append(False)
            continue
        fracs = row / row_sum
        significant = [(j, fracs[j]) for j in range(n_source) if fracs[j] >= threshold]
        if len(significant) <= 1:
            if significant:
                j, _ = significant[0]
                target_colors.append(source_colors[j])
            else:
                target_colors.append("black")
            use_pie.append(False)
        else:
            target_colors.append([(source_colors[j], float(fracs[j])) for j, _ in significant])
            use_pie.append(True)
    return target_colors, use_pie


def _mask_unmatched_source_colors(
    source_colors: List[str],
    T: np.ndarray,
    direction: str,
    match_threshold: float = 0.01,
    unmatched_color: str = "white",
) -> List[str]:
    """
    Return source_colors with unmatched source nodes set to unmatched_color.
    A source node is unmatched if its total mass in T is below match_threshold * expected.
    """
    total = float(T.sum())
    if direction == "source_to_target":
        # T: (n_target, n_source), source = columns
        masses = T.sum(axis=0)
        n_source = T.shape[1]
    else:
        # target_to_source: source = rows of T (barycenter)
        masses = T.sum(axis=1)
        n_source = T.shape[0]
    expected = total / n_source if n_source > 0 and total > 0 else 0.0
    min_valid = match_threshold * expected
    out = list(source_colors)
    if len(out) < n_source:
        out = out + ["#808080"] * (n_source - len(out))
    for j in range(n_source):
        if masses[j] < min_valid:
            out[j] = unmatched_color
    return out


def transfer_colors_composed(
    source_colors_1: List[str],
    p_target: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    threshold: float = 0.01,
    match_threshold: float = 0.01,
) -> Tuple[List[Union[str, List[Tuple[str, float]]]], List[bool]]:
    """
    Transfer colors from source 1 to target 2 via barycenter.
    Composed coupling: (T1.T @ T2) gives flow from source1 to source2.
    T1: (N_center, n1), T2: (N_center, n2).
    """
    # (T1.T @ T2)[j,k] = sum_i T1[i,j] * T2[i,k] = flow from source1 j to source2 k
    composed = T1.T @ T2  # (n1, n2)
    return transfer_colors_via_coupling(
        source_colors_1, p_target, composed, "source_to_target",
        threshold=threshold, match_threshold=match_threshold
    )


def _draw_pie_at(pos: Tuple[float, float], color_fracs: List[Tuple[str, float]], size: float, ax):
    """Draw a pie chart at position. color_fracs: list of (hex_color, fraction)."""
    if not color_fracs:
        return
    total = sum(f for _, f in color_fracs)
    if total <= 0:
        return
    start_deg = 0.0
    for color, frac in color_fracs:
        if frac <= 0:
            continue
        span_deg = 360.0 * frac / total
        w = Wedge(pos, size, start_deg, start_deg + span_deg, facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_patch(w)
        start_deg += span_deg


def _draw_single_contour_tree(
    G: nx.Graph,
    node_colors: Union[Dict, List],
    nodes_order: List,
    use_pie: Optional[List[bool]] = None,
    ax=None,
    title: Optional[str] = None,
    node_size: float = 120,
    pie_size: float = 0.03,
    edge_color: str = "gray",
    edge_linewidth: float = 1.5,
    show_labels: bool = True,
    width: float = 1.0,
    height_scale: float = 1.0,
):
    """
    Draw a single contour tree using plot_contour_tree layout (contour_tree_layout + edge_segments).
    node_colors: dict node->color, or list of colors indexed by nodes_order.
    use_pie: optional list, if True for node i draw pie chart.
    """
    from .contour_tree_visualization import contour_tree_layout

    pos, branches, edge_segments = contour_tree_layout(
        G, root=None, width=width, height_scale=height_scale
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    lc = LineCollection(
        edge_segments,
        colors=edge_color,
        linewidths=edge_linewidth,
        zorder=0,
    )
    ax.add_collection(lc)

    node_to_idx = {n: i for i, n in enumerate(nodes_order)}
    xs_t, ys_t, colors_t = [], [], []
    for n in G.nodes():
        idx = node_to_idx.get(n, -1)
        if idx < 0:
            continue
        pos_n = pos[n]
        if isinstance(node_colors, dict):
            col = node_colors.get(n, "#808080")
        else:
            col = node_colors[idx] if idx < len(node_colors) else "#808080"
        if use_pie is not None and idx < len(use_pie) and use_pie[idx] and isinstance(col, list):
            _draw_pie_at(pos_n, col, pie_size, ax)
        else:
            c = col if isinstance(col, str) else "#808080"
            xs_t.append(pos_n[0])
            ys_t.append(pos_n[1])
            colors_t.append(c)
    if xs_t:
        ax.scatter(xs_t, ys_t, s=node_size, c=colors_t, zorder=1, edgecolors="black", linewidths=0.5)
    if show_labels:
        for n in G.nodes():
            col = node_colors.get(n, "#808080") if isinstance(node_colors, dict) else (
                node_colors[node_to_idx[n]] if node_to_idx.get(n, -1) >= 0 and node_to_idx[n] < len(node_colors) else "#808080"
            )
            is_light = isinstance(col, str) and (col.lower() in ("#ffffff", "#fff", "white") or col.lower().startswith("#fff"))
            lbl_color = "black" if is_light else "white"
            ax.annotate(str(n), pos[n], fontsize=7, ha="center", va="center", color=lbl_color, weight="bold")
    ax.set_aspect("equal")
    ax.autoscale()
    if title:
        ax.set_title(title)
    return ax


def _add_sequential_colorbar(fig, ax, cmap_name: str = "Spectral", label: str = "Sequential order"):
    """Add a colorbar for sequential colormap to the given figure/axes."""
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, label=label)
    return cbar


def plot_color_transfer(
    G_source: nx.Graph,
    G_target: nx.Graph,
    nodes_source: List,
    nodes_target: List,
    target_colors: List[Union[str, List[Tuple[str, float]]]],
    use_pie: List[bool],
    pos_source: Optional[Dict] = None,
    pos_target: Optional[Dict] = None,
    source_colors: Optional[Dict] = None,
    ax=None,
    title: Optional[str] = None,
    node_size: float = 120,
    pie_size: float = 0.03,
    edge_color: str = "gray",
    show_labels: bool = True,
    **kwargs,
):
    """
    Plot source and target trees with color transfer.

    Parameters
    ----------
    G_source, G_target : nx.Graph
        Source and target contour trees.
    nodes_source, nodes_target : list
        Node order for source/target (for layout).
    target_colors : list
        Colors for target nodes (indexed by nodes_target). Either hex or list of (color, frac).
    use_pie : list of bool
        Whether to draw pie chart at each target node.
    pos_source, pos_target : dict, optional
        Precomputed positions. If None, use contour_tree_layout.
    source_colors : dict, optional
        node -> color for source. If None, use G_source.nodes[n]["_plot_color"] or "steelblue".
    """
    from .contour_tree_visualization import contour_tree_layout

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        ax = np.atleast_1d(ax)
    if pos_source is None:
        pos_source, _, _ = contour_tree_layout(G_source, width=1.0, height_scale=1.0)
    if pos_target is None:
        pos_target, _, _ = contour_tree_layout(G_target, width=1.0, height_scale=1.0)

    def _edge_segments(G, pos):
        segs = []
        for u, v in G.edges():
            segs.append([pos[u], pos[v]])
        return segs

    # Plot source (ax must have >= 2 elements for source | target)
    ax0 = np.atleast_1d(ax).flat[0]
    ax1 = np.atleast_1d(ax).flat[1] if np.atleast_1d(ax).size >= 2 else ax0
    segs_s = _edge_segments(G_source, pos_source)
    lc = LineCollection(segs_s, colors=edge_color, linewidths=1.5, zorder=0)
    ax0.add_collection(lc)
    xs = [pos_source[n][0] for n in G_source.nodes()]
    ys = [pos_source[n][1] for n in G_source.nodes()]
    if source_colors is not None:
        source_colors_plot = [source_colors.get(n, "steelblue") for n in G_source.nodes()]
    else:
        source_colors_plot = [G_source.nodes[n].get("_plot_color", "steelblue") for n in G_source.nodes()]
    ax0.scatter(xs, ys, s=node_size, c=source_colors_plot, zorder=1, edgecolors="black", linewidths=0.5)
    if show_labels:
        for n in G_source.nodes():
            ax0.annotate(str(n), pos_source[n], fontsize=7, ha="center", va="center", color="white", weight="bold")
    ax0.set_aspect("equal")
    ax0.autoscale()
    ax0.set_title("Source" if title is None else f"Source: {title}")

    # Plot target
    segs_t = _edge_segments(G_target, pos_target)
    lc = LineCollection(segs_t, colors=edge_color, linewidths=1.5, zorder=0)
    ax1.add_collection(lc)
    node_to_idx = {n: i for i, n in enumerate(nodes_target)}
    xs_t, ys_t, colors_t = [], [], []
    for n in G_target.nodes():
        idx = node_to_idx.get(n, -1)
        if idx < 0:
            continue
        pos = pos_target[n]
        col = target_colors[idx]
        if use_pie[idx] and isinstance(col, list):
            _draw_pie_at(pos, col, pie_size, ax1)
        else:
            c = col if isinstance(col, str) else "#808080"
            xs_t.append(pos[0])
            ys_t.append(pos[1])
            colors_t.append(c)
    if xs_t:
        ax1.scatter(xs_t, ys_t, s=node_size, c=colors_t, zorder=1, edgecolors="black", linewidths=0.5)
    if show_labels:
        for n in G_target.nodes():
            ax1.annotate(str(n), pos_target[n], fontsize=7, ha="center", va="center", color="white", weight="bold")
    ax1.set_aspect("equal")
    ax1.autoscale()
    ax1.set_title("Target" if title is None else f"Target: {title}")

    plt.tight_layout()
    return ax


def color_transfer_pipeline(
    x_1: int,
    x_2: int,
    Gs: List[nx.Graph],
    Cs: List[np.ndarray],
    Ys: List[np.ndarray],
    nodes_list: List[List],
    C_center: np.ndarray,
    X_center: np.ndarray,
    G_center: Optional[nx.Graph] = None,
    C_center_tree: Optional[np.ndarray] = None,
    X_center_tree: Optional[np.ndarray] = None,
    nodes_order_center: Optional[List] = None,
    p_source: Optional[List[np.ndarray]] = None,
    p_center: Optional[np.ndarray] = None,
    Lambda_list: Optional[List[float]] = None,
    alpha: float = 0.5,
    color_mode: str = "categorical",
    color_attr: str = "type",
    sequential_order: Optional[str] = None,
    sequential_cmap: Optional[str] = None,
    threshold: float = 0.01,
    match_threshold: float = 0.01,
    recompute_coupling: bool = True,
    T1: Optional[np.ndarray] = None,
    T2: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (6, 5),
    pie_size: float = 0.03,
) -> Dict:
    """
    Full pipeline: compute couplings, assign colors, transfer, and plot.

    When G_center is postprocessed (e.g. degree-2 saddles suppressed), derive C_center_tree,
    X_center_tree from G_center via matrices_from_contour_tree and pass them here. The
    coupling and color transfer will use the tree matrices for consistency.

    Parameters
    ----------
    x_1, x_2 : int
        Source indices for the three visualizations.
    Gs : list of nx.Graph
        Input contour trees.
    Cs, Ys, nodes_list : from barycenter notebook
    C_center, X_center : barycenter matrices (original, before postprocessing)
    G_center : postprocessed barycenter graph (e.g. after MST + degree-2 saddle suppression)
    C_center_tree, X_center_tree : optional, derived from G_center via matrices_from_contour_tree.
        When provided, used for coupling instead of C_center, X_center.
    nodes_order_center : optional, node order for G_center (from matrices_from_contour_tree).
        When C_center_tree is provided, should match the row order of C_center_tree.
    p_source, p_center : distributions
    Lambda_list : partial GW penalties
    alpha : FGW trade-off
    color_mode : "categorical" | "sequential"
    color_attr : attribute for categorical
    sequential_order : sort order for sequential mode. Examples:
        "height" | "id" | "(x, y, z)" (tuple) | "x+y+z" (sum) | None for default
    sequential_cmap : colormap for sequential mode (default: Spectral). Used for colorbar.
    threshold : pie chart threshold (fraction of row/column sum)
    match_threshold : row sum must be >= match_threshold * expected_row_sum to count as valid match
    recompute_coupling : if True, recompute T before plotting (ignored if T1, T2 provided)
    T1, T2 : optional precomputed couplings (N_center, n_source). If both provided, skip recompute.

    Returns
    -------
    dict with keys: T1, T2, fig, axes, plots
    """
    if not _HAS_GROMOV:
        raise ImportError("gromov module required")
    n_sources = len(Cs)
    if x_1 < 0 or x_1 >= n_sources or x_2 < 0 or x_2 >= n_sources:
        raise ValueError("x_1 and x_2 must be valid source indices")

    # Use tree matrices when provided (for postprocessed G_center)
    use_tree = C_center_tree is not None and X_center_tree is not None
    if use_tree:
        C_use = C_center_tree
        X_use = X_center_tree
        if nodes_order_center is None:
            raise ValueError("nodes_order_center must be provided when C_center_tree/X_center_tree are used")
    else:
        C_use = C_center
        X_use = X_center

    N = C_use.shape[0]
    if X_use.shape[0] != N:
        raise ValueError(
            f"Dimension mismatch: C has {N} rows, X has {X_use.shape[0]}. "
            "Ensure C and X come from the same barycenter."
        )
    if p_center is not None and len(p_center) != N:
        raise ValueError(f"p_center length {len(p_center)} != N_center {N}.")
    if p_center is None:
        p_center = np.ones(N) / N
    if p_source is None:
        p_source = [np.ones(C.shape[0]) for C in Cs]
    if Lambda_list is None:
        Lambda_list = [float(np.max(Cs[s])) for s in range(n_sources)]

    # Compute or use precomputed couplings (both must have same N_center)
    if T1 is not None and T2 is not None:
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)
    else:
        T1 = compute_coupling_source_barycenter(
            x_1, C_use, X_use, Cs[x_1], Ys[x_1], p_center, p_source[x_1],
            Lambda=Lambda_list[x_1], alpha=alpha
        )
        T2 = compute_coupling_source_barycenter(
            x_2, C_use, X_use, Cs[x_2], Ys[x_2], p_center, p_source[x_2],
            Lambda=Lambda_list[x_2], alpha=alpha
        )
    if T1.shape[0] != T2.shape[0]:
        raise ValueError(
            f"Coupling dimension mismatch: T1 has {T1.shape[0]} rows, T2 has {T2.shape[0]}. "
            "Ensure C_center, X_center, p_center are consistent."
        )

    # Build barycenter graph for visualization if not provided
    if G_center is None:
        G_center = nx.Graph()
        for i in range(N):
            h = float(X_use[i, 0]) if X_use.shape[1] > 0 else 0.0
            G_center.add_node(i, height=h)
        for i in range(N):
            for j in range(i + 1, N):
                if C_use[i, j] > 0:
                    G_center.add_edge(i, j, weight=abs(C_use[i, j]))
        nodes_order_center = list(range(N))
    elif nodes_order_center is None:
        nodes_order_center = list(range(N))

    order_key = None
    order_key_center = None
    if color_mode in ("sequential", "serial"):  # "serial" is legacy alias
        order_key = _parse_order_spec(sequential_order, Gs[x_1])
        order_key_center = _parse_order_spec(sequential_order, G_center)

    # Assign source colors for x_1
    seq_cmap = sequential_cmap or "Spectral"
    _, colors_1 = assign_source_colors(
        Gs[x_1], nodes_list[x_1], mode=color_mode, attr=color_attr,
        order_key=order_key, cmap=seq_cmap if color_mode in ("sequential", "serial") else None
    )
    # Unmatched source nodes (no significant mass to target) -> white for (a) and (b)
    colors_1_masked = _mask_unmatched_source_colors(
        colors_1, T1, "source_to_target", match_threshold=match_threshold, unmatched_color="white"
    )
    source_colors_dict = {nodes_list[x_1][i]: colors_1_masked[i] for i in range(len(nodes_list[x_1]))}

    # (a) x_1 -> barycenter
    target_colors_a, use_pie_a = transfer_colors_via_coupling(
        colors_1, p_center, T1, "source_to_target", threshold=threshold, match_threshold=match_threshold
    )
    # (b) barycenter -> x_1: reassign colors to barycenter (source), then transfer to x_1
    _, bary_colors_list = assign_source_colors(
        G_center, nodes_order_center, mode=color_mode, attr=color_attr,
        order_key=order_key_center, cmap=seq_cmap if color_mode in ("sequential", "serial") else None
    )
    bary_colors_masked = _mask_unmatched_source_colors(
        bary_colors_list, T1, "target_to_source", match_threshold=match_threshold, unmatched_color="white"
    )
    target_colors_b, use_pie_b = transfer_colors_via_coupling(
        bary_colors_list, p_source[x_1], T1, "target_to_source", threshold=threshold, match_threshold=match_threshold
    )
    # (c) x_1 -> barycenter (apply transfer), then barycenter -> x_2 (apply transfer, no reassignment)
    # Step 1: barycenter colors from x_1 are target_colors_a (already computed)
    # Step 2: flatten barycenter colors (dominant color per node) and transfer to x_2 via T2
    bary_colors_from_a = []
    for c in target_colors_a:
        if isinstance(c, str):
            bary_colors_from_a.append(c)
        elif isinstance(c, list) and c:
            bary_colors_from_a.append(c[0][0])
        else:
            bary_colors_from_a.append("black")
    target_colors_c, use_pie_c = transfer_colors_via_coupling(
        bary_colors_from_a, p_source[x_2], T2, "target_to_source", threshold=threshold, match_threshold=match_threshold
    )

    # Seven separate figures (each uses plot_contour_tree layout: contour_tree_layout + edge_segments)
    figs = []
    single_figsize = (figsize[0], figsize[1]) if figsize else (6, 5)

    # 1. Source (a): source x_1
    fig1, ax1 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        Gs[x_1], source_colors_dict, nodes_list[x_1],
        use_pie=None, ax=ax1, title=f"Source (a): T_{x_1}",
        node_size=120, pie_size=pie_size,
    )
    if color_mode in ("sequential", "serial"):
        _add_sequential_colorbar(fig1, ax1, cmap_name=seq_cmap)
    figs.append(fig1)

    # 2. Target (a): barycenter with colors from x_1
    fig2, ax2 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        G_center, target_colors_a, nodes_order_center,
        use_pie=use_pie_a, ax=ax2, title="Target (a): barycenter",
        node_size=120, pie_size=pie_size,
    )
    figs.append(fig2)

    # 3. Source (b): barycenter (reassigned colors; unmatched -> white)
    bary_colors_dict = {nodes_order_center[i]: bary_colors_masked[i] for i in range(N)}
    fig3, ax3 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        G_center, bary_colors_dict, nodes_order_center,
        use_pie=None, ax=ax3, title="Source (b): barycenter",
        node_size=120, pie_size=pie_size,
    )
    if color_mode in ("sequential", "serial"):
        _add_sequential_colorbar(fig3, ax3, cmap_name=seq_cmap)
    figs.append(fig3)

    # 4. Target (b): source x_1 with colors from barycenter
    fig4, ax4 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        Gs[x_1], target_colors_b, nodes_list[x_1],
        use_pie=use_pie_b, ax=ax4, title=f"Target (b): T_{x_1}",
        node_size=120, pie_size=pie_size,
    )
    figs.append(fig4)

    # 5. Source (c): source x_1
    fig5, ax5 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        Gs[x_1], source_colors_dict, nodes_list[x_1],
        use_pie=None, ax=ax5, title=f"Source (c): T_{x_1}",
        node_size=120, pie_size=pie_size,
    )
    if color_mode in ("sequential", "serial"):
        _add_sequential_colorbar(fig5, ax5, cmap_name=seq_cmap)
    figs.append(fig5)

    # 6. Intermediate Target (c): barycenter (colors from x_1)
    fig6, ax6 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        G_center, target_colors_a, nodes_order_center,
        use_pie=use_pie_a, ax=ax6, title="Intermediate Target (c): barycenter",
        node_size=120, pie_size=pie_size,
    )
    figs.append(fig6)

    # 7. Final Target (c): source x_2 with colors from x_1 via barycenter
    fig7, ax7 = plt.subplots(1, 1, figsize=single_figsize)
    _draw_single_contour_tree(
        Gs[x_2], target_colors_c, nodes_list[x_2],
        use_pie=use_pie_c, ax=ax7, title=f"Final Target (c): T_{x_2}",
        node_size=120, pie_size=pie_size,
    )
    figs.append(fig7)

    for fig in figs:
        fig.tight_layout()

    return {
        "T1": T1, "T2": T2,
        "figs": figs,
        "target_colors_a": target_colors_a, "target_colors_b": target_colors_b, "target_colors_c": target_colors_c,
        "use_pie_a": use_pie_a, "use_pie_b": use_pie_b, "use_pie_c": use_pie_c,
    }
