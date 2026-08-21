"""
Plotting helpers for Morse-Smale complex and MSC barycenter graphs.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lib.msc_barycenter_threshold import (
    select_thresholded_edges,
    thresholded_barycenter_edge_candidates,
)

# Keep local to avoid circular import with lib.graph_reader.
REGULAR_NODE_TYPE = -1

# MSC visualization colors (node type: 0=min, 1=saddle, 2=max, -1=regular).
MSC_TYPE_COLORS = {
    0: "blue",
    1: "white",
    2: "red",
    -1: "gray",
}


def msc_node_visual_type(attrs: dict) -> int:
    """Return node visualization type (0=min, 1=saddle, 2=max, -1=regular)."""
    return int(attrs.get("type", REGULAR_NODE_TYPE))


def msc_type_color(node_type: int) -> str:
    """Color for an MSC node by visualization type."""
    return MSC_TYPE_COLORS.get(int(node_type), MSC_TYPE_COLORS[REGULAR_NODE_TYPE])


def node_height(G: nx.Graph, n) -> float:
    """Scalar function value at a node (``height`` or legacy ``scalar_value``)."""
    data = G.nodes[n]
    if "height" in data:
        return float(data["height"])
    if "scalar_value" in data:
        return float(data["scalar_value"])
    return 0.0


def draw_msc_nodes(
    G: nx.Graph,
    pos: dict,
    ax,
    *,
    regular_size: float = 40,
    critical_size: float = 40,
    edgecolors: str = "black",
    linewidths: float = 0.5,
) -> None:
    """Draw MSC nodes colored by Morse type (min=blue, saddle=white, max=red, regular=gray)."""
    by_type: dict = {0: [], 1: [], 2: [], REGULAR_NODE_TYPE: []}
    for n in G.nodes():
        t = msc_node_visual_type(G.nodes[n])
        by_type.setdefault(t, []).append(n)

    for t in (REGULAR_NODE_TYPE, 0, 1, 2):
        nodes = by_type.get(t, [])
        if not nodes:
            continue
        size = critical_size if t != REGULAR_NODE_TYPE else regular_size
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=msc_type_color(t),
            node_size=size,
            edgecolors=edgecolors,
            linewidths=linewidths,
            ax=ax,
        )


def plot_msc_barycenter_weight_euclidean_debug(
    G: nx.Graph,
    *,
    relation_rescale: Optional[float] = None,
    relation_offset: Optional[float] = None,
    relation_kind: Optional[str] = None,
    bins: int = 40,
    by_pass: bool = True,
    figsize=(12.0, 4.0),
):
    """
    Plot the distribution of (unscaled edge weight - Euclidean distance).

    Parameters
    ----------
    by_pass : bool
        If True, overlay histograms per ``edge_pass`` (mst / strict / loose).
    relation_rescale, relation_offset, relation_kind : optional
        Inverse scaling for stored edge weights (default: read from ``G.graph``).
    """
    import matplotlib.pyplot as plt

    from lib.msc_barycenter_reconstruction import msc_barycenter_weight_euclidean_debug

    info = msc_barycenter_weight_euclidean_debug(
        G,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        relation_kind=relation_kind,
    )
    diffs = info["differences"]
    records = info["records"]

    if by_pass and records:
        passes = ["mst", "strict", "loose"]
        colors = {"mst": "#444444", "strict": "#2ca02c", "loose": "#d62728"}
        ncols = 2
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        ax_all, ax_pass = axes

        ax_all.hist(diffs, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax_all.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax_all.set_title("All edges: unscaled weight − Euclidean")
        ax_all.set_xlabel("difference")
        ax_all.set_ylabel("count")
        ax_all.grid(False)

        for p in passes:
            sub = [r["difference"] for r in records if r.get("edge_pass") == p]
            if sub:
                ax_pass.hist(
                    sub,
                    bins=bins,
                    alpha=0.55,
                    label=f"{p} (n={len(sub)})",
                    color=colors.get(p, "#888888"),
                    edgecolor="white",
                )
        ax_pass.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax_pass.set_title("By reconstruction pass")
        ax_pass.set_xlabel("difference")
        ax_pass.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )
        ax_pass.grid(False)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
    else:
        fig, ax = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
        ax.hist(diffs, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Unscaled edge weight − Euclidean distance")
        ax.set_xlabel("difference")
        ax.set_ylabel("count")
        ax.grid(False)
        fig.tight_layout()

    plt.show()

    if len(diffs):
        spw = G.graph.get("sp_weighted", True)
        print(
            f"relation unscale: kind={info['relation_kind']}, sp_weighted={spw}, "
            f"rescale={info['relation_rescale']:.6g}, offset={info['relation_offset']:.6g}"
        )
        if info["relation_kind"] == "shortestpath" and not spw:
            print(
                "WARNING: sp_weighted=False uses hop-count paths; "
                "they are not in the same units as 3D Euclidean distance."
            )
        n_below = int(np.sum(diffs < -1e-9))
        print(
            f"unscaled weight − euclidean: n={len(diffs)}, "
            f"min={np.nanmin(diffs):.6g}, max={np.nanmax(diffs):.6g}, "
            f"mean={np.nanmean(diffs):.6g}, std={np.nanstd(diffs):.6g}, "
            f"below_eucl={n_below}"
        )
        ranked = sorted(
            records,
            key=lambda r: r["abs_difference"] if np.isfinite(r["abs_difference"]) else -1.0,
            reverse=True,
        )
        print("Top |difference| edges:")
        for r in ranked[:5]:
            print(
                f"  ({r['u']}, {r['v']}) pass={r['edge_pass']} "
                f"weight_scaled={r['weight_scaled']:.6g} weight_raw={r['weight']:.6g} "
                f"eucl={r['euclidean']:.6g} diff={r['difference']:.6g}"
            )

    return info


# ---------------------------------------------------------------------------
# Input MSC layouts / overlays
# ---------------------------------------------------------------------------


def msc_xy_positions(G: nx.Graph) -> Dict:
    """Layout from node attributes ``x``, ``y`` (as stored in MSC files)."""
    return {
        n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]], dtype=float) for n in G.nodes()
    }


def draw_processed_msc_graphs(
    graphs: Sequence[nx.Graph],
    mesh_ids: Optional[Sequence] = None,
    figsize=(16.0, 10.0),
    critical_size: float = 40,
    regular_size: float = 40,
    edge_width: float = 2.0,
    show_labels: bool = False,
) -> None:
    """
    One figure per graph. Node colors by Morse type; edges drawn in green.
    """
    if mesh_ids is None:
        mesh_ids = list(range(len(graphs)))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=10,
            label="Minimum (type=0)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            label="Saddle (type=1)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=10,
            label="Maximum (type=2)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Regular (type=-1)",
        ),
        Line2D([0], [0], color="#2ca02c", linewidth=2),
    ]

    for i, G in enumerate(graphs):
        fig, ax = plt.subplots(figsize=figsize)
        pos = msc_xy_positions(G)
        edges0 = list(G.edges())
        if edges0:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges0,
                edge_color="#2ca02c",
                width=edge_width,
                alpha=0.9,
                ax=ax,
            )
        draw_msc_nodes(
            G,
            pos,
            ax,
            regular_size=regular_size,
            critical_size=critical_size,
        )
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

        conn = (
            "connected"
            if nx.is_connected(G)
            else f"{nx.number_connected_components(G)} components"
        )
        mid = mesh_ids[i] if i < len(mesh_ids) else i
        ax.set_title(
            f"Subsampled MC/MSC — mesh {mid}\n"
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {conn}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show()


def draw_msc_graphs_overlay(
    graphs: Sequence[nx.Graph],
    mesh_ids: Optional[Sequence] = None,
    edge_colors: Optional[Sequence[str]] = None,
    figsize=(16.0, 8.0),
    edge_width: float = 1.2,
    edge_alpha: float = 0.55,
    show_nodes: bool = False,
    node_size: float = 8,
) -> None:
    """Draw subsampled MSC graphs overlaid for shape comparison."""
    if mesh_ids is None:
        mesh_ids = list(range(len(graphs)))
    if edge_colors is None:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        edge_colors = [palette[i % len(palette)] for i in range(len(graphs))]

    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []

    for i, G in enumerate(graphs):
        pos = msc_xy_positions(G)
        color = edge_colors[i]
        mid = mesh_ids[i] if i < len(mesh_ids) else i
        edgelist = list(G.edges())
        if edgelist:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edgelist,
                edge_color=color,
                width=edge_width,
                alpha=edge_alpha,
                ax=ax,
            )
        if show_nodes and G.number_of_nodes() > 0:
            draw_msc_nodes(
                G, pos, ax, regular_size=node_size, critical_size=node_size
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.5,
                label=(
                    f"mesh {mid} ({G.number_of_nodes()} nodes, "
                    f"{G.number_of_edges()} edges)"
                ),
            )
        )

    ax.set_title("MC/MSC input overlay (subsampled graphs)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    plt.show()


# ---------------------------------------------------------------------------
# Barycenter layouts / drawings
# ---------------------------------------------------------------------------


def barycenter_xy_positions(
    X_center: np.ndarray,
    min_x: float,
    min_y: float,
    domain_size: float,
) -> Dict[int, np.ndarray]:
    """Same (x, y) layout as ``reconstruct_msc_barycenter_mst_graph``."""
    X = np.asarray(X_center, dtype=float)
    return {
        i: np.array(
            [min_x + X[i, 0] * domain_size, min_y + X[i, 1] * domain_size],
            dtype=float,
        )
        for i in range(X.shape[0])
    }


def set_barycenter_axes_limits(ax, pos: Mapping, pad_frac: float = 0.05) -> None:
    """Set xlim/ylim from node positions; equal aspect, data-driven y-span."""
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    xspan = float(xs.max() - xs.min())
    yspan = float(ys.max() - ys.min())
    xpad = xspan * pad_frac if xspan > 0 else 1.0
    ypad = yspan * pad_frac if yspan > 0 else 1.0
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
    ax.set_aspect("equal", adjustable="box")


def draw_msc_barycenter_graph(G: nx.Graph, figsize=(16.0, 12.0)) -> None:
    """Draw an MST (or any) reconstructed barycenter graph in the (x, y) plane."""
    pos = {
        n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]], dtype=float) for n in G.nodes()
    }
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G, pos, width=1.5, alpha=0.7, edge_color="#444444", ax=ax
    )
    draw_msc_nodes(G, pos, ax, regular_size=5, critical_size=5)
    ax.set_title(
        f"PFGW barycenter — {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges (MST)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=[Line2D([0], [0], color="#444444", linewidth=2, label="MST")],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def _kept_edges_for_plot(
    C_center: np.ndarray,
    *,
    relation_kind: str,
    relation_rescale: float,
    relation_offset: float,
    edge_fraction_pct: float,
):
    candidates = thresholded_barycenter_edge_candidates(
        C_center,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
    )
    kept = select_thresholded_edges(candidates, edge_fraction_pct)
    return candidates, kept


def draw_barycenter_thresholded_complete(
    C_center: np.ndarray,
    X_center: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    edge_fraction_pct: float = 10.0,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
    edge_width: float = 1.0,
    edge_color: str = "#1f77b4",
    node_size: float = 5,
    figsize=(16.0, 12.0),
    pos: Optional[Mapping] = None,
    pad_frac: float = 0.05,
    node_attr_graph: Optional[nx.Graph] = None,
) -> Optional[List]:
    """
    Visualize the FGW barycenter as a thresholded complete graph.

    Edge selection matches ``build_thresholded_barycenter_graph``.
    ``node_attr_graph`` (e.g. MST ``G_center``) supplies Morse types for nodes.
    """
    del min_z  # reserved for API symmetry with reconstruction helpers
    X = np.asarray(X_center, dtype=float)
    candidates, kept = _kept_edges_for_plot(
        C_center,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        edge_fraction_pct=edge_fraction_pct,
    )
    if not candidates:
        print("No finite positive edges in C_center.")
        return None

    w_min, w_max = kept[0][0], kept[-1][0]

    def weight_to_alpha(w: float) -> float:
        if w_max <= w_min:
            return alpha_max
        t = (w - w_min) / (w_max - w_min)
        return float(alpha_max - t * (alpha_max - alpha_min))

    if pos is None:
        pos = barycenter_xy_positions(X, min_x, min_y, domain_size)

    fig, ax = plt.subplots(figsize=figsize)
    set_barycenter_axes_limits(ax, pos, pad_frac=pad_frac)

    segments = []
    alphas = []
    rgba_base = mcolors.to_rgba(edge_color)
    for w, u, v in kept:
        segments.append([pos[u], pos[v]])
        a = weight_to_alpha(w)
        alphas.append((rgba_base[0], rgba_base[1], rgba_base[2], a))
    if segments:
        ax.add_collection(LineCollection(segments, colors=alphas, linewidths=edge_width))

    G_nodes = nx.Graph()
    for n in pos:
        attrs = dict(node_attr_graph.nodes[n]) if (
            node_attr_graph is not None and node_attr_graph.has_node(n)
        ) else {}
        G_nodes.add_node(n, **attrs)
    draw_msc_nodes(G_nodes, pos, ax, regular_size=node_size, critical_size=node_size)

    ax.set_title(
        f"Barycenter thresholded complete graph\n"
        f"{len(kept)} / {len(candidates)} edges (smallest {edge_fraction_pct:g}%), "
        f"weight [{w_min:.4g}, {w_max:.4g}]"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()
    print(
        f"Kept {len(kept)} edges ({edge_fraction_pct:g}% of {len(candidates)} candidates); "
        f"alpha in [{alpha_min}, {alpha_max}] by weight"
    )
    return kept


def draw_barycenter_colormap(
    C_center: np.ndarray,
    X_center: np.ndarray,
    node_colors: Sequence,
    *,
    color_mode: str = "bilinear",
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    min_z: float,
    domain_size: float,
    corner_colors: Sequence[str] = ("#11FF22", "#66FFFF", "#FFEE33", "#FF3366"),
    categorical_cmap: str = "tab20",
    categorical_max_colors: int = 20,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    edge_fraction_pct: float = 10.0,
    alpha_min: float = 0.2,
    alpha_max: float = 1.0,
    edge_width: float = 1.0,
    edge_color: str = "black",
    node_size: float = 30,
    figsize=(16.0, 12.0),
    pos: Optional[Mapping] = None,
    pad_frac: float = 0.05,
) -> Optional[Sequence]:
    """
    Thresholded-complete-graph barycenter view with custom per-node colors.

    Edge selection matches ``build_thresholded_barycenter_graph`` /
    ``draw_barycenter_thresholded_complete``.
    """
    del min_z
    X = np.asarray(X_center, dtype=float)
    n = X.shape[0]
    candidates, kept = _kept_edges_for_plot(
        C_center,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        edge_fraction_pct=edge_fraction_pct,
    )
    if not candidates:
        print("No finite positive edges in C_center.")
        return None

    w_min, w_max = kept[0][0], kept[-1][0]

    def weight_to_alpha(w: float) -> float:
        if w_max <= w_min:
            return alpha_max
        t = (w - w_min) / (w_max - w_min)
        return float(alpha_max - t * (alpha_max - alpha_min))

    if pos is None:
        pos = barycenter_xy_positions(X, min_x, min_y, domain_size)

    fig, ax = plt.subplots(figsize=figsize)
    set_barycenter_axes_limits(ax, pos, pad_frac=pad_frac)

    segments = []
    alphas = []
    rgba_base = mcolors.to_rgba(edge_color)
    for w, u, v in kept:
        segments.append([pos[u], pos[v]])
        a = weight_to_alpha(w)
        alphas.append((rgba_base[0], rgba_base[1], rgba_base[2], a))
    if segments:
        ax.add_collection(LineCollection(segments, colors=alphas, linewidths=edge_width))

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(
        xs,
        ys,
        s=node_size,
        c=list(node_colors),
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )

    if color_mode == "bilinear":
        corner_labels = [
            f"({min_x:.3g}, {min_y:.3g})",
            f"({min_x:.3g}, {max_y:.3g})",
            f"({max_x:.3g}, {min_y:.3g})",
            f"({max_x:.3g}, {max_y:.3g})",
        ]
        legend_handles = [
            Patch(facecolor=corner_colors[i], edgecolor="black", label=corner_labels[i])
            for i in range(4)
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=9,
        )
        mode_label = "bilinear (x, y) colormap"
    else:
        mode_label = (
            f"categorical colormap ({categorical_cmap}, "
            f"≤{categorical_max_colors} colors by node id)"
        )

    ax.set_title(
        f"Barycenter — {mode_label}\n"
        f"{len(kept)} / {len(candidates)} edges (smallest {edge_fraction_pct:g}%)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout(rect=[0, 0, 0.82, 1] if color_mode == "bilinear" else None)
    plt.show()
    print(
        f"Colormap mode: {color_mode}; "
        f"kept {len(kept)} edges ({edge_fraction_pct:g}% of {len(candidates)} candidates)"
    )
    return node_colors
