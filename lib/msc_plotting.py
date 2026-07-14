"""
Plotting helpers for Morse-Smale complex and MSC barycenter graphs.
"""

import numpy as np
import networkx as nx
from typing import Optional

from lib.graph_reader import REGULAR_NODE_TYPE

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
