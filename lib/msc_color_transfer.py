"""
Color maps and coupling-based color transfer for MSC FGW barycenters.

Shared by the main barycenter notebook and downstream evaluation notebooks.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Wedge

from lib.msc_plotting import msc_xy_positions

ColorSpec = Union[str, List[Tuple[str, float]]]


def bilinear_xy_rgba(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    corner_rgba: Sequence,
) -> Tuple[float, float, float, float]:
    """
    Bilinear RGBA at world (x, y) from four corner colors:
      corner_rgba[0] at (min_x, min_y), [1] at (min_x, max_y),
      [2] at (max_x, min_y), [3] at (max_x, max_y).
    """
    dx = float(max_x - min_x)
    dy = float(max_y - min_y)
    u = 0.0 if dx <= 0 else float(np.clip((x - min_x) / dx, 0.0, 1.0))
    v = 0.0 if dy <= 0 else float(np.clip((y - min_y) / dy, 0.0, 1.0))
    c00, c01, c10, c11 = corner_rgba
    rgba = (
        (1.0 - u) * (1.0 - v) * np.asarray(c00, dtype=float)
        + u * (1.0 - v) * np.asarray(c10, dtype=float)
        + (1.0 - u) * v * np.asarray(c01, dtype=float)
        + u * v * np.asarray(c11, dtype=float)
    )
    return tuple(np.clip(rgba, 0.0, 1.0))


def barycenter_bilinear_node_colors(
    pos: Mapping,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    corner_colors: Sequence[str],
) -> List[Tuple[float, float, float, float]]:
    """Return list of RGBA tuples for barycenter node indices 0..n-1."""
    corner_rgba = [mcolors.to_rgba(c) for c in corner_colors]
    colors = []
    for i in sorted(pos.keys()):
        x, y = pos[i]
        colors.append(bilinear_xy_rgba(x, y, min_x, max_x, min_y, max_y, corner_rgba))
    return colors


def barycenter_categorical_node_colors(
    n: int,
    *,
    max_colors: int = 20,
    cmap: str = "tab20",
) -> List:
    """Assign up to ``max_colors`` distinct colors by barycenter node index."""
    cmap_obj = plt.get_cmap(cmap)
    n_palette = min(int(max_colors), int(getattr(cmap_obj, "N", max_colors)))
    return [cmap_obj(i % n_palette) for i in range(int(n))]


DEFAULT_CORNER_COLORS = ("#11FF22", "#66FFFF", "#FFEE33", "#FF3366")


def compute_barycenter_node_colors(
    mode: str,
    n: int,
    pos: Mapping,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    corner_colors: Sequence[str] = DEFAULT_CORNER_COLORS,
    categorical_cmap: str = "tab20",
    categorical_max_colors: int = 20,
) -> List:
    """Build per-node colors for bilinear or categorical barycenter colormap."""
    if mode == "bilinear":
        return barycenter_bilinear_node_colors(
            pos, min_x, max_x, min_y, max_y, corner_colors
        )
    if mode == "categorical":
        return barycenter_categorical_node_colors(
            n, max_colors=categorical_max_colors, cmap=categorical_cmap
        )
    raise ValueError(f"Unknown color mode={mode!r}. Use 'bilinear' or 'categorical'.")


def coupling_entry_threshold(T: np.ndarray, p: np.ndarray) -> float:
    """Ignore coupling entries below ``p.sum() / (n_bary * n_source)``."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(p.sum()) / float(T.shape[0] * T.shape[1])


def transfer_barycenter_colors_to_inputs(
    Ts: Sequence[np.ndarray],
    barycenter_colors: Sequence,
    p: np.ndarray,
    *,
    mass_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[List[List[ColorSpec]], List[List[bool]]]:
    """
    Transfer barycenter node colors to each input graph via couplings
    ``Ts[s]`` of shape ``(N_center, n_source)``.

    Returns per-graph lists of node colors (hex or list of (hex, frac) for pie
    charts) and use_pie flags aligned with source node order.
    """
    if barycenter_colors is None:
        raise ValueError("barycenter_colors is required.")

    p = np.asarray(p, dtype=float)
    bary_hex = [mcolors.to_hex(c) for c in barycenter_colors]
    all_node_colors: List[List[ColorSpec]] = []
    all_use_pie: List[List[bool]] = []

    for s, T in enumerate(Ts):
        T = np.asarray(T, dtype=float)
        thr = (
            coupling_entry_threshold(T, p)
            if mass_threshold is None
            else float(mass_threshold)
        )
        n_source = T.shape[1]
        node_colors: List[ColorSpec] = []
        use_pie: List[bool] = []

        for j in range(n_source):
            col = T[:, j]
            mask = col >= thr
            if not np.any(mask):
                node_colors.append("black")
                use_pie.append(False)
                continue

            idxs = np.where(mask)[0]
            weights = col[idxs]
            total = float(weights.sum())
            fracs = weights / total

            if len(idxs) == 1:
                node_colors.append(bary_hex[idxs[0]])
                use_pie.append(False)
            else:
                node_colors.append(
                    [(bary_hex[i], float(f)) for i, f in zip(idxs, fracs)]
                )
                use_pie.append(True)

        all_node_colors.append(node_colors)
        all_use_pie.append(use_pie)
        if verbose:
            print(
                f"Graph {s}: coupling threshold = {thr:.6g}; "
                f"pie nodes = {sum(use_pie)} / {n_source}"
            )

    return all_node_colors, all_use_pie


def _pie_radius_from_pos(pos: Mapping, frac: float = 0.012) -> float:
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-9)
    return frac * span


def _draw_pie_at_xy(ax, center, color_fracs, radius) -> None:
    if not color_fracs:
        return
    total = sum(float(f) for _, f in color_fracs)
    if total <= 0:
        return
    start_deg = 0.0
    for color, frac in color_fracs:
        if frac <= 0:
            continue
        span_deg = 360.0 * float(frac) / total
        ax.add_patch(
            Wedge(
                center,
                radius,
                start_deg,
                start_deg + span_deg,
                facecolor=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=4,
            )
        )
        start_deg += span_deg


def draw_input_graph_coupling_colors(
    G: nx.Graph,
    nodes_order: Sequence,
    node_colors: Sequence[ColorSpec],
    use_pie: Sequence[bool],
    *,
    mesh_id=None,
    figsize=(16.0, 10.0),
    edge_width: float = 2.0,
    edge_color: str = "#2ca02c",
    node_size: float = 40,
    pie_radius: Optional[float] = None,
) -> None:
    """Draw one input MSC graph with coupling-transferred node colors."""
    pos = msc_xy_positions(G)
    if pie_radius is None:
        pie_radius = _pie_radius_from_pos(pos)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_color, width=edge_width, alpha=0.9, ax=ax
    )

    node_to_idx = {n: i for i, n in enumerate(nodes_order)}
    xs, ys, colors = [], [], []
    for n in G.nodes():
        idx = node_to_idx.get(n, -1)
        if idx < 0:
            continue
        xy = pos[n]
        col = node_colors[idx]
        if use_pie[idx] and isinstance(col, list):
            _draw_pie_at_xy(ax, xy, col, pie_radius)
        else:
            c = col if isinstance(col, str) else "black"
            xs.append(xy[0])
            ys.append(xy[1])
            colors.append(c)

    if xs:
        ax.scatter(
            xs,
            ys,
            s=node_size,
            c=colors,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    title = (
        f"Input graph {mesh_id}: colors from barycenter coupling"
        if mesh_id is not None
        else "Input graph: colors from barycenter coupling"
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.show()
