"""
Evaluation helpers for thresholded MSC FGW barycenters.

Functions take explicit arguments (no notebook globals) so they behave
identically when imported from any notebook.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from ot.lp import emd
from ot.utils import dist

from lib.graph_relations import apply_relation_scaling, relation_matrix
from lib.graph_sparsify import _segments_properly_cross_2d
from lib.gromov import (
    MPFGW_dist,
    PFGW_dist_with_penalty,
    _random_transport_matrix_sinkhorn,
    partial_fused_gromov_mass,
    partial_fused_gromov_ver1,
)
from lib.gromov_barycenter import pfgw_barycenters
from lib.msc_barycenter_threshold import (
    _decode_barycenter_nodes,
    build_thresholded_barycenter_graph,
)
from lib.msc_color_transfer import coupling_entry_threshold
from lib.msc_plotting import draw_msc_nodes, msc_node_visual_type, node_height


def pfgw_total_cost(
    C_tmp: np.ndarray,
    X_tmp: np.ndarray,
    Ts: Sequence[np.ndarray],
    *,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    p: np.ndarray,
    ps: Sequence[np.ndarray],
    Lambda_list: Sequence[float],
    lambdas: Sequence[float],
    alpha_fgw: float = 0.5,
    debug: bool = False,
) -> float:
    """Weighted sum of PFGW costs from ``(C_tmp, X_tmp)`` to each input."""
    total = 0.0
    p = np.asarray(p, dtype=float)
    for s in range(len(Cs)):
        M_s = dist(X_tmp, Ys[s])
        d, penalty = PFGW_dist_with_penalty(
            M_s,
            C_tmp,
            Cs[s],
            Ts[s],
            p,
            ps[s],
            Lambda_list[s],
            alpha=alpha_fgw,
        )
        if debug:
            print("d, penalty:", d, penalty)
        total += float(lambdas[s]) * (float(d) + float(penalty))
    return float(total)


def prune_barycenter_nodes_lacking_all_input_support(
    C: np.ndarray,
    X: np.ndarray,
    p: np.ndarray,
    Ts: Sequence[np.ndarray],
    *,
    min_matched_fraction: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Drop barycenter nodes that are not substantially coupled to every input.

    Node ``i`` is substantially coupled to input ``s`` when both:

    - at least one coupling entry exceeds the color-transfer noise floor
      ``coupling_entry_threshold(T[s], p)`` (``p.sum() / (n_bary * n_s)``);
    - the transported row mass ``T[s][i].sum()`` is at least
      ``min_matched_fraction * p[i]``.

    Structure ``C``, features ``X``, masses ``p``, and couplings ``Ts[s]``
    (each of shape ``(n_bary, n_s)``) are restricted to the surviving nodes
    in original order. Masses are not renormalized.
    """
    C = np.asarray(C)
    X = np.asarray(X)
    p = np.asarray(p, dtype=float).reshape(-1)
    n_bary = int(p.shape[0])
    if C.shape[0] != n_bary or C.shape[1] != n_bary:
        raise ValueError(f"C must be ({n_bary}, {n_bary}); got {C.shape}.")
    if X.shape[0] != n_bary:
        raise ValueError(f"X must have {n_bary} rows; got {X.shape}.")
    if min_matched_fraction < 0.0:
        raise ValueError("min_matched_fraction must be nonnegative.")
    if len(Ts) == 0:
        raise ValueError("Ts must contain at least one coupling.")

    keep = np.ones(n_bary, dtype=bool)
    n_unsupported = []
    matched_masses = []
    for s, T in enumerate(Ts):
        T = np.asarray(T, dtype=float)
        if T.shape[0] != n_bary:
            raise ValueError(
                f"T[{s}] must have {n_bary} rows; got {T.shape}."
            )
        entry_thr = coupling_entry_threshold(T, p)
        row_mass = T.sum(axis=1)
        matched_masses.append(row_mass)
        has_entry = np.max(T, axis=1) >= entry_thr
        has_mass = row_mass >= (min_matched_fraction * p)
        supported = has_entry & has_mass
        n_unsupported.append(int(np.count_nonzero(~supported)))
        keep &= supported

    keep_idx = np.flatnonzero(keep)
    n_keep = int(keep_idx.size)
    if n_keep == 0:
        raise ValueError(
            "No barycenter nodes have substantial couplings to all inputs; "
            "lower min_matched_fraction."
        )

    Ts_kept = [np.asarray(T, dtype=float)[keep_idx, :] for T in Ts]
    diagnostics = {
        "n_before": n_bary,
        "n_after": n_keep,
        "n_dropped": n_bary - n_keep,
        "keep_idx": keep_idx,
        "n_unsupported_per_input": n_unsupported,
        "matched_masses": matched_masses,
        "remaining_mass": float(p[keep_idx].sum()),
        "min_matched_fraction": float(min_matched_fraction),
    }
    return (
        C[np.ix_(keep_idx, keep_idx)],
        X[keep_idx],
        p[keep_idx],
        Ts_kept,
        keep_idx,
        diagnostics,
    )


_MSC_TYPE_NAMES = {0: "minimum", 1: "saddle", 2: "maximum", -1: "regular"}


def diagnose_barycenter_morse_types(
    X_center: np.ndarray,
    Ys: Sequence[np.ndarray],
    Ts: Sequence[np.ndarray],
    *,
    msc_type_order: Sequence[int] = (0, 1, 2, -1),
    type_col_start: int = 4,
    eps: float = 1e-6,
    type_scale: Optional[float] = None,
    input_labels: Optional[Sequence[Any]] = None,
    max_print: int = 25,
    plot: bool = True,
) -> Dict[str, Any]:
    """Print Morse-type diagnostics for a PFGW barycenter.

    Uses the one-hot type block of ``X_center`` / ``Ys[s]`` (columns
    ``type_col_start : type_col_start + len(msc_type_order)``). Reports:

    1. Node counts by argmax type.
    2. Ambiguous nodes: two or more one-hot entries strictly above ``eps``.
    3. Cross-type couplings: ``T[s][i, j] > eps`` whose argmax types differ.
    """
    X = np.asarray(X_center, dtype=float)
    type_order = [int(t) for t in msc_type_order]
    n_types = len(type_order)
    type_col_end = int(type_col_start) + n_types
    if X.ndim != 2 or X.shape[1] < type_col_end:
        raise ValueError(
            f"X_center must have at least {type_col_end} columns; got {X.shape}."
        )
    if len(Ys) != len(Ts):
        raise ValueError(
            f"Ys and Ts must have the same length; got {len(Ys)} and {len(Ts)}."
        )

    def _type_name(t: int) -> str:
        return _MSC_TYPE_NAMES.get(int(t), str(int(t)))

    def _onehot(feat: np.ndarray) -> np.ndarray:
        feat = np.asarray(feat, dtype=float)
        if feat.ndim != 2 or feat.shape[1] < type_col_end:
            raise ValueError(
                f"Feature matrix must have at least {type_col_end} columns; "
                f"got {feat.shape}."
            )
        return feat[:, type_col_start:type_col_end]

    def _assigned_types(oh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        finite = np.nan_to_num(oh, nan=0.0)
        n_active = np.sum(finite > eps, axis=1).astype(int)
        argmax_idx = np.argmax(finite, axis=1)
        assigned = np.array([type_order[i] for i in argmax_idx], dtype=int)
        assigned[n_active == 0] = np.iinfo(np.int32).min
        return assigned, n_active

    X_oh = _onehot(X)
    n_bary = int(X_oh.shape[0])
    assigned, n_active = _assigned_types(X_oh)
    untyped_code = int(np.iinfo(np.int32).min)
    n_untyped = int(np.sum(assigned == untyped_code))
    n_ambiguous = int(np.sum(n_active > 1))
    counts = {t: int(np.sum(assigned == t)) for t in type_order}

    print("Barycenter Morse-type diagnostics")
    print(
        f"  n_nodes={n_bary}  eps={eps:g}  "
        f"one-hot X[:, {type_col_start}:{type_col_end}]  order={type_order}"
    )
    if type_scale is not None:
        print(
            f"  type_scale={float(type_scale):.6g}  "
            f"(input one-hots are 0 or type_scale; barycenter may be soft)"
        )

    print()
    print("1) Node counts by argmax type")
    print(f"  {'type':<10} {'code':>6} {'n_nodes':>8} {'frac':>8}")
    for t in type_order:
        n_t = counts[t]
        frac = n_t / max(n_bary, 1)
        print(f"  {_type_name(t):<10} {t:6d} {n_t:8d} {frac:8.2%}")
    if n_untyped:
        print(
            f"  {'untyped':<10} {'':>6} {n_untyped:8d} "
            f"{n_untyped / max(n_bary, 1):8.2%}  (no entry > eps)"
        )

    print()
    print("2) Ambiguous-type nodes (2+ one-hot entries > eps)")
    print(f"  n_ambiguous={n_ambiguous} / {n_bary}")
    amb_idx = np.flatnonzero(n_active > 1)
    if n_ambiguous:
        header_types = "  ".join(f"{_type_name(t):>10}" for t in type_order)
        print(f"  {'i':>6} {'argmax':<10}  one-hot {header_types}")
        order = np.argsort(-n_active[amb_idx], kind="mergesort")
        show = amb_idx[order][: max(0, int(max_print))]
        for i in show:
            tname = _type_name(int(assigned[i]))
            vals = "  ".join(f"{X_oh[i, k]:10.4g}" for k in range(n_types))
            print(f"  {int(i):6d} {tname:<10}          {vals}")
        if amb_idx.size > len(show):
            print(f"  ... {amb_idx.size - len(show)} more omitted")
    else:
        print("  none")

    n_inputs = len(Ts)
    labels = (
        [str(x) for x in input_labels]
        if input_labels is not None and len(input_labels) == n_inputs
        else [f"input {s}" for s in range(n_inputs)]
    )
    type_pair_mass = np.zeros((n_types, n_types), dtype=float)
    type_pair_count = np.zeros((n_types, n_types), dtype=int)
    bary_has_cross = np.zeros((n_bary, n_inputs), dtype=bool)

    total_nz = 0
    total_same_pairs = 0
    total_cross_pairs = 0
    total_same_mass = 0.0
    total_cross_mass = 0.0
    per_input_rows: List[Dict[str, Any]] = []
    top_cross: List[Tuple[float, int, int, int, int, int]] = []
    type_to_idx = {t: i for i, t in enumerate(type_order)}

    print()
    print("3) Cross-type couplings (T_ij > eps and argmax types differ)")
    print(
        f"  {'label':<22} {'T>eps':>8} {'same':>8} {'cross':>8} "
        f"{'same_mass':>12} {'cross_mass':>12} {'n_bary_cross':>13}"
    )
    for s, T in enumerate(Ts):
        T = np.asarray(T, dtype=float)
        Y_oh = _onehot(Ys[s])
        if T.shape != (n_bary, Y_oh.shape[0]):
            raise ValueError(
                f"T[{s}] shape {T.shape} does not match "
                f"(n_bary={n_bary}, n_input={Y_oh.shape[0]})."
            )
        y_assigned, y_active = _assigned_types(Y_oh)
        nz = T > eps
        n_nz = int(np.count_nonzero(nz))
        bary_idx, input_idx = np.nonzero(nz)
        same_mask = assigned[bary_idx] == y_assigned[input_idx]
        typed_mask = (assigned[bary_idx] != untyped_code) & (
            y_assigned[input_idx] != untyped_code
        )
        cross_mask = typed_mask & ~same_mask
        n_same = int(np.count_nonzero(same_mask & typed_mask))
        n_cross = int(np.count_nonzero(cross_mask))
        same_sel = same_mask & typed_mask
        same_mass = (
            float(T[bary_idx[same_sel], input_idx[same_sel]].sum()) if n_same else 0.0
        )
        cross_mass = (
            float(T[bary_idx[cross_mask], input_idx[cross_mask]].sum())
            if n_cross
            else 0.0
        )
        if n_cross:
            bary_has_cross[np.unique(bary_idx[cross_mask]), s] = True
            masses = T[bary_idx[cross_mask], input_idx[cross_mask]]
            for k in np.argsort(-masses)[: max(0, int(max_print))]:
                i = int(bary_idx[cross_mask][k])
                j = int(input_idx[cross_mask][k])
                top_cross.append(
                    (
                        float(masses[k]),
                        s,
                        i,
                        j,
                        int(assigned[i]),
                        int(y_assigned[j]),
                    )
                )
        if np.any(typed_mask):
            ti = np.fromiter(
                (type_to_idx[int(assigned[i])] for i in bary_idx[typed_mask]),
                dtype=int,
                count=int(np.count_nonzero(typed_mask)),
            )
            tj = np.fromiter(
                (type_to_idx[int(y_assigned[j])] for j in input_idx[typed_mask]),
                dtype=int,
                count=int(np.count_nonzero(typed_mask)),
            )
            np.add.at(
                type_pair_mass,
                (ti, tj),
                T[bary_idx[typed_mask], input_idx[typed_mask]],
            )
            np.add.at(type_pair_count, (ti, tj), 1)

        n_bary_cross = int(np.count_nonzero(bary_has_cross[:, s]))
        print(
            f"  {labels[s]:<22} {n_nz:8d} {n_same:8d} {n_cross:8d} "
            f"{same_mass:12.4g} {cross_mass:12.4g} {n_bary_cross:13d}"
        )
        per_input_rows.append(
            {
                "label": labels[s],
                "n_nonzero": n_nz,
                "n_same": n_same,
                "n_cross": n_cross,
                "same_mass": same_mass,
                "cross_mass": cross_mass,
                "n_bary_with_cross": n_bary_cross,
                "n_input_untyped": int(np.sum(y_active == 0)),
            }
        )
        total_nz += n_nz
        total_same_pairs += n_same
        total_cross_pairs += n_cross
        total_same_mass += same_mass
        total_cross_mass += cross_mass

    n_bary_cross_any = int(np.count_nonzero(bary_has_cross.any(axis=1))) if n_inputs else 0
    n_bary_cross_all = int(np.count_nonzero(bary_has_cross.all(axis=1))) if n_inputs else 0
    print(
        f"  {'ALL INPUTS':<22} {total_nz:8d} {total_same_pairs:8d} "
        f"{total_cross_pairs:8d} {total_same_mass:12.4g} {total_cross_mass:12.4g} "
        f"{n_bary_cross_any:13d}"
    )
    if total_cross_pairs == 0:
        print("  no cross-type couplings above eps")
    else:
        print(
            f"  barycenter nodes with a cross-type coupling to "
            f"any input: {n_bary_cross_any} / {n_bary}"
        )
        print(
            f"  barycenter nodes with a cross-type coupling to "
            f"every input: {n_bary_cross_all} / {n_bary}"
        )

    print()
    print("  Coupling mass by (barycenter type, input type), all inputs")
    col_h = "  ".join(f"{_type_name(t):>10}" for t in type_order)
    print(f"  {'bary vs input':<14} {col_h}")
    for i, t in enumerate(type_order):
        vals = "  ".join(f"{type_pair_mass[i, j]:10.4g}" for j in range(n_types))
        print(f"  {_type_name(t):<14} {vals}")

    if total_cross_pairs and top_cross:
        top_cross.sort(key=lambda row: -row[0])
        show_cross = top_cross[: max(0, int(max_print))]
        print()
        print(f"  largest cross-type entries (up to {len(show_cross)})")
        print(
            f"  {'mass':>10}  {'input':<22} {'i':>6} {'j':>6} "
            f"{'bary':<10} {'input_t':<10}"
        )
        for mass, s, i, j, tb, ty in show_cross:
            print(
                f"  {mass:10.4g}  {labels[s]:<22} {i:6d} {j:6d} "
                f"{_type_name(tb):<10} {_type_name(ty):<10}"
            )

    if plot and n_bary > 0:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), constrained_layout=True)
        names = [_type_name(t) for t in type_order]
        bar_colors = {0: "C0", 1: "0.75", 2: "C3", -1: "0.45"}
        colors = [bar_colors.get(t, "0.6") for t in type_order]
        axes[0].bar(names, [counts[t] for t in type_order], color=colors, edgecolor="0.2")
        axes[0].set_title("Barycenter nodes by argmax type")
        axes[0].set_ylabel("number of nodes")
        if n_untyped:
            axes[0].text(
                0.98,
                0.96,
                f"untyped={n_untyped}",
                transform=axes[0].transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        im = axes[1].imshow(type_pair_mass, cmap="viridis", aspect="auto")
        axes[1].set_xticks(range(n_types), names, rotation=30, ha="right")
        axes[1].set_yticks(range(n_types), names)
        axes[1].set_xlabel("input type")
        axes[1].set_ylabel("barycenter type")
        axes[1].set_title("Coupling mass (T_ij > eps)")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle("Barycenter Morse-type diagnostics", fontsize=12)
        plt.show()

    return {
        "n_bary": n_bary,
        "eps": float(eps),
        "type_order": type_order,
        "counts": counts,
        "n_untyped": n_untyped,
        "n_ambiguous": n_ambiguous,
        "ambiguous_idx": amb_idx,
        "assigned_types": assigned,
        "n_active": n_active,
        "onehot": X_oh,
        "n_nonzero_couplings": total_nz,
        "n_same_type_couplings": total_same_pairs,
        "n_cross_type_couplings": total_cross_pairs,
        "same_type_mass": total_same_mass,
        "cross_type_mass": total_cross_mass,
        "n_bary_cross_any_input": n_bary_cross_any,
        "n_bary_cross_all_inputs": n_bary_cross_all,
        "type_pair_mass": type_pair_mass,
        "type_pair_count": type_pair_count,
        "per_input": per_input_rows,
        "bary_has_cross": bary_has_cross,
    }


def recompute_thresholded_relation_matrix(
    G: nx.Graph,
    *,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
    laplacian_base: str = "adjacency",
    laplacian_normalization: str = "none",
    tau: float = 1.0,
    sp_affinity: str = "exp",
    sp_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, List]:
    """Rebuild pairwise relation ``C`` on a thresholded network."""
    nodelist = list(range(G.number_of_nodes()))
    C, nodes = relation_matrix(
        G,
        kind=relation_kind,
        nodelist=nodelist,
        laplacian_base=laplacian_base,
        laplacian_normalization=laplacian_normalization,
        tau=tau,
        sp_weighted=sp_weighted,
        sp_affinity=sp_affinity,
        sp_sigma=sp_sigma,
    )
    if relation_kind == "shortestpath":
        C = apply_relation_scaling(C, relation_kind, relation_rescale, relation_offset)
        C = np.nan_to_num(C, nan=np.inf, posinf=np.inf)
    return C, nodes


def recompute_features_from_graph(
    G: nx.Graph,
    nodelist: Optional[Sequence] = None,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    min_scalar: float,
    scalar_range: float,
    type_to_idx: Mapping[int, int],
    n_types: int,
    type_scale: float,
) -> np.ndarray:
    """Rebuild feature matrix from graph node attributes."""
    if nodelist is None:
        nodelist = list(G.nodes())
    coords = np.array(
        [
            [
                (G.nodes[n]["x"] - min_x) / domain_size,
                (G.nodes[n]["y"] - min_y) / domain_size,
                (G.nodes[n]["z"] - min_z) / domain_size,
            ]
            for n in nodelist
        ],
        dtype=float,
    )
    scalars = np.array(
        [(node_height(G, n) - min_scalar) / scalar_range for n in nodelist],
        dtype=float,
    ).reshape(-1, 1)
    onehot = np.zeros((len(nodelist), n_types), dtype=float)
    for i, n in enumerate(nodelist):
        onehot[i, type_to_idx[msc_node_visual_type(G.nodes[n])]] = type_scale
    return np.column_stack([coords, scalars, onehot])


def solve_pfgw_couplings_to_inputs(
    C_tmp: np.ndarray,
    X_tmp: np.ndarray,
    p_tmp: np.ndarray,
    *,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    Lambda_list: Sequence[float],
    alpha_fgw: float = 0.5,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
) -> Dict[str, List[np.ndarray]]:
    """Solve PFGW from fixed ``(C_tmp, X_tmp, p_tmp)`` to each input."""
    Ts: List[np.ndarray] = []
    for s in range(len(Cs)):
        M_s = dist(X_tmp, Ys[s])
        if gw_restarts <= 1:
            T_best = partial_fused_gromov_ver1(
                M_s,
                C_tmp,
                Cs[s],
                p_tmp,
                ps[s],
                Lambda_list[s],
                alpha=alpha_fgw,
                G0=None,
                numItermax_gw=max_iter,
                tol=1e-5,
                log=False,
                verbose=False,
                seed=gw_seed_start,
            )
        else:
            T_best, best_cost = None, None
            for r in range(gw_restarts):
                T = partial_fused_gromov_ver1(
                    M_s,
                    C_tmp,
                    Cs[s],
                    p_tmp,
                    ps[s],
                    Lambda_list[s],
                    alpha=alpha_fgw,
                    G0=None,
                    numItermax_gw=max_iter,
                    tol=1e-5,
                    log=False,
                    verbose=False,
                    seed=gw_seed_start + r,
                )
                d_val, pen = PFGW_dist_with_penalty(
                    M_s,
                    C_tmp,
                    Cs[s],
                    T,
                    p_tmp,
                    ps[s],
                    Lambda_list[s],
                    alpha=alpha_fgw,
                )
                cost = d_val + pen
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    T_best = T
        Ts.append(T_best)
    return {"T": Ts}


def evaluate_thresholded_barycenter(
    edge_fraction_pct: float,
    *,
    C_center: np.ndarray,
    X_center: np.ndarray,
    p_eval: np.ndarray,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    Lambda_list: Sequence[float],
    lambdas: Sequence[float],
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    min_scalar: float,
    scalar_range: float,
    msc_type_order: Sequence[int],
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
    laplacian_base: str = "adjacency",
    laplacian_normalization: str = "none",
    tau: float = 1.0,
    sp_affinity: str = "exp",
    sp_sigma: Optional[float] = None,
    alpha_fgw: float = 0.5,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    recompute_X: bool = False,
    type_to_idx: Optional[Mapping[int, int]] = None,
    n_types: Optional[int] = None,
    type_scale: float = 1.0,
    seed_cost: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full evaluation for one ``EDGE_FRACTION_PCT``."""
    G_thr, kept, candidates = build_thresholded_barycenter_graph(
        C_center,
        X_center,
        edge_fraction_pct,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        domain_size=domain_size,
        scalar_min=min_scalar,
        scalar_range=scalar_range,
        msc_type_order=msc_type_order,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        sp_weighted=sp_weighted,
    )
    C_thr, nodes = recompute_thresholded_relation_matrix(
        G_thr,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        sp_weighted=sp_weighted,
        laplacian_base=laplacian_base,
        laplacian_normalization=laplacian_normalization,
        tau=tau,
        sp_affinity=sp_affinity,
        sp_sigma=sp_sigma,
    )
    if recompute_X:
        if type_to_idx is None or n_types is None:
            raise ValueError("recompute_X=True requires type_to_idx and n_types")
        X_thr = recompute_features_from_graph(
            G_thr,
            nodes,
            min_x=min_x,
            min_y=min_y,
            min_z=min_z,
            domain_size=domain_size,
            min_scalar=min_scalar,
            scalar_range=scalar_range,
            type_to_idx=type_to_idx,
            n_types=n_types,
            type_scale=type_scale,
        )
    else:
        X_thr = np.asarray(X_center, dtype=float).copy()

    log_thr = solve_pfgw_couplings_to_inputs(
        C_thr,
        X_thr,
        p_eval,
        Cs=Cs,
        Ys=Ys,
        ps=ps,
        Lambda_list=Lambda_list,
        alpha_fgw=alpha_fgw,
        gw_restarts=gw_restarts,
        gw_seed_start=gw_seed_start,
        max_iter=max_iter,
    )
    cost_bary = pfgw_total_cost(
        C_thr,
        X_thr,
        log_thr["T"],
        Cs=Cs,
        Ys=Ys,
        p=p_eval,
        ps=ps,
        Lambda_list=Lambda_list,
        lambdas=lambdas,
        alpha_fgw=alpha_fgw,
        debug=False,
    )
    quality = None if seed_cost is None or seed_cost == 0 else float(cost_bary / seed_cost)

    result = {
        "edge_fraction_pct": float(edge_fraction_pct),
        "n_kept_edges": len(kept),
        "n_candidates": len(candidates),
        "n_components": nx.number_connected_components(G_thr),
        "cost_barycenter": float(cost_bary),
        "cost_seed": None if seed_cost is None else float(seed_cost),
        "quality": quality,
        "recompute_X": bool(recompute_X),
        "G": G_thr,
        "C": C_thr,
        "X": X_thr,
        "log": log_thr,
    }
    if verbose:
        print(
            f"EDGE_FRACTION_PCT={edge_fraction_pct:g}: "
            f"kept {len(kept)}/{len(candidates)} edges, "
            f"components={result['n_components']}, "
            f"bary_cost={cost_bary:.6g}",
            end="",
        )
        if quality is not None:
            print(f", seed_cost={seed_cost:.6g}, quality={quality:.6g}")
        else:
            print()
    return result


def evaluate_seed_network(
    seed_idx: Optional[int] = None,
    *,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    p: np.ndarray,
    Lambda_list: Sequence[float],
    lambdas: Sequence[float],
    alpha_fgw: float = 0.5,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    mesh_ids: Optional[Sequence] = None,
    C_seed: Optional[np.ndarray] = None,
    X_seed: Optional[np.ndarray] = None,
    p_seed: Optional[np.ndarray] = None,
    seed_label: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, List[np.ndarray]]]:
    """
    Baseline: sum of PFGW distances from a seed network to all barycenter inputs.

    In-set seed: pass ``seed_idx`` into ``Cs`` / ``Ys`` / ``ps``.
    Out-of-set seed: pass ``C_seed`` / ``X_seed`` (and optionally ``p_seed``);
    couplings are still solved against the barycenter input family ``Cs``.
    """
    if C_seed is None or X_seed is None:
        if seed_idx is None:
            raise ValueError("Provide seed_idx (in-set) or C_seed/X_seed (out-of-set)")
        if seed_idx < 0 or seed_idx >= len(Cs):
            raise IndexError(f"seed_idx={seed_idx} out of range for {len(Cs)} inputs")
        if len(ps[seed_idx]) != len(p):
            raise ValueError(
                f"Seed network size {len(ps[seed_idx])} != barycenter size {len(p)}. "
                "Pick a seed with N_center nodes (e.g. the init network)."
            )
        C_seed = Cs[seed_idx]
        X_seed = Ys[seed_idx]
        if seed_label is None:
            mid = mesh_ids[seed_idx] if mesh_ids is not None else seed_idx
            seed_label = f"in-set input index {seed_idx} (mesh {mid})"
    else:
        C_seed = np.asarray(C_seed, dtype=float)
        X_seed = np.asarray(X_seed, dtype=float)
        if C_seed.shape[0] != len(p):
            raise ValueError(
                f"Seed network size {C_seed.shape[0]} != barycenter size {len(p)}."
            )
        if seed_label is None:
            seed_label = "out-of-set seed"

    log_seed = solve_pfgw_couplings_to_inputs(
        C_seed,
        X_seed,
        p,
        Cs=Cs,
        Ys=Ys,
        ps=ps,
        Lambda_list=Lambda_list,
        alpha_fgw=alpha_fgw,
        gw_restarts=gw_restarts,
        gw_seed_start=gw_seed_start,
        max_iter=max_iter,
    )
    cost_seed = pfgw_total_cost(
        C_seed,
        X_seed,
        log_seed["T"],
        Cs=Cs,
        Ys=Ys,
        p=p,
        ps=ps,
        Lambda_list=Lambda_list,
        lambdas=lambdas,
        alpha_fgw=alpha_fgw,
        debug=False,
    )
    if verbose:
        print(f"Seed network: {seed_label}, PFGW sum = {cost_seed:.6g}")
    return float(cost_seed), log_seed


def pfgw_pairwise_costs(
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    *,
    lambda_penalty: float,
    alpha_fgw: float = 0.5,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric PFGW costs between every pair of measure networks.

    Only the upper triangle is solved: ``n * (n - 1) / 2`` unique PFGW
    problems. ``D[i, j]`` and ``D[j, i]`` share that cost. The diagonal is 0.

    Returns
    -------
    D : ndarray, shape (n, n)
        Symmetric pairwise PFGW costs.
    row_sums : ndarray, shape (n,)
        ``row_sums[i] = sum_{j != i} D[i, j]``.
    """
    n = len(Cs)
    if n == 0:
        raise ValueError("Need at least one measure network")
    if len(Ys) != n or len(ps) != n:
        raise ValueError("Cs, Ys, and ps must have the same length")

    D = np.zeros((n, n), dtype=float)
    if n == 1:
        return D, np.zeros(1, dtype=float)

    lam = float(lambda_penalty)
    n_pairs = n * (n - 1) // 2
    if verbose:
        print(
            f"Computing {n_pairs} unique PFGW pairs over {n} networks "
            f"(symmetric, gw_restarts={gw_restarts})..."
        )
    for i in range(n):
        others = list(range(i + 1, n))
        if not others:
            continue
        C_i = np.asarray(Cs[i], dtype=float)
        Y_i = np.asarray(Ys[i], dtype=float)
        p_i = np.asarray(ps[i], dtype=float)
        log_i = solve_pfgw_couplings_to_inputs(
            C_i,
            Y_i,
            p_i,
            Cs=[Cs[j] for j in others],
            Ys=[Ys[j] for j in others],
            ps=[ps[j] for j in others],
            Lambda_list=[lam] * len(others),
            alpha_fgw=alpha_fgw,
            gw_restarts=gw_restarts,
            gw_seed_start=gw_seed_start,
            max_iter=max_iter,
        )
        for k, j in enumerate(others):
            M = dist(Y_i, Ys[j])
            d_val, pen = PFGW_dist_with_penalty(
                M,
                C_i,
                Cs[j],
                log_i["T"][k],
                p_i,
                ps[j],
                lam,
                alpha=alpha_fgw,
            )
            cost = float(d_val) + float(pen)
            D[i, j] = cost
            D[j, i] = cost
            if verbose:
                print(f"  pair ({i}, {j}): {cost:.6g}")
    return D, D.sum(axis=1)


def select_seed_by_min_pfgw_sum(
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    *,
    lambda_penalty: float,
    alpha_fgw: float = 0.5,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    verbose: bool = False,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Pick the network with the smallest sum of PFGW costs to all others.

    Ties keep the earliest index in ``Cs``.
    """
    D, row_sums = pfgw_pairwise_costs(
        Cs,
        Ys,
        ps,
        lambda_penalty=lambda_penalty,
        alpha_fgw=alpha_fgw,
        gw_restarts=gw_restarts,
        gw_seed_start=gw_seed_start,
        max_iter=max_iter,
        verbose=verbose,
    )
    seed_idx = int(np.argmin(row_sums))
    if verbose:
        print(
            f"Selected seed index {seed_idx} with PFGW sum "
            f"{float(row_sums[seed_idx]):.6g}"
        )
    return seed_idx, D, row_sums


# ---------------------------------------------------------------------------
# Attribute alignment
# ---------------------------------------------------------------------------


def _feature_type_index(feat_row, *, type_col_start: int = 4) -> Optional[int]:
    row = np.asarray(feat_row, dtype=float)
    if row.size <= type_col_start:
        return None
    return int(np.argmax(row[type_col_start:]))


def _attribute_pair_differences(x_i, y_j, *, type_col_start: int = 4):
    x_i = np.asarray(x_i, dtype=float)
    y_j = np.asarray(y_j, dtype=float)
    eucl = float(np.linalg.norm(x_i[:3] - y_j[:3]))
    scalar = float(np.abs(x_i[3] - y_j[3])) if min(x_i.size, y_j.size) > 3 else np.nan
    ti = _feature_type_index(x_i, type_col_start=type_col_start)
    tj = _feature_type_index(y_j, type_col_start=type_col_start)
    if ti is None or tj is None:
        typ = np.nan
    else:
        typ = 0.0 if ti == tj else 1.0
    return eucl, scalar, typ


def collect_attribute_alignment_diffs(
    Ts: Sequence[np.ndarray],
    X_bary: np.ndarray,
    Ys_list: Sequence[np.ndarray],
    p_bary: np.ndarray,
    *,
    mass_threshold: Optional[float] = None,
    type_col_start: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Collect attribute diffs for couplings above the color-transfer threshold."""
    X_bary = np.asarray(X_bary, dtype=float)
    p_bary = np.asarray(p_bary, dtype=float)

    eucl_diffs: List[float] = []
    scalar_diffs: List[float] = []
    type_diffs: List[float] = []
    pair_meta = []

    for s, T in enumerate(Ts):
        T = np.asarray(T, dtype=float)
        Ys_s = np.asarray(Ys_list[s], dtype=float)
        thr = (
            float(mass_threshold)
            if mass_threshold is not None
            else coupling_entry_threshold(T, p_bary)
        )
        ii, jj = np.where(T >= thr)
        for i, j in zip(ii.tolist(), jj.tolist()):
            eucl, scalar, typ = _attribute_pair_differences(
                X_bary[i], Ys_s[j], type_col_start=type_col_start
            )
            eucl_diffs.append(eucl)
            scalar_diffs.append(scalar)
            type_diffs.append(typ)
            pair_meta.append((s, i, j, float(T[i, j]), thr))
        if verbose:
            print(
                f"Graph {s}: thr={thr:.6g}; significant pairs = {len(ii)} "
                f"/ {T.shape[0] * T.shape[1]}"
            )

    return {
        "euclidean": np.asarray(eucl_diffs, dtype=float),
        "scalar": np.asarray(scalar_diffs, dtype=float),
        "type": np.asarray(type_diffs, dtype=float),
        "pairs": pair_meta,
    }


def attribute_alignment_for_eval_results(
    eval_results: Sequence[Mapping[str, Any]],
    Ys_list: Sequence[np.ndarray],
    p_bary: np.ndarray,
    *,
    type_col_start: int = 4,
    mass_threshold: Optional[float] = None,
) -> Dict[float, Dict[str, Any]]:
    """Run attribute-alignment collection for each thresholded evaluation result."""
    by_pct: Dict[float, Dict[str, Any]] = {}
    for row in eval_results:
        pct = row["edge_fraction_pct"]
        print(f"\n--- Attribute alignment @ EDGE_FRACTION_PCT={pct:g} ---")
        diffs = collect_attribute_alignment_diffs(
            row["log"]["T"],
            row["X"],
            Ys_list,
            p_bary,
            mass_threshold=mass_threshold,
            type_col_start=type_col_start,
        )
        by_pct[pct] = diffs
        n = len(diffs["euclidean"])
        if n == 0:
            print("  (no significant pairs)")
            continue
        print(
            f"  n_pairs={n}; "
            f"eucl median={np.nanmedian(diffs['euclidean']):.4g}, "
            f"scalar median={np.nanmedian(diffs['scalar']):.4g}, "
            f"type mismatch rate={np.nanmean(diffs['type']):.4g}"
        )
    return by_pct


def _select_alignment_pcts(alignment_by_pct, edge_fraction_pcts=None):
    available = sorted(alignment_by_pct.keys())
    if edge_fraction_pcts is None:
        return available
    selected = []
    missing = []
    for pct in edge_fraction_pcts:
        pct = float(pct)
        match = next((k for k in available if abs(float(k) - pct) <= 1e-9), None)
        if match is None:
            missing.append(pct)
        elif match not in selected:
            selected.append(match)
    if missing:
        print(
            f"Warning: requested EDGE_FRACTION_PCT not in results: {missing}. "
            f"Available: {available}"
        )
    if not selected:
        raise ValueError(
            f"No EDGE_FRACTION_PCT selected. Requested={list(edge_fraction_pcts)}, "
            f"available={available}"
        )
    return selected


def summarize_attribute_alignment_distributions(
    alignment_by_pct,
    *,
    edge_fraction_pcts=None,
) -> None:
    """Print per-metric distribution stats for selected EDGE_FRACTION_PCT values."""
    pcts = _select_alignment_pcts(alignment_by_pct, edge_fraction_pcts)
    metrics = [
        ("euclidean", "Euclidean (x,y,z)"),
        ("scalar", "Scalar |diff|"),
        ("type", "Type mismatch"),
    ]
    quantiles = (0.0, 0.25, 0.5, 0.75, 1.0)

    for key, title in metrics:
        print(f"\n=== {title} ===")
        print(
            f"{'pct':>8} {'n':>8} {'mean':>10} {'std':>10} "
            f"{'min':>10} {'q25':>10} {'med':>10} {'q75':>10} {'max':>10}"
        )
        for pct in pcts:
            vals = np.asarray(alignment_by_pct[pct][key], dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            if n == 0:
                print(
                    f"{pct:8.1f} {0:8d} "
                    + " ".join(f"{'—':>10}" for _ in range(7))
                )
                continue
            qs = np.quantile(vals, quantiles)
            print(
                f"{pct:8.1f} {n:8d} {vals.mean():10.4g} {vals.std():10.4g} "
                f"{qs[0]:10.4g} {qs[1]:10.4g} {qs[2]:10.4g} {qs[3]:10.4g} {qs[4]:10.4g}"
            )


def plot_attribute_alignment_boxplots(
    alignment_by_pct,
    figsize=(12, 4),
    *,
    show_fliers: bool = False,
    edge_fraction_pcts=None,
) -> None:
    """Three boxplot panels: Euclidean, scalar |diff|, type mismatch."""
    pcts = _select_alignment_pcts(alignment_by_pct, edge_fraction_pcts)
    labels = [f"{p:g}%" for p in pcts]
    metrics = [
        ("euclidean", "Euclidean distance (x,y,z)", "#1f77b4"),
        ("scalar", "Scalar |diff|", "#ff7f0e"),
        ("type", "Type mismatch (0/1)", "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (key, title, color) in zip(axes, metrics):
        data = []
        for pct in pcts:
            vals = alignment_by_pct[pct][key]
            vals = vals[np.isfinite(vals)]
            data.append(vals if vals.size else np.array([np.nan]))
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showfliers=show_fliers,
            whis=1.5,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        ax.set_title(title)
        ax.set_xlabel("EDGE_FRACTION_PCT")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        if key == "type":
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Attribute alignment under significant couplings\n"
        r"($T_{ij} \geq$ color-transfer threshold)",
        y=1.02,
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Mass-specified FPGW evaluation distance (fixed rho, fixed alpha)
# ---------------------------------------------------------------------------


def _finite_relation_matrix(C: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the largest finite value (0 if none)."""
    C = np.asarray(C, dtype=float).copy()
    finite = np.isfinite(C)
    fill = float(np.max(C[finite])) if np.any(finite) else 0.0
    if not np.all(finite):
        C = np.where(finite, C, fill)
    np.fill_diagonal(C, 0.0)
    return C


def transported_mass_from_rho(
    p: np.ndarray,
    q: np.ndarray,
    rho: float,
) -> float:
    """Absolute mass ``m = rho * min(||p||_1, ||q||_1)``."""
    rho = float(rho)
    if not (0.0 < rho <= 1.0 + 1e-12):
        raise ValueError(f"rho must lie in (0, 1]; got {rho}")
    rho = min(rho, 1.0)
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    return rho * min(float(p.sum()), float(q.sum()))


def mean_transported_mass_fraction(
    Ts: Sequence[np.ndarray],
    p: np.ndarray,
    ps: Sequence[np.ndarray],
) -> float:
    """Mean coupling mass as a fraction of ``min(||p||_1, ||p_s||_1)``."""
    p = np.asarray(p, dtype=float).reshape(-1)
    p_sum = float(p.sum())
    fracs = []
    for T, q in zip(Ts, ps):
        q = np.asarray(q, dtype=float).reshape(-1)
        denom = min(p_sum, float(q.sum()))
        if denom <= 0:
            continue
        fracs.append(float(np.asarray(T, dtype=float).sum()) / denom)
    if not fracs:
        return float("nan")
    return float(np.mean(fracs))


def solve_mpfgw(
    C1: np.ndarray,
    X1: np.ndarray,
    p1: np.ndarray,
    C2: np.ndarray,
    X2: np.ndarray,
    p2: np.ndarray,
    *,
    alpha: float,
    rho: float,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    tol: float = 1e-7,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve mass-specified FPGW and return cost, coupling, and transported mass.

    ``rho`` is the preserved-mass *fraction*; attribute costs use squared
    Euclidean distance on the provided feature matrices (already scaled).
    """
    C1 = _finite_relation_matrix(C1)
    C2 = _finite_relation_matrix(C2)
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    p1 = np.asarray(p1, dtype=float).reshape(-1)
    p2 = np.asarray(p2, dtype=float).reshape(-1)
    M = dist(X1, X2)
    m = transported_mass_from_rho(p1, p2, rho)

    restarts = max(1, int(gw_restarts))
    best_T, best_cost, best_log = None, None, None
    for r in range(restarts):
        G0 = None
        if restarts > 1:
            G0 = _random_transport_matrix_sinkhorn(p1, p2, int(gw_seed_start) + r)
            gsum = float(G0.sum())
            if gsum > 0:
                G0 = G0 * (m / gsum)
        T, log = partial_fused_gromov_mass(
            M,
            C1,
            C2,
            p1,
            p2,
            m=m,
            alpha=float(alpha),
            G0=G0,
            numItermax_gw=int(max_iter),
            tol=float(tol),
            log=True,
            verbose=verbose and restarts == 1,
        )
        cost = float(MPFGW_dist(M, C1, C2, T, alpha=float(alpha)))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_T = T
            best_log = log
    return {
        "cost": float(best_cost),
        "T": best_T,
        "mass": float(best_T.sum()),
        "m_target": float(m),
        "rho": float(rho),
        "alpha": float(alpha),
        "log": best_log,
    }


def mpfgw_pairwise_costs(
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    *,
    alpha: float,
    rho: float,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric mass-specified FPGW costs between every pair of measure networks.

    Only the upper triangle is solved. Diagonal is 0.

    Returns
    -------
    D : ndarray, shape (n, n)
    row_means : ndarray, shape (n,)
        Mean off-diagonal cost in each row (the no-barycenter baseline
        for that held-out graph).
    """
    n = len(Cs)
    if n == 0:
        raise ValueError("Need at least one measure network")
    if len(Ys) != n or len(ps) != n:
        raise ValueError("Cs, Ys, and ps must have the same length")

    D = np.zeros((n, n), dtype=float)
    if n == 1:
        return D, np.zeros(1, dtype=float)

    n_pairs = n * (n - 1) // 2
    if verbose:
        print(
            f"Computing {n_pairs} unique mass-specified FPGW pairs "
            f"(alpha={alpha:g}, rho={rho:g})..."
        )
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            out = solve_mpfgw(
                Cs[i], Ys[i], ps[i], Cs[j], Ys[j], ps[j],
                alpha=alpha,
                rho=rho,
                gw_restarts=gw_restarts,
                gw_seed_start=gw_seed_start,
                max_iter=max_iter,
            )
            D[i, j] = out["cost"]
            D[j, i] = out["cost"]
            done += 1
            if verbose:
                print(f"  pair ({i}, {j}): {out['cost']:.6g}  [{done}/{n_pairs}]")
    off = D.copy()
    np.fill_diagonal(off, np.nan)
    row_means = np.nanmean(off, axis=1)
    return D, row_means


def skeleton_shortestpath_relation(
    C_center: np.ndarray,
    X_center: np.ndarray,
    *,
    edge_fraction_pct: float,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    min_scalar: float,
    scalar_range: float,
    msc_type_order: Sequence[int],
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
    laplacian_base: str = "adjacency",
    laplacian_normalization: str = "none",
    tau: float = 1.0,
    sp_affinity: str = "exp",
    sp_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, nx.Graph]:
    """
    Replace a learned barycenter relation by shortest paths on a thresholded
    skeleton extracted from that relation.
    """
    G_thr, _kept, _cand = build_thresholded_barycenter_graph(
        C_center,
        X_center,
        edge_fraction_pct,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        domain_size=domain_size,
        scalar_min=min_scalar,
        scalar_range=scalar_range,
        msc_type_order=msc_type_order,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        sp_weighted=sp_weighted,
    )
    C_skel, _nodes = recompute_thresholded_relation_matrix(
        G_thr,
        relation_kind=relation_kind,
        relation_rescale=relation_rescale,
        relation_offset=relation_offset,
        sp_weighted=sp_weighted,
        laplacian_base=laplacian_base,
        laplacian_normalization=laplacian_normalization,
        tau=tau,
        sp_affinity=sp_affinity,
        sp_sigma=sp_sigma,
    )
    return _finite_relation_matrix(C_skel), G_thr


def _loo_cache_paths(cache_dir, fold: int):
    from pathlib import Path

    d = Path(cache_dir) / f"fold_{int(fold):02d}"
    return d, d / "C.npy", d / "X.npy", d / "p.npy"


def _try_load_loo_fold(cache_dir, fold: int) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    d, c_path, x_path, p_path = _loo_cache_paths(cache_dir, fold)
    if not (c_path.is_file() and x_path.is_file() and p_path.is_file()):
        return None
    return {
        "fold": int(fold),
        "C": np.load(c_path),
        "X": np.load(x_path),
        "p": np.load(p_path),
        "from_cache": True,
    }


def _save_loo_fold(cache_dir, fold: int, C, X, p) -> None:
    if cache_dir is None:
        return
    d, c_path, x_path, p_path = _loo_cache_paths(cache_dir, fold)
    d.mkdir(parents=True, exist_ok=True)
    np.save(c_path, np.asarray(C, dtype=float))
    np.save(x_path, np.asarray(X, dtype=float))
    np.save(p_path, np.asarray(p, dtype=float))


def loo_pfgw_barycenters(
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    *,
    init_C: np.ndarray,
    init_X: np.ndarray,
    init_p: np.ndarray,
    alpha: float,
    lambda_penalty: float,
    lambdas: Optional[Sequence[float]] = None,
    held_out_indices: Optional[Sequence[int]] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    random_state: int = 0,
    cache_dir: Optional[Any] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Leave-one-out penalized-PFGW barycenters ``B_theta^{(-r)}``.

    Each fold drops input ``r`` and runs ``pfgw_barycenters`` on the rest,
    using the candidate seed ``(init_C, init_X, init_p)`` and the same
    ``alpha`` / ``lambda_penalty`` / node count. Remaining barycentric
    weights are renormalized to sum to 1.
    """
    m = len(Cs)
    if m < 2:
        raise ValueError("Leave-one-out barycenters need at least 2 inputs")
    if len(Ys) != m or len(ps) != m:
        raise ValueError("Cs, Ys, and ps must have the same length")

    init_C = np.asarray(init_C, dtype=float)
    init_X = np.asarray(init_X, dtype=float)
    init_p = np.asarray(init_p, dtype=float).reshape(-1)
    N = int(init_C.shape[0])
    if init_X.shape[0] != N or init_p.shape[0] != N:
        raise ValueError(
            f"Seed C/X/p size mismatch: C={init_C.shape}, X={init_X.shape}, "
            f"p={init_p.shape}"
        )

    if lambdas is None:
        lam = np.ones(m, dtype=float) / m
    else:
        lam = np.asarray(lambdas, dtype=float).reshape(-1)
        if lam.shape[0] != m:
            raise ValueError(f"lambdas length {lam.shape[0]} != n_inputs {m}")

    if held_out_indices is None:
        folds = list(range(m))
    else:
        folds = [int(r) for r in held_out_indices]
        for r in folds:
            if r < 0 or r >= m:
                raise IndexError(f"held-out index {r} out of range for {m} inputs")

    results: List[Dict[str, Any]] = []
    for k, r in enumerate(folds):
        cached = _try_load_loo_fold(cache_dir, r)
        if cached is not None:
            if verbose:
                print(
                    f"LOO fold {k+1}/{len(folds)}: held-out r={r} "
                    f"(loaded cache, C={cached['C'].shape})"
                )
            results.append(cached)
            continue

        keep = [i for i in range(m) if i != r]
        lam_k = lam[keep].copy()
        lam_sum = float(lam_k.sum())
        if lam_sum <= 0:
            lam_k = np.ones(len(keep), dtype=float) / len(keep)
        else:
            lam_k /= lam_sum
        Lambda_list = [float(lambda_penalty)] * len(keep)

        if verbose:
            print(
                f"LOO fold {k+1}/{len(folds)}: held-out r={r}, "
                f"training on {len(keep)} graphs, N={N}, "
                f"alpha={alpha:g}, lambda={lambda_penalty:g}"
            )
        C_bar, X_bar, log = pfgw_barycenters(
            N=N,
            Ys=[Ys[i] for i in keep],
            Cs=[Cs[i] for i in keep],
            ps=[ps[i] for i in keep],
            p=init_p,
            Lambda_list=Lambda_list,
            lambdas=lam_k,
            alpha=float(alpha),
            max_iter=int(max_iter),
            tol=float(tol),
            log=True,
            init_C=init_C,
            init_X=init_X,
            random_state=int(random_state),
            gw_restarts=int(gw_restarts),
            gw_seed_start=int(gw_seed_start),
            debug=False,
        )
        C_bar = np.asarray(C_bar, dtype=float)
        X_bar = np.asarray(X_bar, dtype=float)
        _save_loo_fold(cache_dir, r, C_bar, X_bar, init_p)
        results.append(
            {
                "fold": int(r),
                "C": C_bar,
                "X": X_bar,
                "p": init_p.copy(),
                "log": log,
                "from_cache": False,
            }
        )
        if verbose:
            n_err = len(log.get("err", [])) if isinstance(log, dict) else 0
            last_err = log["err"][-1] if n_err else float("nan")
            print(
                f"  finished r={r}: C={C_bar.shape}, BCD steps={n_err}, "
                f"last err={last_err:.6g}"
            )
    return results


def _score_from_losses(L_bar: float, L_input: float) -> float:
    if not np.isfinite(L_input) or abs(L_input) < 1e-15:
        raise ValueError(f"L_input must be a positive finite baseline; got {L_input}")
    return float(1.0 - L_bar / L_input)


def held_out_fpgw_representativeness(
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    *,
    loo_barycenters: Sequence[Mapping[str, Any]],
    alpha_eval: float,
    rho_eval: float,
    D_input: Optional[np.ndarray] = None,
    skeleton_edge_fraction_pct: Optional[float] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    min_z: Optional[float] = None,
    domain_size: Optional[float] = None,
    min_scalar: Optional[float] = None,
    scalar_range: Optional[float] = None,
    msc_type_order: Optional[Sequence[int]] = None,
    relation_kind: str = "shortestpath",
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
    sp_weighted: bool = True,
    laplacian_base: str = "adjacency",
    laplacian_normalization: str = "none",
    tau: float = 1.0,
    sp_affinity: str = "exp",
    sp_sigma: Optional[float] = None,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Held-out FPGW representativeness skill ``S_rep``.

    Primary score uses each fold's learned barycenter relation ``C``.
    If ``skeleton_edge_fraction_pct`` is set, also reports the paired
    variant that replaces ``C`` by shortest paths on the extracted skeleton.

    Evaluation FPGW (``D_eval``) is mass-specified with fixed ``alpha_eval``
    and ``rho_eval`` — it does not reuse a candidate's training ``lambda``.
    """
    m = len(Cs)
    if m < 2:
        raise ValueError("Need at least 2 input graphs")
    folds = [int(row["fold"]) for row in loo_barycenters]
    if len(set(folds)) != len(folds):
        raise ValueError("Duplicate leave-one-out folds")

    if D_input is None:
        D_input, _row_means = mpfgw_pairwise_costs(
            Cs, Ys, ps,
            alpha=alpha_eval,
            rho=rho_eval,
            gw_restarts=gw_restarts,
            gw_seed_start=gw_seed_start,
            max_iter=max_iter,
            verbose=verbose,
        )
    else:
        D_input = np.asarray(D_input, dtype=float)
        if D_input.shape != (m, m):
            raise ValueError(
                f"D_input shape {D_input.shape} != ({m}, {m})"
            )

    off = D_input.copy()
    np.fill_diagonal(off, np.nan)
    L_input = float(np.nanmean(off))
    L_input_per = np.nanmean(off, axis=1)

    D_bar = np.full(m, np.nan, dtype=float)
    D_bar_skel = np.full(m, np.nan, dtype=float) if skeleton_edge_fraction_pct is not None else None
    masses = np.full(m, np.nan, dtype=float)
    fold_details: List[Dict[str, Any]] = []

    want_skel = skeleton_edge_fraction_pct is not None
    if want_skel:
        missing = [
            name for name, val in [
                ("min_x", min_x), ("min_y", min_y), ("min_z", min_z),
                ("domain_size", domain_size), ("min_scalar", min_scalar),
                ("scalar_range", scalar_range), ("msc_type_order", msc_type_order),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                "skeleton variant requires domain fields: " + ", ".join(missing)
            )

    for row in loo_barycenters:
        r = int(row["fold"])
        C_b = np.asarray(row["C"], dtype=float)
        X_b = np.asarray(row["X"], dtype=float)
        p_b = np.asarray(row["p"], dtype=float).reshape(-1)
        if verbose:
            print(
                f"D_eval(B^{{(-{r})}}, G^{{({r})}})  "
                f"C={C_b.shape}, n_heldout={Cs[r].shape[0]}"
            )
        out = solve_mpfgw(
            C_b, X_b, p_b, Cs[r], Ys[r], ps[r],
            alpha=alpha_eval,
            rho=rho_eval,
            gw_restarts=gw_restarts,
            gw_seed_start=gw_seed_start,
            max_iter=max_iter,
        )
        D_bar[r] = out["cost"]
        masses[r] = out["mass"]
        detail = {
            "fold": r,
            "cost": out["cost"],
            "mass": out["mass"],
            "T": out["T"],
        }

        if want_skel:
            C_skel, G_skel = skeleton_shortestpath_relation(
                C_b,
                X_b,
                edge_fraction_pct=float(skeleton_edge_fraction_pct),
                min_x=float(min_x),
                min_y=float(min_y),
                min_z=float(min_z),
                domain_size=float(domain_size),
                min_scalar=float(min_scalar),
                scalar_range=float(scalar_range),
                msc_type_order=msc_type_order,
                relation_kind=relation_kind,
                relation_rescale=relation_rescale,
                relation_offset=relation_offset,
                sp_weighted=sp_weighted,
                laplacian_base=laplacian_base,
                laplacian_normalization=laplacian_normalization,
                tau=tau,
                sp_affinity=sp_affinity,
                sp_sigma=sp_sigma,
            )
            out_s = solve_mpfgw(
                C_skel, X_b, p_b, Cs[r], Ys[r], ps[r],
                alpha=alpha_eval,
                rho=rho_eval,
                gw_restarts=gw_restarts,
                gw_seed_start=gw_seed_start,
                max_iter=max_iter,
            )
            D_bar_skel[r] = out_s["cost"]
            detail["cost_skeleton"] = out_s["cost"]
            detail["C_skeleton"] = C_skel
            detail["G_skeleton"] = G_skel
            if verbose:
                print(
                    f"  learned C: {out['cost']:.6g};  "
                    f"skeleton SP: {out_s['cost']:.6g}"
                )
        elif verbose:
            print(f"  learned C: {out['cost']:.6g}")
        fold_details.append(detail)

    used = np.array(folds, dtype=int)
    L_bar = float(np.mean(D_bar[used]))
    S_rep = _score_from_losses(L_bar, L_input)

    result: Dict[str, Any] = {
        "S_rep": S_rep,
        "L_bar": L_bar,
        "L_input": L_input,
        "L_input_per_heldout": L_input_per,
        "D_bar": D_bar,
        "D_input": D_input,
        "masses": masses,
        "folds": used,
        "alpha_eval": float(alpha_eval),
        "rho_eval": float(rho_eval),
        "fold_details": fold_details,
        "S_rep_skeleton": None,
        "L_bar_skeleton": None,
        "D_bar_skeleton": D_bar_skel,
        "skeleton_edge_fraction_pct": (
            None if skeleton_edge_fraction_pct is None
            else float(skeleton_edge_fraction_pct)
        ),
    }
    if want_skel:
        L_bar_s = float(np.mean(D_bar_skel[used]))
        result["L_bar_skeleton"] = L_bar_s
        result["S_rep_skeleton"] = _score_from_losses(L_bar_s, L_input)
    if verbose:
        print(
            f"L_bar={L_bar:.6g}, L_input={L_input:.6g}, "
            f"S_rep={S_rep:.4f}"
        )
        if want_skel:
            print(
                f"skeleton L_bar={result['L_bar_skeleton']:.6g}, "
                f"S_rep={result['S_rep_skeleton']:.4f}"
            )
    return result


def plot_held_out_representativeness(
    result: Mapping[str, Any],
    *,
    mesh_ids: Optional[Sequence] = None,
    figsize=(10, 4.5),
) -> None:
    """Paired per-fold distances for Metric 1 (learned C vs input baseline)."""
    folds = np.asarray(result["folds"], dtype=int)
    D_bar = np.asarray(result["D_bar"], dtype=float)[folds]
    L_per = np.asarray(result["L_input_per_heldout"], dtype=float)[folds]
    D_skel = result.get("D_bar_skeleton")
    has_skel = (
        D_skel is not None
        and result.get("S_rep_skeleton") is not None
    )
    if has_skel:
        D_skel = np.asarray(D_skel, dtype=float)[folds]

    if mesh_ids is None:
        labels = [f"r={r}" for r in folds]
    else:
        labels = [f"r={r}\nmesh {mesh_ids[r]}" for r in folds]

    x = np.arange(len(folds), dtype=float)
    n_series = 3 if has_skel else 2
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - (n_series - 1) * width / 2, D_bar, width, label=r"$D_{\mathrm{eval}}(B^{(-r)}, G^{(r)})$", color="#1f77b4")
    if has_skel:
        ax.bar(
            x - (n_series - 1) * width / 2 + width,
            D_skel,
            width,
            label=r"$D_{\mathrm{eval}}(B_{\mathrm{skel}}^{(-r)}, G^{(r)})$",
            color="#ff7f0e",
        )
    ax.bar(
        x + (n_series - 1) * width / 2,
        L_per,
        width,
        label=r"mean$_{s \neq r}\, D_{\mathrm{eval}}(G^{(s)}, G^{(r)})$",
        color="#7f7f7f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$D_{\mathrm{eval}}$")
    ax.set_xlabel("Held-out graph")
    ax.grid(True, axis="y", alpha=0.3)

    s = result["S_rep"]
    title = (
        r"Held-out FPGW representativeness"
        f"\n$S_{{rep}}={s:.3f}$"
        f"  ($L_{{bar}}={result['L_bar']:.4g}$"
        f", $L_{{input}}={result['L_input']:.4g}$"
        f", $\\alpha_{{eval}}={result['alpha_eval']:g}$"
        f", $\\rho_{{eval}}={result['rho_eval']:g}$)"
    )
    if has_skel:
        title += f"\nskeleton $S_{{rep}}={result['S_rep_skeleton']:.3f}$"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Held-out edge-evidence Brier skill (Metric 2)
# ---------------------------------------------------------------------------


def input_binary_adjacency(G: nx.Graph, nodelist: Optional[Sequence] = None) -> np.ndarray:
    """Symmetric 0/1 adjacency in ``nodelist`` order (diagonal 0)."""
    if nodelist is None:
        nodelist = list(G.nodes())
    A, _ = relation_matrix(G, kind="adjacency", nodelist=list(nodelist))
    A = np.asarray(A, dtype=float)
    A = ((A + A.T) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def transported_edge_support(
    pi: np.ndarray,
    K: np.ndarray,
    mu: np.ndarray,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Unconditional transported edge support

        Q = M_B^{-1} π K π^T M_B^{-1},

    with diagonal zeroed. ``pi`` has shape ``(n_B, n_r)``, ``K`` is the input
    adjacency, and ``mu`` is barycenter mass. Dividing by ``mu_i mu_j`` (not
    matched row masses) scores occurrence and adjacency together.
    """
    pi = np.asarray(pi, dtype=float)
    K = np.asarray(K, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if pi.ndim != 2:
        raise ValueError(f"pi must be 2-D; got shape {pi.shape}")
    if pi.shape[0] != mu.shape[0]:
        raise ValueError(
            f"pi rows {pi.shape[0]} != barycenter mass length {mu.shape[0]}"
        )
    if K.shape != (pi.shape[1], pi.shape[1]):
        raise ValueError(
            f"K shape {K.shape} != ({pi.shape[1]}, {pi.shape[1]}) for coupling"
        )
    inv = 1.0 / np.maximum(mu, float(eps))
    P = pi * inv[:, None]
    Q = P @ K @ P.T
    Q = 0.5 * (Q + Q.T)
    np.fill_diagonal(Q, 0.0)
    np.clip(Q, 0.0, 1.0, out=Q)
    return Q


def matched_mass_fraction(pi: np.ndarray, mu: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    """``c_i = (π 1)_i / μ_i``, clipped to ``[0, 1]``."""
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    c = pi.sum(axis=1) / np.maximum(mu, float(eps))
    return np.clip(c, 0.0, 1.0)


def solve_partial_emd(
    M: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    m: float,
    *,
    nb_dummies: int = 1,
    numItermax: int = 100000,
) -> np.ndarray:
    """Mass-specified partial OT (dummy-node EMD) on a feature cost ``M``."""
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    M = np.asarray(M, dtype=float)
    m = float(m)
    p_sum, q_sum = float(p.sum()), float(q.sum())
    m_max = min(p_sum, q_sum)
    if m < 0 or m > m_max + 1e-12:
        raise ValueError(f"Infeasible partial-OT mass m={m}, max={m_max}")
    m = min(m, m_max)
    n1, n2 = p.shape[0], q.shape[0]
    if m >= m_max - 1e-12 and abs(p_sum - q_sum) <= 1e-9:
        return emd(p, q, M, numItermax=numItermax)

    dummy_p = q_sum - m
    dummy_q = p_sum - m
    last_error = None
    nb = max(1, int(nb_dummies))
    while nb <= 301:
        p_ext = np.append(p, [dummy_p / nb] * nb)
        q_ext = np.append(q, [dummy_q / nb] * nb)
        M_emd = np.zeros((n1 + nb, n2 + nb), dtype=float)
        M_emd[:n1, :n2] = M
        fill = float(np.max(M)) if M.size else 0.0
        M_emd[-nb:, -nb:] = fill * 2.0 if fill > 0.0 else (abs(fill) * 1e3 + 1.0)
        try:
            Gc, logemd = emd(p_ext, q_ext, M_emd, numItermax=numItermax, log=True)
        except Exception as exc:
            last_error = exc
            nb += 50
            continue
        warning = logemd.get("warning") if isinstance(logemd, dict) else None
        if warning is not None and nb < 301:
            nb += 50
            continue
        if np.any(Gc[-nb:, -nb:] > 1e-12):
            nb += 50
            continue
        return Gc[:n1, :n2]
    raise RuntimeError(f"Partial EMD failed (m={m}, last error={last_error})")


def _edge_brier_cache_paths(cache_dir, index: int):
    from pathlib import Path

    d = Path(cache_dir)
    return d, d / "cache_meta.json", d / f"T_{int(index):02d}.npy"


def eval_couplings_to_inputs(
    C_center: np.ndarray,
    X_center: np.ndarray,
    p: np.ndarray,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    *,
    alpha_eval: float,
    rho_eval: float,
    correspondence: str = "mpfgw",
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    cache_dir: Optional[Any] = None,
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Fixed evaluation correspondences from the candidate barycenter to each input.

    ``correspondence='mpfgw'`` uses mass-specified FPGW with ``alpha_eval`` and
    ``rho_eval``. ``correspondence='spatial'`` uses partial OT on the first three
    feature columns (normalized xyz), independent of the learned relation.
    """
    correspondence = str(correspondence).lower()
    if correspondence not in ("mpfgw", "spatial"):
        raise ValueError("correspondence must be 'mpfgw' or 'spatial'")

    n_inputs = len(Cs)
    p = np.asarray(p, dtype=float).reshape(-1)
    X_center = np.asarray(X_center, dtype=float)
    C_center = np.asarray(C_center, dtype=float)

    meta = {
        "correspondence": correspondence,
        "alpha_eval": float(alpha_eval),
        "rho_eval": float(rho_eval),
        "n_inputs": int(n_inputs),
        "n_center": int(p.shape[0]),
    }
    cache_ok = False
    if cache_dir is not None:
        import json
        from pathlib import Path

        d, meta_path, _ = _edge_brier_cache_paths(cache_dir, 0)
        d.mkdir(parents=True, exist_ok=True)
        if meta_path.is_file() and all(
            (d / f"T_{s:02d}.npy").is_file() for s in range(n_inputs)
        ):
            try:
                saved = json.loads(meta_path.read_text(encoding="utf-8"))
                cache_ok = all(saved.get(k) == meta[k] for k in meta)
            except Exception:
                cache_ok = False
        if cache_ok:
            Ts = [np.load(d / f"T_{s:02d}.npy") for s in range(n_inputs)]
            if verbose:
                print(f"Loaded {n_inputs} eval couplings from {d.resolve()}")
            return Ts

    Ts: List[np.ndarray] = []
    for s in range(n_inputs):
        if verbose:
            print(
                f"Eval coupling {s + 1}/{n_inputs} "
                f"({correspondence}, alpha={alpha_eval:g}, rho={rho_eval:g})"
            )
        if correspondence == "mpfgw":
            out = solve_mpfgw(
                C_center, X_center, p, Cs[s], Ys[s], ps[s],
                alpha=alpha_eval,
                rho=rho_eval,
                gw_restarts=gw_restarts,
                gw_seed_start=gw_seed_start,
                max_iter=max_iter,
            )
            T = out["T"]
        else:
            M = dist(X_center[:, :3], np.asarray(Ys[s], dtype=float)[:, :3])
            m = transported_mass_from_rho(p, ps[s], rho_eval)
            T = solve_partial_emd(M, p, ps[s], m)
        Ts.append(np.asarray(T, dtype=float))

    if cache_dir is not None:
        import json
        from pathlib import Path

        d, meta_path, _ = _edge_brier_cache_paths(cache_dir, 0)
        d.mkdir(parents=True, exist_ok=True)
        for s, T in enumerate(Ts):
            np.save(d / f"T_{s:02d}.npy", T)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Ts


def _upper_pair_weights(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    n = mu.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    w = mu[iu] * mu[ju]
    return iu, ju, w


def held_out_edge_brier_from_support(
    Qs: Sequence[np.ndarray],
    mu: np.ndarray,
    *,
    graph_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Leave-one-out Brier skill on transported edge support.

    For each held-out ``r``,

        p_ij^(-r) = sum_{s≠r} λ_s q_ij^(s) / sum_{s≠r} λ_s

    with equal ``λ`` by default. Climatology predicts the training-only
    weighted-mean support (a single prevalence) for every pair.
    """
    m = len(Qs)
    if m < 2:
        raise ValueError("Edge Brier skill needs at least 2 input graphs")
    Qs = [np.asarray(Q, dtype=float) for Q in Qs]
    n = Qs[0].shape[0]
    for Q in Qs:
        if Q.shape != (n, n):
            raise ValueError("All support matrices must share shape (n_B, n_B)")

    if graph_weights is None:
        lam = np.ones(m, dtype=float)
    else:
        lam = np.asarray(graph_weights, dtype=float).reshape(-1)
        if lam.shape[0] != m:
            raise ValueError("graph_weights length must match number of Qs")
        if np.any(lam < 0) or float(lam.sum()) <= 0:
            raise ValueError("graph_weights must be nonnegative with positive sum")

    iu, ju, w = _upper_pair_weights(mu)
    if not np.any(w > 0):
        raise ValueError("Pair weights w_ij = mu_i mu_j are all zero")

    bs_num = 0.0
    clim_num = 0.0
    w_sum = 0.0
    sq_err_sum = np.zeros((n, n), dtype=float)
    p_mean = np.zeros((n, n), dtype=float)
    fold_bs = np.zeros(m, dtype=float)
    fold_clim = np.zeros(m, dtype=float)

    p_chunks: List[np.ndarray] = []
    q_chunks: List[np.ndarray] = []
    w_chunks: List[np.ndarray] = []

    for r in range(m):
        train = [s for s in range(m) if s != r]
        lam_t = lam[train]
        lam_t = lam_t / float(lam_t.sum())
        P = np.zeros((n, n), dtype=float)
        q_train_flat = []
        w_train = []
        for a, s in enumerate(train):
            P += lam_t[a] * Qs[s]
            q_train_flat.append(Qs[s][iu, ju])
            w_train.append(w)
        q_train_flat = np.concatenate(q_train_flat)
        w_train = np.concatenate(w_train)
        p_clim = float(np.average(q_train_flat, weights=w_train))

        Qr = Qs[r]
        diff2 = (P - Qr) ** 2
        fold_wsum = float(w.sum())
        fold_bs[r] = float(np.dot(w, diff2[iu, ju])) / fold_wsum
        fold_clim[r] = float(np.dot(w, (p_clim - Qr[iu, ju]) ** 2)) / fold_wsum
        bs_num += float(np.dot(w, diff2[iu, ju]))
        clim_num += float(np.dot(w, (p_clim - Qr[iu, ju]) ** 2))
        w_sum += fold_wsum
        sq_err_sum += diff2
        p_mean += P
        p_chunks.append(P[iu, ju].copy())
        q_chunks.append(Qr[iu, ju].copy())
        w_chunks.append(w.copy())

    BS = bs_num / w_sum
    BS_clim = clim_num / w_sum
    if not np.isfinite(BS_clim) or BS_clim < 1e-15:
        S_edge = 1.0 if BS < 1e-15 else float("nan")
    else:
        S_edge = float(1.0 - BS / BS_clim)

    q_mean = np.mean(np.stack(Qs, axis=0), axis=0)
    np.fill_diagonal(q_mean, 0.0)
    return {
        "S_edge": S_edge,
        "BS_edge": float(BS),
        "BS_clim": float(BS_clim),
        "fold_BS": fold_bs,
        "fold_BS_clim": fold_clim,
        "q_mean": q_mean,
        "p_loo_mean": p_mean / m,
        "sq_err_mean": sq_err_sum / m,
        "Qs": Qs,
        "p_flat": np.concatenate(p_chunks),
        "q_flat": np.concatenate(q_chunks),
        "w_flat": np.concatenate(w_chunks),
        "n_center": int(n),
        "n_inputs": int(m),
    }


def compute_held_out_edge_brier(
    C_center: np.ndarray,
    X_center: np.ndarray,
    p: np.ndarray,
    *,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    Gs: Sequence[nx.Graph],
    nodes_list: Optional[Sequence[Sequence]] = None,
    alpha_eval: float,
    rho_eval: float,
    correspondence: str = "mpfgw",
    graph_weights: Optional[Sequence[float]] = None,
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    cache_dir: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Metric 2: held-out edge-evidence Brier skill for one candidate barycenter.

    Independent of Metric 1: uses the saved barycenter ``(C, X, p)`` and
    recomputes correspondences with fixed evaluation ``alpha`` / ``rho``.
    """
    if Gs is None or len(Gs) != len(Cs):
        raise ValueError("Metric 2 needs input graphs Gs aligned with Cs")
    if nodes_list is None:
        orders = [list(G.nodes()) for G in Gs]
    else:
        orders = [list(nl) for nl in nodes_list]
        if len(orders) != len(Gs):
            raise ValueError("nodes_list length must match Gs")

    Ts = eval_couplings_to_inputs(
        C_center, X_center, p, Cs, Ys, ps,
        alpha_eval=alpha_eval,
        rho_eval=rho_eval,
        correspondence=correspondence,
        gw_restarts=gw_restarts,
        gw_seed_start=gw_seed_start,
        max_iter=max_iter,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    mu = np.asarray(p, dtype=float).reshape(-1)
    Qs: List[np.ndarray] = []
    Ks: List[np.ndarray] = []
    c_fracs: List[np.ndarray] = []
    for s, T in enumerate(Ts):
        if T.shape[0] != mu.shape[0]:
            raise ValueError(
                f"Coupling {s} has {T.shape[0]} barycenter rows, expected {mu.shape[0]}"
            )
        K = input_binary_adjacency(Gs[s], orders[s])
        if T.shape[1] != K.shape[0]:
            raise ValueError(
                f"Coupling {s} has {T.shape[1]} input columns, "
                f"adjacency has {K.shape[0]} nodes"
            )
        Q = transported_edge_support(T, K, mu)
        Qs.append(Q)
        Ks.append(K)
        c_fracs.append(matched_mass_fraction(T, mu))
        if verbose:
            iu, ju, w = _upper_pair_weights(mu)
            print(
                f"  graph {s}: mean q={float(np.average(Q[iu, ju], weights=w)):.4g}, "
                f"mean matched fraction={float(c_fracs[-1].mean()):.4g}"
            )

    result = held_out_edge_brier_from_support(
        Qs, mu, graph_weights=graph_weights
    )
    result.update(
        {
            "alpha_eval": float(alpha_eval),
            "rho_eval": float(rho_eval),
            "correspondence": str(correspondence),
            "matched_fractions": c_fracs,
            "Ks": Ks,
        }
    )
    if verbose:
        print(
            f"BS_edge={result['BS_edge']:.6g}, BS_clim={result['BS_clim']:.6g}, "
            f"S_edge={result['S_edge']:.4f}"
        )
    return result


def plot_edge_brier_reliability(
    result: Mapping[str, Any],
    *,
    n_bins: int = 10,
    figsize=(6.2, 5.4),
) -> None:
    """Reliability diagram: predicted edge evidence vs mean held-out support."""
    p = np.asarray(result["p_flat"], dtype=float)
    q = np.asarray(result["q_flat"], dtype=float)
    w = np.asarray(result["w_flat"], dtype=float)
    finite = np.isfinite(p) & np.isfinite(q) & np.isfinite(w) & (w > 0)
    p, q, w = p[finite], q[finite], w[finite]

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_p = []
    bin_q = []
    bin_n = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isclose(hi, 1.0):
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ww = w[mask]
        bin_p.append(float(np.average(p[mask], weights=ww)))
        bin_q.append(float(np.average(q[mask], weights=ww)))
        bin_n.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1, label="perfect calibration")
    if bin_p:
        sizes = 20 + 40 * np.sqrt(np.asarray(bin_n) / max(max(bin_n), 1))
        ax.scatter(bin_p, bin_q, s=sizes, c="#1f77b4", zorder=3, label="binned mean $q$")
        ax.plot(bin_p, bin_q, color="#1f77b4", lw=1.2, alpha=0.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"predicted evidence $p_{ij}^{(-r)}$")
    ax.set_ylabel(r"held-out support $q_{ij}^{(r)}$")
    ax.grid(True, alpha=0.3)
    s = result["S_edge"]
    ax.set_title(
        "Edge-evidence reliability\n"
        f"$S_{{edge}}={s:.3f}$"
        f"  (BS={result['BS_edge']:.4g}, BS$_{{clim}}$={result['BS_clim']:.4g}"
        f", {result.get('correspondence', 'mpfgw')})"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_edge_brier_error_map(
    result: Mapping[str, Any],
    X_center: np.ndarray,
    *,
    n_edges: int = 150,
    figsize=(6.4, 5.6),
    title: Optional[str] = None,
) -> None:
    """Barycenter xy map of the largest mean leave-one-out squared errors."""
    E = np.asarray(result["sq_err_mean"], dtype=float)
    n = E.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = E[iu, ju]
    k = min(int(n_edges), int(vals.size))
    if k <= 0:
        raise ValueError("No pairs to plot")
    order = np.argpartition(vals, -k)[-k:]
    order = order[np.argsort(vals[order])]
    X = np.asarray(X_center, dtype=float)
    xy = X[:, :2]

    segs = np.stack(
        [xy[iu[order]], xy[ju[order]]], axis=1
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xy[:, 0], xy[:, 1], s=8, c="0.75", linewidths=0, zorder=2)
    lc = LineCollection(
        segs,
        array=vals[order],
        cmap="magma_r",
        linewidths=0.8,
        alpha=0.9,
        zorder=3,
    )
    ax.add_collection(lc)
    pad = 0.03 * max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
    ax.set_xlim(float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad)
    ax.set_ylim(float(xy[:, 1].min()) - pad, float(xy[:, 1].max()) + pad)
    fig.colorbar(lc, ax=ax, label=r"mean$_r\,(p^{(-r)}_{ij}-q^{(r)}_{ij})^2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("normalized x")
    ax.set_ylabel("normalized y")
    ax.set_title(
        title
        or (
            f"Largest {k} edgewise Brier contributions "
            f"(mean over held-out folds)"
        )
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Morse-valid evidence retention (Metric 3)
# ---------------------------------------------------------------------------


class _UF:
    """Union-find with component count for cyclomatic number updates."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)
        self.n_components = int(n)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def same(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.n_components -= 1
        return True


def _seg_min_distance_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    """Minimum Euclidean distance between open-closed segments ab and cd."""

    def _clamp01(t: float) -> float:
        return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

    ab = b - a
    cd = d - c
    ac = a - c
    ab2 = float(np.dot(ab, ab))
    cd2 = float(np.dot(cd, cd))
    if ab2 < 1e-24 and cd2 < 1e-24:
        return float(np.linalg.norm(a - c))
    if ab2 < 1e-24:
        t = _clamp01(-float(np.dot(ac, cd)) / cd2)
        return float(np.linalg.norm(a - (c + t * cd)))
    if cd2 < 1e-24:
        t = _clamp01(float(np.dot(c - a, ab)) / ab2)
        return float(np.linalg.norm(c - (a + t * ab)))
    abcd = float(np.dot(ab, cd))
    denom = ab2 * cd2 - abcd * abcd
    if abs(denom) < 1e-24:
        # nearly parallel: distance from a to cd, b to cd, c to ab, d to ab
        dists = []
        for p, u, v, vv in (
            (a, c, d, cd2),
            (b, c, d, cd2),
            (c, a, b, ab2),
            (d, a, b, ab2),
        ):
            t = _clamp01(float(np.dot(p - u, v - u)) / vv)
            dists.append(float(np.linalg.norm(p - (u + t * (v - u)))))
        return min(dists)
    s = _clamp01((abcd * float(np.dot(cd, ac)) - cd2 * float(np.dot(ab, ac))) / denom)
    t = _clamp01((ab2 * float(np.dot(cd, ac)) - abcd * float(np.dot(ab, ac))) / denom)
    return float(np.linalg.norm((a + s * ab) - (c + t * cd)))


def _projection_overlap_length(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    """Length of overlap of ab and cd when projected onto ab's direction."""
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-15:
        return 0.0
    u = ab / L
    ts = sorted(float(np.dot(p - a, u)) for p in (a, b, c, d))
    # overlap of [0, L] with [min(c,d proj), max(c,d proj)]
    p0, p1 = sorted(float(np.dot(p - a, u)) for p in (c, d))
    lo = max(0.0, p0)
    hi = min(L, p1)
    return max(0.0, hi - lo)


def segments_geometrically_conflict_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    overlap_tol: float,
) -> bool:
    """Proper crossing or near-duplicate overlap at display resolution."""
    if _segments_properly_cross_2d(a, b, c, d):
        return True
    if overlap_tol <= 0.0:
        return False
    dist = _seg_min_distance_2d(a, b, c, d)
    if dist >= overlap_tol:
        return False
    overlap = _projection_overlap_length(a, b, c, d)
    return overlap >= overlap_tol


def _morse_dirs(f_v: float, f_neighbors: Sequence[float], *, eps: float) -> Tuple[int, int, int]:
    n_up = n_down = n_eq = 0
    for fn in f_neighbors:
        d = float(fn) - float(f_v)
        if d > eps:
            n_up += 1
        elif d < -eps:
            n_down += 1
        else:
            n_eq += 1
    return n_up, n_down, n_eq


def morse_node_ok(
    node_type: int,
    scalar: float,
    neighbor_scalars: Sequence[float],
    *,
    scalar_eps: float = 1e-12,
    saddle_degree_cap: int = 4,
    saddle_dir_cap: int = 2,
) -> bool:
    """Local Morse incidence / scalar-direction constraints at one node."""
    n_up, n_down, n_eq = _morse_dirs(scalar, neighbor_scalars, eps=scalar_eps)
    deg = n_up + n_down + n_eq
    t = int(node_type)
    if t == -1:
        if deg > 2 or n_eq > 0:
            return False
        if deg == 2:
            return n_up == 1 and n_down == 1
        return True
    if n_eq > 0:
        return False
    if t == 0:
        return n_down == 0
    if t == 2:
        return n_up == 0
    if t == 1:
        if deg > int(saddle_degree_cap):
            return False
        if n_up > int(saddle_dir_cap) or n_down > int(saddle_dir_cap):
            return False
        return True
    return True


def _beta1_from_edges(edges: Sequence[Tuple[int, int]]) -> int:
    if not edges:
        return 0
    H = nx.Graph()
    H.add_edges_from((int(u), int(v)) for u, v in edges)
    v = H.number_of_nodes()
    e = H.number_of_edges()
    c = nx.number_connected_components(H)
    return int(e - v + c)


def barycenter_mean_edge_evidence(
    C_center: np.ndarray,
    X_center: np.ndarray,
    p: np.ndarray,
    *,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    Gs: Sequence[nx.Graph],
    nodes_list: Optional[Sequence[Sequence]] = None,
    alpha_eval: float,
    rho_eval: float,
    correspondence: str = "mpfgw",
    gw_restarts: int = 1,
    gw_seed_start: int = 0,
    max_iter: int = 200,
    cache_dir: Optional[Any] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Full-sample mean transported support ``(1/m) sum_r Q^{(r)}``."""
    if nodes_list is None:
        orders = [list(G.nodes()) for G in Gs]
    else:
        orders = [list(nl) for nl in nodes_list]
    Ts = eval_couplings_to_inputs(
        C_center, X_center, p, Cs, Ys, ps,
        alpha_eval=alpha_eval,
        rho_eval=rho_eval,
        correspondence=correspondence,
        gw_restarts=gw_restarts,
        gw_seed_start=gw_seed_start,
        max_iter=max_iter,
        cache_dir=cache_dir,
        verbose=verbose,
    )
    mu = np.asarray(p, dtype=float).reshape(-1)
    acc = np.zeros((mu.shape[0], mu.shape[0]), dtype=float)
    for s, T in enumerate(Ts):
        K = input_binary_adjacency(Gs[s], orders[s])
        acc += transported_edge_support(T, K, mu)
    return acc / max(len(Ts), 1)


def _edge_xy_length(pos: Mapping[Any, np.ndarray], u: int, v: int, *, xyz: bool) -> float:
    pu = np.asarray(pos[u], dtype=float)
    pv = np.asarray(pos[v], dtype=float)
    if not xyz:
        pu, pv = pu[:2], pv[:2]
    return float(np.linalg.norm(pu - pv))


def compute_morse_valid_retention(
    X_center: np.ndarray,
    *,
    evidence: np.ndarray,
    min_x: float,
    min_y: float,
    min_z: float,
    domain_size: float,
    min_scalar: float,
    scalar_range: float,
    msc_type_order: Sequence[int],
    evidence_eps: float = 1e-6,
    max_edges: int = 4000,
    b1_plus: int = 10,
    overlap_tol: Optional[float] = None,
    length_mode: str = "euclidean3d",
    scalar_eps: float = 1e-12,
    local_search: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Metric 3: fraction of evidence-weighted curve length retainable in one
    geometrically and combinatorially valid Morse graph.
    """
    import time

    t0 = time.perf_counter()
    X = np.asarray(X_center, dtype=float)
    P = np.asarray(evidence, dtype=float)
    n = X.shape[0]
    if P.shape != (n, n):
        raise ValueError(f"evidence shape {P.shape} != ({n}, {n})")

    G_nodes = nx.Graph()
    _decode_barycenter_nodes(
        G_nodes,
        X,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        domain_size=domain_size,
        scalar_min=min_scalar,
        scalar_range=scalar_range,
        msc_type_order=msc_type_order,
    )
    pos3 = {
        i: np.array(
            [G_nodes.nodes[i]["x"], G_nodes.nodes[i]["y"], G_nodes.nodes[i]["z"]],
            dtype=float,
        )
        for i in range(n)
    }
    pos2 = {i: pos3[i][:2].copy() for i in range(n)}
    scalars = np.array(
        [float(G_nodes.nodes[i].get("height", 0.0)) for i in range(n)],
        dtype=float,
    )
    types = np.array(
        [int(G_nodes.nodes[i].get("type", -1)) for i in range(n)],
        dtype=int,
    )
    use_xyz = str(length_mode).lower() != "xy"
    if overlap_tol is None:
        overlap_tol = 0.002 * float(domain_size) if domain_size > 0 else 1e-4
    overlap_tol = float(overlap_tol)

    candidates: List[Dict[str, Any]] = []
    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu.tolist(), ju.tolist()):
        pe = float(P[i, j])
        if not np.isfinite(pe) or pe <= float(evidence_eps):
            continue
        ell = _edge_xy_length(pos3 if use_xyz else pos2, i, j, xyz=use_xyz)
        if not np.isfinite(ell) or ell <= 0.0:
            continue
        candidates.append(
            {
                "u": int(i),
                "v": int(j),
                "p": pe,
                "length": ell,
                "ink": pe * ell,
                "key": (int(i), int(j)) if i < j else (int(j), int(i)),
            }
        )
    n_raw = len(candidates)
    candidates.sort(key=lambda e: (-e["ink"], -e["p"], e["key"]))
    truncated = False
    if len(candidates) > int(max_edges):
        candidates = candidates[: int(max_edges)]
        truncated = True
        if verbose:
            print(
                f"E+ truncated to top-{max_edges} by p*ell "
                f"({n_raw} pairs had p > {evidence_eps:g})"
            )
    total_ink = float(sum(e["ink"] for e in candidates))
    if total_ink <= 0.0:
        raise ValueError("No positive evidence-weighted length in E+")

    keys = [e["key"] for e in candidates]
    by_key = {e["key"]: e for e in candidates}
    xy = {i: pos2[i] for i in range(n)}

    def _conflict(e, f) -> bool:
        u, v = e["u"], e["v"]
        p, q = f["u"], f["v"]
        if u == p or u == q or v == p or v == q:
            return False
        return segments_geometrically_conflict_2d(
            xy[u], xy[v], xy[p], xy[q], overlap_tol=overlap_tol
        )

    incident: Dict[int, List[int]] = {i: [] for i in range(n)}

    def _node_ok(v: int, extra: Optional[int] = None, without: Optional[int] = None) -> bool:
        nbrs = list(incident[v])
        if without is not None and without in nbrs:
            nbrs.remove(without)
        if extra is not None:
            nbrs.append(extra)
        return morse_node_ok(
            int(types[v]),
            float(scalars[v]),
            [float(scalars[u]) for u in nbrs],
            scalar_eps=scalar_eps,
        )

    selected_keys: List[Tuple[int, int]] = []
    selected_set = set()
    uf = _UF(n)
    n_cycle = 0

    def _can_add(e, *, uf_state=None, cycle_count=None) -> bool:
        u, v = e["u"], e["v"]
        uf_use = uf if uf_state is None else uf_state
        cyc = n_cycle if cycle_count is None else cycle_count
        closes = uf_use.same(u, v)
        if closes and cyc + 1 > int(b1_plus):
            return False
        for sk in selected_keys:
            if _conflict(e, by_key[sk]):
                return False
        if not _node_ok(u, extra=v) or not _node_ok(v, extra=u):
            return False
        return True

    def _add(e) -> None:
        nonlocal n_cycle
        u, v = e["u"], e["v"]
        if uf.same(u, v):
            n_cycle += 1
        else:
            uf.union(u, v)
        selected_keys.append(e["key"])
        selected_set.add(e["key"])
        incident[u].append(v)
        incident[v].append(u)

    for e in candidates:
        if _can_add(e):
            _add(e)

    n_greedy = len(selected_keys)
    n_ls = 0
    if local_search:
        improved = True
        while improved:
            improved = False
            for e in candidates:
                if e["key"] in selected_set:
                    continue
                conflicts = [
                    by_key[sk] for sk in selected_keys if _conflict(e, by_key[sk])
                ]
                conflict_ink = float(sum(c["ink"] for c in conflicts))
                if e["ink"] <= conflict_ink + 1e-15:
                    continue
                conflict_keys = {c["key"] for c in conflicts}
                trial = [k for k in selected_keys if k not in conflict_keys]
                trial.append(e["key"])
                inc_trial: Dict[int, List[int]] = {i: [] for i in range(n)}
                for k in trial:
                    a, b = k
                    inc_trial[a].append(b)
                    inc_trial[b].append(a)
                touched = {e["u"], e["v"]}
                for c in conflicts:
                    touched.add(c["u"])
                    touched.add(c["v"])
                if any(
                    not morse_node_ok(
                        int(types[vtx]),
                        float(scalars[vtx]),
                        [float(scalars[u]) for u in inc_trial[vtx]],
                        scalar_eps=scalar_eps,
                    )
                    for vtx in touched
                ):
                    continue
                if _beta1_from_edges(trial) > int(b1_plus):
                    continue
                for c in conflicts:
                    selected_keys.remove(c["key"])
                    selected_set.remove(c["key"])
                    incident[c["u"]].remove(c["v"])
                    incident[c["v"]].remove(c["u"])
                selected_keys.append(e["key"])
                selected_set.add(e["key"])
                incident[e["u"]].append(e["v"])
                incident[e["v"]].append(e["u"])
                n_ls += 1
                improved = True
                break

    # Verify the retained set is in M
    for i, k in enumerate(selected_keys):
        e = by_key[k]
        for k2 in selected_keys[i + 1 :]:
            if _conflict(e, by_key[k2]):
                raise RuntimeError("Retained set has a geometric conflict")
    for v in range(n):
        if not morse_node_ok(
            int(types[v]),
            float(scalars[v]),
            [float(scalars[u]) for u in incident[v]],
            scalar_eps=scalar_eps,
        ):
            raise RuntimeError(f"Retained set violates Morse constraints at node {v}")
    beta1 = _beta1_from_edges(selected_keys)
    if beta1 > int(b1_plus):
        raise RuntimeError(
            f"Retained set has beta1={beta1} exceeding budget {b1_plus}"
        )

    retained_ink = float(sum(by_key[k]["ink"] for k in selected_keys))
    S = retained_ink / total_ink
    n_cross = 0
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            if _conflict(candidates[a], candidates[b]):
                n_cross += 1

    G_plus = G_nodes.copy()
    for e in candidates:
        G_plus.add_edge(
            e["u"], e["v"],
            evidence=e["p"],
            length=e["length"],
            ink=e["ink"],
            retained=e["key"] in selected_set,
        )
    elapsed = float(time.perf_counter() - t0)
    result = {
        "S_Morse": float(S),
        "retained_ink": retained_ink,
        "total_ink": total_ink,
        "n_E_plus": len(candidates),
        "n_E_plus_untruncated": int(n_raw),
        "truncated": bool(truncated),
        "n_retained": len(selected_keys),
        "n_discarded": len(candidates) - len(selected_keys),
        "n_greedy": int(n_greedy),
        "n_local_search_accepts": int(n_ls),
        "beta1": int(beta1),
        "b1_plus": int(b1_plus),
        "n_conflict_pairs": int(n_cross),
        "overlap_tol": overlap_tol,
        "evidence_eps": float(evidence_eps),
        "solver": "greedy" + ("+local_search" if local_search else ""),
        "optimality": "approximate",
        "elapsed_sec": elapsed,
        "selected_keys": selected_keys,
        "candidates": candidates,
        "G": G_plus,
        "types": types,
        "scalars": scalars,
    }
    if verbose:
        print(
            f"E+={len(candidates)} edges, retained={len(selected_keys)}, "
            f"beta1={beta1} (budget {b1_plus}), "
            f"conflict pairs={n_cross}"
        )
        print(
            f"S_Morse={S:.4f}  (ink {retained_ink:.6g} / {total_ink:.6g}), "
            f"solver={result['solver']}, {elapsed:.2f}s"
        )
    return result


def plot_morse_retention(
    result: Mapping[str, Any],
    *,
    figsize=(7.2, 6.2),
    discarded_alpha: float = 0.22,
) -> None:
    """Retained Morse-valid edges vs discarded conflicting evidence."""
    G: nx.Graph = result["G"]
    pos = {n: np.array([G.nodes[n]["x"], G.nodes[n]["y"]], dtype=float) for n in G.nodes()}
    retained = [(u, v) for u, v, d in G.edges(data=True) if d.get("retained")]
    discarded = [(u, v) for u, v, d in G.edges(data=True) if not d.get("retained")]

    fig, ax = plt.subplots(figsize=figsize)
    if discarded:
        segs_d = np.stack([np.stack([pos[u], pos[v]]) for u, v in discarded])
        lc_d = LineCollection(
            segs_d, colors=(0.85, 0.25, 0.15, discarded_alpha), linewidths=0.7, zorder=2
        )
        ax.add_collection(lc_d)
    if retained:
        evid = np.array([float(G.edges[e]["evidence"]) for e in retained], dtype=float)
        segs_r = np.stack([np.stack([pos[u], pos[v]]) for u, v in retained])
        lw = 0.6 + 2.2 * (evid / max(float(evid.max()), 1e-12))
        lc_r = LineCollection(segs_r, colors="#1f3b73", linewidths=lw, zorder=3)
        ax.add_collection(lc_r)
    draw_msc_nodes(G, pos, ax, regular_size=8, critical_size=18)
    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    pad = 0.03 * max(float(np.ptp(xs)), float(np.ptp(ys)), 1e-6)
    ax.set_xlim(float(xs.min()) - pad, float(xs.max()) + pad)
    ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Morse-valid evidence retention\n"
        f"$S_{{Morse}}={result['S_Morse']:.3f}$  "
        f"(kept {result['n_retained']}/{result['n_E_plus']} edges, "
        f"$\\beta_1={result['beta1']}\\leq {result['b1_plus']}$)"
    )
    ax.plot([], [], color="#1f3b73", lw=2, label="retained")
    ax.plot([], [], color=(0.85, 0.25, 0.15, 0.7), lw=2, label="discarded (conflict)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()
