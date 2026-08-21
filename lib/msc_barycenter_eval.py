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
from ot.utils import dist

from lib.graph_relations import apply_relation_scaling, relation_matrix
from lib.gromov import PFGW_dist_with_penalty, partial_fused_gromov_ver1
from lib.msc_barycenter_threshold import build_thresholded_barycenter_graph
from lib.msc_color_transfer import coupling_entry_threshold
from lib.msc_plotting import msc_node_visual_type, node_height


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
