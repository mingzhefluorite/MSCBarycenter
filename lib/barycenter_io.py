"""
Save / load a fully reconstitutable MSC FGW barycenter bundle.

The on-disk layout under ``outdir`` is::

    meta.json                 # domain, relation, feature, PFGW params
                              # includes exclude_seed / seed_graph_id
    C_center.npy              # dense barycenter relation (pre-thresholding)
    X_center.npy              # barycenter features
    p.npy                     # barycenter mass
    lambdas.npy
    Ts.npz                    # final couplings T_0 .. T_{S-1} (barycenter inputs)
    Cs.npz / Ys.npz / ps.npz  # barycenter-input relations / features / masses
    nodes_list.json           # per-input node order used for matrices
    G_complete.gpickle        # complete (unthresholded) barycenter graph
    G_mst.gpickle             # optional MST reconstruction
    inputs/G_{i}.gpickle      # optional barycenter-input graphs
    seed/                     # only when exclude_seed=True (out-of-set seed)
      C.npy / Y.npy / p.npy
      G.gpickle               # optional
      nodes.json              # optional
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from lib.msc_barycenter_threshold import build_complete_barycenter_graph

PathLike = Union[str, Path]

# Folder / meta stamp: month-day-year-hour-minute-second
BARYCENTER_SAVE_TIME_FMT = "%m-%d-%Y-%H-%M-%S"

# Union of computation knobs written by the barycenter notebooks.
# Inspect notebooks walk every key and report those absent from a bundle.
# ``pipelines`` lists which notebook typically writes the key.
BUNDLE_PARAM_SPEC: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    # group, key, description, pipelines
    ("run", "saved_at", "Bundle save time (mm-dd-yyyy-hh-mm-ss)", ("shortestpath", "lambda_converge")),
    ("run", "source_notebook", "Notebook that wrote this bundle", ("shortestpath", "lambda_converge")),
    ("run", "pipeline", "Computation pipeline identifier", ("shortestpath", "lambda_converge")),
    ("run", "DATA_ROOT", "Project-relative data root", ("shortestpath", "lambda_converge")),
    ("run", "DATA_KEYWORD", "Dataset keyword used in paths / run names", ("shortestpath", "lambda_converge")),
    ("run", "MSC_DIR", "Directory of MSC node/edge files", ("shortestpath", "lambda_converge")),
    ("run", "configured_mesh_ids", "Mesh indices selected before seed hold-out", ("shortestpath", "lambda_converge")),
    ("sampling", "SAMPLE_RATE", "Morse-path subsample rate", ("shortestpath", "lambda_converge")),
    ("sampling", "SAMPLE_MODE", "Subsample mode ('step' | 'geo')", ("shortestpath", "lambda_converge")),
    ("sampling", "SAMPLE_COMPONENT", "Component kept after subsample", ("shortestpath", "lambda_converge")),
    ("seed", "seed_graph_id", "Seed index in the configured mesh sequence", ("shortestpath", "lambda_converge")),
    ("seed", "seed_mesh_id", "Mesh id of the seed (when recorded)", ("shortestpath", "lambda_converge")),
    ("seed", "exclude_seed", "Whether the seed was held out of PFGW inputs", ("shortestpath", "lambda_converge")),
    ("seed", "expand_to_largest", "Expand seed to the largest input node count", ("shortestpath", "lambda_converge")),
    ("relation", "relation_kind", "Pairwise relation used for C", ("shortestpath", "lambda_converge")),
    ("relation", "laplacian_base", "Laplacian base graph (heat-kernel relations)", ("shortestpath", "lambda_converge")),
    ("relation", "laplacian_normalization", "Laplacian normalization", ("shortestpath", "lambda_converge")),
    ("relation", "tau", "Heat-kernel time scale", ("shortestpath", "lambda_converge")),
    ("relation", "sp_weighted", "Weighted shortest paths (3D Euclidean)", ("shortestpath", "lambda_converge")),
    ("relation", "sp_affinity", "Shortest-path affinity transform", ("shortestpath", "lambda_converge")),
    ("relation", "sp_sigma", "Shortest-path affinity bandwidth", ("shortestpath", "lambda_converge")),
    ("relation", "relation_rescale", "Global relation scale applied to C", ("shortestpath", "lambda_converge")),
    ("relation", "relation_offset", "Relation offset applied to C", ("shortestpath", "lambda_converge")),
    ("features", "MSC_TYPE_ORDER", "Critical-type one-hot order", ("shortestpath", "lambda_converge")),
    ("features", "type_scale", "One-hot amplitude; mismatch cost is 2 * type_scale**2", ("shortestpath", "lambda_converge")),
    ("features", "TYPE_SCALE_PERCENTILE", "Percentile x used to set type_scale from same-type costs", ("shortestpath", "lambda_converge")),
    ("features", "type_scale_percentile_value", "Lowest aggregated x-percentile of same-type feature costs", ("shortestpath", "lambda_converge")),
    ("features", "n_types", "Number of Morse types in the one-hot block", ("shortestpath", "lambda_converge")),
    ("features", "min_x", "Domain min x", ("shortestpath", "lambda_converge")),
    ("features", "max_x", "Domain max x", ("shortestpath", "lambda_converge")),
    ("features", "min_y", "Domain min y", ("shortestpath", "lambda_converge")),
    ("features", "max_y", "Domain max y", ("shortestpath", "lambda_converge")),
    ("features", "min_z", "Domain min z", ("shortestpath", "lambda_converge")),
    ("features", "max_z", "Domain max z", ("shortestpath", "lambda_converge")),
    ("features", "domain_size", "Coordinate normalization scale", ("shortestpath", "lambda_converge")),
    ("features", "min_scalar", "Scalar field minimum", ("shortestpath", "lambda_converge")),
    ("features", "max_scalar", "Scalar field maximum", ("shortestpath", "lambda_converge")),
    ("features", "scalar_range", "Scalar normalization range", ("shortestpath", "lambda_converge")),
    ("measure", "node_measure", "Node-mass construction", ("shortestpath", "lambda_converge")),
    ("measure", "node_measure_rescale_factor", "Arc-length measure rescale", ("shortestpath", "lambda_converge")),
    ("measure", "balanced_distribution", "Whether masses are normalized to 1", ("shortestpath", "lambda_converge")),
    ("pfgw", "use_partial", "Partial (vs large-penalty) FGW", ("shortestpath", "lambda_converge")),
    ("pfgw", "n_graphs", "Number of PFGW barycenter inputs", ("shortestpath", "lambda_converge")),
    ("pfgw", "N_maxnode", "Largest input node count (pre-expansion)", ("shortestpath", "lambda_converge")),
    ("pfgw", "N_center", "Barycenter support size after postprocess", ("shortestpath", "lambda_converge")),
    ("pfgw", "n_center", "Saved C_center dimension", ("shortestpath", "lambda_converge")),
    ("pfgw", "n_inputs", "Saved barycenter-input count", ("shortestpath", "lambda_converge")),
    ("pfgw", "alpha_fgw", "FGW trade-off α (structure vs features)", ("shortestpath", "lambda_converge")),
    ("pfgw", "lambda_penalty", "Partial-matching penalty λ (or target λ)", ("shortestpath", "lambda_converge")),
    ("pfgw", "Lambda_list", "Per-input λ used for the saved couplings", ("shortestpath", "lambda_converge")),
    ("pfgw", "max_iter", "PFGW barycenter outer iterations", ("shortestpath", "lambda_converge")),
    ("pfgw", "bary_tol", "PFGW barycenter convergence tolerance", ("shortestpath", "lambda_converge")),
    ("pfgw", "random_state", "RNG seed for PFGW initialization", ("shortestpath", "lambda_converge")),
    ("pfgw", "gw_restarts", "Inner PFGW random restarts", ("shortestpath", "lambda_converge")),
    ("pfgw", "gw_seed_start", "Inner PFGW restart seed offset", ("shortestpath", "lambda_converge")),
    ("pfgw", "loss_fun", "GW loss ('square_loss')", ("shortestpath", "lambda_converge")),
    ("pfgw", "stop_criterion", "PFGW barycenter stop criterion", ("shortestpath", "lambda_converge")),
    ("pfgw", "accumulate_fails_limit", "Inner solver fail budget", ("shortestpath", "lambda_converge")),
    ("pfgw", "pfgw_debug", "Verbose PFGW debug logging", ("shortestpath", "lambda_converge")),
    ("postprocess", "MIN_COUPLING_FRACTION", "Min transported-mass fraction to keep a node", ("shortestpath", "lambda_converge")),
    ("lambda_converge", "lambda_init", "Starting λ for geometric decay", ("lambda_converge",)),
    ("lambda_converge", "lambda_target", "Final λ after decay", ("lambda_converge",)),
    ("lambda_converge", "decay_steps", "Number of geometric λ decay steps", ("lambda_converge",)),
    ("lambda_converge", "lambda_schedule", "Recorded λ values that were solved", ("lambda_converge",)),
    ("lambda_converge", "lambda_converge_loss", "Per-λ FPGW loss sums", ("lambda_converge",)),
    ("lambda_converge", "lambda_converge_transported_mass", "Per-λ transported masses", ("lambda_converge",)),
    ("artifacts", "mesh_ids", "Mesh ids of saved barycenter inputs", ("shortestpath", "lambda_converge")),
    ("artifacts", "has_Ts", "Whether final couplings were saved", ("shortestpath", "lambda_converge")),
    ("artifacts", "has_G_complete", "Whether G_complete was saved", ("shortestpath", "lambda_converge")),
    ("artifacts", "has_G_mst", "Whether MST reconstruction was saved", ("shortestpath", "lambda_converge")),
    ("artifacts", "has_input_graphs", "Whether input graphs were saved", ("shortestpath", "lambda_converge")),
    ("artifacts", "has_seed_bundle", "Whether a held-out seed/ folder was saved", ("shortestpath", "lambda_converge")),
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _save_graph(path: Path, G: nx.Graph) -> None:
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        return pickle.load(f)


def save_barycenter_bundle(
    outdir: PathLike,
    *,
    C_center: np.ndarray,
    X_center: np.ndarray,
    p: np.ndarray,
    Cs: Sequence[np.ndarray],
    Ys: Sequence[np.ndarray],
    ps: Sequence[np.ndarray],
    Ts: Optional[Sequence[np.ndarray]] = None,
    lambdas: Optional[np.ndarray] = None,
    nodes_list: Optional[Sequence[Sequence]] = None,
    Gs: Optional[Sequence[nx.Graph]] = None,
    G_mst: Optional[nx.Graph] = None,
    G_complete: Optional[nx.Graph] = None,
    mesh_ids: Optional[Sequence] = None,
    meta: Optional[Mapping[str, Any]] = None,
    build_complete_if_missing: bool = True,
    exclude_seed: bool = False,
    seed_graph_id: Optional[int] = None,
    C_seed: Optional[np.ndarray] = None,
    Y_seed: Optional[np.ndarray] = None,
    p_seed: Optional[np.ndarray] = None,
    G_seed: Optional[nx.Graph] = None,
    seed_nodes: Optional[Sequence] = None,
    seed_mesh_id: Optional[Any] = None,
    append_timestamp: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Persist a complete (pre-thresholding) barycenter and evaluation inputs.

    ``Cs`` / ``Ys`` / ``ps`` / ``Ts`` / ``lambdas`` / ``Gs`` / ``mesh_ids`` must
    be the **barycenter input set** (aligned with each other). When the seed
    network was held out (``exclude_seed=True``), pass the seed arrays via
    ``C_seed`` / ``Y_seed`` / ``p_seed`` (and optionally ``G_seed``); they are
    written under ``seed/`` and recorded in ``meta.json``.

    If ``G_complete`` is None and ``build_complete_if_missing`` is True, builds
    the complete edge graph from ``C_center`` using domain fields in ``meta``.

    ``saved_at`` (``mm-dd-yyyy-hh-mm-ss``) is always written into ``meta``.
    When ``append_timestamp`` is True and ``outdir`` does not already contain
    that stamp, the stamp is appended to the folder name so later runs do not
    overwrite earlier results.
    """
    meta = dict(meta or {})
    stamp = str(meta.get("saved_at") or barycenter_save_timestamp())
    meta["saved_at"] = stamp
    outdir = Path(outdir)
    if append_timestamp and stamp not in outdir.name:
        outdir = outdir.parent / f"{outdir.name}_{stamp}"
    if (outdir / "meta.json").is_file() and not overwrite:
        raise FileExistsError(
            f"Bundle already exists at {outdir}; pass overwrite=True "
            "or use a timestamped run name"
        )
    outdir = _ensure_dir(outdir)
    exclude_seed = bool(exclude_seed)

    if exclude_seed:
        if C_seed is None or Y_seed is None or p_seed is None:
            raise ValueError(
                "exclude_seed=True requires C_seed, Y_seed, and p_seed "
                "(out-of-set seed used to initialize but not as a PFGW input)"
            )
        if seed_graph_id is None:
            raise ValueError("exclude_seed=True requires seed_graph_id")
    elif C_seed is not None or Y_seed is not None or p_seed is not None:
        raise ValueError(
            "C_seed/Y_seed/p_seed are only valid when exclude_seed=True; "
            "for an in-set seed, index it via seed_graph_id into Cs/Ys/ps"
        )

    n_inputs = len(Cs)
    if not (len(Ys) == n_inputs and len(ps) == n_inputs):
        raise ValueError(
            f"Cs/Ys/ps length mismatch: {len(Cs)}, {len(Ys)}, {len(ps)}"
        )
    if Ts is not None and len(Ts) != n_inputs:
        raise ValueError(f"Ts length {len(Ts)} != n_inputs {n_inputs}")
    if lambdas is not None and len(np.asarray(lambdas)) != n_inputs:
        raise ValueError(
            f"lambdas length {len(np.asarray(lambdas))} != n_inputs {n_inputs}"
        )
    if mesh_ids is not None and len(mesh_ids) != n_inputs:
        raise ValueError(f"mesh_ids length {len(mesh_ids)} != n_inputs {n_inputs}")
    if Gs is not None and len(Gs) != n_inputs:
        raise ValueError(f"Gs length {len(Gs)} != n_inputs {n_inputs}")
    if nodes_list is not None and len(nodes_list) != n_inputs:
        raise ValueError(
            f"nodes_list length {len(nodes_list)} != n_inputs {n_inputs}"
        )

    if not exclude_seed and seed_graph_id is not None:
        if seed_graph_id < 0 or seed_graph_id >= n_inputs:
            raise IndexError(
                f"seed_graph_id={seed_graph_id} out of range for {n_inputs} inputs"
            )

    required_domain = ("min_x", "min_y", "min_z", "domain_size")
    if build_complete_if_missing and G_complete is None:
        missing = [k for k in required_domain if k not in meta]
        if missing:
            raise ValueError(
                f"meta missing {missing}; needed to build complete barycenter graph"
            )
        G_complete, _ = build_complete_barycenter_graph(
            C_center,
            X_center,
            min_x=float(meta["min_x"]),
            min_y=float(meta["min_y"]),
            min_z=float(meta["min_z"]),
            domain_size=float(meta["domain_size"]),
            scalar_min=meta.get("min_scalar"),
            scalar_range=meta.get("scalar_range"),
            msc_type_order=meta.get("MSC_TYPE_ORDER"),
            relation_kind=meta.get("relation_kind", "shortestpath"),
            relation_rescale=float(meta.get("relation_rescale", 1.0)),
            relation_offset=float(meta.get("relation_offset", 0.0)),
            sp_weighted=bool(meta.get("sp_weighted", True)),
        )

    np.save(outdir / "C_center.npy", np.asarray(C_center, dtype=float))
    np.save(outdir / "X_center.npy", np.asarray(X_center, dtype=float))
    np.save(outdir / "p.npy", np.asarray(p, dtype=float))
    if lambdas is None:
        lambdas = np.ones(n_inputs, dtype=float) / max(n_inputs, 1)
    np.save(outdir / "lambdas.npy", np.asarray(lambdas, dtype=float))

    np.savez_compressed(
        outdir / "Cs.npz", **{f"C_{i}": np.asarray(C, dtype=float) for i, C in enumerate(Cs)}
    )
    np.savez_compressed(
        outdir / "Ys.npz", **{f"Y_{i}": np.asarray(Y, dtype=float) for i, Y in enumerate(Ys)}
    )
    np.savez_compressed(
        outdir / "ps.npz", **{f"p_{i}": np.asarray(pi, dtype=float) for i, pi in enumerate(ps)}
    )

    if Ts is not None:
        np.savez_compressed(
            outdir / "Ts.npz",
            **{f"T_{i}": np.asarray(T, dtype=float) for i, T in enumerate(Ts)},
        )
    elif (outdir / "Ts.npz").is_file():
        (outdir / "Ts.npz").unlink()

    if nodes_list is not None:
        _write_json(
            outdir / "nodes_list.json",
            [[_jsonable_node(n) for n in nodes] for nodes in nodes_list],
        )
    elif (outdir / "nodes_list.json").is_file():
        (outdir / "nodes_list.json").unlink()

    if mesh_ids is not None:
        meta = {**meta, "mesh_ids": list(mesh_ids)}

    meta = {
        **meta,
        "exclude_seed": exclude_seed,
        "seed_graph_id": None if seed_graph_id is None else int(seed_graph_id),
        "seed_mesh_id": seed_mesh_id,
        "n_center": int(np.asarray(C_center).shape[0]),
        "n_inputs": int(n_inputs),
        "has_Ts": Ts is not None,
        "has_G_complete": G_complete is not None,
        "has_G_mst": G_mst is not None,
        "has_input_graphs": Gs is not None,
        "has_seed_bundle": exclude_seed,
    }
    _write_json(outdir / "meta.json", meta)

    if G_complete is not None:
        _save_graph(outdir / "G_complete.gpickle", G_complete)
    if G_mst is not None:
        _save_graph(outdir / "G_mst.gpickle", G_mst)

    inputs_dir = outdir / "inputs"
    if inputs_dir.is_dir():
        shutil.rmtree(inputs_dir)
    if Gs is not None:
        inputs_dir = _ensure_dir(inputs_dir)
        for i, G in enumerate(Gs):
            _save_graph(inputs_dir / f"G_{i}.gpickle", G)

    seed_dir = outdir / "seed"
    if seed_dir.is_dir():
        shutil.rmtree(seed_dir)
    if exclude_seed:
        seed_dir = _ensure_dir(seed_dir)
        np.save(seed_dir / "C.npy", np.asarray(C_seed, dtype=float))
        np.save(seed_dir / "Y.npy", np.asarray(Y_seed, dtype=float))
        np.save(seed_dir / "p.npy", np.asarray(p_seed, dtype=float))
        if G_seed is not None:
            _save_graph(seed_dir / "G.gpickle", G_seed)
        if seed_nodes is not None:
            _write_json(
                seed_dir / "nodes.json",
                [_jsonable_node(n) for n in seed_nodes],
            )

    return outdir


def _jsonable_node(n):
    if isinstance(n, (np.integer,)):
        return int(n)
    return n


def _load_npz(path: Path, prefix: str) -> List[np.ndarray]:
    data = np.load(path, allow_pickle=False)
    keys = sorted(data.files, key=lambda k: int(k.split("_", 1)[1]))
    return [data[k] for k in keys if k.startswith(prefix)]


def load_barycenter_bundle(outdir: PathLike) -> Dict[str, Any]:
    """
    Load a bundle written by ``save_barycenter_bundle``.

    Returns a dict with arrays, graphs, and ``meta`` suitable for evaluation /
    visualization notebooks. When the seed was held out, ``seed`` is a dict with
    ``C`` / ``Y`` / ``p`` (and optional ``G`` / ``nodes`` / ``mesh_id``);
    otherwise ``seed`` is None and ``meta["seed_graph_id"]`` indexes into ``Cs``.
    """
    outdir = Path(outdir)
    if not (outdir / "meta.json").is_file():
        raise FileNotFoundError(f"No meta.json in {outdir}")

    meta = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))
    bundle: Dict[str, Any] = {
        "meta": meta,
        "C_center": np.load(outdir / "C_center.npy"),
        "X_center": np.load(outdir / "X_center.npy"),
        "p": np.load(outdir / "p.npy"),
        "lambdas": np.load(outdir / "lambdas.npy"),
        "Cs": _load_npz(outdir / "Cs.npz", "C_"),
        "Ys": _load_npz(outdir / "Ys.npz", "Y_"),
        "ps": _load_npz(outdir / "ps.npz", "p_"),
        "outdir": outdir,
    }

    ts_path = outdir / "Ts.npz"
    if ts_path.is_file():
        bundle["Ts"] = _load_npz(ts_path, "T_")
    else:
        bundle["Ts"] = None

    nodes_path = outdir / "nodes_list.json"
    if nodes_path.is_file():
        bundle["nodes_list"] = json.loads(nodes_path.read_text(encoding="utf-8"))
    else:
        bundle["nodes_list"] = None

    g_complete = outdir / "G_complete.gpickle"
    bundle["G_complete"] = _load_graph(g_complete) if g_complete.is_file() else None

    g_mst = outdir / "G_mst.gpickle"
    bundle["G_mst"] = _load_graph(g_mst) if g_mst.is_file() else None

    inputs_dir = outdir / "inputs"
    if inputs_dir.is_dir():
        g_files = sorted(inputs_dir.glob("G_*.gpickle"), key=lambda p: int(p.stem.split("_")[1]))
        bundle["Gs"] = [_load_graph(p) for p in g_files]
    else:
        bundle["Gs"] = None

    bundle["mesh_ids"] = meta.get("mesh_ids")

    seed_dir = outdir / "seed"
    exclude_seed = bool(meta.get("exclude_seed", False))
    if seed_dir.is_dir() and (seed_dir / "C.npy").is_file():
        seed: Dict[str, Any] = {
            "C": np.load(seed_dir / "C.npy"),
            "Y": np.load(seed_dir / "Y.npy"),
            "p": np.load(seed_dir / "p.npy"),
            "mesh_id": meta.get("seed_mesh_id"),
            "graph_id": meta.get("seed_graph_id"),
            "exclude_seed": True,
        }
        g_seed = seed_dir / "G.gpickle"
        seed["G"] = _load_graph(g_seed) if g_seed.is_file() else None
        nodes_seed = seed_dir / "nodes.json"
        if nodes_seed.is_file():
            seed["nodes"] = json.loads(nodes_seed.read_text(encoding="utf-8"))
        else:
            seed["nodes"] = None
        bundle["seed"] = seed
    else:
        bundle["seed"] = None

    # Legacy bundles sometimes saved the full Cs (including seed) while
    # Ts/lambdas matched only the barycenter inputs. Recover a consistent split.
    Cs, Ys, ps = bundle["Cs"], bundle["Ys"], bundle["ps"]
    Ts, lambdas = bundle["Ts"], bundle["lambdas"]
    n_cs = len(Cs)
    n_bary = None
    if Ts is not None and len(Ts) == n_cs - 1:
        n_bary = len(Ts)
    elif len(np.asarray(lambdas)) == n_cs - 1:
        n_bary = len(np.asarray(lambdas))

    if bundle["seed"] is None and n_bary is not None:
        sid = meta.get("seed_graph_id")
        if sid is None:
            # Prefer an index whose size matches the barycenter.
            n_center = int(bundle["p"].shape[0])
            candidates = [i for i, pi in enumerate(ps) if len(pi) == n_center]
            sid = candidates[0] if candidates else 0
        sid = int(sid)
        if 0 <= sid < n_cs:
            bundle["seed"] = {
                "C": Cs[sid],
                "Y": Ys[sid],
                "p": ps[sid],
                "G": None,
                "nodes": (
                    bundle["nodes_list"][sid]
                    if bundle["nodes_list"] is not None and sid < len(bundle["nodes_list"])
                    else None
                ),
                "mesh_id": meta.get("seed_mesh_id"),
                "graph_id": sid,
                "exclude_seed": True,
            }
            keep = [i for i in range(n_cs) if i != sid]
            bundle["Cs"] = [Cs[i] for i in keep]
            bundle["Ys"] = [Ys[i] for i in keep]
            bundle["ps"] = [ps[i] for i in keep]
            if bundle["nodes_list"] is not None and len(bundle["nodes_list"]) == n_cs:
                bundle["nodes_list"] = [bundle["nodes_list"][i] for i in keep]
            if bundle["Gs"] is not None:
                # Only treat Gs as a full (seed-inclusive) sequence when mesh_ids
                # has the same length; otherwise Gs was already barycenter-only
                # (possibly with stale extra pickles from a prior save).
                if len(bundle["Gs"]) == n_cs and (
                    bundle["mesh_ids"] is None
                    or len(bundle["mesh_ids"]) == n_cs
                ):
                    bundle["seed"]["G"] = bundle["Gs"][sid]
                    bundle["Gs"] = [bundle["Gs"][i] for i in keep]
                elif len(bundle["Gs"]) > n_bary:
                    bundle["Gs"] = bundle["Gs"][:n_bary]
            if bundle["mesh_ids"] is not None and len(bundle["mesh_ids"]) == n_cs:
                bundle["seed"]["mesh_id"] = bundle["mesh_ids"][sid]
                bundle["mesh_ids"] = [bundle["mesh_ids"][i] for i in keep]
            exclude_seed = True
            meta = {
                **meta,
                "exclude_seed": True,
                "seed_graph_id": sid,
                "n_inputs": len(bundle["Cs"]),
                "has_seed_bundle": True,
            }
            bundle["meta"] = meta

    bundle["exclude_seed"] = bool(meta.get("exclude_seed", exclude_seed))
    bundle["seed_graph_id"] = meta.get("seed_graph_id")
    return bundle


def resolve_eval_seed(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Resolve in-set vs out-of-set seed for evaluation notebooks.

    Returns dict with:
      - exclude_seed (bool)
      - seed_graph_id (optional int; original index in the full sequence)
      - seed_idx (int | None): index into barycenter ``Cs`` when in-set
      - C_seed, Y_seed, p_seed
      - seed_mesh_id, seed_label (for plots / logs)
    """
    meta = bundle.get("meta") or {}
    exclude_seed = bool(bundle.get("exclude_seed", meta.get("exclude_seed", False)))
    seed_graph_id = bundle.get("seed_graph_id", meta.get("seed_graph_id"))
    if seed_graph_id is not None:
        seed_graph_id = int(seed_graph_id)

    seed = bundle.get("seed")
    if exclude_seed and seed is not None:
        mesh_id = seed.get("mesh_id", meta.get("seed_mesh_id"))
        return {
            "exclude_seed": True,
            "seed_graph_id": seed_graph_id,
            "seed_idx": None,
            "C_seed": seed["C"],
            "Y_seed": seed["Y"],
            "p_seed": seed["p"],
            "seed_mesh_id": mesh_id,
            "seed_label": (
                f"out-of-set seed (orig idx {seed_graph_id}, mesh {mesh_id})"
                if mesh_id is not None
                else f"out-of-set seed (orig idx {seed_graph_id})"
            ),
        }

    if seed_graph_id is None:
        seed_graph_id = 0
    Cs = bundle["Cs"]
    if seed_graph_id < 0 or seed_graph_id >= len(Cs):
        raise IndexError(
            f"seed_graph_id={seed_graph_id} out of range for {len(Cs)} barycenter inputs"
        )
    mesh_ids = bundle.get("mesh_ids")
    mesh_id = None if mesh_ids is None else mesh_ids[seed_graph_id]
    return {
        "exclude_seed": False,
        "seed_graph_id": seed_graph_id,
        "seed_idx": seed_graph_id,
        "C_seed": Cs[seed_graph_id],
        "Y_seed": bundle["Ys"][seed_graph_id],
        "p_seed": bundle["ps"][seed_graph_id],
        "seed_mesh_id": mesh_id,
        "seed_label": (
            f"in-set seed (input {seed_graph_id}, mesh {mesh_id})"
            if mesh_id is not None
            else f"in-set seed (input {seed_graph_id})"
        ),
    }


def default_barycenter_outdir(
    root: PathLike = "results/barycenters",
    run_name: str = "latest",
) -> Path:
    """Convenience path under ``results/barycenters/<run_name>``."""
    return Path(root) / run_name


def barycenter_save_timestamp(when: Optional[datetime] = None) -> str:
    """Return ``mm-dd-yyyy-hh-mm-ss`` (month, then minute)."""
    return (when or datetime.now()).strftime(BARYCENTER_SAVE_TIME_FMT)


def timestamped_barycenter_outdir(
    run_name: str,
    root: PathLike = "results/barycenters",
    saved_at: Optional[str] = None,
) -> Path:
    """``results/barycenters/<run_name>_<mm-dd-yyyy-hh-mm-ss>``.

    If that folder already exists (same-second collision), a numeric suffix
    is appended so consecutive saves never overlap.
    """
    stamp = saved_at or barycenter_save_timestamp()
    root = Path(root)
    out = root / f"{run_name}_{stamp}"
    extra = 1
    while out.exists():
        out = root / f"{run_name}_{stamp}_{extra:02d}"
        extra += 1
    return out


def _sanitize_meta_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_meta_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _sanitize_meta_value(v) for k, v in value.items()}
    return value


def build_barycenter_run_meta(
    *,
    source_notebook: str,
    pipeline: str,
    saved_at: Optional[str] = None,
    **params: Any,
) -> Dict[str, Any]:
    """Assemble a JSON-safe computation-parameter dict for ``meta.json``."""
    stamp = saved_at or barycenter_save_timestamp()
    meta = {key: _sanitize_meta_value(value) for key, value in params.items()}
    meta["saved_at"] = stamp
    meta["source_notebook"] = source_notebook
    meta["pipeline"] = pipeline
    return meta


def known_bundle_param_keys() -> List[str]:
    """All catalog keys the inspect notebook should try to load."""
    return [key for _, key, _, _ in BUNDLE_PARAM_SPEC]


def inspect_bundle_parameters(
    meta: Optional[Mapping[str, Any]] = None,
    *,
    outdir: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Compare bundle ``meta`` against the union of notebook parameters.

    Returns present / missing / extra keys. Missing keys are those in
    :data:`BUNDLE_PARAM_SPEC` that the saved bundle does not contain — including
    pipeline-specific knobs (e.g. ``lambda_init`` on a shortestpath bundle).
    """
    if meta is None:
        if outdir is None:
            raise ValueError("inspect_bundle_parameters requires meta or outdir")
        bundle = load_barycenter_bundle(outdir)
        meta = bundle["meta"]
        outdir = bundle.get("outdir", outdir)
    meta = dict(meta or {})
    catalog_keys = set(known_bundle_param_keys())
    present: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for group, key, description, pipelines in BUNDLE_PARAM_SPEC:
        row = {
            "group": group,
            "key": key,
            "description": description,
            "pipelines": list(pipelines),
        }
        if key in meta:
            row["value"] = meta[key]
            present.append(row)
        else:
            missing.append(row)
    extra = sorted(k for k in meta.keys() if k not in catalog_keys)
    return {
        "meta": meta,
        "outdir": None if outdir is None else Path(outdir),
        "pipeline": meta.get("pipeline"),
        "source_notebook": meta.get("source_notebook"),
        "saved_at": meta.get("saved_at"),
        "present": present,
        "missing": missing,
        "missing_keys": [row["key"] for row in missing],
        "extra_keys": extra,
        "n_present": len(present),
        "n_missing": len(missing),
        "n_catalog": len(BUNDLE_PARAM_SPEC),
    }


def format_bundle_parameter_report(inspection: Mapping[str, Any]) -> str:
    """Human-readable present / missing parameter listing."""
    lines: List[str] = []
    outdir = inspection.get("outdir")
    if outdir is not None:
        lines.append(f"Bundle: {Path(outdir).resolve()}")
    lines.append(f"pipeline: {inspection.get('pipeline')}")
    lines.append(f"source_notebook: {inspection.get('source_notebook')}")
    lines.append(f"saved_at: {inspection.get('saved_at')}")
    lines.append(
        f"catalog: {inspection.get('n_present')} present, "
        f"{inspection.get('n_missing')} missing "
        f"(of {inspection.get('n_catalog')})"
    )
    present_by_key = {row["key"]: row for row in inspection.get("present") or []}
    current_group = None
    for group, key, _description, pipelines in BUNDLE_PARAM_SPEC:
        if group != current_group:
            current_group = group
            lines.append("")
            lines.append(f"=== {group} ===")
        if key in present_by_key:
            value = present_by_key[key]["value"]
            rendered = json.dumps(value, default=_json_default)
            if len(rendered) > 160:
                rendered = rendered[:157] + "..."
            status = rendered
        else:
            pipes = ", ".join(pipelines)
            status = f"MISSING  (typically written by: {pipes})"
        lines.append(f"  {key:<36} {status}")

    missing_keys = list(inspection.get("missing_keys") or [])
    lines.append("")
    lines.append(f"Missing parameters ({len(missing_keys)}):")
    if missing_keys:
        for key in missing_keys:
            lines.append(f"  - {key}")
    else:
        lines.append("  (none)")

    extra = list(inspection.get("extra_keys") or [])
    lines.append("")
    lines.append(f"Extra keys not in the inspect catalog ({len(extra)}):")
    if extra:
        for key in extra:
            lines.append(f"  - {key}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


_BUNDLE_STAMP_RE = re.compile(r"(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})")


def _infer_bundle_pipeline(run_name: str, meta_pipeline: Optional[Any] = None) -> Optional[str]:
    if meta_pipeline:
        return str(meta_pipeline)
    if "lambda_converge" in str(run_name):
        return "lambda_converge"
    if _BUNDLE_STAMP_RE.search(str(run_name)) or str(run_name).endswith("_latest"):
        return "shortestpath"
    return None


def _bundle_matches_keyword(row: Mapping[str, Any], keyword: str) -> bool:
    kw = str(keyword).strip()
    if not kw:
        return True
    meta_kw = row.get("DATA_KEYWORD")
    if meta_kw is not None and str(meta_kw) == kw:
        return True
    name = str(row.get("run_name") or "")
    return name == kw or name.startswith(kw + "_")


def _bundle_datetime(row: Mapping[str, Any]) -> Optional[datetime]:
    stamp = row.get("saved_at")
    if not stamp:
        match = _BUNDLE_STAMP_RE.search(str(row.get("run_name") or ""))
        stamp = match.group(1) if match else None
    if not stamp:
        return None
    try:
        return datetime.strptime(str(stamp), BARYCENTER_SAVE_TIME_FMT)
    except ValueError:
        return None


def list_barycenter_bundles(
    root: PathLike = "results/barycenters",
    *,
    keyword: Optional[str] = None,
    pipeline: Optional[str] = None,
    newest_first: bool = True,
) -> List[Dict[str, Any]]:
    """List saved bundles under ``results/barycenters`` (any with ``meta.json``).

    ``keyword`` keeps folders for one dataset (``DATA_KEYWORD`` in meta, or a
    folder name starting with ``{keyword}_``). ``pipeline`` further restricts
    to ``shortestpath`` or ``lambda_converge``. Dated runs are listed newest
    first; legacy ``*_latest`` folders without a stamp sort last.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        row = {
            "run_name": path.name,
            "path": path,
            "saved_at": meta.get("saved_at"),
            "pipeline": _infer_bundle_pipeline(path.name, meta.get("pipeline")),
            "source_notebook": meta.get("source_notebook"),
            "DATA_KEYWORD": meta.get("DATA_KEYWORD"),
            "n_inputs": meta.get("n_inputs"),
            "n_center": meta.get("n_center"),
        }
        if keyword is not None and not _bundle_matches_keyword(row, keyword):
            continue
        if pipeline is not None and row.get("pipeline") != pipeline:
            continue
        rows.append(row)
    if newest_first:
        def _sort_key(row: Mapping[str, Any]):
            when = _bundle_datetime(row)
            if when is None:
                return (1, 0.0, str(row["run_name"]))
            try:
                ts = when.timestamp()
            except OSError:
                ts = 0.0
            return (0, -ts, str(row["run_name"]))

        rows.sort(key=_sort_key)
    return rows


def print_barycenter_bundle_menu(
    rows: Sequence[Mapping[str, Any]],
    *,
    keyword: Optional[str] = None,
    pipeline: Optional[str] = None,
) -> None:
    """Print a numbered table so notebooks can set ``BUNDLE_INDEX``."""
    label = "Bundles"
    extras = []
    if keyword:
        extras.append(f"keyword={keyword!r}")
    if pipeline:
        extras.append(f"pipeline={pipeline!r}")
    if extras:
        label = f"Bundles ({', '.join(extras)})"
    if not rows:
        print(f"{label}: (none)")
        return
    print(f"{label}: {len(rows)}")
    for index, row in enumerate(rows):
        print(
            f"  [{index}] {row['run_name']}\n"
            f"      pipeline={row.get('pipeline')}  saved_at={row.get('saved_at')}  "
            f"n_inputs={row.get('n_inputs')}  n_center={row.get('n_center')}"
        )


def select_barycenter_bundle(
    keyword: str,
    *,
    index: Optional[int] = 0,
    run_name: Optional[str] = None,
    pipeline: Optional[str] = None,
    root: PathLike = "results/barycenters",
) -> Dict[str, Any]:
    """Pick one saved bundle for ``keyword``.

    ``run_name`` wins when set (exact folder name). Otherwise ``index`` selects
    from :func:`list_barycenter_bundles` (0 = newest). ``index=None`` also
    selects the newest row.
    """
    rows = list_barycenter_bundles(root, keyword=keyword, pipeline=pipeline)
    if run_name:
        matches = [row for row in rows if row["run_name"] == run_name]
        if not matches:
            matches = [
                row
                for row in list_barycenter_bundles(root)
                if row["run_name"] == run_name
            ]
        if not matches:
            available = ", ".join(row["run_name"] for row in rows) or "(none)"
            raise FileNotFoundError(
                f"No bundle named {run_name!r} for keyword={keyword!r}. "
                f"Available: {available}"
            )
        chosen = dict(matches[0])
        chosen["index"] = next(
            (i for i, row in enumerate(rows) if row["run_name"] == run_name),
            None,
        )
        return chosen
    if not rows:
        detail = f"keyword={keyword!r}"
        if pipeline:
            detail += f", pipeline={pipeline!r}"
        raise FileNotFoundError(
            f"No saved barycenter bundles for {detail} under {Path(root)}"
        )
    if index is None:
        index = 0
    index = int(index)
    if index < 0 or index >= len(rows):
        raise IndexError(
            f"BUNDLE_INDEX={index} out of range for {len(rows)} bundles "
            f"(valid: 0..{len(rows) - 1})"
        )
    chosen = dict(rows[index])
    chosen["index"] = index
    return chosen
