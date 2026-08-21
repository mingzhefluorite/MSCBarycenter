"""Reconstruct MSC barycenter topology from transported separatrix-local connectivity.

Topology is inferred from exact input adjacency and soft connectivity along paths
whose edges share one fixed recorded ``separatrixID``. ``C_center`` is retained
only for optional post-hoc comparison after the graph is fixed.
"""

from __future__ import annotations

import importlib.metadata
import json
import pickle
import platform
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from lib.graph_sparsify import (
    geometric_planarity_allows_edge,
    is_geometrically_planar_2d,
)

PathLike = Union[str, Path]
ConstraintMode = Literal["off", "soft", "hard"]
AffinityMode = Literal["separatrix_hop_decay", "separatrix_length_normalized_decay"]
LengthReferenceMode = Literal[
    "input_median", "separatrix_median", "global_input_family_median"
]
MissingSeparatrixPolicy = Literal["error", "warn", "skip"]
NodeSupportClass = Literal[
    "well_supported", "softly_supported", "weakly_supported", "uncovered"
]

TERMINOLOGY_VERSION = 4
EPS = 1e-15


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class BundleValidationResult:
    """Validated bundle metadata and coupling-aligned node orders."""

    warnings: List[str]
    input_node_orders: List[List[Any]]
    n_center: int
    feature_dim: int
    n_inputs: int
    requested_support_size: Optional[int]
    actual_support_size: int
    has_couplings: bool
    has_input_graphs: bool
    has_node_order_metadata: bool
    has_held_out_seed: bool
    exclude_seed: bool


@dataclass
class DecodedNodesResult:
    """Decoded barycenter node table plus provenance diagnostics."""

    dataframe: pd.DataFrame
    disagreements: pd.DataFrame
    warnings: List[str]
    has_coordinates: bool
    has_scalar: bool
    has_types: bool
    effective_type_constraint_mode: ConstraintMode
    effective_scalar_order_constraint_mode: ConstraintMode
    effective_planarity_mode: ConstraintMode


@dataclass
class SeparatrixMembership:
    """Per-input separatrix membership structures."""

    edge_separatrix_ids: Dict[Any, frozenset]
    node_incident_separatrix_ids: Dict[Any, frozenset]
    separatrix_to_edges: Dict[str, List[Any]]
    subgraphs: Dict[str, nx.Graph]
    warnings: List[str]


@dataclass
class InputAffinityResult:
    """Direct adjacency and separatrix-local affinity for one input."""

    direct_adjacency_matrix: np.ndarray
    separatrix_local_connectivity_affinity: np.ndarray
    best_shared_separatrix_ids: Dict[Tuple[int, int], frozenset]
    hop_distances: Dict[Tuple[int, int], int]
    reference_spacing: Optional[float]
    warnings: List[str]
    diagnostics: Dict[str, Any]


@dataclass
class TransportedConnectivityResult:
    """Aggregated transported connectivity supports and evidence."""

    conditional_direct_adjacency_support: np.ndarray
    conditional_direct_adjacency_support_std: np.ndarray
    conditional_separatrix_connectivity_support: np.ndarray
    conditional_separatrix_connectivity_support_std: np.ndarray
    conditional_soft_extension_support: np.ndarray
    direct_adjacency_evidence: np.ndarray
    separatrix_connectivity_evidence: np.ndarray
    soft_extension_evidence: np.ndarray
    pair_coverage: np.ndarray
    pair_is_covered: np.ndarray
    pair_contributing_inputs: np.ndarray
    matched_fractions: np.ndarray
    input_conditional_direct_adjacency_support: List[np.ndarray]
    input_conditional_separatrix_connectivity_support: List[np.ndarray]
    input_direct_adjacency_matrices: List[np.ndarray]
    input_separatrix_affinities: List[np.ndarray]
    input_annotation_diagnostics: pd.DataFrame
    input_subgraph_diagnostics: pd.DataFrame
    node_separatrix_membership: pd.DataFrame
    edge_separatrix_multiplicity: pd.DataFrame
    input_sampling_diagnostics: pd.DataFrame
    warnings: List[str]
    diagnostics: Dict[str, Any]


@dataclass
class CandidateEdgeResult:
    """Candidate-edge table for reconstruction."""

    all_pairs: pd.DataFrame
    selected_edges: pd.DataFrame
    diagnostics: Dict[str, Any]


@dataclass
class ComponentTarget:
    """Target connected-component count and provenance."""

    count: int
    source: str
    input_component_counts: List[int]
    assumption_used: bool


@dataclass
class ForestResult:
    """Maximum-support forest over candidates."""

    graph: nx.Graph
    selected_edges: pd.DataFrame
    diagnostics: Dict[str, Any]


@dataclass
class AugmentationResult:
    """Consensus augmentation and redundancy filtering."""

    graph: nx.Graph
    added_edges: pd.DataFrame
    rejected_edges: pd.DataFrame
    diagnostics: Dict[str, Any]


@dataclass
class LocalRepairResult:
    """Bounded local repair outcome."""

    graph: nx.Graph
    edit_history: pd.DataFrame
    objective_components: Dict[str, float]
    diagnostics: Dict[str, Any]


@dataclass
class ReconstructionValidationResult:
    """Final topology validation report."""

    metrics: Dict[str, Any]
    selected_edges: pd.DataFrame
    node_support_classes: pd.DataFrame
    warnings: List[str]


@dataclass
class DensityFieldResult:
    """Density field used only for optional geometric routing."""

    density: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class GeometricRoutingResult:
    """Fixed-adjacency graph with density-routed edge polylines."""

    graph: nx.Graph
    route_statistics: pd.DataFrame
    diagnostics: Dict[str, Any]
    density_field: DensityFieldResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionConfig:
    """Configuration for transported separatrix-connectivity reconstruction."""

    run_name: str = "latest"
    reconstruction_name: str = "transported_separatrix_connectivity"
    bundle_dir: Optional[PathLike] = None
    random_seed: int = 20260808

    separatrix_id_attribute: str = "separatrixID"
    separatrix_id_delimiter: str = ","
    missing_separatrix_id_policy: MissingSeparatrixPolicy = "error"
    strict_separatrix_validation: bool = True
    allow_degree_based_chain_fallback: bool = False
    strict_validation: bool = True

    connectivity_affinity_mode: AffinityMode = "separatrix_hop_decay"
    hop_decay_q: float = 0.1
    hop_decay_q_values: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25]
    )
    max_soft_hops: int = 4
    max_soft_hops_values: List[int] = field(default_factory=lambda: [2, 3, 4])

    enable_length_normalized_affinity: bool = True
    length_reference_spacing_mode: LengthReferenceMode = "input_median"
    maximum_effective_soft_distance: float = 4.0
    sampling_irregularity_warning_threshold: float = 0.35
    geometric_length_attribute: Optional[str] = None

    direct_evidence_weight: float = 1.0
    soft_extension_evidence_weight: float = 0.0
    uncertainty_penalty_weight: float = 0.0

    top_k_direct_evidence: int = 3
    top_k_separatrix_evidence: int = 3
    mutual_top_k: int = 3
    minimum_pair_coverage: float = 0.05
    minimum_direct_adjacency_evidence: float = 0.0
    minimum_separatrix_connectivity_evidence: float = 0.0

    type_constraint_mode: ConstraintMode = "soft"
    type_confidence_threshold: float = 0.8
    scalar_order_constraint_mode: ConstraintMode = "soft"
    planarity_mode: ConstraintMode = "soft"
    force_planarity_in_3d: bool = False
    target_component_count: Optional[int] = None

    soft_redundancy_tolerance: float = 0.05
    maximum_local_repair_iterations: int = 50
    objective_edge_penalty_weight: float = 1e-5
    objective_component_penalty_weight: float = 0.05
    objective_type_penalty_weight: float = 0.05
    objective_planarity_penalty_weight: float = 0.05
    objective_redundancy_penalty_weight: float = 0.02

    enable_density_routing: bool = False
    density_grid_resolution_candidates: List[int] = field(
        default_factory=lambda: [128, 192, 256]
    )
    density_bandwidth_candidates: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.04]
    )
    density_sample_spacing_fraction: float = 0.005
    density_padding_fraction: float = 0.05

    report_barycenter_relation_consistency: bool = True
    use_barycenter_relation_for_edge_selection: bool = False
    write_legacy_name_aliases: bool = False

    max_dense_edges_to_plot: int = 10_000
    max_routed_edges_to_plot: int = 2_000

    def __post_init__(self) -> None:
        self._validate_name(self.run_name, "run_name")
        self._validate_name(self.reconstruction_name, "reconstruction_name")
        if self.allow_degree_based_chain_fallback:
            raise ValueError(
                "allow_degree_based_chain_fallback must remain False; degree-based "
                "chain inference is disabled."
            )
        if self.use_barycenter_relation_for_edge_selection:
            raise ValueError(
                "use_barycenter_relation_for_edge_selection must be False; "
                "C_center may only be used post-hoc."
            )
        if not 0.0 <= float(self.hop_decay_q) <= 1.0:
            raise ValueError(f"hop_decay_q must lie in [0, 1]; got {self.hop_decay_q!r}.")
        if float(self.hop_decay_q) == 1.0:
            warnings.warn(
                "hop_decay_q=1 removes decay with hop distance and is intended "
                "only as a diagnostic sensitivity value.",
                UserWarning,
                stacklevel=2,
            )
        for value in self.hop_decay_q_values:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"hop_decay_q_values must lie in [0, 1]; got {value!r}."
                )
            if float(value) == 1.0:
                warnings.warn(
                    "hop_decay_q_values includes 1.0 (diagnostic only; no hop decay).",
                    UserWarning,
                    stacklevel=2,
                )
        if self.max_soft_hops < 1:
            raise ValueError("max_soft_hops must be >= 1.")
        if any(value < 1 for value in self.max_soft_hops_values):
            raise ValueError("max_soft_hops_values must contain integers >= 1.")
        if self.connectivity_affinity_mode not in {
            "separatrix_hop_decay",
            "separatrix_length_normalized_decay",
        }:
            raise ValueError("Unsupported connectivity_affinity_mode.")
        if self.length_reference_spacing_mode not in {
            "input_median",
            "separatrix_median",
            "global_input_family_median",
        }:
            raise ValueError("Unsupported length_reference_spacing_mode.")
        if self.missing_separatrix_id_policy not in {"error", "warn", "skip"}:
            raise ValueError("Unsupported missing_separatrix_id_policy.")
        for mode_name, mode_value in (
            ("type_constraint_mode", self.type_constraint_mode),
            ("scalar_order_constraint_mode", self.scalar_order_constraint_mode),
            ("planarity_mode", self.planarity_mode),
        ):
            if mode_value not in {"off", "soft", "hard"}:
                raise ValueError(f"{mode_name} must be off/soft/hard.")
        self._validate_unit_interval(self.minimum_pair_coverage, "minimum_pair_coverage")
        self._validate_unit_interval(
            self.type_confidence_threshold, "type_confidence_threshold"
        )
        self._validate_unit_interval(
            self.sampling_irregularity_warning_threshold,
            "sampling_irregularity_warning_threshold",
        )
        if self.top_k_direct_evidence < 1 or self.top_k_separatrix_evidence < 1:
            raise ValueError("top-k evidence parameters must be positive.")
        if self.mutual_top_k < 1:
            raise ValueError("mutual_top_k must be positive.")
        if self.maximum_local_repair_iterations < 0:
            raise ValueError("maximum_local_repair_iterations must be nonnegative.")
        if self.soft_redundancy_tolerance < 0:
            raise ValueError("soft_redundancy_tolerance must be nonnegative.")
        if self.maximum_effective_soft_distance < 1:
            raise ValueError("maximum_effective_soft_distance must be >= 1.")
        if self.target_component_count is not None and self.target_component_count < 1:
            raise ValueError("target_component_count must be None or a positive integer.")
        nonnegative = {
            "direct_evidence_weight": self.direct_evidence_weight,
            "soft_extension_evidence_weight": self.soft_extension_evidence_weight,
            "uncertainty_penalty_weight": self.uncertainty_penalty_weight,
            "minimum_direct_adjacency_evidence": self.minimum_direct_adjacency_evidence,
            "minimum_separatrix_connectivity_evidence": (
                self.minimum_separatrix_connectivity_evidence
            ),
        }
        if any(value < 0 for value in nonnegative.values()):
            raise ValueError("Evidence weights and minima must be nonnegative.")

    @staticmethod
    def _validate_name(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError(
                f"{field_name} must be a single directory name, not a path: {value!r}."
            )

    @staticmethod
    def _validate_unit_interval(value: float, field_name: str) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field_name} must lie in [0, 1]; got {value!r}.")

    def resolve_bundle_dir(self, project_root: PathLike = ".") -> Path:
        root = Path(project_root).expanduser().resolve()
        selected = (
            Path(self.bundle_dir).expanduser()
            if self.bundle_dir is not None
            else Path("results") / "barycenters" / self.run_name
        )
        if not selected.is_absolute():
            selected = root / selected
        return selected.resolve()

    def resolve_output_dirs(
        self, project_root: PathLike = ".", *, create: bool = False
    ) -> Tuple[Path, Path]:
        bundle = self.resolve_bundle_dir(project_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = bundle / "reconstruction" / f"{self.reconstruction_name}_{stamp}"
        figures_dir = output_dir / "figures"
        if create:
            figures_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, figures_dir

    def resolved_dict(self, project_root: PathLike = ".") -> Dict[str, Any]:
        values: Dict[str, Any] = asdict(self)
        bundle = self.resolve_bundle_dir(project_root)
        values["bundle_dir"] = str(bundle)
        values["terminology_version"] = TERMINOLOGY_VERSION
        return values


# ---------------------------------------------------------------------------
# Separatrix ID parsing and membership
# ---------------------------------------------------------------------------


def parse_separatrix_ids(
    value: Any,
    *,
    delimiter: str = ",",
    strict: bool = True,
) -> frozenset:
    """Parse an edge ``separatrixID`` value into a frozenset of string IDs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if strict:
            raise ValueError("Missing separatrixID value.")
        return frozenset()
    if isinstance(value, (int, np.integer)):
        return frozenset({str(int(value))})
    if isinstance(value, str):
        if value.strip() == "":
            if strict:
                raise ValueError("Empty separatrixID string.")
            return frozenset()
        tokens = value.split(delimiter)
        cleaned: List[str] = []
        for token in tokens:
            stripped = token.strip()
            if stripped == "":
                if strict:
                    raise ValueError(
                        f"Malformed separatrixID with empty token: {value!r}."
                    )
                continue
            cleaned.append(stripped)
        if not cleaned:
            if strict:
                raise ValueError(f"Malformed/empty separatrixID: {value!r}.")
            return frozenset()
        return frozenset(cleaned)
    if isinstance(value, (set, frozenset, list, tuple)):
        ids: Set[str] = set()
        for item in value:
            if isinstance(item, (int, np.integer)):
                ids.add(str(int(item)))
            elif isinstance(item, str) and item.strip():
                ids.add(item.strip())
            elif strict:
                raise ValueError(f"Malformed separatrixID iterable item: {item!r}.")
        if not ids and strict:
            raise ValueError("Empty separatrixID iterable.")
        return frozenset(ids)
    if strict:
        raise ValueError(f"Unsupported separatrixID type: {type(value)!r}.")
    return frozenset({str(value)})


def format_separatrix_ids_for_display(ids: Iterable[str]) -> str:
    """Numeric-aware sorted display string for a set of separatrix IDs."""

    def sort_key(item: str):
        try:
            return (0, int(item))
        except ValueError:
            return (1, item)

    return ",".join(sorted(ids, key=sort_key))


def _canonical_edge_key(graph: nx.Graph, u: Any, v: Any, key: Any = None) -> Any:
    if graph.is_multigraph():
        return (u, v, key) if u <= v else (v, u, key)
    return (u, v) if u <= v else (v, u)


def build_separatrix_membership(
    graph: nx.Graph,
    *,
    attribute: str = "separatrixID",
    delimiter: str = ",",
    strict: bool = True,
    missing_policy: MissingSeparatrixPolicy = "error",
) -> SeparatrixMembership:
    """Build edge/node membership maps and one subgraph per separatrix ID."""
    edge_ids: Dict[Any, frozenset] = {}
    node_ids: Dict[Any, Set[str]] = {node: set() for node in graph.nodes()}
    separatrix_to_edges: Dict[str, List[Any]] = defaultdict(list)
    warnings_list: List[str] = []
    missing = 0
    malformed = 0

    edge_iter = (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else ((u, v, None, d) for u, v, d in graph.edges(data=True))
    )
    for u, v, key, data in edge_iter:
        raw = data.get(attribute, None)
        try:
            ids = parse_separatrix_ids(raw, delimiter=delimiter, strict=True)
        except ValueError as exc:
            malformed += 1
            if missing_policy == "error" or strict:
                raise ValueError(
                    f"Invalid {attribute} on edge {(u, v)}: {raw!r} ({exc})"
                ) from exc
            warnings_list.append(f"Skipping malformed {attribute} on {(u, v)}: {raw!r}")
            continue
        if not ids:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Missing {attribute} on edge {(u, v)}.")
            if missing_policy == "warn":
                warnings_list.append(f"Missing {attribute} on edge {(u, v)}.")
            continue
        edge_key = _canonical_edge_key(graph, u, v, key)
        edge_ids[edge_key] = ids
        for sid in ids:
            separatrix_to_edges[sid].append(edge_key)
            node_ids[u].add(sid)
            node_ids[v].add(sid)

    subgraphs: Dict[str, nx.Graph] = {}
    for sid, edges in separatrix_to_edges.items():
        H = nx.Graph()
        for edge_key in edges:
            if graph.is_multigraph():
                u, v, _key = edge_key
            else:
                u, v = edge_key
            H.add_edge(u, v)
        components = list(nx.connected_components(H))
        if len(components) > 1:
            message = (
                f"Separatrix ID {sid!r} induces {len(components)} connected components."
            )
            if strict:
                raise ValueError(message)
            warnings_list.append(message)
        subgraphs[sid] = H

    return SeparatrixMembership(
        edge_separatrix_ids=edge_ids,
        node_incident_separatrix_ids={
            node: frozenset(ids) for node, ids in node_ids.items()
        },
        separatrix_to_edges=dict(separatrix_to_edges),
        subgraphs=subgraphs,
        warnings=warnings_list,
    )


def shared_separatrix_ids(
    membership: SeparatrixMembership, u: Any, v: Any
) -> frozenset:
    """Return ``R(u,v) = Sigma(u) intersect Sigma(v)``."""
    return membership.node_incident_separatrix_ids.get(
        u, frozenset()
    ) & membership.node_incident_separatrix_ids.get(v, frozenset())


# ---------------------------------------------------------------------------
# Bundle validation and node decoding
# ---------------------------------------------------------------------------


def _matrix_row_count(value: Any, name: str, errors: List[str]) -> Optional[int]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        errors.append(f"{name} must be square; got shape {array.shape}.")
        return None
    return int(array.shape[0])


def _resolve_input_node_orders(
    graphs: Any,
    nodes_list: Any,
    input_sizes: List[Optional[int]],
    warnings_list: List[str],
) -> Optional[List[List[Any]]]:
    n_inputs = len(input_sizes)
    if graphs is None:
        warnings_list.append("Input graphs are unavailable.")
        return None
    if len(graphs) != n_inputs:
        warnings_list.append(
            f"Input graph count {len(graphs)} does not match {n_inputs} inputs."
        )
        return None
    if nodes_list is not None and len(nodes_list) != n_inputs:
        warnings_list.append(
            f"nodes_list length {len(nodes_list)} does not match {n_inputs} inputs."
        )
        return None

    resolved: List[List[Any]] = []
    for index, (graph, n_s) in enumerate(zip(graphs, input_sizes)):
        if n_s is None:
            return None
        graph_nodes = list(graph.nodes())
        if graph.number_of_nodes() != n_s:
            warnings_list.append(
                f"Input graph {index} has {graph.number_of_nodes()} nodes, "
                f"but C_{index} has {n_s} rows."
            )
            return None
        if nodes_list is None:
            expected = list(range(n_s))
            if set(graph_nodes) != set(expected):
                warnings_list.append(
                    f"Input graph {index} lacks trustworthy nodes_list ordering."
                )
                return None
            resolved.append(expected)
            continue
        order = list(nodes_list[index])
        if len(order) != n_s or len(set(order)) != n_s or set(order) != set(graph_nodes):
            warnings_list.append(
                f"nodes_list[{index}] is not a unique exact ordering of graph {index}."
            )
            return None
        resolved.append(order)
    return resolved


def validate_barycenter_bundle(
    bundle: Mapping[str, Any], *, strict_validation: bool = True
) -> BundleValidationResult:
    """Validate a loaded PFGW bundle for separatrix-connectivity reconstruction.

    Couplings, input graphs, and trustworthy node ordering are required. There is
    no silent ``C_center``-based fallback.
    """
    fatal: List[str] = []
    support_errors: List[str] = []
    C_center = np.asarray(bundle.get("C_center"))
    X_center = np.asarray(bundle.get("X_center"))
    p = np.asarray(bundle.get("p"))
    n_center = _matrix_row_count(C_center, "C_center", fatal)
    n_center_safe = 0 if n_center is None else n_center
    if X_center.ndim != 2 or X_center.shape[0] != n_center_safe:
        fatal.append(
            f"X_center must have {n_center_safe} rows; got shape {X_center.shape}."
        )
    if p.ndim != 1 or p.shape[0] != n_center_safe:
        fatal.append(f"p must have shape ({n_center_safe},); got {p.shape}.")
    elif np.any(~np.isfinite(p)) or np.any(p < 0):
        fatal.append("p must contain only finite nonnegative values.")
    if fatal:
        raise ValueError("Invalid barycenter bundle:\n- " + "\n- ".join(fatal))

    Cs = list(bundle.get("Cs") or [])
    Ys = list(bundle.get("Ys") or [])
    ps = list(bundle.get("ps") or [])
    n_inputs = len(Cs)
    if len(Ys) != n_inputs or len(ps) != n_inputs:
        support_errors.append(
            f"Cs/Ys/ps lengths must match; got {n_inputs}/{len(Ys)}/{len(ps)}."
        )
    lambdas = np.asarray(bundle.get("lambdas"))
    if lambdas.ndim != 1 or lambdas.shape[0] != n_inputs:
        support_errors.append(
            f"lambdas must have shape ({n_inputs},); got {lambdas.shape}."
        )

    input_sizes: List[Optional[int]] = []
    for index, C_s in enumerate(Cs):
        n_s = _matrix_row_count(C_s, f"C_{index}", support_errors)
        input_sizes.append(n_s)
        if n_s is None or index >= len(Ys) or index >= len(ps):
            continue
        Y_s, p_s = np.asarray(Ys[index]), np.asarray(ps[index])
        if Y_s.ndim != 2 or Y_s.shape[0] != n_s:
            support_errors.append(
                f"Y_{index} must have {n_s} rows; got shape {Y_s.shape}."
            )
        if p_s.ndim != 1 or p_s.shape[0] != n_s:
            support_errors.append(
                f"p_{index} must have shape ({n_s},); got {p_s.shape}."
            )

    Ts = bundle.get("Ts")
    has_couplings = Ts is not None
    if Ts is None:
        support_errors.append(
            "Couplings Ts are required; cannot fall back to C_center edge selection."
        )
    elif len(Ts) != n_inputs:
        support_errors.append(
            f"Coupling count {len(Ts)} does not match {n_inputs} inputs."
        )
    else:
        for index, (T_s, n_s) in enumerate(zip(Ts, input_sizes)):
            expected = (n_center_safe, n_s)
            if n_s is not None and np.asarray(T_s).shape != expected:
                support_errors.append(
                    f"T_{index} must have shape {expected}; got {np.asarray(T_s).shape}."
                )

    warnings_list: List[str] = []
    graphs = bundle.get("Gs")
    nodes_list = bundle.get("nodes_list")
    if graphs is None:
        support_errors.append(
            "Input graphs Gs are required; cannot fall back to C_center edge selection."
        )
    node_orders = _resolve_input_node_orders(graphs, nodes_list, input_sizes, warnings_list)
    if node_orders is None:
        support_errors.append(
            "Trustworthy input node ordering is unavailable; refusing C_center fallback."
        )

    if support_errors:
        message = (
            "Cannot reconstruct from transported connectivity:\n- "
            + "\n- ".join(support_errors + warnings_list)
        )
        if strict_validation or support_errors:
            raise ValueError(message)
        warnings_list.extend(support_errors)

    assert node_orders is not None
    meta = dict(bundle.get("meta") or {})
    requested = meta.get("N_center")
    held_out = bool(bundle.get("exclude_seed", False) and bundle.get("seed") is not None)
    return BundleValidationResult(
        warnings=warnings_list,
        input_node_orders=node_orders,
        n_center=n_center_safe,
        feature_dim=int(X_center.shape[1]),
        n_inputs=n_inputs,
        requested_support_size=None if requested is None else int(requested),
        actual_support_size=n_center_safe,
        has_couplings=has_couplings,
        has_input_graphs=graphs is not None,
        has_node_order_metadata=nodes_list is not None,
        has_held_out_seed=held_out,
        exclude_seed=bool(bundle.get("exclude_seed", False)),
    )


def _normalized_type_scores(values: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(values, dtype=float), 0.0, None)
    totals = scores.sum(axis=1, keepdims=True)
    return np.divide(
        scores,
        totals,
        out=np.full_like(scores, np.nan),
        where=totals > np.finfo(float).eps,
    )


def decode_barycenter_nodes(
    bundle: Mapping[str, Any],
    config: Optional[ReconstructionConfig] = None,
    *,
    comparison_tolerance: float = 1e-8,
) -> DecodedNodesResult:
    """Decode every barycenter support point and cross-check ``G_complete``."""
    from lib.msc_barycenter_threshold import _decode_barycenter_nodes

    X = np.asarray(bundle.get("X_center"), dtype=float)
    p = np.asarray(bundle.get("p"), dtype=float)
    meta = dict(bundle.get("meta") or {})
    if X.ndim != 2 or p.ndim != 1 or X.shape[0] != p.shape[0]:
        raise ValueError("X_center must align row-wise with one-dimensional p.")
    n_center, feature_dim = X.shape
    type_order = list(meta.get("MSC_TYPE_ORDER") or [])
    scalar_meta = all(key in meta for key in ("min_scalar", "scalar_range"))
    type_without_scalar = bool(type_order) and feature_dim == 3 + len(type_order)
    scalar_in_features = bool(scalar_meta and feature_dim >= 4 and not type_without_scalar)
    type_start = 4 if scalar_in_features else 3
    type_in_features = bool(type_order and feature_dim >= type_start + len(type_order))
    coordinate_meta = all(key in meta for key in ("min_x", "min_y", "min_z", "domain_size"))

    decoded_graph = nx.Graph()
    if feature_dim >= 3 and coordinate_meta:
        _decode_barycenter_nodes(
            decoded_graph,
            X,
            min_x=float(meta["min_x"]),
            min_y=float(meta["min_y"]),
            min_z=float(meta["min_z"]),
            domain_size=float(meta["domain_size"]),
            scalar_min=float(meta["min_scalar"]) if scalar_in_features else None,
            scalar_range=float(meta["scalar_range"]) if scalar_in_features else None,
            msc_type_order=type_order if type_in_features else None,
        )

    G_complete = bundle.get("G_complete")
    graph_available = isinstance(G_complete, nx.Graph)
    normalized_types = (
        _normalized_type_scores(X[:, type_start : type_start + len(type_order)])
        if type_in_features
        else None
    )
    records: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    warnings_list: List[str] = []

    for node_id in range(n_center):
        feature_attrs = decoded_graph.nodes[node_id] if node_id in decoded_graph else {}
        graph_attrs = (
            G_complete.nodes[node_id]
            if graph_available and node_id in G_complete
            else {}
        )
        coordinates = []
        coordinate_source = "unavailable"
        for name in ("x", "y", "z"):
            if name in feature_attrs:
                coordinates.append(float(feature_attrs[name]))
                coordinate_source = "features"
            elif name in graph_attrs:
                coordinates.append(float(graph_attrs[name]))
                coordinate_source = "G_complete"
            else:
                coordinates.append(np.nan)

        scalar_normalized = float(X[node_id, 3]) if scalar_in_features else np.nan
        scalar_value = feature_attrs.get("scalar_value", np.nan)
        scalar_source = "features" if np.isfinite(scalar_value) else "unavailable"
        if not np.isfinite(scalar_value):
            graph_scalar = graph_attrs.get(
                "scalar_value", graph_attrs.get("height", np.nan)
            )
            if np.isfinite(graph_scalar):
                scalar_value = float(graph_scalar)
                scalar_source = "G_complete"
                if scalar_meta and float(meta["scalar_range"]) != 0:
                    scalar_normalized = (
                        scalar_value - float(meta["min_scalar"])
                    ) / float(meta["scalar_range"])

        morse_type: Any = np.nan
        type_confidence = np.nan
        type_source = "unavailable"
        if normalized_types is not None and np.any(np.isfinite(normalized_types[node_id])):
            row = normalized_types[node_id]
            best = int(np.nanargmax(row))
            sorted_scores = np.sort(row[np.isfinite(row)])
            second = sorted_scores[-2] if sorted_scores.size > 1 else 0.0
            morse_type = type_order[best]
            type_confidence = float(row[best] - second)
            type_source = "features"
        elif "type" in graph_attrs:
            morse_type = graph_attrs["type"]
            type_confidence = 1.0
            type_source = "G_complete"

        records.append(
            {
                "node_id": node_id,
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
                "scalar_normalized": scalar_normalized,
                "scalar_value": scalar_value,
                "morse_type": morse_type,
                "type_confidence": type_confidence,
                "mass": float(p[node_id]),
                "coordinate_source": coordinate_source,
                "scalar_source": scalar_source,
                "type_source": type_source,
            }
        )
        if graph_attrs:
            for attribute in ("x", "y", "z", "scalar_value", "type"):
                decoded_name = "morse_type" if attribute == "type" else attribute
                decoded_value = records[-1].get(decoded_name, np.nan)
                graph_value = graph_attrs.get(
                    attribute,
                    graph_attrs.get("height", np.nan)
                    if attribute == "scalar_value"
                    else np.nan,
                )
                if pd.isna(decoded_value) or pd.isna(graph_value):
                    continue
                different = (
                    decoded_value != graph_value
                    if attribute == "type"
                    else not np.isclose(
                        float(decoded_value),
                        float(graph_value),
                        rtol=comparison_tolerance,
                        atol=comparison_tolerance,
                    )
                )
                if different:
                    disagreements.append(
                        {
                            "node_id": node_id,
                            "attribute": attribute,
                            "decoded_value": decoded_value,
                            "G_complete_value": graph_value,
                        }
                    )

    dataframe = pd.DataFrame.from_records(records)
    disagreement_frame = pd.DataFrame.from_records(
        disagreements,
        columns=["node_id", "attribute", "decoded_value", "G_complete_value"],
    )
    has_coordinates = bool(dataframe[["x", "y", "z"]].notna().to_numpy().all())
    has_scalar = bool(dataframe["scalar_value"].notna().all())
    has_types = bool(dataframe["morse_type"].notna().all())
    requested_type_mode = config.type_constraint_mode if config else "off"
    requested_scalar_mode = (
        config.scalar_order_constraint_mode if config else "off"
    )
    requested_planarity_mode = config.planarity_mode if config else "off"
    effective_type_mode: ConstraintMode = requested_type_mode if has_types else "off"
    effective_scalar_mode: ConstraintMode = (
        requested_scalar_mode if has_types and has_scalar else "off"
    )
    effective_planarity_mode: ConstraintMode = (
        requested_planarity_mode
        if dataframe[["x", "y"]].notna().to_numpy().all()
        else "off"
    )
    return DecodedNodesResult(
        dataframe=dataframe,
        disagreements=disagreement_frame,
        warnings=warnings_list,
        has_coordinates=has_coordinates,
        has_scalar=has_scalar,
        has_types=has_types,
        effective_type_constraint_mode=effective_type_mode,
        effective_scalar_order_constraint_mode=effective_scalar_mode,
        effective_planarity_mode=effective_planarity_mode,
    )


# ---------------------------------------------------------------------------
# Direct adjacency and separatrix-local affinity
# ---------------------------------------------------------------------------


def build_direct_adjacency_matrix(
    graph: nx.Graph, node_order: Sequence[Any]
) -> np.ndarray:
    """Symmetric binary adjacency in coupling-aligned node order."""
    order = list(node_order)
    index = {node: position for position, node in enumerate(order)}
    if len(index) != len(order):
        raise ValueError("node_order must contain unique node ids.")
    if set(graph.nodes()) != set(order):
        raise ValueError("node_order must contain exactly every graph node.")
    adjacency = np.zeros((len(order), len(order)), dtype=float)
    for u, v in graph.edges():
        if u == v:
            continue
        i, j = index[u], index[v]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _affinity_from_distance(distance: float, q: float, max_hops: float) -> float:
    if not np.isfinite(distance) or distance < 1 or distance > max_hops:
        return 0.0
    if q == 0.0:
        return 1.0 if distance == 1.0 else 0.0
    return float(q ** (distance - 1.0))


def _node_xyz(graph: nx.Graph, node: Any) -> np.ndarray:
    data = graph.nodes[node]
    return np.asarray([data["x"], data["y"], data.get("z", 0.0)], dtype=float)


def _edge_geometric_length(
    graph: nx.Graph,
    u: Any,
    v: Any,
    *,
    length_attribute: Optional[str] = None,
) -> float:
    data = graph.get_edge_data(u, v)
    if isinstance(data, dict) and graph.is_multigraph():
        # Multi-edge dict keyed by edge key; use the first edge payload.
        data = next(iter(data.values())) if data else {}
    if length_attribute and isinstance(data, dict) and length_attribute in data:
        return float(data[length_attribute])
    return float(np.linalg.norm(_node_xyz(graph, u) - _node_xyz(graph, v)))


def _unique_edge_lengths(
    graph: nx.Graph, *, length_attribute: Optional[str] = None
) -> np.ndarray:
    lengths = []
    seen = set()
    for u, v in graph.edges():
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        lengths.append(_edge_geometric_length(graph, u, v, length_attribute=length_attribute))
    return np.asarray(lengths, dtype=float)


def build_separatrix_local_connectivity_affinity(
    graph: nx.Graph,
    node_order: Sequence[Any],
    membership: SeparatrixMembership,
    *,
    q: float,
    max_soft_hops: int,
    mode: AffinityMode = "separatrix_hop_decay",
    length_reference_spacing_mode: LengthReferenceMode = "input_median",
    length_attribute: Optional[str] = None,
    maximum_effective_soft_distance: float = 4.0,
    global_reference_spacing: Optional[float] = None,
) -> InputAffinityResult:
    """Build soft separatrix-local connectivity affinity ``K``.

    Soft paths remain inside one fixed separatrix-ID subgraph. Pairwise shared
    membership is never transitively closed.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1].")
    if q == 1.0:
        warnings.warn(
            "q=1 assigns affinity one to all pairs within the hop cutoff.",
            UserWarning,
            stacklevel=2,
        )
    order = list(node_order)
    index = {node: i for i, node in enumerate(order)}
    n = len(order)
    affinity = np.zeros((n, n), dtype=float)
    best_ids: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    hop_distances: Dict[Tuple[int, int], int] = {}
    warnings_list: List[str] = []

    reference_spacing = None
    if mode == "separatrix_length_normalized_decay":
        lengths = _unique_edge_lengths(graph, length_attribute=length_attribute)
        if lengths.size == 0:
            raise ValueError("Cannot compute length-normalized affinity without edges.")
        if length_reference_spacing_mode == "input_median":
            reference_spacing = float(np.median(lengths))
        elif length_reference_spacing_mode == "global_input_family_median":
            if global_reference_spacing is None:
                raise ValueError("global_reference_spacing is required.")
            reference_spacing = float(global_reference_spacing)
        else:
            reference_spacing = float(np.median(lengths))
        if reference_spacing <= 0:
            raise ValueError("Reference spacing must be positive.")

    for sid, subgraph in membership.subgraphs.items():
        if mode == "separatrix_length_normalized_decay":
            assert reference_spacing is not None
            if length_reference_spacing_mode == "separatrix_median":
                local_lengths = _unique_edge_lengths(
                    subgraph, length_attribute=length_attribute
                )
                local_delta = (
                    float(np.median(local_lengths))
                    if local_lengths.size
                    else reference_spacing
                )
            else:
                local_delta = reference_spacing
            # Weighted geometric shortest paths within this ID only.
            weighted = nx.Graph()
            for u, v in subgraph.edges():
                length = _edge_geometric_length(
                    graph, u, v, length_attribute=length_attribute
                )
                weighted.add_edge(u, v, weight=max(length, EPS))
            for source in list(weighted.nodes()):
                if source not in index:
                    continue
                lengths_map = nx.single_source_dijkstra_path_length(
                    weighted, source, weight="weight"
                )
                i = index[source]
                for target, geo_length in lengths_map.items():
                    if target == source or target not in index:
                        continue
                    j = index[target]
                    if i == j:
                        continue
                    # Direct neighbors get affinity 1.
                    if subgraph.has_edge(source, target):
                        effective = 1.0
                        hop_for_diag = 1
                    else:
                        effective = max(1.0, float(geo_length) / local_delta)
                        if effective > maximum_effective_soft_distance:
                            continue
                        hop_for_diag = int(np.ceil(effective))
                    value = _affinity_from_distance(
                        effective, q, maximum_effective_soft_distance
                    )
                    if value <= 0:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    if value > affinity[i, j] + 1e-15:
                        affinity[i, j] = affinity[j, i] = value
                        best_ids[(a, b)] = {sid}
                        hop_distances[(a, b)] = hop_for_diag
                    elif np.isclose(value, affinity[i, j]):
                        best_ids[(a, b)].add(sid)
            continue

        # Hop-decay mode: BFS with cutoff inside this separatrix subgraph only.
        ordered_nodes = [node for node in subgraph.nodes() if node in index]
        ordered_nodes.sort(key=lambda node: index[node])
        for source in ordered_nodes:
            i = index[source]
            distances = nx.single_source_shortest_path_length(
                subgraph, source, cutoff=max_soft_hops
            )
            for target, hop in distances.items():
                if hop < 1 or target == source or target not in index:
                    continue
                j = index[target]
                if j <= i:
                    continue
                value = _affinity_from_distance(float(hop), q, float(max_soft_hops))
                if value <= 0:
                    continue
                if value > affinity[i, j] + 1e-15:
                    affinity[i, j] = affinity[j, i] = value
                    best_ids[(i, j)] = {sid}
                    hop_distances[(i, j)] = int(hop)
                elif np.isclose(value, affinity[i, j]):
                    best_ids[(i, j)].add(sid)

    np.fill_diagonal(affinity, 0.0)
    if not np.allclose(affinity, affinity.T):
        raise AssertionError("Affinity matrix is unexpectedly asymmetric.")
    if np.any(affinity < -1e-12) or np.any(affinity > 1.0 + 1e-12):
        raise AssertionError("Affinity values must lie in [0, 1].")

    direct = build_direct_adjacency_matrix(graph, order)
    if mode == "separatrix_hop_decay" and max_soft_hops >= 1 and np.any(direct > 0):
        if np.any(affinity + 1e-12 < direct):
            # Incomplete annotations can violate K >= A; report rather than invent IDs.
            warnings_list.append(
                "Affinity is elementwise below direct adjacency for some pairs; "
                "annotations may be incomplete."
            )

    return InputAffinityResult(
        direct_adjacency_matrix=direct,
        separatrix_local_connectivity_affinity=affinity,
        best_shared_separatrix_ids={
            key: frozenset(values) for key, values in best_ids.items()
        },
        hop_distances=hop_distances,
        reference_spacing=reference_spacing,
        warnings=warnings_list,
        diagnostics={
            "q": float(q),
            "max_soft_hops": int(max_soft_hops),
            "mode": mode,
            "n_positive_affinity_pairs": int(np.count_nonzero(np.triu(affinity, 1))),
            "n_direct_pairs": int(np.count_nonzero(np.triu(direct, 1))),
        },
    )


# ---------------------------------------------------------------------------
# Annotation / sampling diagnostics
# ---------------------------------------------------------------------------


def _cycle_rank(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    components = nx.number_connected_components(graph)
    return int(graph.number_of_edges() - graph.number_of_nodes() + components)


def diagnose_separatrix_annotations(
    graph: nx.Graph,
    input_index: int,
    membership: SeparatrixMembership,
    *,
    length_attribute: Optional[str] = None,
    sampling_irregularity_warning_threshold: float = 0.35,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return compact annotation, subgraph, membership, and sampling diagnostics."""
    lengths = _unique_edge_lengths(graph, length_attribute=length_attribute)
    median_length = float(np.median(lengths)) if lengths.size else np.nan
    mean_length = float(np.mean(lengths)) if lengths.size else np.nan
    std_length = float(np.std(lengths)) if lengths.size else np.nan
    cv = (
        float(std_length / mean_length)
        if lengths.size and mean_length > EPS
        else np.nan
    )
    short_fraction = (
        float(np.mean(lengths < 0.5 * median_length))
        if lengths.size and np.isfinite(median_length)
        else np.nan
    )

    multi_id_counts = [len(ids) for ids in membership.edge_separatrix_ids.values()]
    nodes_without_id = sum(
        1 for ids in membership.node_incident_separatrix_ids.values() if not ids
    )
    disconnected = 0
    non_path_like = 0
    subgraph_rows = []
    for sid, subgraph in membership.subgraphs.items():
        degrees = [d for _, d in subgraph.degree()]
        max_degree = int(max(degrees)) if degrees else 0
        degree_one = int(sum(d == 1 for d in degrees))
        n_components = int(nx.number_connected_components(subgraph))
        if n_components > 1:
            disconnected += 1
        cycle = _cycle_rank(subgraph)
        if max_degree > 2 or cycle > 0 or n_components != 1:
            non_path_like += 1
        shared = 0
        for edge_key in membership.separatrix_to_edges[sid]:
            if len(membership.edge_separatrix_ids[edge_key]) > 1:
                shared += 1
        total_length = 0.0
        local_lengths = []
        for u, v in subgraph.edges():
            length = _edge_geometric_length(
                graph, u, v, length_attribute=length_attribute
            )
            total_length += length
            local_lengths.append(length)
        subgraph_rows.append(
            {
                "input_index": input_index,
                "separatrix_id": sid,
                "node_count": subgraph.number_of_nodes(),
                "edge_count": subgraph.number_of_edges(),
                "connected_component_count": n_components,
                "maximum_degree": max_degree,
                "degree_one_node_count": degree_one,
                "cycle_rank": cycle,
                "total_geometric_length": total_length,
                "median_sampled_edge_length": (
                    float(np.median(local_lengths)) if local_lengths else np.nan
                ),
                "shared_edge_count": shared,
                "shared_edge_fraction": (
                    shared / subgraph.number_of_edges()
                    if subgraph.number_of_edges()
                    else 0.0
                ),
            }
        )

    annotation = {
        "input_index": input_index,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "unique_separatrix_id_count": len(membership.subgraphs),
        "missing_separatrix_id_count": nodes_without_id,  # nodes without incident IDs tracked below
        "malformed_value_count": 0,
        "nodes_with_no_incident_id": nodes_without_id,
        "multi_id_edge_fraction": (
            float(np.mean([c > 1 for c in multi_id_counts])) if multi_id_counts else 0.0
        ),
        "maximum_ids_on_one_edge": int(max(multi_id_counts)) if multi_id_counts else 0,
        "disconnected_per_id_subgraph_count": disconnected,
        "non_path_like_per_id_subgraph_count": non_path_like,
        "sampled_edge_length_median": median_length,
        "sampled_edge_length_cv": cv,
        "fraction_edges_substantially_shorter_than_median": short_fraction,
    }
    if (
        np.isfinite(cv)
        and cv >= sampling_irregularity_warning_threshold
    ):
        membership.warnings.append(
            f"Input {input_index}: sampled-edge-length CV={cv:.3f} exceeds "
            f"threshold {sampling_irregularity_warning_threshold}."
        )

    node_rows = [
        {
            "input_index": input_index,
            "node": node,
            "incident_separatrix_ids": format_separatrix_ids_for_display(ids),
            "incident_separatrix_id_count": len(ids),
        }
        for node, ids in membership.node_incident_separatrix_ids.items()
    ]
    edge_rows = [
        {
            "input_index": input_index,
            "edge": str(edge_key),
            "separatrix_ids": format_separatrix_ids_for_display(ids),
            "separatrix_id_count": len(ids),
        }
        for edge_key, ids in membership.edge_separatrix_ids.items()
    ]
    sampling = {
        "input_index": input_index,
        "edge_count_unique": int(lengths.size),
        "median_length": median_length,
        "mean_length": mean_length,
        "std_length": std_length,
        "cv": cv,
        "short_edge_fraction": short_fraction,
    }
    return (
        annotation,
        pd.DataFrame(subgraph_rows),
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
        sampling,
    )


# ---------------------------------------------------------------------------
# Transport through couplings
# ---------------------------------------------------------------------------


def row_normalize_coupling(
    coupling: np.ndarray, *, epsilon: float = EPS
) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(coupling, dtype=float)
    matched_mass = T.sum(axis=1)
    Q = np.zeros_like(T)
    positive = matched_mass > epsilon
    Q[positive] = T[positive] / matched_mass[positive, None]
    return Q, matched_mass


def transport_matrix(
    matrix: np.ndarray, conditional_correspondence: np.ndarray
) -> np.ndarray:
    """Return ``Q M Q.T`` with symmetrization and zero diagonal."""
    M = np.asarray(matrix, dtype=float)
    Q = np.asarray(conditional_correspondence, dtype=float)
    transported = Q @ M @ Q.T
    transported = 0.5 * (transported + transported.T)
    np.fill_diagonal(transported, 0.0)
    return transported


def compute_transported_separatrix_connectivity(
    bundle: Mapping[str, Any],
    input_node_orders: Sequence[Sequence[Any]],
    config: ReconstructionConfig,
    *,
    q: Optional[float] = None,
    max_soft_hops: Optional[int] = None,
    affinity_mode: Optional[AffinityMode] = None,
    epsilon: float = EPS,
    tiny_overshoot_tolerance: float = 1e-8,
) -> TransportedConnectivityResult:
    """Transport direct adjacency and separatrix-local affinity through PFGW couplings.

    Held-out seed graphs are never included.
    """
    q_value = float(config.hop_decay_q if q is None else q)
    hops = int(config.max_soft_hops if max_soft_hops is None else max_soft_hops)
    mode = affinity_mode or config.connectivity_affinity_mode

    p = np.asarray(bundle["p"], dtype=float)
    graphs = list(bundle["Gs"])
    Ts = list(bundle["Ts"])
    lambdas = np.asarray(bundle["lambdas"], dtype=float)
    n_center = p.shape[0]
    n_inputs = len(graphs)
    if not (
        len(Ts)
        == len(input_node_orders)
        == lambdas.shape[0]
        == n_inputs
    ):
        raise ValueError("Gs, Ts, input_node_orders, and lambdas must align.")
    lambda_sum = float(np.sum(lambdas))
    if lambda_sum <= epsilon:
        raise ValueError("lambdas must have a positive sum.")

    # Optional family-wide median spacing for length-normalized sensitivity.
    global_reference = None
    if mode == "separatrix_length_normalized_decay":
        all_lengths = []
        for graph in graphs:
            all_lengths.append(
                _unique_edge_lengths(
                    graph, length_attribute=config.geometric_length_attribute
                )
            )
        stacked = np.concatenate(all_lengths) if all_lengths else np.asarray([])
        if stacked.size:
            global_reference = float(np.median(stacked))

    direct_num = np.zeros((n_center, n_center), dtype=float)
    sep_num = np.zeros_like(direct_num)
    direct_second = np.zeros_like(direct_num)
    sep_second = np.zeros_like(direct_num)
    coverage_weight_sum = np.zeros_like(direct_num)
    contributing = np.zeros((n_center, n_center), dtype=np.int32)

    matched_fractions: List[np.ndarray] = []
    input_direct_supports: List[np.ndarray] = []
    input_sep_supports: List[np.ndarray] = []
    input_directs: List[np.ndarray] = []
    input_affinities: List[np.ndarray] = []
    annotation_rows: List[Dict[str, Any]] = []
    subgraph_frames: List[pd.DataFrame] = []
    node_frames: List[pd.DataFrame] = []
    edge_frames: List[pd.DataFrame] = []
    sampling_rows: List[Dict[str, Any]] = []
    warnings_list: List[str] = []
    substantial_overshoots = 0
    maximum_overshoot = 0.0

    for input_index, (graph, T_s, node_order, weight) in enumerate(
        zip(graphs, Ts, input_node_orders, lambdas)
    ):
        membership = build_separatrix_membership(
            graph,
            attribute=config.separatrix_id_attribute,
            delimiter=config.separatrix_id_delimiter,
            strict=config.strict_separatrix_validation,
            missing_policy=config.missing_separatrix_id_policy,
        )
        warnings_list.extend(membership.warnings)
        annotation, subgraph_df, node_df, edge_df, sampling = diagnose_separatrix_annotations(
            graph,
            input_index,
            membership,
            length_attribute=config.geometric_length_attribute,
            sampling_irregularity_warning_threshold=(
                config.sampling_irregularity_warning_threshold
            ),
        )
        annotation_rows.append(annotation)
        subgraph_frames.append(subgraph_df)
        node_frames.append(node_df)
        edge_frames.append(edge_df)
        sampling_rows.append(sampling)

        affinity_result = build_separatrix_local_connectivity_affinity(
            graph,
            node_order,
            membership,
            q=q_value,
            max_soft_hops=hops,
            mode=mode,
            length_reference_spacing_mode=config.length_reference_spacing_mode,
            length_attribute=config.geometric_length_attribute,
            maximum_effective_soft_distance=config.maximum_effective_soft_distance,
            global_reference_spacing=global_reference,
        )
        warnings_list.extend(affinity_result.warnings)
        A = affinity_result.direct_adjacency_matrix
        K = affinity_result.separatrix_local_connectivity_affinity

        T = np.asarray(T_s, dtype=float)
        if T.shape != (n_center, A.shape[0]):
            raise ValueError(
                f"T_{input_index} must have shape {(n_center, A.shape[0])}; got {T.shape}."
            )
        T = np.where((T < 0) & (T >= -tiny_overshoot_tolerance), 0.0, T)
        Q, matched_mass = row_normalize_coupling(T, epsilon=epsilon)
        fractions = matched_mass / np.maximum(p, epsilon)
        tiny = (fractions > 1.0) & (fractions <= 1.0 + tiny_overshoot_tolerance)
        fractions[tiny] = 1.0
        substantial = fractions > 1.0 + tiny_overshoot_tolerance
        if np.any(substantial):
            substantial_overshoots += int(np.count_nonzero(substantial))
            maximum_overshoot = max(
                maximum_overshoot, float(np.max(fractions[substantial] - 1.0))
            )
        fractions = np.clip(fractions, 0.0, 1.0)

        S_direct = transport_matrix(A, Q)
        S_sep = transport_matrix(K, Q)
        if np.any(S_direct < -1e-8) or np.any(S_direct > 1.0 + 1e-8):
            raise AssertionError("Conditional direct support escaped [0, 1].")
        if np.any(S_sep < -1e-8) or np.any(S_sep > 1.0 + 1e-8):
            raise AssertionError("Conditional separatrix support escaped [0, 1].")
        if np.any(S_sep + 1e-8 < S_direct):
            warnings_list.append(
                f"Input {input_index}: some conditional separatrix supports are "
                "below direct supports."
            )

        coverage_s = np.outer(fractions, fractions)
        weight_s = float(weight) * coverage_s
        np.fill_diagonal(weight_s, 0.0)
        direct_num += weight_s * S_direct
        sep_num += weight_s * S_sep
        direct_second += weight_s * S_direct * S_direct
        sep_second += weight_s * S_sep * S_sep
        coverage_weight_sum += weight_s
        contributing += (weight_s > epsilon).astype(np.int32)

        matched_fractions.append(fractions)
        input_direct_supports.append(S_direct)
        input_sep_supports.append(S_sep)
        input_directs.append(A)
        input_affinities.append(K)

    pair_is_covered = coverage_weight_sum > epsilon
    pair_coverage = coverage_weight_sum / lambda_sum
    direct_evidence = direct_num / lambda_sum
    sep_evidence = sep_num / lambda_sum
    soft_extension_evidence = sep_evidence - direct_evidence

    direct_support = np.full_like(pair_coverage, np.nan)
    sep_support = np.full_like(pair_coverage, np.nan)
    direct_support[pair_is_covered] = (
        direct_num[pair_is_covered] / coverage_weight_sum[pair_is_covered]
    )
    sep_support[pair_is_covered] = (
        sep_num[pair_is_covered] / coverage_weight_sum[pair_is_covered]
    )
    soft_extension_support = sep_support - direct_support

    def _std(second: np.ndarray, mean: np.ndarray) -> np.ndarray:
        out = np.full_like(mean, np.nan)
        second_moment = np.zeros_like(mean)
        second_moment[pair_is_covered] = (
            second[pair_is_covered] / coverage_weight_sum[pair_is_covered]
        )
        var = second_moment[pair_is_covered] - mean[pair_is_covered] ** 2
        out[pair_is_covered] = np.sqrt(np.maximum(var, 0.0))
        return out

    direct_std = _std(direct_second, direct_support)
    sep_std = _std(sep_second, sep_support)

    for matrix in (
        pair_coverage,
        pair_is_covered,
        contributing,
        direct_evidence,
        sep_evidence,
        soft_extension_evidence,
    ):
        np.fill_diagonal(matrix, 0)

    if np.any(soft_extension_evidence < -1e-8):
        warnings_list.append(
            "soft_extension_evidence has negative values beyond numerical tolerance."
        )
    soft_extension_evidence = np.maximum(soft_extension_evidence, 0.0)

    if not np.allclose(
        direct_evidence[pair_is_covered],
        (pair_coverage * np.nan_to_num(direct_support, nan=0.0))[pair_is_covered],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise AssertionError("direct evidence identity failed.")
    if not np.allclose(
        sep_evidence[pair_is_covered],
        (pair_coverage * np.nan_to_num(sep_support, nan=0.0))[pair_is_covered],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise AssertionError("separatrix evidence identity failed.")

    covered_threshold = pair_coverage >= config.minimum_pair_coverage
    pair_is_covered = pair_is_covered & covered_threshold
    # Keep evidence for weakly covered pairs at zero for selection purposes.
    # Conditional supports remain NaN where originally uncovered; for below-threshold
    # coverage we still expose the conditional values but mark pair_is_covered False.

    upper = np.triu(np.ones((n_center, n_center), dtype=bool), k=1)
    diagnostics = {
        "n_inputs": n_inputs,
        "n_center": n_center,
        "lambda_sum": lambda_sum,
        "q": q_value,
        "max_soft_hops": hops,
        "affinity_mode": mode,
        "covered_pair_count": int(np.count_nonzero(pair_is_covered & upper)),
        "mean_conditional_direct_adjacency_support": (
            float(np.nanmean(direct_support[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "mean_conditional_separatrix_connectivity_support": (
            float(np.nanmean(sep_support[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "mean_pair_coverage": float(np.mean(pair_coverage[upper])),
        "mean_direct_adjacency_evidence": float(np.mean(direct_evidence[upper])),
        "mean_separatrix_connectivity_evidence": float(np.mean(sep_evidence[upper])),
        "substantial_matched_fraction_overshoots": substantial_overshoots,
        "maximum_matched_fraction_overshoot": maximum_overshoot,
    }
    if substantial_overshoots:
        warnings_list.append(
            f"Found {substantial_overshoots} matched-fraction overshoots; "
            f"max={maximum_overshoot:.3g}."
        )

    return TransportedConnectivityResult(
        conditional_direct_adjacency_support=direct_support,
        conditional_direct_adjacency_support_std=direct_std,
        conditional_separatrix_connectivity_support=sep_support,
        conditional_separatrix_connectivity_support_std=sep_std,
        conditional_soft_extension_support=soft_extension_support,
        direct_adjacency_evidence=direct_evidence,
        separatrix_connectivity_evidence=sep_evidence,
        soft_extension_evidence=soft_extension_evidence,
        pair_coverage=pair_coverage,
        pair_is_covered=pair_is_covered,
        pair_contributing_inputs=contributing,
        matched_fractions=np.vstack(matched_fractions),
        input_conditional_direct_adjacency_support=input_direct_supports,
        input_conditional_separatrix_connectivity_support=input_sep_supports,
        input_direct_adjacency_matrices=input_directs,
        input_separatrix_affinities=input_affinities,
        input_annotation_diagnostics=pd.DataFrame(annotation_rows),
        input_subgraph_diagnostics=pd.concat(subgraph_frames, ignore_index=True)
        if subgraph_frames
        else pd.DataFrame(),
        node_separatrix_membership=pd.concat(node_frames, ignore_index=True)
        if node_frames
        else pd.DataFrame(),
        edge_separatrix_multiplicity=pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame(),
        input_sampling_diagnostics=pd.DataFrame(sampling_rows),
        warnings=warnings_list,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Candidate edges, forest, augmentation, repair
# ---------------------------------------------------------------------------


def _euclidean_distance(nodes: pd.DataFrame, i: int, j: int) -> float:
    indexed = nodes.set_index("node_id")
    a = indexed.loc[i, ["x", "y", "z"]].to_numpy(dtype=float)
    b = indexed.loc[j, ["x", "y", "z"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))


def _type_compatibility_from_values(
    ti: Any,
    tj: Any,
    ci: Any,
    cj: Any,
    *,
    mode: ConstraintMode,
    confidence_threshold: float,
) -> str:
    if pd.isna(ti) or pd.isna(tj):
        return "unknown"
    ti_i, tj_i = int(ti), int(tj)
    if ti_i == -1 or tj_i == -1:
        return "compatible"
    high_conf = (
        np.isfinite(ci)
        and np.isfinite(cj)
        and float(ci) >= confidence_threshold
        and float(cj) >= confidence_threshold
    )
    if ti_i == tj_i and ti_i in {0, 1, 2}:
        if high_conf and mode != "off":
            return "violation"
        return "soft_violation" if mode == "soft" else "compatible"
    return "compatible"


def _scalar_order_from_values(ti: Any, tj: Any, si: Any, sj: Any) -> str:
    if pd.isna(ti) or pd.isna(tj) or pd.isna(si) or pd.isna(sj):
        return "unknown"
    ti_i, tj_i = int(ti), int(tj)
    if {ti_i, tj_i} == {0, 2}:
        lo = si if ti_i == 0 else sj
        hi = sj if ti_i == 0 else si
        if float(lo) > float(hi) + 1e-9:
            return "violation"
    return "compatible"


def _type_compatibility(
    nodes: pd.DataFrame,
    i: int,
    j: int,
    *,
    mode: ConstraintMode,
    confidence_threshold: float,
) -> str:
    indexed = nodes.set_index("node_id")
    return _type_compatibility_from_values(
        indexed.loc[i, "morse_type"],
        indexed.loc[j, "morse_type"],
        indexed.loc[i, "type_confidence"],
        indexed.loc[j, "type_confidence"],
        mode=mode,
        confidence_threshold=confidence_threshold,
    )


def _scalar_order_compatibility(nodes: pd.DataFrame, i: int, j: int) -> str:
    indexed = nodes.set_index("node_id")
    return _scalar_order_from_values(
        indexed.loc[i, "morse_type"],
        indexed.loc[j, "morse_type"],
        indexed.loc[i, "scalar_value"],
        indexed.loc[j, "scalar_value"],
    )


def connectivity_selection_score(
    direct_evidence: float,
    soft_extension_evidence: float,
    uncertainty: float,
    config: ReconstructionConfig,
) -> float:
    return float(
        config.direct_evidence_weight * direct_evidence
        + config.soft_extension_evidence_weight * soft_extension_evidence
        - config.uncertainty_penalty_weight * uncertainty
    )


def construct_candidate_edge_table(
    transported: TransportedConnectivityResult,
    nodes: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    barycenter_relation: Optional[np.ndarray] = None,
) -> CandidateEdgeResult:
    """Build the all-pairs evidence table and candidate subset.

    ``barycenter_relation`` is accepted only for audit columns and never used for
    candidate generation or ranking.
    """
    n = len(nodes)
    E_d = transported.direct_adjacency_evidence
    E_s = transported.separatrix_connectivity_evidence
    E_e = transported.soft_extension_evidence
    S_d = transported.conditional_direct_adjacency_support
    S_s = transported.conditional_separatrix_connectivity_support
    S_e = transported.conditional_soft_extension_support
    std_d = transported.conditional_direct_adjacency_support_std
    std_s = transported.conditional_separatrix_connectivity_support_std
    O = transported.pair_coverage
    covered = transported.pair_is_covered
    contrib = transported.pair_contributing_inputs

    # Two-step explanation: R_ij^(2)=max_k min(E_ik^sep, E_kj^sep), k not in {i,j}.
    explanation = np.zeros((n, n), dtype=float)
    for k in range(n):
        vals = np.minimum(E_s[:, k][:, None], E_s[k, :][None, :])
        vals[k, :] = 0.0
        vals[:, k] = 0.0
        np.fill_diagonal(vals, 0.0)
        np.maximum(explanation, vals, out=explanation)

    ii, jj = np.triu_indices(n, k=1)
    indexed_nodes = nodes.set_index("node_id").sort_index()
    coords = indexed_nodes.loc[range(n), ["x", "y", "z"]].to_numpy(dtype=float)
    masses = indexed_nodes.loc[range(n), "mass"].to_numpy(dtype=float)
    types = indexed_nodes.loc[range(n), "morse_type"].to_numpy()
    conf = indexed_nodes.loc[range(n), "type_confidence"].to_numpy(dtype=float)
    scalars = indexed_nodes.loc[range(n), "scalar_value"].to_numpy(dtype=float)
    euclidean = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    unc = np.nan_to_num(std_s[ii, jj], nan=0.0)
    scores = (
        config.direct_evidence_weight * E_d[ii, jj]
        + config.soft_extension_evidence_weight * E_e[ii, jj]
        - config.uncertainty_penalty_weight * unc
    )
    relations = (
        barycenter_relation[ii, jj]
        if barycenter_relation is not None
        else np.full(ii.shape, np.nan)
    )
    type_compat = [
        _type_compatibility_from_values(
            types[i],
            types[j],
            conf[i],
            conf[j],
            mode=config.type_constraint_mode,
            confidence_threshold=config.type_confidence_threshold,
        )
        for i, j in zip(ii, jj)
    ]
    scalar_compat = [
        _scalar_order_from_values(types[i], types[j], scalars[i], scalars[j])
        for i, j in zip(ii, jj)
    ]
    all_pairs = pd.DataFrame(
        {
            "source": ii,
            "target": jj,
            "conditional_direct_adjacency_support": S_d[ii, jj],
            "conditional_separatrix_connectivity_support": S_s[ii, jj],
            "conditional_soft_extension_support": S_e[ii, jj],
            "conditional_direct_adjacency_support_std": std_d[ii, jj],
            "conditional_separatrix_connectivity_support_std": std_s[ii, jj],
            "pair_coverage": O[ii, jj],
            "pair_is_covered": covered[ii, jj].astype(bool),
            "n_contributing_inputs": contrib[ii, jj].astype(int),
            "direct_adjacency_evidence": E_d[ii, jj],
            "separatrix_connectivity_evidence": E_s[ii, jj],
            "soft_extension_evidence": E_e[ii, jj],
            "connectivity_selection_score": scores,
            "two_step_connectivity_explanation": explanation[ii, jj],
            "euclidean_endpoint_distance": euclidean,
            "barycenter_relation_distance": relations,
            "source_mass": masses[ii],
            "target_mass": masses[jj],
            "source_type": types[ii],
            "target_type": types[jj],
            "source_type_confidence": conf[ii],
            "target_type_confidence": conf[jj],
            "type_compatibility": type_compat,
            "scalar_order_compatibility": scalar_compat,
        }
    )

    candidate_flags = {
        "candidate_from_direct_top_k": set(),
        "candidate_from_separatrix_top_k": set(),
        "candidate_from_mutual_top_k": set(),
        "candidate_from_direct_threshold": set(),
        "candidate_from_separatrix_threshold": set(),
        "candidate_from_node_coverage_repair": set(),
    }

    def _pair(i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    top_mutual_direct: Dict[int, Set[int]] = {}
    top_mutual_sep: Dict[int, Set[int]] = {}
    for i in range(n):
        covered_targets = np.flatnonzero(covered[i] & (np.arange(n) != i))
        if covered_targets.size == 0:
            top_mutual_direct[i] = set()
            top_mutual_sep[i] = set()
            continue
        direct_order = covered_targets[
            np.argsort(-E_d[i, covered_targets], kind="mergesort")
        ]
        sep_order = covered_targets[
            np.argsort(-E_s[i, covered_targets], kind="mergesort")
        ]
        for j in direct_order[: config.top_k_direct_evidence]:
            candidate_flags["candidate_from_direct_top_k"].add(_pair(i, int(j)))
        for j in sep_order[: config.top_k_separatrix_evidence]:
            candidate_flags["candidate_from_separatrix_top_k"].add(_pair(i, int(j)))
        top_mutual_direct[i] = set(map(int, direct_order[: config.mutual_top_k]))
        top_mutual_sep[i] = set(map(int, sep_order[: config.mutual_top_k]))

    for i in range(n):
        for j in top_mutual_direct[i]:
            if i in top_mutual_direct.get(j, ()):
                candidate_flags["candidate_from_mutual_top_k"].add(_pair(i, j))
        for j in top_mutual_sep[i]:
            if i in top_mutual_sep.get(j, ()):
                candidate_flags["candidate_from_mutual_top_k"].add(_pair(i, j))

    covered_upper = np.triu(covered, 1)
    positive_direct = covered_upper & (E_d > 0)
    positive_sep = covered_upper & (E_s > 0)
    direct_threshold = (
        float(np.quantile(E_d[positive_direct], 0.9))
        if np.any(positive_direct)
        else np.inf
    )
    sep_threshold = (
        float(np.quantile(E_s[positive_sep], 0.9)) if np.any(positive_sep) else np.inf
    )
    strong_direct = positive_direct & (
        E_d >= max(config.minimum_direct_adjacency_evidence, 1e-12)
    ) & (E_d >= direct_threshold)
    strong_sep = (
        positive_sep
        & (E_s >= max(config.minimum_separatrix_connectivity_evidence, 1e-12))
        & (O >= config.minimum_pair_coverage)
        & (E_s >= sep_threshold)
    )
    for i, j in zip(*np.where(strong_direct)):
        if i < j:
            candidate_flags["candidate_from_direct_threshold"].add((int(i), int(j)))
    for i, j in zip(*np.where(strong_sep)):
        if i < j:
            candidate_flags["candidate_from_separatrix_threshold"].add((int(i), int(j)))

    selected_pairs: Set[Tuple[int, int]] = set()
    for values in candidate_flags.values():
        selected_pairs |= values
    for i in range(n):
        has_covered = bool(np.any(covered[i]))
        if not has_covered:
            continue
        if any(i in pair for pair in selected_pairs):
            continue
        covered_targets = np.flatnonzero(covered[i] & (np.arange(n) != i))
        if covered_targets.size == 0:
            continue
        best_j = int(
            covered_targets[
                np.argmax(
                    E_s[i, covered_targets] * 1e6
                    + E_d[i, covered_targets] * 1e3
                    + O[i, covered_targets]
                )
            ]
        )
        pair = _pair(i, best_j)
        candidate_flags["candidate_from_node_coverage_repair"].add(pair)
        selected_pairs.add(pair)

    for name, values in candidate_flags.items():
        all_pairs[name] = [
            (int(s), int(t)) in values
            for s, t in zip(all_pairs["source"], all_pairs["target"])
        ]
    all_pairs["candidate_selected"] = all_pairs[list(candidate_flags)].any(axis=1)
    selected = all_pairs[all_pairs["candidate_selected"]].copy()
    diagnostics = {
        "total_pair_count": int(len(all_pairs)),
        "candidate_count": int(len(selected)),
        "inclusion_reason_counts": {
            name: int(all_pairs[name].sum()) for name in candidate_flags
        },
        "nodes_with_no_candidates": int(
            sum(
                1
                for i in range(n)
                if not any(
                    (min(i, j), max(i, j)) in selected_pairs for j in range(n) if j != i
                )
            )
        ),
    }
    return CandidateEdgeResult(
        all_pairs=all_pairs, selected_edges=selected, diagnostics=diagnostics
    )


def infer_target_component_count(
    bundle: Mapping[str, Any], configured_count: Optional[int] = None
) -> ComponentTarget:
    graphs = bundle.get("Gs") or []
    lambdas = np.asarray(bundle.get("lambdas"), dtype=float)
    counts = []
    for graph in graphs:
        undirected = graph.to_undirected() if graph.is_directed() else graph
        counts.append(int(nx.number_connected_components(undirected)))
    if configured_count is not None:
        return ComponentTarget(
            count=int(configured_count),
            source="configured target_component_count",
            input_component_counts=counts,
            assumption_used=False,
        )
    if counts and len(set(counts)) == 1:
        return ComponentTarget(
            count=counts[0],
            source="unanimous input graph component count",
            input_component_counts=counts,
            assumption_used=False,
        )
    if counts and lambdas.shape[0] == len(counts):
        # Weighted median of component counts.
        order = np.argsort(counts)
        sorted_counts = np.asarray(counts, dtype=float)[order]
        sorted_weights = lambdas[order]
        cumulative = np.cumsum(sorted_weights)
        median = int(sorted_counts[np.searchsorted(cumulative, 0.5 * cumulative[-1])])
        return ComponentTarget(
            count=median,
            source="weighted-median input component count",
            input_component_counts=counts,
            assumption_used=False,
        )
    return ComponentTarget(
        count=1,
        source="default connected-component assumption",
        input_component_counts=counts,
        assumption_used=True,
    )


def _init_graph_with_nodes(nodes: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for _, row in nodes.iterrows():
        graph.add_node(
            int(row["node_id"]),
            x=float(row["x"]) if pd.notna(row["x"]) else np.nan,
            y=float(row["y"]) if pd.notna(row["y"]) else np.nan,
            z=float(row["z"]) if pd.notna(row["z"]) else np.nan,
            scalar_value=float(row["scalar_value"])
            if pd.notna(row["scalar_value"])
            else np.nan,
            morse_type=row["morse_type"],
            type=row["morse_type"],
            type_confidence=float(row["type_confidence"])
            if pd.notna(row["type_confidence"])
            else np.nan,
            mass=float(row["mass"]),
        )
    return graph


def _edge_attrs_from_row(row: Mapping[str, Any], source_stage: str) -> Dict[str, Any]:
    euclidean = float(row["euclidean_endpoint_distance"])
    return {
        "connectivity_selection_score": float(row["connectivity_selection_score"]),
        "conditional_direct_adjacency_support": float(
            row["conditional_direct_adjacency_support"]
        )
        if pd.notna(row["conditional_direct_adjacency_support"])
        else np.nan,
        "conditional_separatrix_connectivity_support": float(
            row["conditional_separatrix_connectivity_support"]
        )
        if pd.notna(row["conditional_separatrix_connectivity_support"])
        else np.nan,
        "pair_coverage": float(row["pair_coverage"]),
        "direct_adjacency_evidence": float(row["direct_adjacency_evidence"]),
        "separatrix_connectivity_evidence": float(
            row["separatrix_connectivity_evidence"]
        ),
        "soft_extension_evidence": float(row["soft_extension_evidence"]),
        "two_step_connectivity_explanation": float(
            row.get("two_step_connectivity_explanation", np.nan)
        ),
        "euclidean_length": euclidean,
        "routed_length": euclidean,
        "barycenter_relation_distance": float(row["barycenter_relation_distance"])
        if pd.notna(row["barycenter_relation_distance"])
        else np.nan,
        "weight": euclidean,
        "source_stage": source_stage,
        "type_compatibility": row.get("type_compatibility", "unknown"),
        "scalar_order_compatibility": row.get("scalar_order_compatibility", "unknown"),
    }


def _hard_constraints_allow(
    graph: nx.Graph,
    row: Mapping[str, Any],
    config: ReconstructionConfig,
    planarity_mode: ConstraintMode,
) -> bool:
    if (
        config.type_constraint_mode == "hard"
        and row.get("type_compatibility") == "violation"
    ):
        return False
    if (
        config.scalar_order_constraint_mode == "hard"
        and row.get("scalar_order_compatibility") == "violation"
    ):
        return False
    if planarity_mode == "hard":
        u, v = int(row["source"]), int(row["target"])
        if not geometric_planarity_allows_edge(graph, u, v):
            return False
    return True


def build_maximum_support_forest(
    candidates: pd.DataFrame,
    nodes: pd.DataFrame,
    config: ReconstructionConfig,
    component_target: ComponentTarget,
    *,
    effective_planarity_mode: ConstraintMode = "soft",
) -> ForestResult:
    """Kruskal-style maximum spanning forest over connectivity scores."""
    graph = _init_graph_with_nodes(nodes)
    parent = {int(n): int(n) for n in graph.nodes()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    sorted_edges = candidates.sort_values(
        by=[
            "connectivity_selection_score",
            "direct_adjacency_evidence",
            "pair_coverage",
            "euclidean_endpoint_distance",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected_rows = []
    target = int(component_target.count)
    components = graph.number_of_nodes()
    for _, row in sorted_edges.iterrows():
        if components <= target:
            break
        u, v = int(row["source"]), int(row["target"])
        if find(u) == find(v):
            continue
        if not _hard_constraints_allow(graph, row, config, effective_planarity_mode):
            continue
        if components - 1 < target:
            continue
        attrs = _edge_attrs_from_row(row, "maximum_support_forest")
        graph.add_edge(u, v, **attrs)
        union(u, v)
        components -= 1
        selected_rows.append({**dict(row), **attrs})

    selected = pd.DataFrame(selected_rows)
    diagnostics = {
        "target_component_count": component_target.count,
        "target_source": component_target.source,
        "final_component_count": int(nx.number_connected_components(graph)),
        "forest_edge_count": int(graph.number_of_edges()),
        "assumption_used": component_target.assumption_used,
        "input_component_counts": component_target.input_component_counts,
    }
    return ForestResult(graph=graph, selected_edges=selected, diagnostics=diagnostics)


def _is_soft_dominated(row: Mapping[str, Any]) -> bool:
    return float(row["soft_extension_evidence"]) > float(
        row["direct_adjacency_evidence"]
    ) + 1e-12


def _soft_edge_redundant(
    graph: nx.Graph,
    row: Mapping[str, Any],
    config: ReconstructionConfig,
) -> bool:
    if float(row["direct_adjacency_evidence"]) > 1e-8 and not _is_soft_dominated(row):
        return False
    u, v = int(row["source"]), int(row["target"])
    explanation = float(row.get("two_step_connectivity_explanation", 0.0))
    sep = float(row["separatrix_connectivity_evidence"])
    if explanation + config.soft_redundancy_tolerance >= sep:
        # Check whether a short selected path already exists.
        if nx.has_path(graph, u, v):
            try:
                path = nx.shortest_path(graph, u, v)
            except nx.NetworkXNoPath:
                return False
            if len(path) <= 4:
                strengths = []
                for a, b in zip(path[:-1], path[1:]):
                    strengths.append(
                        float(
                            graph.edges[a, b].get(
                                "separatrix_connectivity_evidence", 0.0
                            )
                        )
                    )
                if strengths and min(strengths) + config.soft_redundancy_tolerance >= sep:
                    return True
        # Local triangle / clique risk even without an existing path of selected edges.
        common = set(graph.neighbors(u)) & set(graph.neighbors(v))
        if common:
            return True
    return False


def augment_consensus_edges(
    forest: nx.Graph,
    candidates: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    effective_planarity_mode: ConstraintMode = "soft",
    stability_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> AugmentationResult:
    """Add consensus-supported cycle edges and skip redundant soft chords."""
    graph = forest.copy()
    remaining = candidates.copy()
    existing = {tuple(sorted(edge)) for edge in graph.edges()}
    remaining = remaining[
        ~remaining.apply(
            lambda r: (int(r.source), int(r.target)) in existing, axis=1
        )
    ]

    def rank_key(row):
        stable = (
            1
            if stability_pairs is not None
            and (int(row.source), int(row.target)) in stability_pairs
            else 0
        )
        return (
            float(row.direct_adjacency_evidence),
            float(row.pair_coverage) * float(row.separatrix_connectivity_evidence),
            stable,
            float(row.connectivity_selection_score),
        )

    # Keep augmentation tractable: only inspect the strongest omitted candidates.
    ordered = remaining.sort_values(
        by=[
            "direct_adjacency_evidence",
            "separatrix_connectivity_evidence",
            "pair_coverage",
            "connectivity_selection_score",
        ],
        ascending=False,
        kind="mergesort",
    ).head(max(1000, 10 * graph.number_of_nodes()))
    added_rows = []
    rejected_rows = []
    for _, row in ordered.iterrows():
        u, v = int(row["source"]), int(row["target"])
        if graph.has_edge(u, v):
            continue
        if not _hard_constraints_allow(graph, row, config, effective_planarity_mode):
            rejected_rows.append({**dict(row), "reject_reason": "hard_constraint"})
            continue
        soft_dominated = _is_soft_dominated(row)
        if soft_dominated and _soft_edge_redundant(graph, row, config):
            rejected_rows.append({**dict(row), "reject_reason": "soft_redundancy"})
            continue
        # Require nontrivial evidence.
        if (
            float(row["direct_adjacency_evidence"]) < 1e-8
            and float(row["separatrix_connectivity_evidence"]) < 1e-8
        ):
            continue
        stage = (
            "direct_consensus_augmentation"
            if float(row["direct_adjacency_evidence"])
            >= float(row["soft_extension_evidence"])
            else "separatrix_soft_augmentation"
        )
        # Keep soft augmentation conservative.
        if stage == "separatrix_soft_augmentation":
            if float(row["pair_coverage"]) < config.minimum_pair_coverage:
                rejected_rows.append({**dict(row), "reject_reason": "low_coverage"})
                continue
            if float(row["separatrix_connectivity_evidence"]) < max(
                config.minimum_separatrix_connectivity_evidence, 1e-6
            ):
                rejected_rows.append({**dict(row), "reject_reason": "weak_separatrix"})
                continue
        attrs = _edge_attrs_from_row(row, stage)
        graph.add_edge(u, v, **attrs)
        added_rows.append({**dict(row), **attrs})

    return AugmentationResult(
        graph=graph,
        added_edges=pd.DataFrame(added_rows),
        rejected_edges=pd.DataFrame(rejected_rows),
        diagnostics={
            "added_edge_count": len(added_rows),
            "rejected_edge_count": len(rejected_rows),
        },
    )


def graph_objective_components(
    graph: nx.Graph,
    config: ReconstructionConfig,
    *,
    target_components: int,
    planarity_mode: ConstraintMode,
) -> Dict[str, float]:
    score_sum = sum(
        float(data.get("connectivity_selection_score", 0.0))
        for _, _, data in graph.edges(data=True)
    )
    components = int(nx.number_connected_components(graph))
    component_penalty = abs(components - target_components)
    type_penalty = sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("type_compatibility") == "violation"
    )
    planar_penalty = 0.0
    if planarity_mode == "hard":
        planar_penalty = 0.0 if is_geometrically_planar_2d(graph) else 1.0
    redundancy_penalty = sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("source_stage") == "separatrix_soft_augmentation"
        and float(data.get("two_step_connectivity_explanation", 0.0))
        + config.soft_redundancy_tolerance
        >= float(data.get("separatrix_connectivity_evidence", 0.0))
    )
    return {
        "neg_score_sum": -score_sum,
        "edge_count_penalty": config.objective_edge_penalty_weight
        * graph.number_of_edges(),
        "component_penalty": config.objective_component_penalty_weight
        * component_penalty,
        "type_penalty": config.objective_type_penalty_weight * type_penalty,
        "planarity_penalty": config.objective_planarity_penalty_weight * planar_penalty,
        "redundancy_penalty": config.objective_redundancy_penalty_weight
        * redundancy_penalty,
        "total": (
            -score_sum
            + config.objective_edge_penalty_weight * graph.number_of_edges()
            + config.objective_component_penalty_weight * component_penalty
            + config.objective_type_penalty_weight * type_penalty
            + config.objective_planarity_penalty_weight * planar_penalty
            + config.objective_redundancy_penalty_weight * redundancy_penalty
        ),
    }


def local_repair_topology(
    graph: nx.Graph,
    candidates: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    target_components: int,
    effective_planarity_mode: ConstraintMode = "soft",
) -> LocalRepairResult:
    """Deterministic bounded local repair without metric-stress optimization."""
    G = graph.copy()
    history = []
    candidate_lookup = {
        (int(r.source), int(r.target)): r
        for _, r in candidates.sort_values(
            "direct_adjacency_evidence", ascending=False
        )
        .head(max(500, 5 * graph.number_of_nodes()))
        .iterrows()
    }
    objective = graph_objective_components(
        G, config, target_components=target_components, planarity_mode=effective_planarity_mode
    )

    for iteration in range(config.maximum_local_repair_iterations):
        changed = False
        # Remove weak redundant non-bridge soft edges.
        bridges = (
            {tuple(sorted(edge)) for edge in nx.bridges(G)}
            if G.number_of_edges()
            else set()
        )
        for u, v, data in list(G.edges(data=True)):
            if data.get("source_stage") not in {
                "separatrix_soft_augmentation",
            }:
                continue
            if float(data.get("direct_adjacency_evidence", 0.0)) > 1e-8:
                continue
            if tuple(sorted((u, v))) in bridges:
                continue
            row = {
                "source": u,
                "target": v,
                "direct_adjacency_evidence": data.get("direct_adjacency_evidence", 0.0),
                "separatrix_connectivity_evidence": data.get(
                    "separatrix_connectivity_evidence", 0.0
                ),
                "soft_extension_evidence": data.get("soft_extension_evidence", 0.0),
                "two_step_connectivity_explanation": data.get(
                    "two_step_connectivity_explanation", 0.0
                ),
            }
            if not _soft_edge_redundant(G, row, config):
                continue
            old_stage = data.get("source_stage")
            G.remove_edge(u, v)
            new_obj = graph_objective_components(
                G,
                config,
                target_components=target_components,
                planarity_mode=effective_planarity_mode,
            )
            if new_obj["total"] <= objective["total"] + 1e-12:
                history.append(
                    {
                        "iteration": iteration,
                        "operation": "remove_redundant_soft_edge",
                        "source": u,
                        "target": v,
                        "old_source_stage": old_stage,
                        "new_source_stage": None,
                        "reason": "soft_redundancy",
                        **{f"delta_{k}": new_obj[k] - objective[k] for k in objective},
                    }
                )
                objective = new_obj
                changed = True
                break
            G.add_edge(u, v, **data)

        if changed:
            continue

        # Add omitted high-direct-evidence edges.
        for pair, row in sorted(
            candidate_lookup.items(),
            key=lambda item: float(item[1]["direct_adjacency_evidence"]),
            reverse=True,
        ):
            u, v = pair
            if G.has_edge(u, v):
                continue
            if float(row["direct_adjacency_evidence"]) < 1e-6:
                break
            if not _hard_constraints_allow(G, row, config, effective_planarity_mode):
                continue
            attrs = _edge_attrs_from_row(row, "local_repair_direct_add")
            G.add_edge(u, v, **attrs)
            new_obj = graph_objective_components(
                G,
                config,
                target_components=target_components,
                planarity_mode=effective_planarity_mode,
            )
            if new_obj["total"] < objective["total"] - 1e-12:
                history.append(
                    {
                        "iteration": iteration,
                        "operation": "add_direct_evidence_edge",
                        "source": u,
                        "target": v,
                        "old_source_stage": None,
                        "new_source_stage": "local_repair_direct_add",
                        "reason": "high_direct_evidence",
                        "direct_adjacency_evidence": float(
                            row["direct_adjacency_evidence"]
                        ),
                        "separatrix_connectivity_evidence": float(
                            row["separatrix_connectivity_evidence"]
                        ),
                        **{f"delta_{k}": new_obj[k] - objective[k] for k in objective},
                    }
                )
                objective = new_obj
                changed = True
                break
            G.remove_edge(u, v)

        if not changed:
            break

    return LocalRepairResult(
        graph=G,
        edit_history=pd.DataFrame(history),
        objective_components=objective,
        diagnostics={"iterations_completed": len({h["iteration"] for h in history})},
    )


# ---------------------------------------------------------------------------
# Validation, node support classes, sensitivity, baselines
# ---------------------------------------------------------------------------


def classify_node_support(
    graph: nx.Graph,
    transported: TransportedConnectivityResult,
    *,
    coverage_threshold: float,
) -> pd.DataFrame:
    rows = []
    n = transported.pair_coverage.shape[0]
    for i in range(n):
        incident_direct = [
            transported.direct_adjacency_evidence[i, j]
            for j in range(n)
            if j != i
        ]
        incident_sep = [
            transported.separatrix_connectivity_evidence[i, j]
            for j in range(n)
            if j != i
        ]
        incident_cov = [
            transported.pair_coverage[i, j] for j in range(n) if j != i
        ]
        max_direct = max(incident_direct) if incident_direct else 0.0
        max_sep = max(incident_sep) if incident_sep else 0.0
        max_cov = max(incident_cov) if incident_cov else 0.0
        degree = graph.degree(i)
        isolated = degree == 0
        if max_cov < coverage_threshold:
            label: NodeSupportClass = "uncovered"
        elif max_direct >= 0.05 and degree > 0:
            label = "well_supported"
        elif max_sep >= 0.05 and degree > 0:
            label = "softly_supported"
        else:
            label = "weakly_supported"
        rows.append(
            {
                "node_id": i,
                "support_class": label,
                "degree": int(degree),
                "isolated": isolated,
                "max_incident_direct_evidence": max_direct,
                "max_incident_separatrix_evidence": max_sep,
                "max_incident_pair_coverage": max_cov,
            }
        )
    return pd.DataFrame(rows)


def validate_reconstructed_topology(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    transported: TransportedConnectivityResult,
    candidates: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    effective_planarity_mode: ConstraintMode,
) -> ReconstructionValidationResult:
    warnings_list: List[str] = []
    n = len(nodes)
    if graph.number_of_nodes() != n or set(graph.nodes()) != set(nodes["node_id"]):
        raise AssertionError("Final graph must retain every barycenter node.")
    selected_rows = []
    for u, v, data in graph.edges(data=True):
        a, b = (u, v) if u < v else (v, u)
        selected_rows.append(
            {
                "source": a,
                "target": b,
                **{
                    key: data.get(key)
                    for key in (
                        "connectivity_selection_score",
                        "conditional_direct_adjacency_support",
                        "conditional_separatrix_connectivity_support",
                        "pair_coverage",
                        "direct_adjacency_evidence",
                        "separatrix_connectivity_evidence",
                        "soft_extension_evidence",
                        "euclidean_length",
                        "routed_length",
                        "barycenter_relation_distance",
                        "weight",
                        "source_stage",
                    )
                },
            }
        )
    selected = pd.DataFrame(selected_rows)
    node_classes = classify_node_support(
        graph, transported, coverage_threshold=config.minimum_pair_coverage
    )
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    covered = transported.pair_is_covered & upper
    components = int(nx.number_connected_components(graph))
    cycle_rank = _cycle_rank(graph)
    degrees = [d for _, d in graph.degree()]
    planar = is_geometrically_planar_2d(graph)
    direct_dominated = 0
    soft_dominated = 0
    if not selected.empty:
        direct_dominated = int(
            (
                selected["direct_adjacency_evidence"]
                >= selected["soft_extension_evidence"]
            ).sum()
        )
        soft_dominated = int(len(selected) - direct_dominated)

    metrics = {
        "original_node_count": n,
        "final_node_count": graph.number_of_nodes(),
        "missing_nodes": 0,
        "extra_nodes": 0,
        "node_retention_fraction": 1.0,
        "edge_count": graph.number_of_edges(),
        "component_count": components,
        "cycle_rank": cycle_rank,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "edge_to_node_ratio": graph.number_of_edges() / max(n, 1),
        "geometric_planarity": bool(planar),
        "mean_conditional_direct_adjacency_support": float(
            np.nanmean(transported.conditional_direct_adjacency_support[covered])
        )
        if np.any(covered)
        else None,
        "mean_conditional_separatrix_connectivity_support": float(
            np.nanmean(transported.conditional_separatrix_connectivity_support[covered])
        )
        if np.any(covered)
        else None,
        "mean_pair_coverage": float(np.mean(transported.pair_coverage[upper])),
        "mean_direct_adjacency_evidence": float(
            np.mean(transported.direct_adjacency_evidence[upper])
        ),
        "mean_separatrix_connectivity_evidence": float(
            np.mean(transported.separatrix_connectivity_evidence[upper])
        ),
        "selected_direct_dominated_fraction": (
            direct_dominated / len(selected) if len(selected) else None
        ),
        "selected_soft_dominated_fraction": (
            soft_dominated / len(selected) if len(selected) else None
        ),
        "weakly_supported_node_count": int(
            (node_classes["support_class"] == "weakly_supported").sum()
        ),
        "uncovered_node_count": int(
            (node_classes["support_class"] == "uncovered").sum()
        ),
        "effective_planarity_mode": effective_planarity_mode,
        "candidate_count": int(len(candidates)),
    }
    return ReconstructionValidationResult(
        metrics=metrics,
        selected_edges=selected,
        node_support_classes=node_classes,
        warnings=warnings_list,
    )


def run_q_hop_sensitivity(
    bundle: Mapping[str, Any],
    input_node_orders: Sequence[Sequence[Any]],
    nodes: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    effective_planarity_mode: ConstraintMode,
) -> pd.DataFrame:
    """Evaluate a grid of ``(q, H)`` settings without changing the configured defaults."""
    rows = []
    component_target = infer_target_component_count(
        bundle, config.target_component_count
    )
    for q in config.hop_decay_q_values:
        for hops in config.max_soft_hops_values:
            transported = compute_transported_separatrix_connectivity(
                bundle,
                input_node_orders,
                config,
                q=q,
                max_soft_hops=hops,
            )
            candidates = construct_candidate_edge_table(transported, nodes, config)
            forest = build_maximum_support_forest(
                candidates.selected_edges,
                nodes,
                config,
                component_target,
                effective_planarity_mode=effective_planarity_mode,
            )
            augmented = augment_consensus_edges(
                forest.graph,
                candidates.selected_edges,
                config,
                effective_planarity_mode=effective_planarity_mode,
            )
            repaired = local_repair_topology(
                augmented.graph,
                candidates.selected_edges,
                config,
                target_components=component_target.count,
                effective_planarity_mode=effective_planarity_mode,
            )
            validation = validate_reconstructed_topology(
                repaired.graph,
                nodes,
                transported,
                candidates.selected_edges,
                config,
                effective_planarity_mode=effective_planarity_mode,
            )
            rows.append(
                {
                    "hop_decay_q": q,
                    "max_soft_hops": hops,
                    "mean_conditional_direct_support": validation.metrics[
                        "mean_conditional_direct_adjacency_support"
                    ],
                    "mean_conditional_separatrix_support": validation.metrics[
                        "mean_conditional_separatrix_connectivity_support"
                    ],
                    "candidate_edge_count": validation.metrics["candidate_count"],
                    "forest_edge_count": forest.diagnostics["forest_edge_count"],
                    "final_edge_count": validation.metrics["edge_count"],
                    "component_count": validation.metrics["component_count"],
                    "cycle_rank": validation.metrics["cycle_rank"],
                    "geometric_planarity": validation.metrics["geometric_planarity"],
                    "weakly_supported_node_count": validation.metrics[
                        "weakly_supported_node_count"
                    ],
                    "uncovered_node_count": validation.metrics["uncovered_node_count"],
                    "selected_direct_dominated_fraction": validation.metrics[
                        "selected_direct_dominated_fraction"
                    ],
                    "selected_soft_dominated_fraction": validation.metrics[
                        "selected_soft_dominated_fraction"
                    ],
                    "edge_set": frozenset(
                        tuple(sorted(edge)) for edge in repaired.graph.edges()
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    # Jaccard similarity with neighboring settings.
    jaccard = []
    for idx, row in frame.iterrows():
        neighbors = frame[
            (
                (frame.hop_decay_q == row.hop_decay_q)
                & (frame.max_soft_hops.isin([row.max_soft_hops - 1, row.max_soft_hops + 1]))
            )
            | (
                (frame.max_soft_hops == row.max_soft_hops)
                & (
                    np.isclose(frame.hop_decay_q, row.hop_decay_q - 0.25)
                    | np.isclose(frame.hop_decay_q, row.hop_decay_q + 0.25)
                )
            )
        ]
        if neighbors.empty:
            jaccard.append(np.nan)
            continue
        scores = []
        for _, other in neighbors.iterrows():
            a, b = row.edge_set, other.edge_set
            scores.append(len(a & b) / len(a | b) if a or b else 1.0)
        jaccard.append(float(np.mean(scores)))
    frame["neighbor_edge_jaccard"] = jaccard
    return frame.drop(columns=["edge_set"])


def run_sampling_mode_sensitivity(
    bundle: Mapping[str, Any],
    input_node_orders: Sequence[Sequence[Any]],
    nodes: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    effective_planarity_mode: ConstraintMode,
) -> pd.DataFrame:
    if not config.enable_length_normalized_affinity:
        return pd.DataFrame()
    rows = []
    component_target = infer_target_component_count(
        bundle, config.target_component_count
    )
    for mode in ("separatrix_hop_decay", "separatrix_length_normalized_decay"):
        transported = compute_transported_separatrix_connectivity(
            bundle,
            input_node_orders,
            config,
            affinity_mode=mode,
        )
        candidates = construct_candidate_edge_table(transported, nodes, config)
        forest = build_maximum_support_forest(
            candidates.selected_edges,
            nodes,
            config,
            component_target,
            effective_planarity_mode=effective_planarity_mode,
        )
        augmented = augment_consensus_edges(
            forest.graph,
            candidates.selected_edges,
            config,
            effective_planarity_mode=effective_planarity_mode,
        )
        repaired = local_repair_topology(
            augmented.graph,
            candidates.selected_edges,
            config,
            target_components=component_target.count,
            effective_planarity_mode=effective_planarity_mode,
        )
        edge_set = frozenset(tuple(sorted(edge)) for edge in repaired.graph.edges())
        rows.append(
            {
                "connectivity_affinity_mode": mode,
                "final_edge_count": repaired.graph.number_of_edges(),
                "component_count": nx.number_connected_components(repaired.graph),
                "edge_set": edge_set,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) == 2:
        a, b = frame.loc[0, "edge_set"], frame.loc[1, "edge_set"]
        jaccard = len(a & b) / len(a | b) if a or b else 1.0
        frame["edge_set_jaccard_vs_other_mode"] = jaccard
        frame["changed_edge_count"] = len(a ^ b)
    return frame.drop(columns=["edge_set"])


def compare_posthoc_baselines(
    *,
    final_graph: nx.Graph,
    forest_graph: nx.Graph,
    nodes: pd.DataFrame,
    transported: TransportedConnectivityResult,
    bundle: Mapping[str, Any],
    config: ReconstructionConfig,
    direct_only_graph: Optional[nx.Graph] = None,
) -> pd.DataFrame:
    """Compare fixed reconstruction against post-hoc baselines.

    Does not modify the reconstruction.
    """
    rows = []

    def summarize(name: str, graph: nx.Graph) -> Dict[str, Any]:
        selected_direct = []
        selected_sep = []
        selected_cov = []
        for u, v in graph.edges():
            selected_direct.append(transported.direct_adjacency_evidence[u, v])
            selected_sep.append(transported.separatrix_connectivity_evidence[u, v])
            selected_cov.append(transported.pair_coverage[u, v])
        return {
            "baseline": name,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "component_count": nx.number_connected_components(graph),
            "cycle_rank": _cycle_rank(graph),
            "mean_direct_evidence": float(np.mean(selected_direct))
            if selected_direct
            else np.nan,
            "mean_separatrix_evidence": float(np.mean(selected_sep))
            if selected_sep
            else np.nan,
            "mean_pair_coverage": float(np.mean(selected_cov))
            if selected_cov
            else np.nan,
            "geometric_planarity": is_geometrically_planar_2d(graph),
        }

    rows.append(summarize("final_transported_connectivity", final_graph))
    rows.append(summarize("maximum_support_forest", forest_graph))
    if direct_only_graph is not None:
        rows.append(summarize("direct_only_q0", direct_only_graph))
    if isinstance(bundle.get("G_mst"), nx.Graph):
        rows.append(summarize("saved_G_mst", bundle["G_mst"]))

    # C_center threshold / matched-count baselines (post-hoc only).
    C = np.asarray(bundle.get("C_center"), dtype=float)
    if C.ndim == 2 and C.shape[0] == len(nodes):
        threshold_graph = _init_graph_with_nodes(nodes)
        finite = np.isfinite(C) & (C > 0)
        vals = C[np.triu(finite, 1)]
        if vals.size:
            thr = float(np.quantile(vals, 0.1))
            for i in range(C.shape[0]):
                for j in range(i + 1, C.shape[0]):
                    if finite[i, j] and C[i, j] <= thr:
                        threshold_graph.add_edge(
                            i,
                            j,
                            weight=float(C[i, j]),
                            barycenter_relation_distance=float(C[i, j]),
                        )
            rows.append(summarize("C_center_threshold_baseline", threshold_graph))

            matched = _init_graph_with_nodes(nodes)
            target_edges = final_graph.number_of_edges()
            pairs = [
                (float(C[i, j]), i, j)
                for i in range(C.shape[0])
                for j in range(i + 1, C.shape[0])
                if finite[i, j]
            ]
            pairs.sort()
            for dist, i, j in pairs[:target_edges]:
                matched.add_edge(
                    i,
                    j,
                    weight=dist,
                    barycenter_relation_distance=dist,
                )
            rows.append(summarize("C_center_edge_count_matched", matched))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional density routing (topology-preserving)
# ---------------------------------------------------------------------------


def build_evidence_density_field(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    resolution: int,
    bandwidth: float,
    mass_column: str = "connectivity_selection_score",
) -> DensityFieldResult:
    from scipy.ndimage import gaussian_filter

    indexed = nodes.set_index("node_id")
    xy = indexed.loc[range(len(nodes)), ["x", "y"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(xy)):
        raise ValueError("Finite x-y coordinates are required for density routing.")
    masses = np.maximum(edges[mass_column].fillna(0.0).to_numpy(dtype=float), 0.0)
    xy_min, xy_max = np.min(xy, axis=0), np.max(xy, axis=0)
    span = np.maximum(xy_max - xy_min, 1e-12)
    padding = config.density_padding_fraction * span
    lower, upper = xy_min - padding, xy_max + padding
    x_coordinates = np.linspace(lower[0], upper[0], resolution)
    y_coordinates = np.linspace(lower[1], upper[1], resolution)
    dx = float(x_coordinates[1] - x_coordinates[0])
    dy = float(y_coordinates[1] - y_coordinates[0])
    sample_spacing = config.density_sample_spacing_fraction * float(np.max(span))
    mass_grid = np.zeros((resolution, resolution), dtype=float)
    for row, mass in zip(edges.to_dict(orient="records"), masses):
        if mass <= 0:
            continue
        start = xy[int(row["source"])]
        end = xy[int(row["target"])]
        length = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(length / max(sample_spacing, EPS))) + 1)
        parameter = np.linspace(0.0, 1.0, sample_count)
        samples = start[None, :] + parameter[:, None] * (end - start)[None, :]
        x_index = np.clip(
            np.rint((samples[:, 0] - lower[0]) / dx).astype(int), 0, resolution - 1
        )
        y_index = np.clip(
            np.rint((samples[:, 1] - lower[1]) / dy).astype(int), 0, resolution - 1
        )
        np.add.at(mass_grid, (y_index, x_index), mass / sample_count)
    sigma_physical = bandwidth * float(np.max(span))
    sigma_pixels = (sigma_physical / dy, sigma_physical / dx)
    smoothed = gaussian_filter(mass_grid, sigma=sigma_pixels, mode="constant")
    maximum = float(np.max(smoothed))
    density = smoothed / maximum if maximum > 0 else smoothed
    return DensityFieldResult(
        density=density,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        metadata={
            "resolution": resolution,
            "bandwidth_fraction": bandwidth,
            "x_min": float(lower[0]),
            "x_max": float(upper[0]),
            "y_min": float(lower[1]),
            "y_max": float(upper[1]),
            "dx": dx,
            "dy": dy,
            "mass_column": mass_column,
        },
    )


def route_graph_through_density(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    density_field: DensityFieldResult,
) -> GeometricRoutingResult:
    """Route existing edges through a density field without changing adjacency."""
    G = graph.copy()
    stats = []
    meta = density_field.metadata
    density = density_field.density
    x_coords = density_field.x_coordinates
    y_coords = density_field.y_coordinates
    indexed = nodes.set_index("node_id")

    def to_index(xy: np.ndarray) -> Tuple[int, int]:
        ix = int(np.clip(np.argmin(np.abs(x_coords - xy[0])), 0, len(x_coords) - 1))
        iy = int(np.clip(np.argmin(np.abs(y_coords - xy[1])), 0, len(y_coords) - 1))
        return iy, ix

    height, width = density.shape
    for u, v, data in list(G.edges(data=True)):
        start = indexed.loc[u, ["x", "y"]].to_numpy(dtype=float)
        end = indexed.loc[v, ["x", "y"]].to_numpy(dtype=float)
        # Simple dense A* on cost = 1 / (eps + density)
        start_idx = to_index(start)
        end_idx = to_index(end)
        costs = 1.0 / (1e-3 + density)
        try:
            # Grid graph path via networkx for robustness.
            grid = nx.grid_2d_graph(height, width)
            for node in grid.nodes():
                grid.nodes[node]["weight"] = float(costs[node])
            path = nx.astar_path(
                grid,
                start_idx,
                end_idx,
                heuristic=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
                weight=lambda a, b, _attrs: 0.5
                * (grid.nodes[a]["weight"] + grid.nodes[b]["weight"]),
            )
            polyline = np.asarray(
                [[x_coords[ix], y_coords[iy]] for iy, ix in path], dtype=float
            )
        except Exception:
            polyline = np.vstack([start, end])
        diffs = np.diff(polyline, axis=0)
        routed_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        G.edges[u, v]["route_polyline"] = polyline
        G.edges[u, v]["routed_length"] = routed_length
        G.edges[u, v]["weight"] = routed_length
        stats.append(
            {
                "source": u,
                "target": v,
                "euclidean_length": float(data.get("euclidean_length", np.nan)),
                "routed_length": routed_length,
                "n_route_samples": int(len(polyline)),
            }
        )
    return GeometricRoutingResult(
        graph=G,
        route_statistics=pd.DataFrame(stats),
        diagnostics={"n_routed_edges": len(stats)},
        density_field=density_field,
    )


# ---------------------------------------------------------------------------
# Serialization and reload verification
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, frozenset):
        return sorted(value)
    return value


def save_graph_pickle(graph: nx.Graph, path: PathLike) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_graph_pickle(path: PathLike) -> nx.Graph:
    source = Path(path)
    with source.open("rb") as stream:
        graph = pickle.load(stream)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected NetworkX graph, got {type(graph)!r}.")
    return graph


def save_transported_arrays(
    output_dir: PathLike, transported: TransportedConnectivityResult
) -> None:
    output = Path(output_dir)
    np.save(output / "conditional_direct_adjacency_support.npy", transported.conditional_direct_adjacency_support)
    np.save(output / "conditional_direct_adjacency_support_std.npy", transported.conditional_direct_adjacency_support_std)
    np.save(output / "conditional_separatrix_connectivity_support.npy", transported.conditional_separatrix_connectivity_support)
    np.save(output / "conditional_separatrix_connectivity_support_std.npy", transported.conditional_separatrix_connectivity_support_std)
    np.save(output / "conditional_soft_extension_support.npy", transported.conditional_soft_extension_support)
    np.save(output / "direct_adjacency_evidence.npy", transported.direct_adjacency_evidence)
    np.save(output / "separatrix_connectivity_evidence.npy", transported.separatrix_connectivity_evidence)
    np.save(output / "soft_extension_evidence.npy", transported.soft_extension_evidence)
    np.save(output / "pair_coverage.npy", transported.pair_coverage)
    np.save(output / "pair_is_covered.npy", transported.pair_is_covered)
    np.save(output / "pair_contributing_inputs.npy", transported.pair_contributing_inputs)


def save_reconstruction_bundle(
    output_dir: PathLike,
    *,
    config: ReconstructionConfig,
    source_barycenter_dir: PathLike,
    source_metadata: Mapping[str, Any],
    nodes: pd.DataFrame,
    transported: TransportedConnectivityResult,
    candidates: CandidateEdgeResult,
    forest: ForestResult,
    augmented: AugmentationResult,
    repaired: LocalRepairResult,
    validation: ReconstructionValidationResult,
    sensitivity: pd.DataFrame,
    sampling_sensitivity: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    routed: Optional[GeometricRoutingResult] = None,
    posthoc_relation_metrics: Optional[Mapping[str, Any]] = None,
    warnings_and_fallbacks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    resolved = config.resolved_dict(Path.cwd())
    resolved["output_dir"] = str(output)
    (output / "reconstruction_config.json").write_text(
        json.dumps(_json_safe(resolved), indent=2), encoding="utf-8"
    )
    nodes.to_csv(output / "decoded_nodes.csv", index=False)
    transported.input_annotation_diagnostics.to_csv(
        output / "separatrix_annotation_diagnostics.csv", index=False
    )
    transported.input_subgraph_diagnostics.to_csv(
        output / "separatrix_subgraph_diagnostics.csv", index=False
    )
    transported.node_separatrix_membership.to_csv(
        output / "node_separatrix_membership.csv", index=False
    )
    transported.edge_separatrix_multiplicity.to_csv(
        output / "edge_separatrix_multiplicity.csv", index=False
    )
    transported.input_sampling_diagnostics.to_csv(
        output / "input_sampling_diagnostics.csv", index=False
    )
    save_transported_arrays(output, transported)
    candidates.selected_edges.to_csv(output / "candidate_edges.csv", index=False)
    validation.selected_edges.to_csv(output / "selected_edges.csv", index=False)
    repaired.edit_history.to_csv(output / "edge_edit_history.csv", index=False)
    save_graph_pickle(forest.graph, output / "G_maximum_support_forest.gpickle")
    save_graph_pickle(augmented.graph, output / "G_consensus_augmented.gpickle")
    save_graph_pickle(repaired.graph, output / "G_topology.gpickle")
    if routed is not None:
        save_graph_pickle(routed.graph, output / "G_routed.gpickle")
        routed.route_statistics.to_csv(output / "route_statistics.csv", index=False)
    sensitivity.to_csv(output / "q_hop_sensitivity.csv", index=False)
    sampling_sensitivity.to_csv(output / "sampling_mode_sensitivity.csv", index=False)
    baseline_comparison.to_csv(output / "baseline_comparison.csv", index=False)
    (output / "reconstruction_metrics.json").write_text(
        json.dumps(_json_safe(validation.metrics), indent=2), encoding="utf-8"
    )
    (output / "posthoc_relation_metrics.json").write_text(
        json.dumps(_json_safe(posthoc_relation_metrics or {}), indent=2),
        encoding="utf-8",
    )

    edge_weight_semantics = (
        "density_routed_polyline_length"
        if routed is not None
        else "euclidean_endpoint_length"
    )
    package_names = ["numpy", "pandas", "networkx", "matplotlib", "scipy"]
    versions = {"python": platform.python_version()}
    for package in package_names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    created_files = sorted(
        {
            str(path.relative_to(output).as_posix())
            for path in output.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        | {"manifest.json"}
    )
    manifest = {
        "reconstruction_method": "transported_separatrix_connectivity",
        "separatrix_membership_source": "edge_attribute",
        "separatrix_id_attribute": config.separatrix_id_attribute,
        "separatrix_id_delimiter": config.separatrix_id_delimiter,
        "separatrix_pair_relation_is_transitive": False,
        "soft_paths_restricted_to_single_separatrix_id": True,
        "connectivity_affinity_mode": config.connectivity_affinity_mode,
        "connectivity_affinity_formula": "max_r q ** (d_r - 1)",
        "uses_degree_based_chain_inference": False,
        "uses_barycenter_relation_for_edge_selection": False,
        "terminology_version": TERMINOLOGY_VERSION,
        "selected_hop_decay_q": config.hop_decay_q,
        "selected_max_soft_hops": config.max_soft_hops,
        "length_normalized_sensitivity_run": bool(config.enable_length_normalized_affinity),
        "source_barycenter_directory": str(Path(source_barycenter_dir).resolve()),
        "reconstruction_directory": str(output.resolve()),
        "source_metadata": _json_safe(dict(source_metadata)),
        "exclude_seed": bool(dict(source_metadata).get("exclude_seed", False)),
        "edge_weight_semantics": edge_weight_semantics,
        "node_count": int(validation.metrics["final_node_count"]),
        "edge_count": int(validation.metrics["edge_count"]),
        "files_created": created_files,
        "library_versions": versions,
        "validation_result": _json_safe(validation.metrics),
        "warnings_and_fallbacks": sorted(set(warnings_and_fallbacks or [])),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": _json_safe(resolved),
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
    )
    return manifest


def verify_reloaded_topology(
    graph: nx.Graph,
    *,
    expected_nodes: Set[Any],
    expected_edge_count: int,
    saved_metrics: Mapping[str, Any],
    edge_weight_semantics: str,
) -> Dict[str, Any]:
    required_node_attributes = {
        "x",
        "y",
        "z",
        "scalar_value",
        "morse_type",
        "type_confidence",
        "mass",
    }
    required_edge_attributes = {
        "weight",
        "connectivity_selection_score",
        "direct_adjacency_evidence",
        "separatrix_connectivity_evidence",
        "soft_extension_evidence",
        "pair_coverage",
        "euclidean_length",
        "source_stage",
    }
    if set(graph.nodes()) != set(expected_nodes):
        raise AssertionError("Reloaded node set mismatch.")
    if graph.number_of_edges() != expected_edge_count:
        raise AssertionError("Reloaded edge count mismatch.")
    for node, data in graph.nodes(data=True):
        missing = required_node_attributes - set(data)
        if missing:
            raise AssertionError(f"Node {node} missing attributes {missing}.")
    for u, v, data in graph.edges(data=True):
        missing = required_edge_attributes - set(data)
        if missing:
            raise AssertionError(f"Edge {(u, v)} missing attributes {missing}.")
        if "barycenter_relation_distance" in data:
            # Audit-only: must not equal selection score storage semantics.
            pass
        if edge_weight_semantics == "euclidean_endpoint_length":
            if not np.isclose(
                float(data["weight"]),
                float(data["euclidean_length"]),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            ):
                raise AssertionError("weight must equal euclidean_length.")
        elif edge_weight_semantics == "density_routed_polyline_length":
            if not np.isclose(
                float(data["weight"]),
                float(data.get("routed_length", np.nan)),
                rtol=1e-8,
                atol=1e-10,
                equal_nan=True,
            ):
                raise AssertionError("weight must equal routed_length.")
    recomputed = {
        "edge_count": graph.number_of_edges(),
        "component_count": nx.number_connected_components(graph),
        "cycle_rank": _cycle_rank(graph),
        "final_node_count": graph.number_of_nodes(),
    }
    for key, value in recomputed.items():
        if key in saved_metrics and saved_metrics[key] != value:
            raise AssertionError(
                f"Metric mismatch for {key}: saved={saved_metrics[key]} recomputed={value}."
            )
    return {"ok": True, "recomputed": recomputed}


# ---------------------------------------------------------------------------
# Synthetic tests A–L
# ---------------------------------------------------------------------------


def run_synthetic_separatrix_tests(*, tolerance: float = 1e-12) -> Dict[str, bool]:
    """Compact synthetic checks for parsing, affinity, transport, and independence."""
    results: Dict[str, bool] = {}

    # Test A
    assert parse_separatrix_ids("0,1,3") == frozenset({"0", "1", "3"})
    assert parse_separatrix_ids(" 0, 3, 1 ") == frozenset({"0", "1", "3"})
    assert parse_separatrix_ids("1,1,3") == frozenset({"1", "3"})
    results["A_id_parsing"] = True

    # Test B
    G = nx.Graph()
    G.add_edge(0, 1, separatrixID="0,1,3")
    membership = build_separatrix_membership(G, strict=True)
    assert set(membership.subgraphs) == {"0", "1", "3"}
    for sid in ("0", "1", "3"):
        assert membership.subgraphs[sid].has_edge(0, 1)
    results["B_multi_id_membership"] = True

    # Test C
    G = nx.Graph()
    G.add_edge("u", "a", separatrixID="7")
    G.add_edge("a", "v", separatrixID="7")
    for node, (x, y) in {"u": (0, 0), "a": (1, 0), "v": (2, 0)}.items():
        G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"] = x, y, 0
    membership = build_separatrix_membership(G)
    order = ["u", "a", "v"]
    affinity = build_separatrix_local_connectivity_affinity(
        G, order, membership, q=0.5, max_soft_hops=4
    )
    K = affinity.separatrix_local_connectivity_affinity
    assert np.isclose(K[0, 1], 1.0, atol=tolerance)
    assert np.isclose(K[0, 2], 0.5, atol=tolerance)
    results["C_same_separatrix_hop_decay"] = True

    # Test D
    G = nx.Graph()
    G.add_edge("u", "v", separatrixID="0")
    G.add_edge("v", "w", separatrixID="1")
    for node, (x, y) in {"u": (0, 0), "v": (1, 0), "w": (2, 0)}.items():
        G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"] = x, y, 0
    membership = build_separatrix_membership(G)
    assert shared_separatrix_ids(membership, "u", "v") == frozenset({"0"})
    assert shared_separatrix_ids(membership, "v", "w") == frozenset({"1"})
    assert shared_separatrix_ids(membership, "u", "w") == frozenset()
    affinity = build_separatrix_local_connectivity_affinity(
        G, ["u", "v", "w"], membership, q=0.5, max_soft_hops=4
    )
    assert affinity.separatrix_local_connectivity_affinity[0, 2] == 0.0
    results["D_nontransitive_membership"] = True

    # Test E
    G = nx.Graph()
    G.add_edge("a", "b", separatrixID="0,1")
    G.add_edge("b", "c0", separatrixID="0")
    G.add_edge("b", "c1", separatrixID="1")
    for i, node in enumerate(["a", "b", "c0", "c1"]):
        G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"] = i, 0, 0
    membership = build_separatrix_membership(G)
    affinity = build_separatrix_local_connectivity_affinity(
        G, ["a", "b", "c0", "c1"], membership, q=0.5, max_soft_hops=4
    )
    K = affinity.separatrix_local_connectivity_affinity
    # a-c0 along ID 0 has hop 2 => q; a-c1 along ID 1 has hop 2 => q
    assert np.isclose(K[0, 2], 0.5, atol=tolerance)
    assert np.isclose(K[0, 3], 0.5, atol=tolerance)
    # c0-c1 share no ID path
    assert K[2, 3] == 0.0
    results["E_overlapping_separatrices"] = True

    # Test F
    G = nx.Graph()
    G.add_edge("u", "v", separatrixID="1")  # hop 1 via ID 1
    G.add_edge("u", "m", separatrixID="2")
    G.add_edge("m", "v", separatrixID="2")  # hop 2 via ID 2
    for i, node in enumerate(["u", "v", "m"]):
        G.nodes[node]["x"], G.nodes[node]["y"], G.nodes[node]["z"] = i, 0, 0
    membership = build_separatrix_membership(G)
    affinity = build_separatrix_local_connectivity_affinity(
        G, ["u", "v", "m"], membership, q=0.5, max_soft_hops=4
    )
    assert np.isclose(affinity.separatrix_local_connectivity_affinity[0, 1], 1.0)
    results["F_multiple_common_ids"] = True

    # Test G
    G = nx.Graph()
    G.add_edge(0, 1, separatrixID="9")
    G.add_edge(2, 3, separatrixID="9")
    for i in range(4):
        G.nodes[i]["x"], G.nodes[i]["y"], G.nodes[i]["z"] = i, 0, 0
    try:
        build_separatrix_membership(G, strict=True)
        raised = False
    except ValueError:
        raised = True
    assert raised
    membership = build_separatrix_membership(G, strict=False)
    affinity = build_separatrix_local_connectivity_affinity(
        G, [0, 1, 2, 3], membership, q=0.5, max_soft_hops=4
    )
    assert affinity.separatrix_local_connectivity_affinity[0, 2] == 0.0
    results["G_disconnected_repeated_id"] = True

    # Test H
    G = nx.path_graph([0, 1, 2])
    nx.set_edge_attributes(G, "5", "separatrixID")
    for i in range(3):
        G.nodes[i]["x"], G.nodes[i]["y"], G.nodes[i]["z"] = i, 0, 0
    membership = build_separatrix_membership(G)
    affinity = build_separatrix_local_connectivity_affinity(
        G, [0, 1, 2], membership, q=0.0, max_soft_hops=4
    )
    assert np.allclose(
        affinity.separatrix_local_connectivity_affinity,
        affinity.direct_adjacency_matrix,
    )
    results["H_binary_limit"] = True

    # Test I
    A = affinity.direct_adjacency_matrix
    K = build_separatrix_local_connectivity_affinity(
        G, [0, 1, 2], membership, q=0.5, max_soft_hops=4
    ).separatrix_local_connectivity_affinity
    Q = np.eye(3)
    assert np.allclose(transport_matrix(A, Q), A)
    assert np.allclose(transport_matrix(K, Q), K)
    results["I_identity_coupling"] = True

    # Test J
    config = ReconstructionConfig(
        reconstruction_name="synthetic_test",
        hop_decay_q=0.5,
        max_soft_hops=4,
        enable_length_normalized_affinity=False,
    )
    graph = nx.Graph()
    graph.add_edge(0, 1, separatrixID="0")
    for i in range(2):
        graph.nodes[i]["x"], graph.nodes[i]["y"], graph.nodes[i]["z"] = i, 0, 0
    bundle = {
        "p": np.ones(2),
        "Gs": [graph],
        "Ts": [np.diag([0.2, 0.3])],
        "lambdas": np.ones(1),
        "Cs": [np.zeros((2, 2))],
    }
    transported = compute_transported_separatrix_connectivity(
        bundle, [[0, 1]], config
    )
    assert np.isclose(transported.pair_coverage[0, 1], 0.06, atol=tolerance)
    assert np.isclose(
        transported.direct_adjacency_evidence[0, 1],
        transported.pair_coverage[0, 1]
        * transported.conditional_direct_adjacency_support[0, 1],
        atol=tolerance,
    )
    assert np.isclose(
        transported.separatrix_connectivity_evidence[0, 1],
        transported.pair_coverage[0, 1]
        * transported.conditional_separatrix_connectivity_support[0, 1],
        atol=tolerance,
    )
    results["J_partial_coupling"] = True

    # Test K
    G_short = nx.Graph()
    G_short.add_edge("u", "a", separatrixID="1")
    G_short.add_edge("a", "v", separatrixID="1")
    coords_short = {"u": (0.0, 0.0), "a": (1.0, 0.0), "v": (2.0, 0.0)}
    for node, (x, y) in coords_short.items():
        G_short.nodes[node]["x"], G_short.nodes[node]["y"], G_short.nodes[node]["z"] = (
            x,
            y,
            0,
        )
    G_long = nx.Graph()
    path = ["u", "b", "c", "d", "v"]
    for a, b in zip(path[:-1], path[1:]):
        G_long.add_edge(a, b, separatrixID="1")
    coords_long = {
        "u": (0.0, 0.0),
        "b": (0.5, 0.0),
        "c": (1.0, 0.0),
        "d": (1.5, 0.0),
        "v": (2.0, 0.0),
    }
    for node, (x, y) in coords_long.items():
        G_long.nodes[node]["x"], G_long.nodes[node]["y"], G_long.nodes[node]["z"] = (
            x,
            y,
            0,
        )
    m_short = build_separatrix_membership(G_short)
    m_long = build_separatrix_membership(G_long)
    hop_short = build_separatrix_local_connectivity_affinity(
        G_short, ["u", "a", "v"], m_short, q=0.5, max_soft_hops=4
    )
    hop_long = build_separatrix_local_connectivity_affinity(
        G_long, path, m_long, q=0.5, max_soft_hops=4
    )
    len_short = build_separatrix_local_connectivity_affinity(
        G_short,
        ["u", "a", "v"],
        m_short,
        q=0.5,
        max_soft_hops=4,
        mode="separatrix_length_normalized_decay",
    )
    len_long = build_separatrix_local_connectivity_affinity(
        G_long,
        path,
        m_long,
        q=0.5,
        max_soft_hops=4,
        mode="separatrix_length_normalized_decay",
    )
    hop_diff = abs(
        hop_short.separatrix_local_connectivity_affinity[0, 2]
        - hop_long.separatrix_local_connectivity_affinity[0, 4]
    )
    len_diff = abs(
        len_short.separatrix_local_connectivity_affinity[0, 2]
        - len_long.separatrix_local_connectivity_affinity[0, 4]
    )
    assert hop_diff > len_diff - tolerance
    results["K_oversampled_equivalent"] = True

    # Test L: selection helpers do not accept C_center
    import inspect

    for fn in (
        construct_candidate_edge_table,
        build_maximum_support_forest,
        augment_consensus_edges,
        local_repair_topology,
    ):
        params = inspect.signature(fn).parameters
        assert "C_center" not in params
        assert "barycenter_relation" not in params or fn is construct_candidate_edge_table
    # construct_candidate_edge_table may take barycenter_relation for audit only.
    results["L_no_metric_dependency"] = True
    return results


__all__ = [
    "ReconstructionConfig",
    "TERMINOLOGY_VERSION",
    "parse_separatrix_ids",
    "build_separatrix_membership",
    "validate_barycenter_bundle",
    "decode_barycenter_nodes",
    "build_direct_adjacency_matrix",
    "build_separatrix_local_connectivity_affinity",
    "compute_transported_separatrix_connectivity",
    "construct_candidate_edge_table",
    "build_maximum_support_forest",
    "augment_consensus_edges",
    "local_repair_topology",
    "validate_reconstructed_topology",
    "run_q_hop_sensitivity",
    "run_sampling_mode_sensitivity",
    "compare_posthoc_baselines",
    "save_reconstruction_bundle",
    "verify_reloaded_topology",
    "run_synthetic_separatrix_tests",
]
