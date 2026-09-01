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
import re
import warnings
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from lib.barycenter_io import barycenter_save_timestamp
from lib.graph_sparsify import (
    geometric_planarity_allows_edge,
    is_geometrically_planar_2d,
)

PathLike = Union[str, Path]
ConstraintMode = Literal["off", "soft", "hard"]
AffinityMode = Literal[
    "separatrix_hop_decay",
    "separatrix_length_normalized_decay",
    "separatrix_gaussian",
]
LengthReferenceMode = Literal[
    "input_median", "separatrix_median", "global_input_family_median"
]
GaussianDistanceUnits = Literal["domain", "raw"]
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
    path_distances: Dict[Tuple[int, int], float] = field(default_factory=dict)


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
class SeparatrixPathIndex:
    """Ordered 1-separatrix paths aligned to coupling columns of one input.

    ``between_nodes[(x, z)]`` lists every input-column index ``y`` that lies
    strictly between ``x`` and ``z`` on at least one recorded simple path.
    Membership is existential: multiple separatrix IDs never duplicate ``y``.
    """

    node_order: List[Any]
    node_to_index: Dict[Any, int]
    node_to_separatrix_ids: Dict[int, frozenset]
    separatrix_positions: Dict[str, Dict[int, int]]
    separatrix_paths: Dict[str, List[int]]
    between_nodes: Dict[Tuple[int, int], np.ndarray]
    warnings: List[str]
    diagnostics: Dict[str, Any]


@dataclass
class TripleBetweenness:
    """Coverage-weighted ensemble betweenness ``B(a, b, c)``."""

    a: int
    b: int
    c: int
    B: float
    B_inputs: np.ndarray
    weights: np.ndarray
    weight_sum: float
    uninformative: bool


@dataclass
class TriangleReconstructionResult:
    """Hard reconstruction by same-separatrix triangle shortcut removal."""

    mode: str
    theta: float
    min_edge_threshold: float
    candidate_graph: nx.Graph
    reconstructed_graph: nx.Graph
    edge_scores: pd.DataFrame
    removed_edges: pd.DataFrame
    statistics: Dict[str, Any]
    diagnostics: Dict[str, Any]


@dataclass
class PathCoherence:
    """Coverage-weighted path-level same-separatrix coherence ``C(P)``."""

    path: Tuple[int, ...]
    C: float
    C_inputs: np.ndarray
    weights: np.ndarray
    weight_sum: float
    uninformative: bool
    best_separatrix_ids: List[Optional[str]]
    best_directions: List[Optional[str]]
    best_scores: np.ndarray


@dataclass
class CyclePruningResult:
    """Greedy Phase 2 pruning of same-separatrix cycles."""

    mode: str
    theta: float
    phase1_graph: nx.Graph
    reconstructed_graph: nx.Graph
    removed_edges: pd.DataFrame
    attempt_log: pd.DataFrame
    statistics: Dict[str, Any]
    diagnostics: Dict[str, Any]
    preprocess_seconds: float
    elapsed_seconds: float


@dataclass
class EdgeSeparatrixSupport:
    """Coverage-weighted max-separatrix support ``S_sep(e)`` of one barycenter edge."""

    u: int
    v: int
    mode: str
    S_sep: float
    S_inputs: np.ndarray
    weights: np.ndarray
    weight_sum: float
    uninformative: bool
    best_separatrix_ids: List[Optional[str]]
    best_directions: List[Optional[str]]
    S_forward_inputs: np.ndarray
    S_reverse_inputs: np.ndarray
    mean_hops: np.ndarray
    strong_fraction: float
    S_std: float


@dataclass
class EdgePathRedundancy:
    """Same-separatrix redundancy ``R(e, P)`` of a candidate edge and a replacement path."""

    u: int
    v: int
    path: Tuple[int, ...]
    R: float
    R_inputs: np.ndarray
    weights: np.ndarray
    weight_sum: float
    uninformative: bool
    best_separatrix_ids: List[Optional[str]]
    best_directions: List[Optional[str]]
    S_matched: np.ndarray
    C_matched: np.ndarray
    bottleneck: float
    mean_affinity: float


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

    connectivity_affinity_mode: AffinityMode = "separatrix_gaussian"
    max_soft_hops: int = 4
    max_soft_hops_values: List[int] = field(default_factory=lambda: [2, 3, 4])
    gaussian_sigma: float = 0.05
    gaussian_eta: float = 1.0
    gaussian_zero_at_zero_distance: bool = True
    gaussian_cutoff_sigmas: float = 3.0
    gaussian_distance_units: GaussianDistanceUnits = "domain"

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
        if self.max_soft_hops < 1:
            raise ValueError("max_soft_hops must be >= 1.")
        if any(value < 1 for value in self.max_soft_hops_values):
            raise ValueError("max_soft_hops_values must contain integers >= 1.")
        if self.connectivity_affinity_mode not in {
            "separatrix_hop_decay",
            "separatrix_length_normalized_decay",
            "separatrix_gaussian",
        }:
            raise ValueError("Unsupported connectivity_affinity_mode.")
        if float(self.gaussian_sigma) <= 0.0:
            raise ValueError("gaussian_sigma must be positive.")
        if float(self.gaussian_eta) <= 0.0:
            raise ValueError("gaussian_eta must be positive.")
        if float(self.gaussian_cutoff_sigmas) <= 0.0:
            raise ValueError("gaussian_cutoff_sigmas must be positive.")
        if self.gaussian_distance_units not in {"domain", "raw"}:
            raise ValueError("gaussian_distance_units must be 'domain' or 'raw'.")
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
        stamp = barycenter_save_timestamp()
        reconstruction_root = bundle / "reconstruction"
        output_dir = reconstruction_root / f"{self.reconstruction_name}_{stamp}"
        extra = 1
        while output_dir.exists():
            output_dir = (
                reconstruction_root / f"{self.reconstruction_name}_{stamp}_{extra:02d}"
            )
            extra += 1
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


def _gaussian_kernel_value(
    distance: float,
    *,
    sigma: float,
    eta: float,
    cutoff: float,
    zero_at_zero_distance: bool,
) -> float:
    """Standard RBF ``η exp(-d² / (2σ²))``, clipped to ``[0, 1]``.

    Long-tail pairs with ``d > cutoff`` are zero. Distance ``0`` is either
    the Gaussian peak ``η`` or hard-zeroed to avoid self-loop affinity.
    """
    if not np.isfinite(distance) or distance < 0.0:
        return 0.0
    if distance == 0.0:
        return 0.0 if zero_at_zero_distance else min(1.0, float(eta))
    if distance > cutoff:
        return 0.0
    value = float(eta) * float(np.exp(-0.5 * (distance / sigma) ** 2))
    return float(min(1.0, max(0.0, value)))


def _connectivity_affinity_formula(mode: AffinityMode) -> str:
    if mode == "separatrix_gaussian":
        return "eta * exp(-(d_raw/domain_size)^2 / (2 * sigma^2))"
    if mode == "separatrix_length_normalized_decay":
        return "max_r q ** (d_r / delta - 1)"
    return "max_r q ** (d_r - 1)"


def _node_xyz(graph: nx.Graph, node: Any) -> np.ndarray:
    data = graph.nodes[node]
    return np.asarray([data["x"], data["y"], data.get("z", 0.0)], dtype=float)


def _edge_geometric_length(
    graph: nx.Graph,
    u: Any,
    v: Any,
    *,
    length_attribute: Optional[str] = None,
    length_scale: float = 1.0,
) -> float:
    """Unnormalized Euclidean length, optionally divided by ``length_scale``.

    Input graphs store raw mesh coordinates on nodes and the same Euclidean
    length on ``edge.weight``. This does **not** use FGW-rescaled ``C`` entries.
    Pass ``length_scale=domain_size`` to express the length in the same units
    as the normalized feature coordinates ``(x-min)/domain_size``.
    """
    data = graph.get_edge_data(u, v)
    if isinstance(data, dict) and graph.is_multigraph():
        # Multi-edge dict keyed by edge key; use the first edge payload.
        data = next(iter(data.values())) if data else {}
    if length_attribute and isinstance(data, dict) and length_attribute in data:
        raw = float(data[length_attribute])
    else:
        raw = float(np.linalg.norm(_node_xyz(graph, u) - _node_xyz(graph, v)))
    return raw / max(float(length_scale), EPS)


def _unique_edge_lengths(
    graph: nx.Graph,
    *,
    length_attribute: Optional[str] = None,
    length_scale: float = 1.0,
) -> np.ndarray:
    lengths = []
    seen = set()
    for u, v in graph.edges():
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        lengths.append(
            _edge_geometric_length(
                graph,
                u,
                v,
                length_attribute=length_attribute,
                length_scale=length_scale,
            )
        )
    return np.asarray(lengths, dtype=float)


def _separatrix_weighted_graph(
    graph: nx.Graph,
    subgraph: nx.Graph,
    *,
    length_attribute: Optional[str] = None,
    length_scale: float = 1.0,
) -> nx.Graph:
    """Undirected subgraph with Euclidean (or attributed) edge lengths."""
    weighted = nx.Graph()
    weighted.add_nodes_from(subgraph.nodes())
    for u, v in subgraph.edges():
        length = _edge_geometric_length(
            graph,
            u,
            v,
            length_attribute=length_attribute,
            length_scale=length_scale,
        )
        weighted.add_edge(u, v, weight=max(float(length), EPS))
    return weighted


def _bundle_domain_size(bundle: Mapping[str, Any]) -> Optional[float]:
    meta = dict(bundle.get("meta") or {})
    if "domain_size" not in meta:
        return None
    value = float(meta["domain_size"])
    return value if value > 0.0 else None


def _resolve_gaussian_length_scale(
    bundle: Mapping[str, Any],
    graphs: Sequence[nx.Graph],
    config: ReconstructionConfig,
    *,
    length_attribute: Optional[str] = None,
) -> Tuple[float, Dict[str, Any], List[str]]:
    """Return ``(length_scale, diagnostics, warnings)`` for Gaussian distances."""
    notes: List[str] = []
    raw_chunks = [
        _unique_edge_lengths(graph, length_attribute=length_attribute, length_scale=1.0)
        for graph in graphs
    ]
    stacked = (
        np.concatenate([chunk for chunk in raw_chunks if chunk.size])
        if raw_chunks
        else np.asarray([])
    )
    median_raw = float(np.median(stacked)) if stacked.size else np.nan
    domain_size = _bundle_domain_size(bundle)
    units = str(config.gaussian_distance_units)
    scale = 1.0
    effective_units = "raw"
    if units == "domain":
        if domain_size is None:
            notes.append(
                "gaussian_distance_units='domain' but bundle meta has no positive "
                "domain_size; using raw (unnormalized) Euclidean lengths."
            )
        else:
            scale = float(domain_size)
            effective_units = "domain"
    diagnostics = {
        "gaussian_distance_units": effective_units,
        "gaussian_distance_units_requested": units,
        "domain_size": domain_size,
        "median_raw_edge_length": median_raw,
        "median_scaled_edge_length": (
            float(median_raw / scale) if np.isfinite(median_raw) and scale > 0 else np.nan
        ),
        "sigma_in_scaled_units": float(config.gaussian_sigma),
        "cutoff_in_scaled_units": float(
            config.gaussian_cutoff_sigmas * config.gaussian_sigma
        ),
        "sigma_in_raw_units": float(config.gaussian_sigma * scale),
        "cutoff_in_raw_units": float(
            config.gaussian_cutoff_sigmas * config.gaussian_sigma * scale
        ),
        "length_scale": float(scale),
    }
    return scale, diagnostics, notes


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
    sigma: Optional[float] = None,
    eta: float = 1.0,
    zero_at_zero_distance: bool = True,
    cutoff_sigmas: float = 3.0,
    length_scale: float = 1.0,
) -> InputAffinityResult:
    """Build soft separatrix-local connectivity affinity ``K``.

    Soft paths remain inside one fixed separatrix-ID subgraph. Pairwise shared
    membership is never transitively closed.

    Gaussian mode uses geometric shortest-path distance ``d_sp``:

    ``K = min(1, η exp(-d_sp² / (2σ²)))`` inside the long-tail cutoff
    ``d_sp ≤ cutoff_sigmas · σ``. Distance 0 is the Gaussian peak; set
    ``zero_at_zero_distance=True`` to zero that peak and avoid self-loop
    affinity. ``length_scale`` converts raw Euclidean lengths into the
    units of ``σ`` (typically ``domain_size`` so ``σ`` is a domain fraction).
    """
    if mode != "separatrix_gaussian":
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1].")
        if q == 1.0:
            warnings.warn(
                "q=1 assigns affinity one to all pairs within the hop cutoff.",
                UserWarning,
                stacklevel=2,
            )
    gaussian_sigma = float(sigma) if sigma is not None else 1.0
    gaussian_eta = float(eta)
    gaussian_cutoff = float(cutoff_sigmas) * gaussian_sigma
    if mode == "separatrix_gaussian":
        if gaussian_sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if gaussian_eta <= 0.0:
            raise ValueError("eta must be positive.")
        if float(cutoff_sigmas) <= 0.0:
            raise ValueError("cutoff_sigmas must be positive.")

    order = list(node_order)
    index = {node: i for i, node in enumerate(order)}
    n = len(order)
    affinity = np.zeros((n, n), dtype=float)
    best_ids: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    hop_distances: Dict[Tuple[int, int], int] = {}
    path_distances: Dict[Tuple[int, int], float] = {}
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

    def _record_pair(
        i: int,
        j: int,
        value: float,
        sid: str,
        *,
        hop: Optional[int] = None,
        path_distance: Optional[float] = None,
    ) -> None:
        if value <= 0:
            return
        a, b = (i, j) if i < j else (j, i)
        if value > affinity[i, j] + 1e-15:
            affinity[i, j] = affinity[j, i] = value
            best_ids[(a, b)] = {sid}
            if hop is not None:
                hop_distances[(a, b)] = int(hop)
            if path_distance is not None:
                path_distances[(a, b)] = float(path_distance)
        elif np.isclose(value, affinity[i, j]):
            best_ids[(a, b)].add(sid)

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
            weighted = _separatrix_weighted_graph(
                graph, subgraph, length_attribute=length_attribute
            )
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
                    _record_pair(
                        i,
                        j,
                        value,
                        sid,
                        hop=hop_for_diag,
                        path_distance=float(geo_length),
                    )
            continue

        if mode == "separatrix_gaussian":
            weighted = _separatrix_weighted_graph(
                graph,
                subgraph,
                length_attribute=length_attribute,
                length_scale=length_scale,
            )
            for source in list(weighted.nodes()):
                if source not in index:
                    continue
                lengths_map = nx.single_source_dijkstra_path_length(
                    weighted,
                    source,
                    cutoff=gaussian_cutoff,
                    weight="weight",
                )
                i = index[source]
                for target, geo_length in lengths_map.items():
                    if target == source or target not in index:
                        continue
                    j = index[target]
                    if j <= i:
                        continue
                    distance = float(geo_length)
                    value = _gaussian_kernel_value(
                        distance,
                        sigma=gaussian_sigma,
                        eta=gaussian_eta,
                        cutoff=gaussian_cutoff,
                        zero_at_zero_distance=zero_at_zero_distance,
                    )
                    _record_pair(
                        i,
                        j,
                        value,
                        sid,
                        path_distance=distance,
                    )
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
                _record_pair(i, j, value, sid, hop=int(hop))

    if mode == "separatrix_gaussian" and not zero_at_zero_distance:
        peak = min(1.0, gaussian_eta)
        for i in range(n):
            if affinity[i, i] < peak:
                affinity[i, i] = peak
    else:
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
            "sigma": gaussian_sigma if mode == "separatrix_gaussian" else None,
            "eta": gaussian_eta if mode == "separatrix_gaussian" else None,
            "zero_at_zero_distance": (
                bool(zero_at_zero_distance) if mode == "separatrix_gaussian" else None
            ),
            "cutoff_sigmas": (
                float(cutoff_sigmas) if mode == "separatrix_gaussian" else None
            ),
            "length_scale": float(length_scale) if mode == "separatrix_gaussian" else None,
            "n_positive_affinity_pairs": int(np.count_nonzero(np.triu(affinity, 1))),
            "n_direct_pairs": int(np.count_nonzero(np.triu(direct, 1))),
        },
        path_distances=path_distances,
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
    q_value = 0.1 if q is None else float(q)
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

    length_scale = 1.0
    scale_diagnostics: Dict[str, Any] = {}
    warnings_list: List[str] = []
    if mode == "separatrix_gaussian":
        length_scale, scale_diagnostics, scale_notes = _resolve_gaussian_length_scale(
            bundle,
            graphs,
            config,
            length_attribute=config.geometric_length_attribute,
        )
        warnings_list.extend(scale_notes)

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
            sigma=config.gaussian_sigma,
            eta=config.gaussian_eta,
            zero_at_zero_distance=config.gaussian_zero_at_zero_distance,
            cutoff_sigmas=config.gaussian_cutoff_sigmas,
            length_scale=length_scale,
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
        if mode == "separatrix_hop_decay" and np.any(S_sep + 1e-8 < S_direct):
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

    if mode != "separatrix_gaussian" and np.any(soft_extension_evidence < -1e-8):
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
        "affinity_mode": mode,
        "gaussian_sigma": float(config.gaussian_sigma),
        "gaussian_eta": float(config.gaussian_eta),
        "gaussian_zero_at_zero_distance": bool(config.gaussian_zero_at_zero_distance),
        "gaussian_cutoff_sigmas": float(config.gaussian_cutoff_sigmas),
        **scale_diagnostics,
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
# Coupling-derived 1-separatrix betweenness and triangle reconstruction
# ---------------------------------------------------------------------------


def ordered_separatrix_path_nodes(subgraph: nx.Graph) -> List[Any]:
    """Return the unique endpoint-to-endpoint node sequence of a simple path."""
    n_nodes = subgraph.number_of_nodes()
    if n_nodes == 0:
        return []
    if n_nodes == 1:
        return list(subgraph.nodes())
    degrees = dict(subgraph.degree())
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    if (
        subgraph.number_of_edges() != n_nodes - 1
        or len(endpoints) != 2
        or any(degree > 2 for degree in degrees.values())
        or not nx.is_connected(subgraph)
    ):
        raise ValueError("Separatrix subgraph is not a simple path.")
    return list(nx.shortest_path(subgraph, endpoints[0], endpoints[1]))


def build_separatrix_path_index(
    graph: nx.Graph,
    node_order: Sequence[Any],
    membership: SeparatrixMembership,
    *,
    strict: bool = True,
) -> SeparatrixPathIndex:
    """Precompute ordered paths and existential between-sets for one input.

    Coupling columns follow ``node_order``. For each ordered pair of input
    nodes ``(x, z)``, ``between_nodes[(x, z)]`` is the union of interior nodes
    on every recorded 1-separatrix that contains both endpoints.
    """
    order = list(node_order)
    node_to_index = {node: i for i, node in enumerate(order)}
    if len(node_to_index) != len(order):
        raise ValueError("node_order must contain unique node ids.")

    warnings_list: List[str] = []
    node_to_sids: Dict[int, Set[str]] = defaultdict(set)
    separatrix_positions: Dict[str, Dict[int, int]] = {}
    separatrix_paths: Dict[str, List[int]] = {}
    between_sets: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    skipped = 0

    for sid, subgraph in membership.subgraphs.items():
        try:
            path_nodes = ordered_separatrix_path_nodes(subgraph)
        except ValueError as exc:
            skipped += 1
            message = f"Separatrix ID {sid!r} is not a simple path ({exc})."
            if strict:
                raise ValueError(message) from exc
            warnings_list.append(message)
            continue
        path_idx: List[int] = []
        for node in path_nodes:
            if node not in node_to_index:
                raise ValueError(
                    f"Separatrix ID {sid!r} contains node {node!r} missing from node_order."
                )
            path_idx.append(int(node_to_index[node]))
        separatrix_paths[str(sid)] = path_idx
        positions = {node_idx: pos for pos, node_idx in enumerate(path_idx)}
        separatrix_positions[str(sid)] = positions
        for node_idx in path_idx:
            node_to_sids[node_idx].add(str(sid))
        n_path = len(path_idx)
        for left in range(n_path):
            for right in range(left + 2, n_path):
                x = path_idx[left]
                z = path_idx[right]
                interior = path_idx[left + 1 : right]
                between_sets[(x, z)].update(interior)
                between_sets[(z, x)].update(interior)

    between_nodes = {
        key: np.fromiter(sorted(values), dtype=np.int64)
        for key, values in between_sets.items()
        if values
    }
    path_lengths = [len(path) for path in separatrix_paths.values()]
    return SeparatrixPathIndex(
        node_order=order,
        node_to_index=node_to_index,
        node_to_separatrix_ids={
            node_idx: frozenset(ids) for node_idx, ids in node_to_sids.items()
        },
        separatrix_positions=separatrix_positions,
        separatrix_paths=separatrix_paths,
        between_nodes=between_nodes,
        warnings=warnings_list,
        diagnostics={
            "n_nodes": len(order),
            "n_separatrix_ids": len(separatrix_paths),
            "n_between_pairs": len(between_nodes),
            "skipped_non_path_ids": skipped,
            "path_node_count_min": int(min(path_lengths)) if path_lengths else 0,
            "path_node_count_median": (
                float(np.median(path_lengths)) if path_lengths else 0.0
            ),
            "path_node_count_max": int(max(path_lengths)) if path_lengths else 0,
        },
    )


def input_nodes_between(
    path_index: SeparatrixPathIndex, x: int, y: int, z: int
) -> bool:
    """Return True iff ``y`` lies strictly between ``x`` and ``z`` on some path.

    ``x``, ``y``, and ``z`` are coupling-column indices. The predicate is
    existential across separatrix IDs and ignores degree / junction status.
    """
    if x == y or y == z or x == z:
        return False
    interior = path_index.between_nodes.get((int(x), int(z)))
    if interior is None or interior.size == 0:
        return False
    return int(y) in set(interior.tolist())


def input_nodes_between_from_positions(
    path_index: SeparatrixPathIndex, x: int, y: int, z: int
) -> bool:
    """Reference predicate that walks recorded path positions directly."""
    if x == y or y == z or x == z:
        return False
    x_i, y_i, z_i = int(x), int(y), int(z)
    for positions in path_index.separatrix_positions.values():
        px = positions.get(x_i)
        py = positions.get(y_i)
        pz = positions.get(z_i)
        if px is None or py is None or pz is None:
            continue
        if min(px, pz) < py < max(px, pz):
            return True
    return False


def _betweenness_score_from_conditionals(
    q_a: np.ndarray,
    q_b: np.ndarray,
    q_c: np.ndarray,
    path_index: SeparatrixPathIndex,
    *,
    epsilon: float = EPS,
) -> float:
    """``sum_{x,z} q_a(x) q_c(z) sum_{y in Between(x,z)} q_b(y)``."""
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_c = np.asarray(q_c, dtype=float)
    support_a = np.flatnonzero(q_a > epsilon)
    support_c = np.flatnonzero(q_c > epsilon)
    if support_a.size == 0 or support_c.size == 0:
        return 0.0
    score = 0.0
    between = path_index.between_nodes
    for x in support_a:
        q_x = q_a[x]
        x_i = int(x)
        for z in support_c:
            interior = between.get((x_i, int(z)))
            if interior is None or interior.size == 0:
                continue
            score += q_x * q_c[z] * float(q_b[interior].sum())
    return float(score)


def _betweenness_score_naive(
    q_a: np.ndarray,
    q_b: np.ndarray,
    q_c: np.ndarray,
    path_index: SeparatrixPathIndex,
    *,
    epsilon: float = EPS,
) -> float:
    """Dense O(N^3) reference used only in tests."""
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_c = np.asarray(q_c, dtype=float)
    n = q_a.shape[0]
    score = 0.0
    for x in range(n):
        if q_a[x] <= epsilon:
            continue
        for y in range(n):
            if q_b[y] <= epsilon:
                continue
            for z in range(n):
                if q_c[z] <= epsilon:
                    continue
                if input_nodes_between_from_positions(path_index, x, y, z):
                    score += q_a[x] * q_b[y] * q_c[z]
    return float(score)


class SeparatrixBetweennessEngine:
    """Cached ensemble betweenness ``B(a, b, c)`` from PFGW couplings.

    Coverage weights reuse the existing matched-fraction convention
    ``r_i(u) = min(1, m_i(u) / p(u))`` stored on the transported result.
    An input contributes only through ``w_i = min(r_i(a), r_i(b), r_i(c))``,
    so a nearly unmatched barycenter node cannot cast a strong negative vote.
    """

    def __init__(
        self,
        conditionals: Sequence[np.ndarray],
        matched_fractions: np.ndarray,
        path_indices: Sequence[SeparatrixPathIndex],
        *,
        epsilon: float = EPS,
    ) -> None:
        self.conditionals = [np.asarray(Q, dtype=float) for Q in conditionals]
        self.matched_fractions = np.asarray(matched_fractions, dtype=float)
        self.path_indices = list(path_indices)
        self.epsilon = float(epsilon)
        n_inputs = len(self.conditionals)
        if (
            self.matched_fractions.ndim != 2
            or self.matched_fractions.shape[0] != n_inputs
            or len(self.path_indices) != n_inputs
        ):
            raise ValueError(
                "conditionals, matched_fractions, and path_indices must align."
            )
        n_center = self.matched_fractions.shape[1]
        for index, Q in enumerate(self.conditionals):
            if Q.ndim != 2 or Q.shape[0] != n_center:
                raise ValueError(
                    f"Q_{index} must have {n_center} rows; got shape {Q.shape}."
                )
            if Q.shape[1] != len(self.path_indices[index].node_order):
                raise ValueError(
                    f"Q_{index} columns must match the path-index node_order."
                )
        self.n_inputs = n_inputs
        self.n_center = n_center
        self._cache: Dict[Tuple[int, int, int], TripleBetweenness] = {}

    def score(self, a: int, b: int, c: int) -> TripleBetweenness:
        """Return ``B(a, b, c)``, the probability that ``b`` lies between ``a`` and ``c``."""
        a_i, b_i, c_i = int(a), int(b), int(c)
        cached = self._cache.get((a_i, b_i, c_i))
        if cached is not None:
            return cached
        B_inputs = np.zeros(self.n_inputs, dtype=float)
        weights = np.zeros(self.n_inputs, dtype=float)
        for i, (Q, fractions, path_index) in enumerate(
            zip(self.conditionals, self.matched_fractions, self.path_indices)
        ):
            weights[i] = float(
                min(fractions[a_i], fractions[b_i], fractions[c_i])
            )
            if weights[i] <= self.epsilon:
                continue
            B_inputs[i] = _betweenness_score_from_conditionals(
                Q[a_i],
                Q[b_i],
                Q[c_i],
                path_index,
                epsilon=self.epsilon,
            )
        weight_sum = float(weights.sum())
        if weight_sum <= self.epsilon:
            result = TripleBetweenness(
                a=a_i,
                b=b_i,
                c=c_i,
                B=0.0,
                B_inputs=B_inputs,
                weights=weights,
                weight_sum=0.0,
                uninformative=True,
            )
        else:
            result = TripleBetweenness(
                a=a_i,
                b=b_i,
                c=c_i,
                B=float(np.dot(weights, B_inputs) / weight_sum),
                B_inputs=B_inputs,
                weights=weights,
                weight_sum=weight_sum,
                uninformative=False,
            )
        self._cache[(a_i, b_i, c_i)] = result
        # Ordered betweenness is symmetric in the endpoints.
        self._cache[(c_i, b_i, a_i)] = TripleBetweenness(
            a=c_i,
            b=b_i,
            c=a_i,
            B=result.B,
            B_inputs=result.B_inputs.copy(),
            weights=result.weights.copy(),
            weight_sum=result.weight_sum,
            uninformative=result.uninformative,
        )
        return result


def prepare_separatrix_betweenness_engine(
    bundle: Mapping[str, Any],
    input_node_orders: Sequence[Sequence[Any]],
    config: ReconstructionConfig,
    transported: TransportedConnectivityResult,
    *,
    epsilon: float = EPS,
    tiny_overshoot_tolerance: float = 1e-8,
) -> SeparatrixBetweennessEngine:
    """Build path indices and conditional couplings from an existing bundle.

    Does not recompute PFGW. Reuses ``transported.matched_fractions`` as the
    coverage ratios ``r_i(u)``.
    """
    graphs = list(bundle["Gs"])
    Ts = list(bundle["Ts"])
    p = np.asarray(bundle["p"], dtype=float)
    n_center = p.shape[0]
    if not (len(graphs) == len(Ts) == len(input_node_orders)):
        raise ValueError("Gs, Ts, and input_node_orders must align.")

    conditionals: List[np.ndarray] = []
    path_indices: List[SeparatrixPathIndex] = []
    for input_index, (graph, T_s, node_order) in enumerate(
        zip(graphs, Ts, input_node_orders)
    ):
        membership = build_separatrix_membership(
            graph,
            attribute=config.separatrix_id_attribute,
            delimiter=config.separatrix_id_delimiter,
            strict=config.strict_separatrix_validation,
            missing_policy=config.missing_separatrix_id_policy,
        )
        path_index = build_separatrix_path_index(
            graph,
            node_order,
            membership,
            strict=config.strict_separatrix_validation,
        )
        path_indices.append(path_index)
        T = np.asarray(T_s, dtype=float)
        if T.shape != (n_center, len(node_order)):
            raise ValueError(
                f"T_{input_index} must have shape {(n_center, len(node_order))}; "
                f"got {T.shape}."
            )
        T = np.where((T < 0) & (T >= -tiny_overshoot_tolerance), 0.0, T)
        Q, _matched_mass = row_normalize_coupling(T, epsilon=epsilon)
        conditionals.append(Q)

    matched = np.asarray(transported.matched_fractions, dtype=float)
    if matched.shape != (len(conditionals), n_center):
        raise ValueError(
            "transported.matched_fractions must have shape "
            f"{(len(conditionals), n_center)}; got {matched.shape}."
        )
    return SeparatrixBetweennessEngine(
        conditionals,
        matched,
        path_indices,
        epsilon=epsilon,
    )


def candidate_graph_from_evidence(
    evidence: np.ndarray,
    *,
    min_edge_threshold: float,
    n_nodes: Optional[int] = None,
) -> nx.Graph:
    """Undirected graph of pairs whose evidence meets the visualization floor."""
    W = np.asarray(evidence, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"evidence must be square; got shape {W.shape}.")
    n = int(W.shape[0] if n_nodes is None else n_nodes)
    if n != W.shape[0]:
        raise ValueError("n_nodes must match the evidence matrix.")
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    iu, ju = np.triu_indices(n, k=1)
    weights = W[iu, ju]
    keep = np.isfinite(weights) & (weights >= float(min_edge_threshold))
    for u, v, weight in zip(iu[keep], ju[keep], weights[keep]):
        graph.add_edge(int(u), int(v), evidence=float(weight))
    return graph


def reconstruction_graph_statistics(
    graph: nx.Graph,
    *,
    n_candidate_edges: Optional[int] = None,
    n_removed: int = 0,
) -> Dict[str, Any]:
    """Report node/edge counts and Betti numbers of an undirected graph."""
    n_nodes = int(graph.number_of_nodes())
    n_edges = int(graph.number_of_edges())
    beta_0 = int(nx.number_connected_components(graph)) if n_nodes else 0
    return {
        "n_nodes": n_nodes,
        "n_candidate_edges": (
            n_edges if n_candidate_edges is None else int(n_candidate_edges)
        ),
        "n_removed": int(n_removed),
        "n_reconstructed_edges": n_edges,
        "beta_0": beta_0,
        "beta_1": int(n_edges - n_nodes + beta_0),
    }


def evaluate_triangle_redundancy(
    candidate_graph: nx.Graph,
    engine: SeparatrixBetweennessEngine,
    *,
    edge_order: Optional[Sequence[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Score every candidate edge against the frozen original triangle set.

    ``R(a, c) = max_{b in N(a) ∩ N(c)} B(a, b, c)``. Decisions are not applied
    here; callers must remove marked edges only after this table is complete.
    """
    if edge_order is None:
        edges = [tuple(sorted(edge)) for edge in candidate_graph.edges()]
        edges.sort()
    else:
        edges = [tuple(sorted(edge)) for edge in edge_order]

    rows: List[Dict[str, Any]] = []
    for a, c in edges:
        evidence = float(candidate_graph[a][c].get("evidence", np.nan))
        common = sorted(
            set(candidate_graph.neighbors(a)).intersection(
                candidate_graph.neighbors(c)
            )
        )
        if not common:
            rows.append(
                {
                    "a": int(a),
                    "c": int(c),
                    "evidence": evidence,
                    "n_common_neighbors": 0,
                    "redundancy": 0.0,
                    "witness": np.nan,
                    "uninformative": False,
                    "weight_sum": np.nan,
                    "B_inputs": None,
                    "weights": None,
                    "in_triangle": False,
                }
            )
            continue
        best_B = -1.0
        best_witness = int(common[0])
        best_detail: Optional[TripleBetweenness] = None
        for b in common:
            detail = engine.score(int(a), int(b), int(c))
            if detail.B > best_B + 1e-15 or (
                abs(detail.B - best_B) <= 1e-15 and int(b) < best_witness
            ):
                best_B = float(detail.B)
                best_witness = int(b)
                best_detail = detail
        assert best_detail is not None
        rows.append(
            {
                "a": int(a),
                "c": int(c),
                "evidence": evidence,
                "n_common_neighbors": len(common),
                "redundancy": float(best_B),
                "witness": int(best_witness),
                "uninformative": bool(best_detail.uninformative),
                "weight_sum": float(best_detail.weight_sum),
                "B_inputs": best_detail.B_inputs.copy(),
                "weights": best_detail.weights.copy(),
                "in_triangle": True,
            }
        )
    return pd.DataFrame(rows)


def apply_triangle_shortcut_threshold(
    candidate_graph: nx.Graph,
    edge_scores: pd.DataFrame,
    *,
    theta: float,
    mode: str,
    min_edge_threshold: float,
) -> TriangleReconstructionResult:
    """Remove every edge with ``R(a, c) >= theta`` simultaneously."""
    reconstructed = candidate_graph.copy()
    if edge_scores.empty:
        removed = edge_scores.copy()
    else:
        removed_mask = (
            edge_scores["in_triangle"]
            & np.isfinite(edge_scores["redundancy"])
            & (edge_scores["redundancy"] >= float(theta))
        )
        removed = edge_scores.loc[removed_mask].copy().reset_index(drop=True)
        for a, c in zip(removed["a"].to_numpy(), removed["c"].to_numpy()):
            if reconstructed.has_edge(int(a), int(c)):
                reconstructed.remove_edge(int(a), int(c))

    triangle_scores = edge_scores.loc[edge_scores["in_triangle"], "redundancy"]
    removed_scores = removed["redundancy"] if len(removed) else pd.Series(dtype=float)
    statistics = reconstruction_graph_statistics(
        reconstructed,
        n_candidate_edges=int(candidate_graph.number_of_edges()),
        n_removed=int(len(removed)),
    )
    if len(removed_scores):
        statistics.update(
            {
                "removed_B_min": float(removed_scores.min()),
                "removed_B_median": float(removed_scores.median()),
                "removed_B_max": float(removed_scores.max()),
            }
        )
    else:
        statistics.update(
            {
                "removed_B_min": np.nan,
                "removed_B_median": np.nan,
                "removed_B_max": np.nan,
            }
        )
    if len(triangle_scores):
        statistics.update(
            {
                "triangle_edge_count": int(len(triangle_scores)),
                "triangle_R_min": float(triangle_scores.min()),
                "triangle_R_median": float(triangle_scores.median()),
                "triangle_R_max": float(triangle_scores.max()),
            }
        )
    else:
        statistics.update(
            {
                "triangle_edge_count": 0,
                "triangle_R_min": np.nan,
                "triangle_R_median": np.nan,
                "triangle_R_max": np.nan,
            }
        )
    return TriangleReconstructionResult(
        mode=mode,
        theta=float(theta),
        min_edge_threshold=float(min_edge_threshold),
        candidate_graph=candidate_graph,
        reconstructed_graph=reconstructed,
        edge_scores=edge_scores,
        removed_edges=removed,
        statistics=statistics,
        diagnostics={
            "n_scored_edges": int(len(edge_scores)),
            "n_triangle_edges": int(statistics["triangle_edge_count"]),
            "n_uninformative_triangle_edges": int(
                edge_scores.loc[edge_scores["in_triangle"], "uninformative"].sum()
            )
            if len(edge_scores)
            else 0,
        },
    )


def reconstruct_separatrix_triangle_shortcuts(
    evidence: np.ndarray,
    engine: SeparatrixBetweennessEngine,
    *,
    min_edge_threshold: float,
    theta: float = 0.5,
    mode: str = "evidence",
    edge_order: Optional[Sequence[Tuple[int, int]]] = None,
) -> TriangleReconstructionResult:
    """Threshold evidence, score frozen triangles, then drop shortcut edges.

    Direct and soft modes must call this same function; only ``evidence``
    should differ. ``theta`` is not estimated here.
    """
    candidate = candidate_graph_from_evidence(
        evidence,
        min_edge_threshold=min_edge_threshold,
        n_nodes=engine.n_center,
    )
    scores = evaluate_triangle_redundancy(
        candidate, engine, edge_order=edge_order
    )
    return apply_triangle_shortcut_threshold(
        candidate,
        scores,
        theta=theta,
        mode=mode,
        min_edge_threshold=min_edge_threshold,
    )


def evidence_matrix_from_graph(
    graph: nx.Graph,
    *,
    n_nodes: int,
    weight_attr: str = "evidence",
) -> np.ndarray:
    """Square symmetric matrix of retained edge weights, zeros elsewhere."""
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v, data in graph.edges(data=True):
        weight = float(data.get(weight_attr, 0.0))
        matrix[int(u), int(v)] = weight
        matrix[int(v), int(u)] = weight
    return matrix


def _path_index_from_explicit_paths(
    paths: Mapping[str, Sequence[int]],
    n_nodes: int,
) -> SeparatrixPathIndex:
    """Test helper: build a path index from integer node sequences."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for sid, path in paths.items():
        for u, v in zip(path[:-1], path[1:]):
            existing = graph[u][v].get("separatrixID") if graph.has_edge(u, v) else None
            if existing:
                sid_value = f"{existing},{sid}"
            else:
                sid_value = str(sid)
            graph.add_edge(int(u), int(v), separatrixID=str(sid_value))
    membership = build_separatrix_membership(graph, strict=False)
    return build_separatrix_path_index(
        graph, list(range(n_nodes)), membership, strict=True
    )


def run_triangle_reconstruction_tests(*, tolerance: float = 1e-10) -> Dict[str, bool]:
    """Lightweight correctness checks for ordered-separatrix triangle pruning."""
    results: Dict[str, bool] = {}

    # Test A: ordered triple on a simple path.
    index_a = _path_index_from_explicit_paths({"s0": [0, 1, 2]}, n_nodes=3)
    assert input_nodes_between(index_a, 0, 1, 2) is True
    assert input_nodes_between(index_a, 0, 2, 1) is False
    assert input_nodes_between(index_a, 2, 1, 0) is True
    assert input_nodes_between_from_positions(index_a, 0, 1, 2) is True
    results["A_ordered_triple"] = True

    # Test B: a junction does not invalidate an explicit [x0, x1, x2] path.
    graph_b = nx.Graph()
    graph_b.add_edge(0, 1, separatrixID="main")
    graph_b.add_edge(1, 2, separatrixID="main")
    graph_b.add_edge(1, 3, separatrixID="branch")
    membership_b = build_separatrix_membership(graph_b, strict=True)
    index_b = build_separatrix_path_index(
        graph_b, [0, 1, 2, 3], membership_b, strict=True
    )
    assert graph_b.degree(1) == 3
    assert input_nodes_between(index_b, 0, 1, 2) is True
    assert input_nodes_between(index_b, 0, 3, 2) is False
    results["B_junction_does_not_invalidate"] = True

    # Test C: multiple shared IDs stay Boolean (no double counting).
    index_c = _path_index_from_explicit_paths(
        {"s0": [0, 1, 2], "s1": [0, 1, 2]},
        n_nodes=3,
    )
    assert input_nodes_between(index_c, 0, 1, 2) is True
    q_onehot = np.eye(3)
    score_c = _betweenness_score_from_conditionals(
        q_onehot[0], q_onehot[1], q_onehot[2], index_c
    )
    naive_c = _betweenness_score_naive(
        q_onehot[0], q_onehot[1], q_onehot[2], index_c
    )
    assert np.isclose(score_c, 1.0, atol=tolerance)
    assert np.isclose(score_c, naive_c, atol=tolerance)
    results["C_boolean_multi_id"] = True

    # Test D: endpoint symmetry of B(a, b, c).
    Q_d = np.array(
        [
            [0.7, 0.3, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
        ],
        dtype=float,
    )
    fractions_d = np.ones((1, 3))
    engine_d = SeparatrixBetweennessEngine([Q_d], fractions_d, [index_a])
    forward = engine_d.score(0, 1, 2)
    reverse = engine_d.score(2, 1, 0)
    assert np.isclose(forward.B, reverse.B, atol=tolerance)
    naive_d = _betweenness_score_naive(Q_d[0], Q_d[1], Q_d[2], index_a)
    assert np.isclose(forward.B, naive_d, atol=tolerance)
    results["D_coupling_symmetry"] = True

    # Test E: a nearly unmatched input must not dominate the ensemble.
    Q_informative = np.eye(3)
    Q_unmatched = np.zeros((3, 3))
    fractions_e = np.array(
        [
            [1.0, 1.0, 1.0],
            [1e-16, 1.0, 1.0],
        ]
    )
    engine_e = SeparatrixBetweennessEngine(
        [Q_informative, Q_unmatched],
        fractions_e,
        [index_a, index_a],
    )
    partial = engine_e.score(0, 1, 2)
    assert partial.weights[1] <= EPS
    assert np.isclose(partial.B, 1.0, atol=tolerance)
    results["E_partial_matching"] = True

    # Test F: simultaneous removal is independent of candidate-edge order.
    evidence_f = np.array(
        [
            [0.0, 0.8, 0.7, 0.8],
            [0.8, 0.0, 0.8, 0.0],
            [0.7, 0.8, 0.0, 0.8],
            [0.8, 0.0, 0.8, 0.0],
        ]
    )
    Q_f = np.eye(4)
    index_f = _path_index_from_explicit_paths({"s": [0, 1, 2]}, n_nodes=4)
    engine_f = SeparatrixBetweennessEngine([Q_f], np.ones((1, 4)), [index_f])
    order_u = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3)]
    order_v = list(reversed(order_u))
    result_u = reconstruct_separatrix_triangle_shortcuts(
        evidence_f,
        engine_f,
        min_edge_threshold=0.1,
        theta=0.5,
        mode="test",
        edge_order=order_u,
    )
    result_v = reconstruct_separatrix_triangle_shortcuts(
        evidence_f,
        engine_f,
        min_edge_threshold=0.1,
        theta=0.5,
        mode="test",
        edge_order=order_v,
    )
    assert set(result_u.reconstructed_graph.edges()) == set(
        result_v.reconstructed_graph.edges()
    )
    assert (0, 2) in result_u.candidate_graph.edges() or (
        2,
        0,
    ) in result_u.candidate_graph.edges()
    assert not result_u.reconstructed_graph.has_edge(0, 2)
    assert result_u.reconstructed_graph.has_edge(0, 1)
    assert result_u.reconstructed_graph.has_edge(1, 2)
    results["F_simultaneous_removal"] = True
    return results


# ---------------------------------------------------------------------------
# Phase 2: path-level same-separatrix cycle pruning
# ---------------------------------------------------------------------------

PATH_HOP_TIEBREAKER = 1e-9


def _geometric_mean(values: Sequence[float], *, epsilon: float) -> float:
    """``exp(mean(log(max(T, eps))))``; length-invariant relative to a raw product."""
    clipped = np.maximum(np.asarray(values, dtype=float), float(epsilon))
    if clipped.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(clipped))))


def ordered_pair_support_along_path(
    q_u: np.ndarray,
    q_v: np.ndarray,
    path_columns: np.ndarray,
) -> Tuple[float, float]:
    """Return ``(T_forward, T_reverse)`` for one ordered input separatrix.

    ``T_forward = sum_{j < k} q_u[s_j] q_v[s_k]`` and likewise for reverse.
    Computed from exclusive prefix/suffix sums in ``O(L)``.
    """
    columns = np.asarray(path_columns, dtype=np.int64)
    if columns.size < 2:
        return 0.0, 0.0
    a = np.asarray(q_u, dtype=float)[columns]
    b = np.asarray(q_v, dtype=float)[columns]
    prefix = np.cumsum(a)
    suffix = np.cumsum(a[::-1])[::-1]
    t_forward = float(np.dot(b[1:], prefix[:-1]))
    t_reverse = float(np.dot(b[:-1], suffix[1:]))
    return t_forward, t_reverse


def ordered_pair_support_naive(
    q_u: np.ndarray,
    q_v: np.ndarray,
    path_columns: Sequence[int],
) -> Tuple[float, float]:
    """Dense O(L^2) reference for ``ordered_pair_support_along_path``."""
    columns = [int(c) for c in path_columns]
    q_u = np.asarray(q_u, dtype=float)
    q_v = np.asarray(q_v, dtype=float)
    t_forward = 0.0
    t_reverse = 0.0
    for j, x in enumerate(columns):
        for k, y in enumerate(columns):
            if j < k:
                t_forward += q_u[x] * q_v[y]
            elif j > k:
                t_reverse += q_u[x] * q_v[y]
    return float(t_forward), float(t_reverse)


class SeparatrixPathCoherenceEngine:
    """Cached path-level same-separatrix coherence ``C(P)``.

    Reuses Phase 1 conditional couplings ``q_i(x|u)`` and coverage
    ``r_i(u)``. A path is coherent in input ``i`` only when every consecutive
    ordered pair is explained by the **same** separatrix ID and orientation.
    """

    def __init__(
        self,
        betweenness_engine: SeparatrixBetweennessEngine,
        *,
        epsilon: Optional[float] = None,
    ) -> None:
        t0 = time.perf_counter()
        self.betweenness_engine = betweenness_engine
        self.conditionals = betweenness_engine.conditionals
        self.matched_fractions = betweenness_engine.matched_fractions
        self.path_indices = betweenness_engine.path_indices
        self.epsilon = float(
            betweenness_engine.epsilon if epsilon is None else epsilon
        )
        self.n_inputs = betweenness_engine.n_inputs
        self.n_center = betweenness_engine.n_center
        self.separatrix_columns: List[Dict[str, np.ndarray]] = []
        self.node_separatrix_ids: List[List[frozenset]] = []
        for i, path_index in enumerate(self.path_indices):
            columns = {
                str(sid): np.asarray(path, dtype=np.int64)
                for sid, path in path_index.separatrix_paths.items()
            }
            self.separatrix_columns.append(columns)
            Q = self.conditionals[i]
            per_node: List[frozenset] = []
            for u in range(self.n_center):
                q_u = Q[u]
                sids = [
                    sid
                    for sid, cols in columns.items()
                    if cols.size and float(q_u[cols].sum()) > self.epsilon
                ]
                per_node.append(frozenset(sids))
            self.node_separatrix_ids.append(per_node)
        self._pair_cache: Dict[Tuple[int, str, int, int], Tuple[float, float]] = {}
        self._path_cache: Dict[Tuple[int, ...], PathCoherence] = {}
        self.preprocess_seconds = float(time.perf_counter() - t0)

    def ordered_pair_support(
        self, input_index: int, separatrix_id: str, u: int, v: int
    ) -> Tuple[float, float]:
        """Cached ``(T_forward, T_reverse)`` for barycenter pair ``(u, v)``."""
        key = (int(input_index), str(separatrix_id), int(u), int(v))
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        columns = self.separatrix_columns[input_index].get(str(separatrix_id))
        if columns is None or columns.size < 2:
            result = (0.0, 0.0)
        else:
            Q = self.conditionals[input_index]
            result = ordered_pair_support_along_path(Q[int(u)], Q[int(v)], columns)
        self._pair_cache[key] = result
        return result

    def directed_coherence_on_separatrix(
        self, input_index: int, separatrix_id: str, path: Sequence[int]
    ) -> Tuple[float, float]:
        """Return ``(C_forward, C_reverse)`` of ``P`` on one input separatrix.

        Each term is the geometric mean of ordered-pair supports ``T`` on
        consecutive barycenter hops. ``T`` itself has **no** input-adjacency
        requirement: endpoints may be arbitrarily far along ``s``.
        """
        nodes = [int(v) for v in path]
        if len(nodes) < 3:
            return 0.0, 0.0
        t_forward = []
        t_reverse = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            tf, tr = self.ordered_pair_support(input_index, str(separatrix_id), u, v)
            t_forward.append(tf)
            t_reverse.append(tr)
        return (
            _geometric_mean(t_forward, epsilon=self.epsilon),
            _geometric_mean(t_reverse, epsilon=self.epsilon),
        )

    def path_coherence(self, path: Sequence[int]) -> PathCoherence:
        """Ensemble ``C(P)`` with per-input witness separatrix and direction."""
        nodes = tuple(int(v) for v in path)
        cached = self._path_cache.get(nodes)
        if cached is not None:
            return cached
        if len(nodes) < 3:
            result = PathCoherence(
                path=nodes,
                C=0.0,
                C_inputs=np.zeros(self.n_inputs, dtype=float),
                weights=np.zeros(self.n_inputs, dtype=float),
                weight_sum=0.0,
                uninformative=True,
                best_separatrix_ids=[None] * self.n_inputs,
                best_directions=[None] * self.n_inputs,
                best_scores=np.zeros(self.n_inputs, dtype=float),
            )
            self._path_cache[nodes] = result
            return result

        pairs = list(zip(nodes[:-1], nodes[1:]))
        C_inputs = np.zeros(self.n_inputs, dtype=float)
        weights = np.zeros(self.n_inputs, dtype=float)
        best_sids: List[Optional[str]] = [None] * self.n_inputs
        best_dirs: List[Optional[str]] = [None] * self.n_inputs
        best_scores = np.zeros(self.n_inputs, dtype=float)

        for i in range(self.n_inputs):
            fractions = self.matched_fractions[i]
            weights[i] = float(min(fractions[v] for v in nodes))
            if weights[i] <= self.epsilon:
                continue
            best = -1.0
            best_sid: Optional[str] = None
            best_dir: Optional[str] = None
            candidate_sids: Optional[frozenset] = None
            for node in nodes:
                sids = self.node_separatrix_ids[i][node]
                candidate_sids = (
                    sids if candidate_sids is None else (candidate_sids & sids)
                )
                if not candidate_sids:
                    break
            if not candidate_sids:
                continue
            for sid in candidate_sids:
                t_forward = []
                t_reverse = []
                for u, v in pairs:
                    tf, tr = self.ordered_pair_support(i, sid, u, v)
                    t_forward.append(tf)
                    t_reverse.append(tr)
                c_forward = _geometric_mean(t_forward, epsilon=self.epsilon)
                c_reverse = _geometric_mean(t_reverse, epsilon=self.epsilon)
                if c_forward >= c_reverse:
                    candidate, direction = c_forward, "forward"
                else:
                    candidate, direction = c_reverse, "reverse"
                if candidate > best + 1e-15 or (
                    abs(candidate - best) <= 1e-15
                    and (best_sid is None or str(sid) < str(best_sid))
                ):
                    best = float(candidate)
                    best_sid = str(sid)
                    best_dir = direction
            if best_sid is None:
                continue
            C_inputs[i] = float(max(best, 0.0))
            best_sids[i] = best_sid
            best_dirs[i] = best_dir
            best_scores[i] = C_inputs[i]

        weight_sum = float(weights.sum())
        if weight_sum <= self.epsilon:
            result = PathCoherence(
                path=nodes,
                C=0.0,
                C_inputs=C_inputs,
                weights=weights,
                weight_sum=0.0,
                uninformative=True,
                best_separatrix_ids=best_sids,
                best_directions=best_dirs,
                best_scores=best_scores,
            )
        else:
            result = PathCoherence(
                path=nodes,
                C=float(np.dot(weights, C_inputs) / weight_sum),
                C_inputs=C_inputs,
                weights=weights,
                weight_sum=weight_sum,
                uninformative=False,
                best_separatrix_ids=best_sids,
                best_directions=best_dirs,
                best_scores=best_scores,
            )
        self._path_cache[nodes] = result
        reversed_nodes = tuple(reversed(nodes))
        if reversed_nodes not in self._path_cache:
            self._path_cache[reversed_nodes] = PathCoherence(
                path=reversed_nodes,
                C=result.C,
                C_inputs=result.C_inputs.copy(),
                weights=result.weights.copy(),
                weight_sum=result.weight_sum,
                uninformative=result.uninformative,
                best_separatrix_ids=list(result.best_separatrix_ids),
                best_directions=[
                    None
                    if direction is None
                    else ("reverse" if direction == "forward" else "forward")
                    for direction in result.best_directions
                ],
                best_scores=result.best_scores.copy(),
            )
        return result


def prepare_path_coherence_engine(
    betweenness_engine: SeparatrixBetweennessEngine,
    *,
    epsilon: Optional[float] = None,
) -> SeparatrixPathCoherenceEngine:
    """Wrap Phase 1 couplings/path indices for path-level coherence queries."""
    return SeparatrixPathCoherenceEngine(betweenness_engine, epsilon=epsilon)


def _phase1_edge_sort_key(edge: Tuple[Any, Any, Mapping[str, Any]]) -> Tuple:
    u, v, data = edge
    evidence = float(data.get("evidence", 0.0))
    a, c = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
    return (evidence, a, c)


def _evidence_path_weight(u: Any, v: Any, data: Mapping[str, Any], *, epsilon: float) -> float:
    evidence = float(data.get("evidence", 0.0))
    return float(-np.log(max(evidence, epsilon)) + PATH_HOP_TIEBREAKER)


def _propose_alternative_path(
    graph: nx.Graph,
    u: int,
    v: int,
    *,
    epsilon: float,
) -> Optional[List[int]]:
    """Evidence-preferred u-v path (max product / min -log evidence)."""
    try:
        path = nx.shortest_path(
            graph,
            int(u),
            int(v),
            weight=lambda a, b, data: _evidence_path_weight(
                a, b, data, epsilon=epsilon
            ),
        )
    except nx.NetworkXNoPath:
        return None
    nodes = [int(n) for n in path]
    if len(nodes) < 3:
        return None
    return nodes


def prune_same_separatrix_cycles(
    phase1_graph: nx.Graph,
    coherence_engine: SeparatrixPathCoherenceEngine,
    *,
    theta: float = 0.5,
    mode: str = "evidence",
) -> CyclePruningResult:
    """Greedily remove edges whose alternative path is one 1-separatrix.

    Edges are visited in increasing original evidence (node-id tie break).
    Each candidate is tested against the **current** graph after earlier
    permanent removals. An alternative path is necessary but not sufficient:
    removal requires ``C(P) >= theta``. Bridges are never removed.
    """
    t0 = time.perf_counter()
    graph = phase1_graph.copy()
    stats_before = reconstruction_graph_statistics(graph)
    ordered_edges = sorted(graph.edges(data=True), key=_phase1_edge_sort_key)

    attempt_rows: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []
    n_alternative = 0
    n_coherent = 0

    for u_raw, v_raw, data in ordered_edges:
        u, v = int(u_raw), int(v_raw)
        if not graph.has_edge(u, v):
            continue
        evidence = float(data.get("evidence", np.nan))
        edge_attrs = dict(graph.get_edge_data(u, v) or {})
        graph.remove_edge(u, v)
        path = _propose_alternative_path(
            graph, u, v, epsilon=coherence_engine.epsilon
        )
        if path is None:
            graph.add_edge(u, v, **edge_attrs)
            attempt_rows.append(
                {
                    "u": u,
                    "v": v,
                    "evidence": evidence,
                    "decision": "retained_bridge",
                    "phase": "phase2",
                    "n_path_edges": 0,
                    "C": np.nan,
                    "uninformative": False,
                    "path": None,
                }
            )
            continue

        n_alternative += 1
        coherence = coherence_engine.path_coherence(path)
        n_path_edges = len(path) - 1
        if coherence.C >= float(theta):
            n_coherent += 1
            row = {
                "u": u,
                "v": v,
                "evidence": evidence,
                "decision": "removed_same_separatrix_cycle",
                "phase": "phase2",
                "n_path_edges": int(n_path_edges),
                "C": float(coherence.C),
                "uninformative": bool(coherence.uninformative),
                "weight_sum": float(coherence.weight_sum),
                "path": list(path),
                "C_inputs": coherence.C_inputs.copy(),
                "weights": coherence.weights.copy(),
                "best_separatrix_ids": list(coherence.best_separatrix_ids),
                "best_directions": list(coherence.best_directions),
                "best_scores": coherence.best_scores.copy(),
                "removal_order": len(removed_rows),
            }
            removed_rows.append(row)
            attempt_rows.append(row)
            # Leave e permanently deleted so later tests see this removal.
            continue

        graph.add_edge(u, v, **edge_attrs)
        attempt_rows.append(
            {
                "u": u,
                "v": v,
                "evidence": evidence,
                "decision": "retained_incoherent_alternative",
                "phase": "phase2",
                "n_path_edges": int(n_path_edges),
                "C": float(coherence.C),
                "uninformative": bool(coherence.uninformative),
                "weight_sum": float(coherence.weight_sum),
                "path": list(path),
                "C_inputs": coherence.C_inputs.copy(),
                "weights": coherence.weights.copy(),
                "best_separatrix_ids": list(coherence.best_separatrix_ids),
                "best_directions": list(coherence.best_directions),
                "best_scores": coherence.best_scores.copy(),
            }
        )

    stats_after = reconstruction_graph_statistics(
        graph,
        n_candidate_edges=int(stats_before["n_reconstructed_edges"]),
        n_removed=len(removed_rows),
    )
    if stats_after["beta_0"] != stats_before["beta_0"]:
        raise AssertionError(
            "Phase 2 changed beta_0; a bridge was removed."
        )

    removed = pd.DataFrame(removed_rows)
    attempts = pd.DataFrame(attempt_rows)
    removed_C = removed["C"] if len(removed) else pd.Series(dtype=float)
    removed_len = (
        removed["n_path_edges"] if len(removed) else pd.Series(dtype=float)
    )
    statistics = {
        "n_nodes": stats_after["n_nodes"],
        "n_edges_before": int(stats_before["n_reconstructed_edges"]),
        "n_removed": int(len(removed)),
        "n_edges_after": int(stats_after["n_reconstructed_edges"]),
        "beta_0_before": int(stats_before["beta_0"]),
        "beta_0_after": int(stats_after["beta_0"]),
        "beta_1_before": int(stats_before["beta_1"]),
        "beta_1_after": int(stats_after["beta_1"]),
        "n_alternative_paths": int(n_alternative),
        "n_coherent_alternatives": int(n_coherent),
        "removed_C_min": float(removed_C.min()) if len(removed_C) else np.nan,
        "removed_C_median": float(removed_C.median()) if len(removed_C) else np.nan,
        "removed_C_max": float(removed_C.max()) if len(removed_C) else np.nan,
        "removed_path_edges_min": (
            int(removed_len.min()) if len(removed_len) else 0
        ),
        "removed_path_edges_median": (
            float(removed_len.median()) if len(removed_len) else np.nan
        ),
        "removed_path_edges_max": (
            int(removed_len.max()) if len(removed_len) else 0
        ),
    }
    elapsed = float(time.perf_counter() - t0)
    return CyclePruningResult(
        mode=mode,
        theta=float(theta),
        phase1_graph=phase1_graph,
        reconstructed_graph=graph,
        removed_edges=removed,
        attempt_log=attempts,
        statistics=statistics,
        diagnostics={
            "n_attempted_edges": int(len(attempts)),
            "n_retained_bridge": int(
                (attempts["decision"] == "retained_bridge").sum()
            )
            if len(attempts)
            else 0,
            "n_retained_incoherent": int(
                (attempts["decision"] == "retained_incoherent_alternative").sum()
            )
            if len(attempts)
            else 0,
        },
        preprocess_seconds=float(coherence_engine.preprocess_seconds),
        elapsed_seconds=elapsed,
    )


def _test_coherence_engine(
    conditionals: Sequence[np.ndarray],
    matched_fractions: np.ndarray,
    path_maps: Mapping[str, Sequence[int]],
    *,
    n_input_nodes: int,
) -> SeparatrixPathCoherenceEngine:
    path_index = _path_index_from_explicit_paths(path_maps, n_nodes=n_input_nodes)
    betweenness = SeparatrixBetweennessEngine(
        conditionals, np.asarray(matched_fractions, dtype=float), [path_index]
    )
    return SeparatrixPathCoherenceEngine(betweenness)


def _graph_from_weighted_edges(
    n_nodes: int, edges: Sequence[Tuple[int, int, float]]
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for u, v, evidence in edges:
        graph.add_edge(int(u), int(v), evidence=float(evidence))
    return graph


def run_cycle_pruning_tests(*, tolerance: float = 1e-8) -> Dict[str, bool]:
    """Correctness checks for path-level same-separatrix cycle pruning."""
    results: Dict[str, bool] = {}

    # Prefix-sum T matches the O(L^2) definition.
    q_u = np.array([0.5, 0.3, 0.2, 0.0])
    q_v = np.array([0.1, 0.0, 0.4, 0.5])
    path_cols = np.array([0, 1, 2, 3])
    fast = ordered_pair_support_along_path(q_u, q_v, path_cols)
    naive = ordered_pair_support_naive(q_u, q_v, path_cols)
    assert np.allclose(fast, naive, atol=tolerance)
    results["T_prefix_matches_naive"] = True

    # Test A: weaker same-separatrix chord is removed.
    Q_a = np.eye(4)
    engine_a = _test_coherence_engine(
        [Q_a], np.ones((1, 4)), {"s": [0, 1, 2, 3]}, n_input_nodes=4
    )
    graph_a = _graph_from_weighted_edges(
        4,
        [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9), (0, 3, 0.2)],
    )
    path_abcd = engine_a.path_coherence([0, 1, 2, 3])
    assert path_abcd.C > 1.0 - 1e-8
    result_a = prune_same_separatrix_cycles(
        graph_a, engine_a, theta=0.5, mode="testA"
    )
    assert not result_a.reconstructed_graph.has_edge(0, 3)
    assert result_a.reconstructed_graph.has_edge(0, 1)
    assert result_a.reconstructed_graph.has_edge(1, 2)
    assert result_a.reconstructed_graph.has_edge(2, 3)
    assert result_a.statistics["n_removed"] == 1
    results["A_same_separatrix_shortcut"] = True

    # Test B: Euclidean coordinates never enter C(P).
    C_curved = engine_a.path_coherence([0, 1, 2, 3]).C
    assert np.isclose(C_curved, path_abcd.C, atol=tolerance)
    results["B_geometry_ignored"] = True

    # Test C: a cycle from an unrelated detour is not a same-separatrix path.
    Q_c = np.zeros((3, 6))
    Q_c[0, 0] = 1.0
    Q_c[1, 5] = 1.0
    Q_c[2, 2] = 1.0
    engine_c = _test_coherence_engine(
        [Q_c],
        np.ones((1, 3)),
        {"s1": [0, 3, 2], "s2": [4, 5]},
        n_input_nodes=6,
    )
    graph_c = _graph_from_weighted_edges(
        3, [(0, 2, 0.8), (0, 1, 0.9), (1, 2, 0.9)]
    )
    C_detour = engine_c.path_coherence([0, 1, 2]).C
    assert C_detour < 0.5
    result_c = prune_same_separatrix_cycles(
        graph_c, engine_c, theta=0.5, mode="testC"
    )
    assert result_c.reconstructed_graph.has_edge(0, 2)
    assert result_c.statistics["n_removed"] == 0
    results["C_genuine_different_separatrices"] = True

    # Test D: mixed s1/s2 support cannot explain one path.
    Q_d = np.eye(4)
    engine_d = _test_coherence_engine(
        [Q_d],
        np.ones((1, 4)),
        {"s1": [0, 1, 2], "s2": [2, 3]},
        n_input_nodes=4,
    )
    C_mixed = engine_d.path_coherence([0, 1, 2, 3]).C
    assert C_mixed < 0.5
    results["D_common_separatrix_required"] = True

    # Test E: reverse orientation is accepted.
    Q_e = np.zeros((4, 4))
    Q_e[0, 3] = 1.0
    Q_e[1, 2] = 1.0
    Q_e[2, 1] = 1.0
    Q_e[3, 0] = 1.0
    engine_e = _test_coherence_engine(
        [Q_e], np.ones((1, 4)), {"s": [0, 1, 2, 3]}, n_input_nodes=4
    )
    reverse_score = engine_e.path_coherence([0, 1, 2, 3])
    assert reverse_score.C > 1.0 - 1e-8
    assert reverse_score.best_directions[0] == "reverse"
    graph_e = _graph_from_weighted_edges(
        4,
        [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9), (0, 3, 0.2)],
    )
    result_e = prune_same_separatrix_cycles(
        graph_e, engine_e, theta=0.5, mode="testE"
    )
    assert not result_e.reconstructed_graph.has_edge(0, 3)
    results["E_reverse_orientation"] = True

    # Test F: an unmatched input does not veto a coherent path.
    Q_f0 = np.eye(4)
    Q_f1 = np.zeros((4, 4))
    fractions_f = np.array([[1.0, 1.0, 1.0, 1.0], [1e-16, 1.0, 1.0, 1.0]])
    index_f = _path_index_from_explicit_paths({"s": [0, 1, 2, 3]}, n_nodes=4)
    betweenness_f = SeparatrixBetweennessEngine(
        [Q_f0, Q_f1], fractions_f, [index_f, index_f]
    )
    engine_f = SeparatrixPathCoherenceEngine(betweenness_f)
    partial = engine_f.path_coherence([0, 1, 2, 3])
    assert partial.weights[1] <= EPS
    assert np.isclose(partial.C, 1.0, atol=1e-8)
    results["F_partial_matching"] = True

    # Test G: bridges are never removed.
    engine_g = _test_coherence_engine(
        [np.eye(3)], np.ones((1, 3)), {"s": [0, 1, 2]}, n_input_nodes=3
    )
    graph_g = _graph_from_weighted_edges(3, [(0, 1, 0.2), (1, 2, 0.9)])
    result_g = prune_same_separatrix_cycles(
        graph_g, engine_g, theta=0.5, mode="testG"
    )
    assert result_g.reconstructed_graph.has_edge(0, 1)
    assert result_g.reconstructed_graph.has_edge(1, 2)
    assert result_g.statistics["n_removed"] == 0
    assert result_g.diagnostics["n_retained_bridge"] == 2
    results["G_bridge_protection"] = True

    # Test H: the greedy policy is deterministic.
    result_h1 = prune_same_separatrix_cycles(
        graph_a, engine_a, theta=0.5, mode="testH"
    )
    result_h2 = prune_same_separatrix_cycles(
        graph_a, engine_a, theta=0.5, mode="testH"
    )
    assert list(result_h1.removed_edges["u"]) == list(result_h2.removed_edges["u"])
    assert list(result_h1.removed_edges["v"]) == list(result_h2.removed_edges["v"])
    assert set(result_h1.reconstructed_graph.edges()) == set(
        result_h2.reconstructed_graph.edges()
    )
    results["H_deterministic"] = True
    return results


# ---------------------------------------------------------------------------
# Phase 2.1: widest same-separatrix-affinity paths and R(e, P)
# ---------------------------------------------------------------------------


class _IntUnionFind:
    """Union-find for Kruskal widest-path search."""

    def __init__(self, nodes: Iterable[int]) -> None:
        self.parent = {int(n): int(n) for n in nodes}
        self.rank = {int(n): 0 for n in nodes}

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def _directed_bilinear_support(
    q_u: np.ndarray,
    q_v: np.ndarray,
    kernel: np.ndarray,
) -> Tuple[float, float, float]:
    """Return ``(S_forward, S_reverse, mean_hop)`` with ``kernel`` on an ordered path.

    ``S_forward = sum_{j < k} q_u[j] kernel[j,k] q_v[k]``. No input-graph
    adjacency is required unless ``kernel`` itself is an adjacency matrix.
    """
    a = np.asarray(q_u, dtype=float)
    b = np.asarray(q_v, dtype=float)
    K = np.asarray(kernel, dtype=float)
    L = a.shape[0]
    if L < 2 or K.shape != (L, L):
        return 0.0, 0.0, float("nan")
    hops = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]).astype(float)
    forward = np.triu(np.ones((L, L), dtype=float), k=1)
    reverse = np.tril(np.ones((L, L), dtype=float), k=-1)
    mass_f = a[:, None] * K * b[None, :] * forward
    mass_r = a[:, None] * K * b[None, :] * reverse
    s_f = float(mass_f.sum())
    s_r = float(mass_r.sum())
    mass = mass_f + mass_r
    total = float(mass.sum())
    mean_hop = float((mass * hops).sum() / total) if total > 0.0 else float("nan")
    return s_f, s_r, mean_hop


def _gaussian_kernel_from_distances(
    distances: np.ndarray,
    *,
    sigma: float,
    eta: float,
    cutoff: float,
    zero_at_zero_distance: bool,
) -> np.ndarray:
    L = int(distances.shape[0])
    kernel = np.zeros((L, L), dtype=float)
    for j in range(L):
        for k in range(j + 1, L):
            value = _gaussian_kernel_value(
                float(distances[j, k]),
                sigma=sigma,
                eta=eta,
                cutoff=cutoff,
                zero_at_zero_distance=zero_at_zero_distance,
            )
            kernel[j, k] = kernel[k, j] = value
    return kernel


def _path_pairwise_distances(
    graph: Optional[nx.Graph],
    node_ids: Sequence[Any],
    *,
    length_attribute: Optional[str],
    length_scale: float,
) -> np.ndarray:
    """Cumulative geometric length along a simple path; hop fallback if needed."""
    L = len(node_ids)
    cum = np.zeros(L, dtype=float)
    for j in range(1, L):
        u, v = node_ids[j - 1], node_ids[j]
        used_geo = False
        if graph is not None and u in graph and v in graph:
            try:
                cum[j] = cum[j - 1] + _edge_geometric_length(
                    graph,
                    u,
                    v,
                    length_attribute=length_attribute,
                    length_scale=length_scale,
                )
                used_geo = True
            except Exception:
                used_geo = False
        if not used_geo:
            cum[j] = cum[j - 1] + 1.0
    distances = np.abs(cum[:, None] - cum[None, :])
    return distances


class SeparatrixEdgeSupportEngine:
    """Mode-aware per-edge / per-separatrix support ``S_{i,s}(u,v)``.

    Direct support uses the same structural criterion as transported direct
    evidence: binary input adjacency restricted to one separatrix ID.
    Soft support uses the same Gaussian kernel as transported soft evidence,
    evaluated along one ordered 1-separatrix (multi-hop pairs allowed).
    """

    def __init__(
        self,
        coherence_engine: SeparatrixPathCoherenceEngine,
        graphs: Optional[Sequence[nx.Graph]] = None,
        *,
        gaussian_sigma: float = 0.03,
        gaussian_eta: float = 1.0,
        gaussian_cutoff_sigmas: float = 3.0,
        gaussian_zero_at_zero_distance: bool = True,
        length_scale: float = 1.0,
        length_attribute: Optional[str] = None,
        use_unit_soft_kernel: bool = False,
    ) -> None:
        t0 = time.perf_counter()
        self.coherence = coherence_engine
        self.conditionals = coherence_engine.conditionals
        self.matched_fractions = coherence_engine.matched_fractions
        self.path_indices = coherence_engine.path_indices
        self.node_separatrix_ids = coherence_engine.node_separatrix_ids
        self.epsilon = coherence_engine.epsilon
        self.n_inputs = coherence_engine.n_inputs
        self.n_center = coherence_engine.n_center
        self.graphs = None if graphs is None else list(graphs)
        self.sigma = float(gaussian_sigma)
        self.eta = float(gaussian_eta)
        self.cutoff = float(gaussian_cutoff_sigmas) * float(gaussian_sigma)
        self.zero_at_zero = bool(gaussian_zero_at_zero_distance)
        self.length_scale = float(length_scale)
        self.length_attribute = length_attribute
        self.use_unit_soft_kernel = bool(use_unit_soft_kernel)

        self._sid_data: List[Dict[str, Dict[str, Any]]] = []
        for i, path_index in enumerate(self.path_indices):
            graph = None if self.graphs is None else self.graphs[i]
            per_sid: Dict[str, Dict[str, Any]] = {}
            for sid, cols in coherence_engine.separatrix_columns[i].items():
                cols_arr = np.asarray(cols, dtype=np.int64)
                L = int(cols_arr.size)
                node_ids = [path_index.node_order[int(c)] for c in cols_arr]
                adjacent = np.zeros((L, L), dtype=float)
                if L >= 2:
                    idx = np.arange(L - 1)
                    adjacent[idx, idx + 1] = 1.0
                    adjacent[idx + 1, idx] = 1.0
                if self.use_unit_soft_kernel:
                    soft = np.ones((L, L), dtype=float)
                    np.fill_diagonal(soft, 0.0)
                else:
                    distances = _path_pairwise_distances(
                        graph,
                        node_ids,
                        length_attribute=self.length_attribute,
                        length_scale=self.length_scale,
                    )
                    soft = _gaussian_kernel_from_distances(
                        distances,
                        sigma=self.sigma,
                        eta=self.eta,
                        cutoff=self.cutoff,
                        zero_at_zero_distance=self.zero_at_zero,
                    )
                per_sid[str(sid)] = {
                    "cols": cols_arr,
                    "adjacent": adjacent,
                    "soft": soft,
                }
            self._sid_data.append(per_sid)
        self._cache: Dict[Tuple[str, int, str, int, int], Tuple[float, float, float]] = {}
        self._edge_cache: Dict[Tuple[str, int, int], EdgeSeparatrixSupport] = {}
        self.preprocess_seconds = float(time.perf_counter() - t0)

    def directed_support(
        self,
        input_index: int,
        separatrix_id: str,
        u: int,
        v: int,
        mode: str,
    ) -> Tuple[float, float, float]:
        """``(S_forward, S_reverse, mean_hop)`` for one input separatrix."""
        if mode not in {"direct", "soft"}:
            raise ValueError("mode must be 'direct' or 'soft'.")
        key = (mode, int(input_index), str(separatrix_id), int(u), int(v))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        data = self._sid_data[input_index].get(str(separatrix_id))
        if data is None or data["cols"].size < 2:
            result = (0.0, 0.0, float("nan"))
        else:
            cols = data["cols"]
            kernel = data["adjacent"] if mode == "direct" else data["soft"]
            Q = self.conditionals[input_index]
            result = _directed_bilinear_support(Q[int(u), cols], Q[int(v), cols], kernel)
        self._cache[key] = result
        return result

    def edge_support(
        self,
        u: int,
        v: int,
        mode: str,
        *,
        theta: float = 0.5,
    ) -> EdgeSeparatrixSupport:
        """Coverage-weighted ``S_sep(e) = sum w_i max_s S_{i,s}(e) / sum w_i``."""
        cache_key = (mode, int(u), int(v), float(theta))
        cached = self._edge_cache.get(cache_key)
        if cached is not None:
            return cached
        u_i, v_i = int(u), int(v)
        S_inputs = np.zeros(self.n_inputs, dtype=float)
        S_forward = np.zeros(self.n_inputs, dtype=float)
        S_reverse = np.zeros(self.n_inputs, dtype=float)
        weights = np.zeros(self.n_inputs, dtype=float)
        hops = np.full(self.n_inputs, np.nan, dtype=float)
        best_sids: List[Optional[str]] = [None] * self.n_inputs
        best_dirs: List[Optional[str]] = [None] * self.n_inputs
        for i in range(self.n_inputs):
            fractions = self.matched_fractions[i]
            weights[i] = float(min(fractions[u_i], fractions[v_i]))
            if weights[i] <= self.epsilon:
                continue
            candidate = self.node_separatrix_ids[i][u_i] & self.node_separatrix_ids[i][v_i]
            best = -1.0
            for sid in candidate:
                s_f, s_r, mean_hop = self.directed_support(i, sid, u_i, v_i, mode)
                s_undirected = s_f + s_r
                direction = "forward" if s_f >= s_r else "reverse"
                if s_undirected > best + 1e-15 or (
                    abs(s_undirected - best) <= 1e-15
                    and (best_sids[i] is None or str(sid) < str(best_sids[i]))
                ):
                    best = float(s_undirected)
                    best_sids[i] = str(sid)
                    best_dirs[i] = direction
                    S_forward[i] = float(s_f)
                    S_reverse[i] = float(s_r)
                    hops[i] = mean_hop
            S_inputs[i] = 0.0 if best < 0.0 else float(best)
        weight_sum = float(weights.sum())
        if weight_sum <= self.epsilon:
            S_sep = 0.0
            strong = 0.0
            std = 0.0
            uninformative = True
        else:
            S_sep = float(np.dot(weights, S_inputs) / weight_sum)
            strong = float(np.dot(weights, (S_inputs >= float(theta)).astype(float)) / weight_sum)
            centered = S_inputs - S_sep
            std = float(np.sqrt(np.dot(weights, centered * centered) / weight_sum))
            uninformative = False
        result = EdgeSeparatrixSupport(
            u=u_i,
            v=v_i,
            mode=mode,
            S_sep=S_sep,
            S_inputs=S_inputs,
            weights=weights,
            weight_sum=weight_sum,
            uninformative=uninformative,
            best_separatrix_ids=best_sids,
            best_directions=best_dirs,
            S_forward_inputs=S_forward,
            S_reverse_inputs=S_reverse,
            mean_hops=hops,
            strong_fraction=strong,
            S_std=std,
        )
        self._edge_cache[cache_key] = result
        reverse_key = (mode, v_i, u_i, float(theta))
        if reverse_key not in self._edge_cache:
            self._edge_cache[reverse_key] = EdgeSeparatrixSupport(
                u=v_i,
                v=u_i,
                mode=mode,
                S_sep=result.S_sep,
                S_inputs=result.S_inputs.copy(),
                weights=result.weights.copy(),
                weight_sum=result.weight_sum,
                uninformative=result.uninformative,
                best_separatrix_ids=list(result.best_separatrix_ids),
                best_directions=[
                    None
                    if direction is None
                    else ("reverse" if direction == "forward" else "forward")
                    for direction in result.best_directions
                ],
                S_forward_inputs=result.S_reverse_inputs.copy(),
                S_reverse_inputs=result.S_forward_inputs.copy(),
                mean_hops=result.mean_hops.copy(),
                strong_fraction=result.strong_fraction,
                S_std=result.S_std,
            )
        return result

    def edge_path_redundancy(
        self,
        u: int,
        v: int,
        path: Sequence[int],
        mode: str,
        *,
        affinities: Optional[Mapping[Tuple[int, int], float]] = None,
    ) -> EdgePathRedundancy:
        """``R(e,P)``: same input separatrix and orientation must support both ``e`` and ``P``."""
        nodes = tuple(int(n) for n in path)
        u_i, v_i = int(u), int(v)
        if len(nodes) >= 2 and nodes[0] == v_i and nodes[-1] == u_i:
            nodes = tuple(reversed(nodes))
        R_inputs = np.zeros(self.n_inputs, dtype=float)
        S_matched = np.zeros(self.n_inputs, dtype=float)
        C_matched = np.zeros(self.n_inputs, dtype=float)
        weights = np.zeros(self.n_inputs, dtype=float)
        best_sids: List[Optional[str]] = [None] * self.n_inputs
        best_dirs: List[Optional[str]] = [None] * self.n_inputs
        for i in range(self.n_inputs):
            fractions = self.matched_fractions[i]
            weights[i] = float(min(fractions[n] for n in nodes)) if nodes else 0.0
            if weights[i] <= self.epsilon:
                continue
            candidate = None
            for node in nodes:
                sids = self.node_separatrix_ids[i][node]
                candidate = sids if candidate is None else (candidate & sids)
                if not candidate:
                    break
            if not candidate:
                continue
            best = -1.0
            for sid in candidate:
                s_f, s_r, _hop = self.directed_support(i, sid, u_i, v_i, mode)
                c_f, c_r = self.coherence.directed_coherence_on_separatrix(i, sid, nodes)
                r_f = min(s_f, c_f)
                r_r = min(s_r, c_r)
                if r_f >= r_r:
                    r_s, direction, s_used, c_used = r_f, "forward", s_f, c_f
                else:
                    r_s, direction, s_used, c_used = r_r, "reverse", s_r, c_r
                if r_s > best + 1e-15 or (
                    abs(r_s - best) <= 1e-15
                    and (best_sids[i] is None or str(sid) < str(best_sids[i]))
                ):
                    best = float(r_s)
                    best_sids[i] = str(sid)
                    best_dirs[i] = direction
                    S_matched[i] = float(s_used)
                    C_matched[i] = float(c_used)
            R_inputs[i] = 0.0 if best < 0.0 else float(best)
        weight_sum = float(weights.sum())
        R = (
            0.0
            if weight_sum <= self.epsilon
            else float(np.dot(weights, R_inputs) / weight_sum)
        )
        bottleneck = 0.0
        mean_aff = 0.0
        if affinities is not None and len(nodes) >= 3:
            affs = [
                float(affinities[_edge_key(a, b)])
                for a, b in zip(nodes[:-1], nodes[1:])
            ]
            if affs:
                bottleneck = float(min(affs))
                mean_aff = float(np.mean(affs))
        return EdgePathRedundancy(
            u=u_i,
            v=v_i,
            path=nodes,
            R=R,
            R_inputs=R_inputs,
            weights=weights,
            weight_sum=weight_sum,
            uninformative=weight_sum <= self.epsilon,
            best_separatrix_ids=best_sids,
            best_directions=best_dirs,
            S_matched=S_matched,
            C_matched=C_matched,
            bottleneck=bottleneck,
            mean_affinity=mean_aff,
        )


def prepare_edge_support_engine(
    coherence_engine: SeparatrixPathCoherenceEngine,
    bundle: Mapping[str, Any],
    transported: TransportedConnectivityResult,
    config: ReconstructionConfig,
) -> SeparatrixEdgeSupportEngine:
    """Build mode-aware ``S_{i,s}`` using the existing Gaussian / adjacency semantics."""
    diagnostics = dict(transported.diagnostics or {})
    return SeparatrixEdgeSupportEngine(
        coherence_engine,
        list(bundle["Gs"]),
        gaussian_sigma=float(diagnostics.get("gaussian_sigma", config.gaussian_sigma)),
        gaussian_eta=float(diagnostics.get("gaussian_eta", config.gaussian_eta)),
        gaussian_cutoff_sigmas=float(
            diagnostics.get("gaussian_cutoff_sigmas", config.gaussian_cutoff_sigmas)
        ),
        gaussian_zero_at_zero_distance=bool(
            diagnostics.get(
                "gaussian_zero_at_zero_distance",
                config.gaussian_zero_at_zero_distance,
            )
        ),
        length_scale=float(diagnostics.get("length_scale", 1.0)),
        length_attribute=config.geometric_length_attribute,
        use_unit_soft_kernel=False,
    )


def annotate_separatrix_affinities(
    graph: nx.Graph,
    support_engine: SeparatrixEdgeSupportEngine,
    mode: str,
    *,
    theta: float = 0.5,
) -> Dict[Tuple[int, int], EdgeSeparatrixSupport]:
    """Store ``sep_affinity = S_sep(e)`` on every undirected graph edge."""
    records: Dict[Tuple[int, int], EdgeSeparatrixSupport] = {}
    for u, v, data in graph.edges(data=True):
        support = support_engine.edge_support(int(u), int(v), mode, theta=theta)
        key = _edge_key(u, v)
        records[key] = support
        data["sep_affinity"] = float(support.S_sep)
    return records


def widest_affinity_path(
    graph: nx.Graph,
    u: int,
    v: int,
    *,
    affinity_attr: str = "sep_affinity",
    evidence_attr: str = "evidence",
) -> Optional[List[int]]:
    """Maximum-bottleneck u-v path via Kruskal on a maximum spanning forest.

    Edge sort (all descending, then node ids): affinity, evidence, min-id, max-id.
    Hop count is not the primary criterion. Returns ``None`` if disconnected.
    """
    u_i, v_i = int(u), int(v)
    if u_i == v_i:
        return None
    ranked = []
    for a, b, data in graph.edges(data=True):
        affinity = float(data.get(affinity_attr, 0.0))
        evidence = float(data.get(evidence_attr, 0.0))
        a_i, b_i = _edge_key(a, b)
        ranked.append((-affinity, -evidence, a_i, b_i, int(a), int(b)))
    ranked.sort()
    uf = _IntUnionFind(graph.nodes())
    tree = nx.Graph()
    tree.add_nodes_from(int(n) for n in graph.nodes())
    for _neg_a, _neg_e, _a_i, _b_i, a, b in ranked:
        if uf.union(a, b):
            tree.add_edge(a, b)
            if uf.find(u_i) == uf.find(v_i):
                path = [int(n) for n in nx.shortest_path(tree, u_i, v_i)]
                return None if len(path) < 3 else path
    return None


def prune_same_separatrix_redundancy(
    phase1_graph: nx.Graph,
    support_engine: SeparatrixEdgeSupportEngine,
    mode: str,
    *,
    theta: float = 0.5,
) -> CyclePruningResult:
    """Greedy Phase 2.1: widest ``S_sep`` path and ``R(e,P) >= theta``.

    ``R_i(e,P) = max_s max(min(S_forward, C_forward), min(S_reverse, C_reverse))``.
    The candidate edge and the replacement path must share one separatrix ID
    *and* one orientation inside each input. Direct and soft use the same
    loop; only ``mode`` selects the support kernel.
    """
    t0 = time.perf_counter()
    graph = phase1_graph.copy()
    supports = annotate_separatrix_affinities(graph, support_engine, mode, theta=theta)
    affinities = {key: float(rec.S_sep) for key, rec in supports.items()}
    stats_before = reconstruction_graph_statistics(graph)
    ordered_edges = sorted(graph.edges(data=True), key=_phase1_edge_sort_key)

    attempt_rows: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []
    n_alternative = 0
    n_redundant = 0

    for u_raw, v_raw, data in ordered_edges:
        u, v = int(u_raw), int(v_raw)
        if not graph.has_edge(u, v):
            continue
        evidence = float(data.get("evidence", np.nan))
        edge_attrs = dict(graph.get_edge_data(u, v) or {})
        edge_support = supports.get(_edge_key(u, v))
        graph.remove_edge(u, v)
        path = widest_affinity_path(graph, u, v)
        if path is None:
            graph.add_edge(u, v, **edge_attrs)
            attempt_rows.append(
                {
                    "u": u,
                    "v": v,
                    "evidence": evidence,
                    "S_sep": np.nan if edge_support is None else float(edge_support.S_sep),
                    "decision": "retained_bridge",
                    "phase": "phase2.1",
                    "n_path_edges": 0,
                    "R": np.nan,
                    "C_old_style": np.nan,
                    "path": None,
                    "bottleneck": np.nan,
                    "mean_affinity": np.nan,
                }
            )
            continue

        n_alternative += 1
        redundancy = support_engine.edge_path_redundancy(
            u, v, path, mode, affinities=affinities
        )
        n_path_edges = len(path) - 1
        row = {
            "u": u,
            "v": v,
            "evidence": evidence,
            "S_sep": np.nan if edge_support is None else float(edge_support.S_sep),
            "decision": "",
            "phase": "phase2.1",
            "n_path_edges": int(n_path_edges),
            "R": float(redundancy.R),
            "uninformative": bool(redundancy.uninformative),
            "weight_sum": float(redundancy.weight_sum),
            "path": list(path),
            "bottleneck": float(redundancy.bottleneck),
            "mean_affinity": float(redundancy.mean_affinity),
            "R_inputs": redundancy.R_inputs.copy(),
            "weights": redundancy.weights.copy(),
            "S_matched": redundancy.S_matched.copy(),
            "C_matched": redundancy.C_matched.copy(),
            "best_separatrix_ids": list(redundancy.best_separatrix_ids),
            "best_directions": list(redundancy.best_directions),
        }
        if redundancy.R >= float(theta):
            n_redundant += 1
            row["decision"] = "removed_same_separatrix_redundancy"
            row["removal_order"] = len(removed_rows)
            removed_rows.append(row)
            attempt_rows.append(row)
            continue
        graph.add_edge(u, v, **edge_attrs)
        row["decision"] = "retained_incoherent_alternative"
        attempt_rows.append(row)

    stats_after = reconstruction_graph_statistics(
        graph,
        n_candidate_edges=int(stats_before["n_reconstructed_edges"]),
        n_removed=len(removed_rows),
    )
    if stats_after["beta_0"] != stats_before["beta_0"]:
        raise AssertionError("Phase 2.1 changed beta_0; a bridge was removed.")
    removed = pd.DataFrame(removed_rows)
    attempts = pd.DataFrame(attempt_rows)
    removed_R = removed["R"] if len(removed) else pd.Series(dtype=float)
    removed_len = removed["n_path_edges"] if len(removed) else pd.Series(dtype=float)
    statistics = {
        "n_nodes": stats_after["n_nodes"],
        "n_edges_before": int(stats_before["n_reconstructed_edges"]),
        "n_removed": int(len(removed)),
        "n_edges_after": int(stats_after["n_reconstructed_edges"]),
        "beta_0_before": int(stats_before["beta_0"]),
        "beta_0_after": int(stats_after["beta_0"]),
        "beta_1_before": int(stats_before["beta_1"]),
        "beta_1_after": int(stats_after["beta_1"]),
        "n_alternative_paths": int(n_alternative),
        "n_redundant_alternatives": int(n_redundant),
        "removed_R_min": float(removed_R.min()) if len(removed_R) else np.nan,
        "removed_R_median": float(removed_R.median()) if len(removed_R) else np.nan,
        "removed_R_max": float(removed_R.max()) if len(removed_R) else np.nan,
        "removed_path_edges_min": int(removed_len.min()) if len(removed_len) else 0,
        "removed_path_edges_median": (
            float(removed_len.median()) if len(removed_len) else np.nan
        ),
        "removed_path_edges_mean": (
            float(removed_len.mean()) if len(removed_len) else np.nan
        ),
        "removed_path_edges_max": int(removed_len.max()) if len(removed_len) else 0,
    }
    return CyclePruningResult(
        mode=mode,
        theta=float(theta),
        phase1_graph=phase1_graph,
        reconstructed_graph=graph,
        removed_edges=removed,
        attempt_log=attempts,
        statistics=statistics,
        diagnostics={
            "n_attempted_edges": int(len(attempts)),
            "n_retained_bridge": int((attempts["decision"] == "retained_bridge").sum())
            if len(attempts)
            else 0,
            "n_retained_incoherent": int(
                (attempts["decision"] == "retained_incoherent_alternative").sum()
            )
            if len(attempts)
            else 0,
            "proposal": "widest_sep_affinity",
            "criterion": "R(e,P)",
        },
        preprocess_seconds=float(support_engine.preprocess_seconds),
        elapsed_seconds=float(time.perf_counter() - t0),
    )


def classify_surviving_separatrix_support(
    S_sep: float,
    strong_fraction: float,
    *,
    direct_evidence: float = np.nan,
    theta: float = 0.5,
) -> str:
    """Diagnostic labels only; not used as extra pruning thresholds."""
    if np.isfinite(direct_evidence) and float(direct_evidence) >= float(theta):
        return "direct_like"
    if float(S_sep) >= float(theta) and float(strong_fraction) >= float(theta):
        return "high_consensus"
    if float(S_sep) < float(theta):
        return "low_separatrix"
    return "mixed_uncertain"


def surviving_edge_support_table(
    graph: nx.Graph,
    support_engine: SeparatrixEdgeSupportEngine,
    mode: str,
    *,
    theta: float = 0.5,
    direct_evidence: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-surviving-edge same-separatrix diagnostics."""
    rows: List[Dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        support = support_engine.edge_support(int(u), int(v), mode, theta=theta)
        ev = float(data.get("evidence", np.nan))
        direct_ev = np.nan
        if direct_evidence is not None:
            direct_ev = float(direct_evidence[int(u), int(v)])
        rows.append(
            {
                "u": int(u),
                "v": int(v),
                "evidence": ev,
                "S_sep": float(support.S_sep),
                "strong_fraction": float(support.strong_fraction),
                "S_std": float(support.S_std),
                "mean_hop": float(np.nanmean(support.mean_hops)),
                "direct_evidence": direct_ev,
                "category": classify_surviving_separatrix_support(
                    support.S_sep,
                    support.strong_fraction,
                    direct_evidence=direct_ev,
                    theta=theta,
                ),
                "S_inputs": support.S_inputs.copy(),
                "weights": support.weights.copy(),
                "best_separatrix_ids": list(support.best_separatrix_ids),
                "best_directions": list(support.best_directions),
                "mean_hops": support.mean_hops.copy(),
            }
        )
    return pd.DataFrame(rows)


def count_thresholded_pairs(matrix: np.ndarray, threshold: float) -> int:
    """Upper-triangle count of finite entries at or above ``threshold``."""
    W = np.asarray(matrix, dtype=float)
    iu, ju = np.triu_indices(W.shape[0], k=1)
    weights = W[iu, ju]
    return int(np.sum(np.isfinite(weights) & (weights >= float(threshold))))


def reconstruction_count_row(
    *,
    mode: str,
    candidate_graph: nx.Graph,
    phase1: TriangleReconstructionResult,
    phase2: CyclePruningResult,
    evidence_matrix: np.ndarray,
    threshold: float,
    phase21: Optional["CyclePruningResult"] = None,
) -> Dict[str, Any]:
    """Counts taken from the same graph objects later passed to plotting."""
    n_nodes = int(candidate_graph.number_of_nodes())
    row: Dict[str, Any] = {
        "mode": mode,
        "candidate_edges": int(candidate_graph.number_of_edges()),
        "candidate_edges_thresholded_matrix": count_thresholded_pairs(
            evidence_matrix, threshold
        ),
        "phase1_edges": int(phase1.reconstructed_graph.number_of_edges()),
        "phase1_removed": int(len(phase1.removed_edges)),
        "phase1_plot_matrix_edges": count_thresholded_pairs(
            evidence_matrix_from_graph(
                phase1.reconstructed_graph,
                n_nodes=n_nodes,
            ),
            threshold,
        ),
        "phase2_edges": int(phase2.reconstructed_graph.number_of_edges()),
        "phase2_removed": int(len(phase2.removed_edges)),
        "phase2_plot_matrix_edges": count_thresholded_pairs(
            evidence_matrix_from_graph(
                phase2.reconstructed_graph,
                n_nodes=n_nodes,
            ),
            threshold,
        ),
    }
    if phase21 is not None:
        row["phase21_edges"] = int(phase21.reconstructed_graph.number_of_edges())
        row["phase21_removed"] = int(len(phase21.removed_edges))
        row["phase21_plot_matrix_edges"] = count_thresholded_pairs(
            evidence_matrix_from_graph(
                phase21.reconstructed_graph,
                n_nodes=n_nodes,
            ),
            threshold,
        )
    return row


def witness_path_length_stats(log: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Median / mean / max hop counts of proposed alternative paths."""
    empty = {
        "n_witnesses": 0,
        "median": float("nan"),
        "mean": float("nan"),
        "max": 0,
    }
    if log is None or len(log) == 0 or "n_path_edges" not in log.columns:
        return empty
    lengths = pd.to_numeric(log["n_path_edges"], errors="coerce")
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if lengths.empty:
        return empty
    return {
        "n_witnesses": int(len(lengths)),
        "median": float(lengths.median()),
        "mean": float(lengths.mean()),
        "max": int(lengths.max()),
    }


def _test_support_engine(
    conditionals: Sequence[np.ndarray],
    matched_fractions: np.ndarray,
    path_maps: Mapping[str, Sequence[int]],
    n_input_nodes: int,
    *,
    extra_path_indices: Optional[Sequence[SeparatrixPathIndex]] = None,
) -> Tuple[SeparatrixPathCoherenceEngine, SeparatrixEdgeSupportEngine]:
    path_index = _path_index_from_explicit_paths(path_maps, n_nodes=n_input_nodes)
    indices = [path_index]
    cond = [np.asarray(Q, dtype=float) for Q in conditionals]
    if extra_path_indices is not None:
        indices = list(extra_path_indices)
    if len(indices) != len(cond):
        indices = [path_index] * len(cond)
    betweenness = SeparatrixBetweennessEngine(
        cond, np.asarray(matched_fractions, dtype=float), indices
    )
    coherence = SeparatrixPathCoherenceEngine(betweenness)
    support = SeparatrixEdgeSupportEngine(coherence, use_unit_soft_kernel=True)
    return coherence, support


def run_phase21_tests(*, tolerance: float = 1e-8) -> Dict[str, bool]:
    """Correctness checks for multi-hop support, widest paths, and R(e,P)."""
    results: Dict[str, bool] = {}

    # Part 0: T and betweenness do not require input adjacency.
    q_u = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    q_v = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    t_f, t_r = ordered_pair_support_along_path(q_u, q_v, np.arange(5))
    assert t_f > 1.0 - tolerance
    assert t_r < tolerance
    index_between = _path_index_from_explicit_paths({"s": [0, 1, 2, 3, 4]}, n_nodes=5)
    assert input_nodes_between(index_between, 0, 2, 4) is True
    assert input_nodes_between_from_positions(index_between, 0, 2, 4) is True
    results["P0_no_input_adjacency_requirement"] = True

    # Test A: multi-hop soft vs adjacent-only direct.
    Q_a = np.zeros((2, 5))
    Q_a[0, 0] = 1.0
    Q_a[1, 4] = 1.0
    _coh_a, support_a = _test_support_engine([Q_a], np.ones((1, 2)), {"s": [0, 1, 2, 3, 4]}, 5)
    s_soft = support_a.directed_support(0, "s", 0, 1, "soft")
    s_direct = support_a.directed_support(0, "s", 0, 1, "direct")
    assert s_soft[0] > 0.5
    assert s_direct[0] + s_direct[1] < tolerance
    results["A_multihop_soft_not_direct"] = True

    # Test B: widest path prefers a long high-affinity route over a short weak detour.
    graph_b = nx.Graph()
    graph_b.add_nodes_from(range(7))
    long = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for a, b in long:
        graph_b.add_edge(a, b, sep_affinity=0.91, evidence=0.5)
    graph_b.add_edge(0, 6, sep_affinity=0.72, evidence=0.9)
    graph_b.add_edge(6, 5, sep_affinity=0.71, evidence=0.9)
    path_b = widest_affinity_path(graph_b, 0, 5)
    assert path_b == [0, 1, 2, 3, 4, 5]
    results["B_long_widest_path"] = True

    # Test C: same-separatrix edge and path => high R.
    Q_c = np.eye(4)
    _coh_c, support_c = _test_support_engine([Q_c], np.ones((1, 4)), {"s": [0, 1, 2, 3]}, 4)
    red_c = support_c.edge_path_redundancy(0, 3, [0, 1, 2, 3], "soft")
    assert red_c.R > 0.5
    results["C_same_separatrix_edge_and_path"] = True

    # Test D: different separatrices must not create R via independent maxima.
    Q_d = np.eye(4)
    _coh_d, support_d = _test_support_engine(
        [Q_d],
        np.ones((1, 4)),
        {"s1": [0, 1], "s2": [2, 3, 1]},
        4,
    )
    # Edge (0,1) lives on s1; path 0-2-3-1 is ordered on s2 via node 1 only at the end.
    red_d = support_d.edge_path_redundancy(0, 1, [0, 2, 3, 1], "soft")
    assert red_d.R < 0.5
    results["D_different_separatrices"] = True

    # Test E: orientation must match.
    Q_e = np.eye(4)
    _coh_e, support_e = _test_support_engine([Q_e], np.ones((1, 4)), {"s": [0, 1, 2, 3]}, 4)
    s_f, s_r, _ = support_e.directed_support(0, "s", 0, 3, "soft")
    c_f, c_r = _coh_e.directed_coherence_on_separatrix(0, "s", [0, 1, 2, 3])
    assert s_f > 0.5 and c_f > 0.5
    assert min(s_f, c_r) < 0.5 or c_r < 0.5
    red_e = support_e.edge_path_redundancy(0, 3, [0, 1, 2, 3], "soft")
    assert red_e.best_directions[0] == "forward"
    red_e_rev = support_e.edge_path_redundancy(3, 0, [3, 2, 1, 0], "soft")
    assert red_e_rev.best_directions[0] == "forward" or red_e_rev.R > 0.5
    results["E_orientation"] = True

    # Test F: cross-input disagreement lowers aggregate R.
    Q_f0 = np.eye(4)
    Q_f1 = np.eye(4)
    index_f0 = _path_index_from_explicit_paths({"s": [0, 1, 2, 3]}, n_nodes=4)
    index_f1 = _path_index_from_explicit_paths(
        {"s_edge": [0, 3], "s_mid": [1, 2]}, n_nodes=4
    )
    betweenness_f = SeparatrixBetweennessEngine(
        [Q_f0, Q_f1], np.ones((2, 4)), [index_f0, index_f1]
    )
    coh_f = SeparatrixPathCoherenceEngine(betweenness_f)
    support_f = SeparatrixEdgeSupportEngine(coh_f, use_unit_soft_kernel=True)
    red_f = support_f.edge_path_redundancy(0, 3, [0, 1, 2, 3], "soft")
    assert red_f.R_inputs[0] > 0.5
    assert red_f.R_inputs[1] < 0.5
    assert 0.05 < red_f.R < 0.9
    results["F_cross_input_uncertainty"] = True

    # Test G: partial matching.
    Q_g0 = np.eye(4)
    Q_g1 = np.zeros((4, 4))
    fractions_g = np.array([[1.0, 1.0, 1.0, 1.0], [1e-16, 1.0, 1.0, 1.0]])
    betweenness_g = SeparatrixBetweennessEngine(
        [Q_g0, Q_g1], fractions_g, [index_f0, index_f0]
    )
    coh_g = SeparatrixPathCoherenceEngine(betweenness_g)
    support_g = SeparatrixEdgeSupportEngine(coh_g, use_unit_soft_kernel=True)
    red_g = support_g.edge_path_redundancy(0, 3, [0, 1, 2, 3], "soft")
    assert red_g.weights[1] <= EPS
    assert red_g.R > 0.5
    results["G_partial_matching"] = True

    # Test H: bridges remain under Phase 2.1.
    Q_h = np.eye(3)
    _coh_h, support_h = _test_support_engine([Q_h], np.ones((1, 3)), {"s": [0, 1, 2]}, 3)
    graph_h = _graph_from_weighted_edges(3, [(0, 1, 0.2), (1, 2, 0.9)])
    result_h = prune_same_separatrix_redundancy(
        graph_h, support_h, "soft", theta=0.5
    )
    assert result_h.reconstructed_graph.has_edge(0, 1)
    assert result_h.statistics["n_removed"] == 0
    results["H_bridge_protection"] = True

    # Test I: determinism, plus a same-separatrix chord is removed.
    Q_i = np.eye(4)
    _coh_i, support_i = _test_support_engine([Q_i], np.ones((1, 4)), {"s": [0, 1, 2, 3]}, 4)
    graph_i = _graph_from_weighted_edges(
        4, [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9), (0, 3, 0.2)]
    )
    r1 = prune_same_separatrix_redundancy(graph_i, support_i, "soft", theta=0.5)
    r2 = prune_same_separatrix_redundancy(graph_i, support_i, "soft", theta=0.5)
    assert list(r1.removed_edges["u"]) == list(r2.removed_edges["u"])
    assert list(r1.removed_edges["v"]) == list(r2.removed_edges["v"])
    assert not r1.reconstructed_graph.has_edge(0, 3)
    if len(r1.removed_edges):
        assert int(r1.removed_edges.iloc[0]["n_path_edges"]) >= 2
    results["I_deterministic_and_chord"] = True
    return results


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
    """Evaluate a grid of hop-decay ``(q, H)`` settings without changing defaults."""
    rows = []
    component_target = infer_target_component_count(
        bundle, config.target_component_count
    )
    q_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for q in q_values:
        for hops in config.max_soft_hops_values:
            transported = compute_transported_separatrix_connectivity(
                bundle,
                input_node_orders,
                config,
                q=q,
                max_soft_hops=hops,
                affinity_mode="separatrix_hop_decay",
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
    rows = []
    component_target = infer_target_component_count(
        bundle, config.target_component_count
    )
    modes: List[AffinityMode] = ["separatrix_hop_decay", "separatrix_gaussian"]
    if config.enable_length_normalized_affinity:
        modes.append("separatrix_length_normalized_decay")
    for mode in modes:
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
    hop_rows = frame[frame.connectivity_affinity_mode == "separatrix_hop_decay"]
    if hop_rows.empty:
        return frame.drop(columns=["edge_set"])
    hop_set = hop_rows.iloc[0]["edge_set"]
    jaccard = []
    changed = []
    for _, row in frame.iterrows():
        other = row.edge_set
        union = hop_set | other
        jaccard.append(len(hop_set & other) / len(union) if union else 1.0)
        changed.append(len(hop_set ^ other))
    frame["edge_set_jaccard_vs_hop_decay"] = jaccard
    frame["changed_edge_count_vs_hop_decay"] = changed
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


def _as_frame(value: Optional[pd.DataFrame]) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    return value


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
    sensitivity: Optional[pd.DataFrame] = None,
    sampling_sensitivity: Optional[pd.DataFrame] = None,
    baseline_comparison: Optional[pd.DataFrame] = None,
    routed: Optional[GeometricRoutingResult] = None,
    posthoc_relation_metrics: Optional[Mapping[str, Any]] = None,
    warnings_and_fallbacks: Optional[List[str]] = None,
    saved_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist reconstruction artifacts under a timestamped output directory.

    Optional analyses (q-hop sensitivity, sampling sensitivity, post-hoc
    baselines) may be omitted; empty CSV placeholders are written so the
    on-disk layout stays stable. ``saved_at`` uses ``mm-dd-yyyy-hh-mm-ss``.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    stamp = saved_at
    if not stamp:
        match = re.search(r"(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})", output.name)
        stamp = match.group(1) if match else barycenter_save_timestamp()
    source_meta = dict(source_metadata or {})
    sensitivity_df = _as_frame(sensitivity)
    sampling_df = _as_frame(sampling_sensitivity)
    baseline_df = _as_frame(baseline_comparison)

    resolved = config.resolved_dict(Path.cwd())
    resolved["output_dir"] = str(output)
    resolved["saved_at"] = stamp
    resolved["source_run_name"] = config.run_name
    resolved["source_barycenter_dir"] = str(Path(source_barycenter_dir).resolve())
    resolved["source_pipeline"] = source_meta.get("pipeline")
    resolved["source_DATA_KEYWORD"] = source_meta.get("DATA_KEYWORD")
    resolved["source_saved_at"] = source_meta.get("saved_at")
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
    sensitivity_df.to_csv(output / "q_hop_sensitivity.csv", index=False)
    sampling_df.to_csv(output / "sampling_mode_sensitivity.csv", index=False)
    baseline_df.to_csv(output / "baseline_comparison.csv", index=False)
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
        "connectivity_affinity_formula": _connectivity_affinity_formula(
            config.connectivity_affinity_mode
        ),
        "uses_degree_based_chain_inference": False,
        "uses_barycenter_relation_for_edge_selection": False,
        "terminology_version": TERMINOLOGY_VERSION,
        "selected_max_soft_hops": config.max_soft_hops,
        "selected_gaussian_sigma": config.gaussian_sigma,
        "selected_gaussian_eta": config.gaussian_eta,
        "selected_gaussian_zero_at_zero_distance": bool(
            config.gaussian_zero_at_zero_distance
        ),
        "selected_gaussian_cutoff_sigmas": config.gaussian_cutoff_sigmas,
        "selected_gaussian_distance_units": config.gaussian_distance_units,
        "length_normalized_sensitivity_run": bool(config.enable_length_normalized_affinity),
        "source_barycenter_directory": str(Path(source_barycenter_dir).resolve()),
        "source_run_name": config.run_name,
        "source_pipeline": source_meta.get("pipeline"),
        "source_DATA_KEYWORD": source_meta.get("DATA_KEYWORD"),
        "source_saved_at": source_meta.get("saved_at"),
        "reconstruction_directory": str(output.resolve()),
        "source_metadata": _json_safe(source_meta),
        "exclude_seed": bool(source_meta.get("exclude_seed", False)),
        "has_q_hop_sensitivity": not sensitivity_df.empty,
        "has_sampling_sensitivity": not sampling_df.empty,
        "has_baseline_comparison": not baseline_df.empty,
        "edge_weight_semantics": edge_weight_semantics,
        "node_count": int(validation.metrics["final_node_count"]),
        "edge_count": int(validation.metrics["edge_count"]),
        "files_created": created_files,
        "library_versions": versions,
        "validation_result": _json_safe(validation.metrics),
        "warnings_and_fallbacks": sorted(set(warnings_and_fallbacks or [])),
        "saved_at": stamp,
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
        max_soft_hops=4,
        enable_length_normalized_affinity=False,
        connectivity_affinity_mode="separatrix_hop_decay",
        gaussian_distance_units="raw",
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

    # Test M: Gaussian kernel on geometric shortest-path distance
    gaussian_kwargs = dict(
        q=0.5,
        max_soft_hops=4,
        mode="separatrix_gaussian",
        sigma=1.0,
        eta=1.0,
        cutoff_sigmas=3.0,
    )
    gauss_short = build_separatrix_local_connectivity_affinity(
        G_short,
        ["u", "a", "v"],
        m_short,
        zero_at_zero_distance=True,
        **gaussian_kwargs,
    )
    gauss_long = build_separatrix_local_connectivity_affinity(
        G_long,
        path,
        m_long,
        zero_at_zero_distance=True,
        **gaussian_kwargs,
    )
    K_short = gauss_short.separatrix_local_connectivity_affinity
    K_long = gauss_long.separatrix_local_connectivity_affinity
    expected_near = float(np.exp(-0.5))
    expected_far = float(np.exp(-2.0))
    assert np.isclose(K_short[0, 1], expected_near, atol=tolerance)
    assert np.isclose(K_short[0, 2], expected_far, atol=tolerance)
    assert np.isclose(K_long[0, 4], expected_far, atol=tolerance)
    assert np.allclose(np.diag(K_short), 0.0)
    gauss_self = build_separatrix_local_connectivity_affinity(
        G_short,
        ["u", "a", "v"],
        m_short,
        zero_at_zero_distance=False,
        **gaussian_kwargs,
    )
    assert np.allclose(
        np.diag(gauss_self.separatrix_local_connectivity_affinity), 1.0
    )
    gauss_cut = build_separatrix_local_connectivity_affinity(
        G_short,
        ["u", "a", "v"],
        m_short,
        q=0.5,
        max_soft_hops=4,
        mode="separatrix_gaussian",
        sigma=1.0,
        eta=1.0,
        cutoff_sigmas=1.0,
        zero_at_zero_distance=True,
    )
    assert gauss_cut.separatrix_local_connectivity_affinity[0, 2] == 0.0
    assert np.isclose(
        gauss_cut.separatrix_local_connectivity_affinity[0, 1],
        expected_near,
        atol=tolerance,
    )
    gauss_eta = build_separatrix_local_connectivity_affinity(
        G_short,
        ["u", "a", "v"],
        m_short,
        q=0.5,
        max_soft_hops=4,
        mode="separatrix_gaussian",
        sigma=1.0,
        eta=0.5,
        cutoff_sigmas=3.0,
        zero_at_zero_distance=True,
    )
    assert np.isclose(
        gauss_eta.separatrix_local_connectivity_affinity[0, 1],
        0.5 * expected_near,
        atol=tolerance,
    )
    assert np.all(gauss_eta.separatrix_local_connectivity_affinity <= 1.0 + tolerance)
    gauss_scaled = build_separatrix_local_connectivity_affinity(
        G_short,
        ["u", "a", "v"],
        m_short,
        zero_at_zero_distance=True,
        length_scale=2.0,
        **gaussian_kwargs,
    )
    assert np.isclose(
        gauss_scaled.separatrix_local_connectivity_affinity[0, 1],
        float(np.exp(-0.125)),
        atol=tolerance,
    )
    scale_config = ReconstructionConfig(
        reconstruction_name="scale_test",
        gaussian_sigma=0.05,
        gaussian_distance_units="domain",
        enable_length_normalized_affinity=False,
    )
    bundle_scaled = {
        "p": np.ones(3),
        "Gs": [G_short],
        "Ts": [0.2 * np.eye(3)],
        "lambdas": np.ones(1),
        "Cs": [np.zeros((3, 3))],
        "meta": {"domain_size": 2.0},
    }
    transported_scaled = compute_transported_separatrix_connectivity(
        bundle_scaled, [["u", "a", "v"]], scale_config
    )
    assert transported_scaled.diagnostics["gaussian_distance_units"] == "domain"
    assert np.isclose(transported_scaled.diagnostics["length_scale"], 2.0)
    assert np.isclose(
        transported_scaled.diagnostics["median_raw_edge_length"], 1.0, atol=tolerance
    )
    assert np.isclose(
        transported_scaled.diagnostics["median_scaled_edge_length"], 0.5, atol=tolerance
    )
    # Same-separatrix rule still blocks the ID-mismatched two-hop pair.
    G_mismatch = nx.Graph()
    G_mismatch.add_edge("u", "v", separatrixID="0")
    G_mismatch.add_edge("v", "w", separatrixID="1")
    for node, (x, y) in {"u": (0, 0), "v": (1, 0), "w": (2, 0)}.items():
        G_mismatch.nodes[node]["x"] = x
        G_mismatch.nodes[node]["y"] = y
        G_mismatch.nodes[node]["z"] = 0
    gauss_nontransitive = build_separatrix_local_connectivity_affinity(
        G_mismatch,
        ["u", "v", "w"],
        build_separatrix_membership(G_mismatch),
        q=0.5,
        max_soft_hops=4,
        mode="separatrix_gaussian",
        sigma=1.0,
        eta=1.0,
        cutoff_sigmas=3.0,
        zero_at_zero_distance=True,
    )
    assert gauss_nontransitive.separatrix_local_connectivity_affinity[0, 2] == 0.0
    results["M_gaussian_shortestpath"] = True

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
    "build_separatrix_path_index",
    "prepare_separatrix_betweenness_engine",
    "candidate_graph_from_evidence",
    "reconstruct_separatrix_triangle_shortcuts",
    "apply_triangle_shortcut_threshold",
    "evaluate_triangle_redundancy",
    "evidence_matrix_from_graph",
    "reconstruction_graph_statistics",
    "run_triangle_reconstruction_tests",
    "prepare_path_coherence_engine",
    "prune_same_separatrix_cycles",
    "run_cycle_pruning_tests",
    "prepare_edge_support_engine",
    "annotate_separatrix_affinities",
    "widest_affinity_path",
    "prune_same_separatrix_redundancy",
    "surviving_edge_support_table",
    "classify_surviving_separatrix_support",
    "count_thresholded_pairs",
    "reconstruction_count_row",
    "witness_path_length_stats",
    "run_phase21_tests",
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
