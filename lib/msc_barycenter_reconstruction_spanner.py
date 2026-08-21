"""Conditional-adjacency-aware reconstruction helpers for saved PFGW barycenters.

This module is intentionally separate from the legacy MST reconstruction.  The
configuration defined here is shared by the reconstruction notebook and the
algorithms added in later reconstruction stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from lib.graph_sparsify import geometric_planarity_allows_edge


PathLike = Union[str, Path]
ConstraintMode = Literal["off", "soft", "hard"]
MetricRepairMode = Literal["shortest_path_closure"]
ReconstructionMode = Literal["full_support", "metric_only"]
DensitySelectionRule = Literal[
    "minimum_intersections_then_inflation",
    "minimum_inflation_then_intersections",
]


@dataclass
class BundleValidationResult:
    """Validated bundle metadata and safe input-node matrix orderings."""

    mode: ReconstructionMode
    warnings: List[str]
    input_node_orders: Optional[List[List[Any]]]
    n_center: int
    feature_dim: int
    n_inputs: int
    requested_support_size: Optional[int]
    actual_support_size: int
    has_couplings: bool
    has_input_graphs: bool
    has_node_order_metadata: bool
    has_held_out_seed: bool

    @property
    def full_support_available(self) -> bool:
        """Whether coupling-transported input adjacency can be computed safely."""
        return self.mode == "full_support"


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
class DistanceMatrixResult:
    """Sanitized barycenter distances, metric repair, and diagnostics."""

    D_raw: np.ndarray
    D_metric: np.ndarray
    finite_pair_mask: np.ndarray
    relative_repair: np.ndarray
    repaired_pairs: pd.DataFrame
    diagnostics: Dict[str, Any]
    warnings: List[str]
    source: str


@dataclass
class ConditionalAdjacencySupportResult:
    """Transported adjacency quantities on the barycenter support."""

    conditional_adjacency_support: np.ndarray
    conditional_adjacency_support_variance: np.ndarray
    conditional_adjacency_support_std: np.ndarray
    pair_coverage: np.ndarray
    pair_is_covered: np.ndarray
    adjacency_evidence: np.ndarray
    pair_contributing_inputs: np.ndarray
    matched_fractions: np.ndarray
    input_adjacencies: List[np.ndarray]
    input_conditional_adjacency_support: List[np.ndarray]
    input_pair_coverage: List[np.ndarray]
    warnings: List[str]
    diagnostics: Dict[str, Any]


@dataclass
class MetricDecompositionResult:
    """Exact two-hop metric decomposition quantities."""

    metric_decomposition_distance: np.ndarray
    metric_decomposition_gap: np.ndarray
    metric_relative_gap: np.ndarray
    metric_directness: np.ndarray
    finite_pair_mask: np.ndarray
    diagnostics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


@dataclass
class CandidateEdgeResult:
    """Interpretable all-pairs evidence table and selected candidate graph."""

    all_pairs: pd.DataFrame
    selected_edges: pd.DataFrame
    graph: nx.Graph
    diagnostics: Dict[str, Any]
    warnings: List[str]
    evidence_formula: str


@dataclass
class ComponentTarget:
    """Resolved component-count target and its provenance."""

    count: int
    source: str
    input_component_counts: List[int]
    assumption_used: bool


@dataclass
class SpannerResult:
    """One constrained conditional-adjacency-aware metric spanner."""

    graph: nx.Graph
    stretch: float
    target_component_count: int
    effective_planarity_mode: ConstraintMode
    fallback_bridges: pd.DataFrame
    diagnostics: Dict[str, Any]
    rejected_nonplanar_edges: int


@dataclass
class ConsensusAugmentationResult:
    """Spanner after conditional-support-quantile edge augmentation."""

    graph: nx.Graph
    conditional_support_quantile: float
    conditional_support_threshold: Optional[float]
    added_edges: pd.DataFrame
    diagnostics: Dict[str, Any]


@dataclass
class LocalRepairResult:
    """Deterministically locally repaired topology and accepted edit history."""

    graph: nx.Graph
    edit_history: pd.DataFrame
    objective_before: Dict[str, float]
    objective_after: Dict[str, float]
    iterations_completed: int


@dataclass
class ReconstructionValidationResult:
    """Graph-level validation for a Morse(-Smale)-like reconstruction."""

    summary: Dict[str, Any]
    stretch_values: np.ndarray
    selected_edges: pd.DataFrame
    degree_by_type: pd.DataFrame
    unexpected_degree_nodes: pd.DataFrame
    low_confidence_violation_nodes: pd.DataFrame
    strong_omitted_candidates: pd.DataFrame
    questionable_edges: pd.DataFrame
    warnings: List[str]


@dataclass
class DensityFieldResult:
    """Weighted candidate-segment density on a padded x-y grid."""

    density: np.ndarray
    raw_mass_grid: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    bandwidth: float
    resolution: int
    evidence_mass_total: float
    metadata: Dict[str, Any]


@dataclass
class GeometricRoutingResult:
    """Fixed-adjacency graph with density-routed edge polylines."""

    graph: nx.Graph
    route_statistics: pd.DataFrame
    diagnostics: Dict[str, Any]
    density_field: DensityFieldResult


@dataclass
class ReconstructionConfig:
    """Configuration for a Morse(-Smale)-like barycenter reconstruction.

    Parameters are initial exploration values, not fitted or assumed-optimal
    settings.  ``bundle_dir`` may override the standard
    ``results/barycenters/<run_name>`` input location.
    """

    run_name: str = "latest"
    reconstruction_name: str = "conditional_adjacency_aware_spanner"
    bundle_dir: Optional[PathLike] = None
    random_seed: int = 20260804
    metric_repair_mode: MetricRepairMode = "shortest_path_closure"
    strict_validation: bool = True
    top_k_conditional_support: int = field(
        default=12, metadata={"description": "Per-node conditional-support candidates."}
    )
    top_k_metric_relative_gap: int = field(
        default=12, metadata={"description": "Per-node relative-gap candidates."}
    )
    minimum_pair_coverage: float = field(
        default=0.05, metadata={"description": "Minimum transported pair coverage."}
    )
    spanner_stretch_values: List[float] = field(
        default_factory=lambda: [1.02, 1.05, 1.10, 1.20]
    )
    type_constraint_mode: ConstraintMode = "soft"
    type_confidence_threshold: float = 0.75
    scalar_order_constraint_mode: ConstraintMode = "soft"
    planarity_mode: ConstraintMode = "soft"
    force_planarity_in_3d: bool = False
    target_component_count: Optional[int] = None
    consensus_conditional_support_quantiles: List[float] = field(
        default_factory=lambda: [0.75, 0.90, 0.95],
        metadata={"description": "Conditional-support augmentation quantiles."},
    )
    global_conditional_support_quantile: float = field(
        default=0.95, metadata={"description": "Global conditional-support quantile."}
    )
    evidence_conditional_support_weight: float = field(
        default=0.45, metadata={"description": "Conditional-support score weight."}
    )
    evidence_metric_relative_gap_weight: float = field(
        default=0.35, metadata={"description": "Metric-relative-gap score weight."}
    )
    evidence_pair_coverage_weight: float = field(
        default=0.20, metadata={"description": "Pair-coverage score weight."}
    )
    evidence_uncertainty_penalty: float = 0.20
    type_violation_penalty: float = 0.15
    scalar_order_violation_penalty: float = 0.10
    stretch_diagnostic_quantile: float = 0.95
    local_repair_max_iterations: int = 25
    local_repair_metric_tolerance: float = 0.002
    local_repair_low_conditional_support_quantile: float = field(
        default=0.10,
        metadata={"description": "Low conditional-support repair quantile."},
    )
    objective_edge_penalty_weight: float = 1e-5
    objective_conditional_support_reward_weight: float = field(
        default=0.02,
        metadata={"description": "Conditional-support objective reward weight."},
    )
    objective_uncertainty_penalty_weight: float = 0.01
    objective_constraint_penalty_weight: float = 0.05
    structural_threshold_edge_fraction: float = 0.07
    enable_density_routing: bool = False
    density_grid_resolution_candidates: List[int] = field(
        default_factory=lambda: [128, 192, 256]
    )
    density_bandwidth_candidates: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.04]
    )
    density_selection_rule: DensitySelectionRule = (
        "minimum_intersections_then_inflation"
    )
    density_sample_spacing_fraction: float = 0.005
    density_padding_fraction: float = 0.05
    density_conditional_support_weight: float = field(
        default=0.50,
        metadata={"description": "Conditional-support density weight."},
    )
    density_metric_relative_gap_weight: float = field(
        default=0.25,
        metadata={"description": "Metric-relative-gap density weight."},
    )
    density_pair_coverage_weight: float = field(
        default=0.25, metadata={"description": "Pair-coverage density weight."}
    )
    density_uncertainty_penalty: float = 0.25
    density_length_penalty: float = 0.05
    density_occupancy_penalty: float = 0.0
    density_high_region_quantile: float = 0.50
    max_dense_edges_to_plot: int = 10_000
    max_routed_edges_to_plot: int = 2_000
    write_legacy_name_aliases: bool = field(
        default=False,
        metadata={"description": "Write deprecated terminology aliases when saving."},
    )

    def __post_init__(self) -> None:
        """Validate values early so notebook failures have useful messages."""
        self._validate_name(self.run_name, "run_name")
        self._validate_name(self.reconstruction_name, "reconstruction_name")
        if self.metric_repair_mode != "shortest_path_closure":
            raise ValueError(
                "metric_repair_mode must currently be 'shortest_path_closure'; "
                f"got {self.metric_repair_mode!r}."
            )
        self._validate_constraint_mode(self.type_constraint_mode, "type_constraint_mode")
        self._validate_constraint_mode(
            self.scalar_order_constraint_mode, "scalar_order_constraint_mode"
        )
        self._validate_constraint_mode(self.planarity_mode, "planarity_mode")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if (
            self.top_k_conditional_support < 1
            or self.top_k_metric_relative_gap < 1
        ):
            raise ValueError(
                "top_k_conditional_support and top_k_metric_relative_gap must be positive."
            )
        self._validate_unit_interval(
            self.minimum_pair_coverage, "minimum_pair_coverage"
        )
        self._validate_unit_interval(
            self.type_confidence_threshold, "type_confidence_threshold"
        )
        if not self.spanner_stretch_values or any(
            value < 1.0 for value in self.spanner_stretch_values
        ):
            raise ValueError("spanner_stretch_values must contain values >= 1.")
        if not self.consensus_conditional_support_quantiles:
            raise ValueError("consensus_conditional_support_quantiles must not be empty.")
        for value in self.consensus_conditional_support_quantiles:
            self._validate_unit_interval(value, "consensus_conditional_support_quantiles")
        self._validate_unit_interval(
            self.global_conditional_support_quantile,
            "global_conditional_support_quantile",
        )
        self._validate_unit_interval(
            self.stretch_diagnostic_quantile, "stretch_diagnostic_quantile"
        )
        self._validate_unit_interval(
            self.local_repair_low_conditional_support_quantile,
            "local_repair_low_conditional_support_quantile",
        )
        self._validate_unit_interval(
            self.structural_threshold_edge_fraction,
            "structural_threshold_edge_fraction",
        )
        if self.structural_threshold_edge_fraction <= 0:
            raise ValueError("structural_threshold_edge_fraction must be positive.")
        if self.local_repair_max_iterations < 0:
            raise ValueError("local_repair_max_iterations must be nonnegative.")
        if self.local_repair_metric_tolerance < 0:
            raise ValueError("local_repair_metric_tolerance must be nonnegative.")
        score_parameters = {
            "evidence_conditional_support_weight": self.evidence_conditional_support_weight,
            "evidence_metric_relative_gap_weight": self.evidence_metric_relative_gap_weight,
            "evidence_pair_coverage_weight": self.evidence_pair_coverage_weight,
            "evidence_uncertainty_penalty": self.evidence_uncertainty_penalty,
            "type_violation_penalty": self.type_violation_penalty,
            "scalar_order_violation_penalty": self.scalar_order_violation_penalty,
            "objective_edge_penalty_weight": self.objective_edge_penalty_weight,
            "objective_conditional_support_reward_weight": (
                self.objective_conditional_support_reward_weight
            ),
            "objective_uncertainty_penalty_weight": self.objective_uncertainty_penalty_weight,
            "objective_constraint_penalty_weight": self.objective_constraint_penalty_weight,
        }
        if any(value < 0 for value in score_parameters.values()):
            raise ValueError(
                "Evidence weights and violation penalties must be nonnegative."
            )
        if self.target_component_count is not None and self.target_component_count < 1:
            raise ValueError("target_component_count must be None or a positive integer.")
        if not self.density_grid_resolution_candidates or any(
            value < 8 for value in self.density_grid_resolution_candidates
        ):
            raise ValueError(
                "density_grid_resolution_candidates must contain integers >= 8."
            )
        if not self.density_bandwidth_candidates or any(
            value <= 0 for value in self.density_bandwidth_candidates
        ):
            raise ValueError("density_bandwidth_candidates must be strictly positive.")
        if self.density_selection_rule not in {
            "minimum_intersections_then_inflation",
            "minimum_inflation_then_intersections",
        }:
            raise ValueError("Unsupported density_selection_rule.")
        self._validate_unit_interval(
            self.density_high_region_quantile, "density_high_region_quantile"
        )
        if self.density_sample_spacing_fraction <= 0:
            raise ValueError("density_sample_spacing_fraction must be positive.")
        if self.density_padding_fraction < 0:
            raise ValueError("density_padding_fraction must be nonnegative.")
        density_parameters = {
            "density_conditional_support_weight": self.density_conditional_support_weight,
            "density_metric_relative_gap_weight": self.density_metric_relative_gap_weight,
            "density_pair_coverage_weight": self.density_pair_coverage_weight,
            "density_uncertainty_penalty": self.density_uncertainty_penalty,
            "density_length_penalty": self.density_length_penalty,
            "density_occupancy_penalty": self.density_occupancy_penalty,
        }
        if any(value < 0 for value in density_parameters.values()):
            raise ValueError("Density weights and penalties must be nonnegative.")
        if self.max_dense_edges_to_plot < 1 or self.max_routed_edges_to_plot < 1:
            raise ValueError("Plotting edge limits must be positive integers.")

    @staticmethod
    def _validate_name(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError(
                f"{field_name} must be a single directory name, not a path: {value!r}."
            )

    @staticmethod
    def _validate_constraint_mode(value: str, field_name: str) -> None:
        allowed = {"off", "soft", "hard"}
        if value not in allowed:
            raise ValueError(f"{field_name} must be one of {sorted(allowed)}; got {value!r}.")

    @staticmethod
    def _validate_unit_interval(value: float, field_name: str) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field_name} must lie in [0, 1]; got {value!r}.")

    def resolve_bundle_dir(self, project_root: PathLike = ".") -> Path:
        """Return the absolute saved-bundle path without creating or changing it."""
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
        """Resolve, and optionally create, reconstruction and figure directories.

        Only the new ``reconstruction/<reconstruction_name>`` subtree is created;
        no saved bundle arrays, metadata, couplings, or baseline graphs are
        opened for writing.
        """
        bundle = self.resolve_bundle_dir(project_root)
        output_dir = bundle / "reconstruction" / self.reconstruction_name
        figures_dir = output_dir / "figures"
        if create:
            figures_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, figures_dir

    def resolved_dict(self, project_root: PathLike = ".") -> Dict[str, Any]:
        """Return JSON/table-friendly settings including resolved paths."""
        values: Dict[str, Any] = asdict(self)
        bundle = self.resolve_bundle_dir(project_root)
        output, figures = self.resolve_output_dirs(project_root, create=False)
        values["bundle_dir"] = str(bundle)
        values["output_dir"] = str(output)
        values["figures_dir"] = str(figures)
        return values


def _matrix_row_count(value: Any, name: str, errors: List[str]) -> Optional[int]:
    """Validate a square input relation matrix and return its row count."""
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        errors.append(f"{name} must be square; got shape {array.shape}.")
        return None
    return int(array.shape[0])


def _resolve_input_node_orders(
    graphs: Any,
    nodes_list: Any,
    input_sizes: List[Optional[int]],
    warnings: List[str],
) -> Optional[List[List[Any]]]:
    """Return explicit matrix-index node orders, or ``None`` if ambiguous."""
    n_inputs = len(input_sizes)
    if graphs is None:
        warnings.append("Input graphs are unavailable; transported adjacency is disabled.")
        return None
    if len(graphs) != n_inputs:
        warnings.append(
            f"Input graph count {len(graphs)} does not match {n_inputs} inputs."
        )
        return None

    if nodes_list is not None and len(nodes_list) != n_inputs:
        warnings.append(
            f"nodes_list length {len(nodes_list)} does not match {n_inputs} inputs."
        )
        return None

    resolved: List[List[Any]] = []
    for index, (graph, n_s) in enumerate(zip(graphs, input_sizes)):
        if n_s is None:
            return None
        graph_nodes = list(graph.nodes())
        if graph.number_of_nodes() != n_s:
            warnings.append(
                f"Input graph {index} has {graph.number_of_nodes()} nodes, "
                f"but C_{index} has {n_s} rows."
            )
            return None

        if nodes_list is None:
            expected = list(range(n_s))
            if set(graph_nodes) != set(expected):
                warnings.append(
                    f"Input graph {index} lacks node-order metadata and its node labels "
                    "are not exactly matrix indices 0..n_s-1; ordering is ambiguous."
                )
                return None
            resolved.append(expected)
            continue

        order = list(nodes_list[index])
        try:
            order_is_unique = len(set(order)) == n_s
            order_matches_graph = set(order) == set(graph_nodes)
        except TypeError:
            warnings.append(
                f"nodes_list[{index}] contains non-hashable node ids and cannot be "
                "matched safely to the input graph."
            )
            return None
        if len(order) != n_s or not order_is_unique:
            warnings.append(
                f"nodes_list[{index}] must contain {n_s} unique ordered node ids."
            )
            return None
        if not order_matches_graph:
            warnings.append(
                f"nodes_list[{index}] does not match the nodes of input graph {index}."
            )
            return None
        resolved.append(order)
    return resolved


def validate_barycenter_bundle(
    bundle: Mapping[str, Any], *, strict_validation: bool = True
) -> BundleValidationResult:
    """Validate a loaded shortest-path PFGW bundle for reconstruction.

    Center-matrix failures and a non-shortest-path relation are always fatal.
    With strict validation enabled, malformed aligned input arrays/couplings are
    also fatal. Missing or ambiguous graph ordering is nonfatal but explicitly
    selects metric-only fallback mode.
    """
    meta = dict(bundle.get("meta") or {})
    relation_kind = str(meta.get("relation_kind", "")).lower()
    if relation_kind != "shortestpath":
        raise NotImplementedError(
            "Support-aware reconstruction requires meta['relation_kind'] == "
            f"'shortestpath' because C_center is interpreted as a distance matrix; "
            f"got {relation_kind!r}."
        )

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
    if Ts is not None:
        if len(Ts) != n_inputs:
            support_errors.append(
                f"Coupling count {len(Ts)} does not match {n_inputs} inputs."
            )
        else:
            for index, (T_s, n_s) in enumerate(zip(Ts, input_sizes)):
                expected = (n_center_safe, n_s)
                if n_s is not None and np.asarray(T_s).shape != expected:
                    support_errors.append(
                        f"T_{index} must have shape {expected}; "
                        f"got {np.asarray(T_s).shape}."
                    )

    warnings: List[str] = []
    if support_errors and strict_validation:
        raise ValueError(
            "Invalid aligned PFGW input data:\n- " + "\n- ".join(support_errors)
        )
    warnings.extend(support_errors)
    if Ts is None:
        warnings.append("Couplings Ts are unavailable; transported adjacency is disabled.")

    graphs = bundle.get("Gs")
    nodes_list = bundle.get("nodes_list")
    node_orders = _resolve_input_node_orders(graphs, nodes_list, input_sizes, warnings)
    full_support = not support_errors and Ts is not None and node_orders is not None
    requested = meta.get("N_center")
    requested_size = None if requested is None else int(requested)
    held_out = bool(bundle.get("exclude_seed", False) and bundle.get("seed") is not None)
    return BundleValidationResult(
        mode="full_support" if full_support else "metric_only",
        warnings=warnings,
        input_node_orders=node_orders if full_support else None,
        n_center=n_center_safe,
        feature_dim=int(X_center.shape[1]),
        n_inputs=n_inputs,
        requested_support_size=requested_size,
        actual_support_size=n_center_safe,
        has_couplings=has_couplings,
        has_input_graphs=graphs is not None,
        has_node_order_metadata=nodes_list is not None,
        has_held_out_seed=held_out,
    )


def _normalized_type_scores(values: np.ndarray) -> np.ndarray:
    """Normalize nonnegative soft type scores row-wise."""
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
    """Decode every barycenter support point and cross-check ``G_complete``.

    Existing repository node decoding is used for feature-to-physical
    coordinates, scalar values, and hard type labels. This function augments it
    with normalized scalar values, soft type confidence, mass, provenance, and
    disagreements against the saved complete graph.
    """
    from lib.msc_barycenter_threshold import _decode_barycenter_nodes

    X = np.asarray(bundle.get("X_center"), dtype=float)
    p = np.asarray(bundle.get("p"), dtype=float)
    meta = dict(bundle.get("meta") or {})
    if X.ndim != 2 or p.ndim != 1 or X.shape[0] != p.shape[0]:
        raise ValueError(
            "X_center must be two-dimensional and aligned row-wise with one-dimensional p."
        )
    if comparison_tolerance < 0:
        raise ValueError("comparison_tolerance must be nonnegative.")

    n_center, feature_dim = X.shape
    type_order = list(meta.get("MSC_TYPE_ORDER") or [])
    scalar_meta = all(key in meta for key in ("min_scalar", "scalar_range"))
    type_without_scalar = bool(type_order) and feature_dim == 3 + len(type_order)
    scalar_in_features = bool(
        scalar_meta and feature_dim >= 4 and not type_without_scalar
    )
    type_start = 4 if scalar_in_features else 3
    type_in_features = bool(
        type_order and feature_dim >= type_start + len(type_order)
    )
    coordinate_meta = all(
        key in meta for key in ("min_x", "min_y", "min_z", "domain_size")
    )

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
    warnings: List[str] = []
    recovered_scalar = False
    recovered_type = False

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
                recovered_scalar = True
                if scalar_meta and float(meta["scalar_range"]) != 0:
                    scalar_normalized = (
                        scalar_value - float(meta["min_scalar"])
                    ) / float(meta["scalar_range"])

        morse_type: Any = np.nan
        type_confidence = np.nan
        type_source = "unavailable"
        if normalized_types is not None and np.any(
            np.isfinite(normalized_types[node_id])
        ):
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
            recovered_type = True

        record = {
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
        records.append(record)

        if graph_attrs:
            for attribute in ("x", "y", "z", "scalar_value", "type"):
                decoded_name = "morse_type" if attribute == "type" else attribute
                decoded_value = record.get(decoded_name, np.nan)
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
    if recovered_scalar:
        warnings.append("Missing scalar feature values were recovered from G_complete.")
    if recovered_type:
        warnings.append(
            "Missing type feature values were recovered as hard G_complete labels "
            "with confidence 1."
        )
    if not disagreement_frame.empty:
        warnings.append(
            f"Found {len(disagreement_frame)} decoded/G_complete attribute disagreements."
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
    if requested_type_mode != "off" and effective_type_mode == "off":
        warnings.append("Type constraints were disabled because types are unavailable.")
    if requested_scalar_mode != "off" and effective_scalar_mode == "off":
        warnings.append(
            "Scalar-order constraints were disabled because types or scalar values "
            "are unavailable."
        )
    if requested_planarity_mode != "off" and effective_planarity_mode == "off":
        warnings.append(
            "Planarity constraints were disabled because x-y coordinates are unavailable."
        )
    return DecodedNodesResult(
        dataframe=dataframe,
        disagreements=disagreement_frame,
        warnings=warnings,
        has_coordinates=has_coordinates,
        has_scalar=has_scalar,
        has_types=has_types,
        effective_type_constraint_mode=effective_type_mode,
        effective_scalar_order_constraint_mode=effective_scalar_mode,
        effective_planarity_mode=effective_planarity_mode,
    )


def shortest_path_metric_closure(
    distances: np.ndarray,
    finite_pair_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an all-pairs shortest-path closure that only reduces valid pairs.

    Invalid or unobserved pairs remain ``NaN`` rather than being filled with an
    arbitrary large value. The diagonal is always zero.
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distances must be square; got shape {D.shape}.")
    n = D.shape[0]
    if finite_pair_mask is None:
        mask = np.isfinite(D) & (D > 0)
        np.fill_diagonal(mask, False)
    else:
        mask = np.asarray(finite_pair_mask, dtype=bool)
        if mask.shape != D.shape:
            raise ValueError(
                f"finite_pair_mask shape {mask.shape} does not match {D.shape}."
            )
        mask = mask & mask.T
        np.fill_diagonal(mask, False)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, j in zip(*np.where(np.triu(mask, k=1))):
        weight = float(D[i, j])
        if np.isfinite(weight) and weight > 0:
            graph.add_edge(int(i), int(j), weight=weight)
    closure = np.asarray(
        nx.floyd_warshall_numpy(graph, nodelist=range(n), weight="weight"),
        dtype=float,
    )
    repaired = np.full_like(D, np.nan, dtype=float)
    valid = mask & np.isfinite(closure)
    repaired[valid] = np.minimum(D[valid], closure[valid])
    np.fill_diagonal(repaired, 0.0)
    return repaired


def _distances_from_complete_graph(
    graph: Any, n_center: int
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Extract a complete index-aligned weight matrix, or explain rejection."""
    if not isinstance(graph, nx.Graph):
        return None, "G_complete is unavailable."
    expected_nodes = set(range(n_center))
    if set(graph.nodes()) != expected_nodes:
        return None, "G_complete nodes are not exactly indexed 0..n_center-1."
    expected_edges = n_center * (n_center - 1) // 2
    if graph.number_of_edges() != expected_edges:
        return None, (
            f"G_complete has {graph.number_of_edges()} edges; "
            f"a complete graph requires {expected_edges}."
        )
    matrix = np.zeros((n_center, n_center), dtype=float)
    for i in range(n_center):
        for j in range(i + 1, n_center):
            if not graph.has_edge(i, j) or "weight" not in graph.edges[i, j]:
                return None, f"G_complete edge ({i}, {j}) lacks a structural weight."
            weight = float(graph.edges[i, j]["weight"])
            if not np.isfinite(weight) or weight <= 0:
                return None, (
                    f"G_complete edge ({i}, {j}) has invalid distance {weight!r}."
                )
            matrix[i, j] = matrix[j, i] = weight
    return matrix, None


def _decode_center_relation(bundle: Mapping[str, Any]) -> np.ndarray:
    """Decode ``C_center`` with the repository's inverse-relation helper."""
    from lib.msc_barycenter_reconstruction import _inverse_relation_weight

    C = np.asarray(bundle.get("C_center"), dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C_center must be square; got shape {C.shape}.")
    meta = dict(bundle.get("meta") or {})
    relation_kind = str(meta.get("relation_kind", "shortestpath"))
    rescale = float(meta.get("relation_rescale", 1.0))
    offset = float(meta.get("relation_offset", 0.0))
    decoded = np.empty_like(C, dtype=float)
    for index in np.ndindex(C.shape):
        decoded[index] = _inverse_relation_weight(
            float(C[index]),
            relation_kind=relation_kind,
            relation_rescale=rescale,
            relation_offset=offset,
        )
    np.fill_diagonal(decoded, 0.0)
    return decoded


def extract_barycenter_distance_matrix(
    bundle: Mapping[str, Any],
    *,
    metric_repair_mode: MetricRepairMode = "shortest_path_closure",
    material_pair_fraction: float = 0.05,
    material_relative_repair: float = 0.01,
    tolerance: float = 1e-12,
) -> DistanceMatrixResult:
    """Extract, sanitize, diagnose, and metrically repair barycenter distances."""
    if metric_repair_mode != "shortest_path_closure":
        raise NotImplementedError(
            "Only metric_repair_mode='shortest_path_closure' is implemented."
        )
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")
    C = np.asarray(bundle.get("C_center"), dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C_center must be square; got shape {C.shape}.")
    n_center = C.shape[0]
    decoded_C = _decode_center_relation(bundle)
    graph_D, graph_rejection = _distances_from_complete_graph(
        bundle.get("G_complete"), n_center
    )
    warnings: List[str] = []
    if graph_D is not None:
        selected = graph_D.copy()
        source = "G_complete.weight"
    else:
        selected = decoded_C.copy()
        source = "decoded C_center"
        if graph_rejection:
            warnings.append(graph_rejection + " Falling back to decoded C_center.")

    upper = np.triu_indices(n_center, k=1)
    comparison = np.array([], dtype=float)
    if graph_D is not None:
        comparable = np.isfinite(graph_D[upper]) & np.isfinite(decoded_C[upper])
        comparison = np.abs(graph_D[upper][comparable] - decoded_C[upper][comparable])

    off_diagonal = ~np.eye(n_center, dtype=bool)
    finite_selected = np.isfinite(selected)
    overlap = finite_selected & finite_selected.T
    asymmetry_mask = overlap & off_diagonal
    asymmetry_values = np.abs(
        selected[asymmetry_mask] - selected.T[asymmetry_mask]
    )
    max_asymmetry = (
        float(np.max(asymmetry_values)) if asymmetry_values.size else 0.0
    )
    negative_entries = int(np.count_nonzero((selected < 0) & off_diagonal))
    zero_entries = int(np.count_nonzero((selected == 0) & off_diagonal))
    nonfinite_entries = int(np.count_nonzero(~finite_selected & off_diagonal))
    valid_entries = finite_selected & (selected > 0) & off_diagonal
    finite_pair_mask = valid_entries & valid_entries.T
    invalid_entries = int(np.count_nonzero(off_diagonal & ~valid_entries))

    D_raw = np.full_like(selected, np.nan, dtype=float)
    D_raw[finite_pair_mask] = (
        selected[finite_pair_mask] + selected.T[finite_pair_mask]
    ) / 2.0
    np.fill_diagonal(D_raw, 0.0)
    D_metric = shortest_path_metric_closure(D_raw, finite_pair_mask)

    pair_mask_upper = np.triu(finite_pair_mask, k=1)
    reductions = np.full_like(D_raw, np.nan)
    reductions[finite_pair_mask] = D_raw[finite_pair_mask] - D_metric[finite_pair_mask]
    relative = np.full_like(D_raw, np.nan)
    relative[finite_pair_mask] = np.divide(
        reductions[finite_pair_mask],
        D_raw[finite_pair_mask],
        out=np.zeros(np.count_nonzero(finite_pair_mask), dtype=float),
        where=D_raw[finite_pair_mask] > 0,
    )
    modified = pair_mask_upper & (reductions > tolerance)
    n_pairs = int(np.count_nonzero(pair_mask_upper))
    n_modified = int(np.count_nonzero(modified))
    fraction_modified = float(n_modified / n_pairs) if n_pairs else 0.0
    raw_pair_values = D_raw[pair_mask_upper]
    reduction_values = reductions[modified]
    relative_values = relative[modified]
    relative_total = (
        float(np.nansum(reductions[pair_mask_upper]) / np.nansum(raw_pair_values))
        if raw_pair_values.size and np.nansum(raw_pair_values) > 0
        else 0.0
    )

    pair_i, pair_j = np.where(modified)
    repaired_pairs = pd.DataFrame(
        {
            "node_i": pair_i,
            "node_j": pair_j,
            "raw_distance": D_raw[modified],
            "metric_distance": D_metric[modified],
            "absolute_repair": reductions[modified],
            "relative_repair": relative[modified],
        }
    ).sort_values(
        ["relative_repair", "absolute_repair"], ascending=False, ignore_index=True
    )

    finite_values = D_raw[pair_mask_upper]
    diagnostics: Dict[str, Any] = {
        "source": source,
        "n_center": int(n_center),
        "maximum_asymmetry": max_asymmetry,
        "invalid_entry_count": invalid_entries,
        "negative_entry_count": negative_entries,
        "nonfinite_entry_count": nonfinite_entries,
        "zero_off_diagonal_entry_count": zero_entries,
        "finite_pair_count": n_pairs,
        "off_diagonal_minimum": float(np.min(finite_values)) if n_pairs else None,
        "off_diagonal_median": float(np.median(finite_values)) if n_pairs else None,
        "off_diagonal_mean": float(np.mean(finite_values)) if n_pairs else None,
        "off_diagonal_maximum": float(np.max(finite_values)) if n_pairs else None,
        "triangle_shortcut_violation_count": n_modified,
        "maximum_triangle_shortcut": (
            float(np.max(reduction_values)) if n_modified else 0.0
        ),
        "median_triangle_shortcut": (
            float(np.median(reduction_values)) if n_modified else 0.0
        ),
        "fraction_pairs_modified": fraction_modified,
        "relative_total_repair": relative_total,
        "median_relative_repair_modified": (
            float(np.median(relative_values)) if n_modified else 0.0
        ),
        "maximum_relative_repair": (
            float(np.max(relative_values)) if n_modified else 0.0
        ),
        "graph_vs_center_comparable_pairs": int(comparison.size),
        "graph_vs_center_maximum_disagreement": (
            float(np.max(comparison)) if comparison.size else None
        ),
        "graph_vs_center_median_disagreement": (
            float(np.median(comparison)) if comparison.size else None
        ),
        "metric_repair_mode": metric_repair_mode,
    }
    if (
        fraction_modified >= material_pair_fraction
        or relative_total >= material_relative_repair
    ):
        warnings.append(
            "Metric closure materially changes the input relation "
            f"({fraction_modified:.1%} of finite pairs; "
            f"{relative_total:.1%} total relative reduction)."
        )
    return DistanceMatrixResult(
        D_raw=D_raw,
        D_metric=D_metric,
        finite_pair_mask=finite_pair_mask,
        relative_repair=relative,
        repaired_pairs=repaired_pairs,
        diagnostics=diagnostics,
        warnings=warnings,
        source=source,
    )


def build_aligned_binary_adjacency(
    graph: nx.Graph, node_order: List[Any]
) -> np.ndarray:
    """Build symmetric direct adjacency in an explicit matrix node order.

    Parallel edges collapse to one binary adjacency entry. Edge lengths and
    other attributes are intentionally ignored.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"graph must be a NetworkX graph; got {type(graph)!r}.")
    order = list(node_order)
    try:
        index = {node: position for position, node in enumerate(order)}
    except TypeError as exc:
        raise ValueError("node_order contains a non-hashable node id.") from exc
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
    if not np.array_equal(adjacency, adjacency.T):
        raise AssertionError("Constructed direct adjacency is unexpectedly asymmetric.")
    return adjacency


def transport_adjacency_probability(
    adjacency: np.ndarray, conditional_correspondence: np.ndarray
) -> np.ndarray:
    """Return input conditional adjacency support ``S_s = Q_s A_s Q_s.T``.

    ``A_s`` is binary with shape ``(n_s, n_s)`` and row-stochastic ``Q_s`` has
    shape ``(n_center, n_s)``. Zero-mass rows of ``Q_s`` remain zero. The
    symmetric output has shape ``(n_center, n_center)`` and range ``[0, 1]``.
    """
    A = np.asarray(adjacency, dtype=float)
    Q = np.asarray(conditional_correspondence, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency must be square; got shape {A.shape}.")
    if Q.ndim != 2 or Q.shape[1] != A.shape[0]:
        raise ValueError(
            f"Q must have shape (n_center, {A.shape[0]}); got {Q.shape}."
        )
    if not np.allclose(A, A.T):
        raise ValueError("adjacency must be symmetric.")
    transported = Q @ A @ Q.T
    transported = (transported + transported.T) / 2.0
    np.fill_diagonal(transported, 0.0)
    return transported


def row_normalize_coupling(
    coupling: np.ndarray, *, epsilon: float = 1e-15
) -> Tuple[np.ndarray, np.ndarray]:
    """Row-normalize a coupling and return ``(Q_s, m_i^s)``.

    For ``T_s`` of shape ``(n_center, n_s)``, ``m_i^s=sum_u T_s(i,u)`` and
    ``Q_s(i,u)=T_s(i,u)/m_i^s`` when ``m_i^s>epsilon``. Zero-mass rows are
    zero, so the returned ``Q_s`` is finite and nonnegative.
    """
    T = np.asarray(coupling, dtype=float)
    if T.ndim != 2:
        raise ValueError(f"coupling must be two-dimensional; got {T.shape}.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    matched_mass = T.sum(axis=1)
    Q = np.zeros_like(T)
    positive = matched_mass > epsilon
    Q[positive] = T[positive] / matched_mass[positive, None]
    return Q, matched_mass


def compute_conditional_adjacency_support(
    bundle: Mapping[str, Any],
    input_node_orders: Optional[List[List[Any]]],
    *,
    epsilon: float = 1e-15,
    tiny_overshoot_tolerance: float = 1e-8,
) -> ConditionalAdjacencySupportResult:
    """Aggregate transported conditional adjacency, coverage, and evidence.

    For each aligned input ``s``, ``m_i^s=sum_u T_s(i,u)``, rows of ``Q_s`` are
    ``T_s(i,:)/m_i^s`` for positive mass, and
    ``S_s=Q_s A_s Q_s.T``. Matched fractions are
    ``c_i^s=m_i^s/max(p_i,epsilon)`` and input pair coverage is
    ``o_s(i,j)=c_i^s c_j^s``. With ``L=sum_s lambda_s``,
    ``O=(1/L) sum_s lambda_s o_s`` and
    ``E=(1/L) sum_s lambda_s o_s S_s``. Conditional support is
    ``S=sum_s lambda_s o_s S_s / sum_s lambda_s o_s`` on covered pairs and
    zero elsewhere; therefore ``E=O*S``. The conditional-support standard
    deviation uses the same normalized coverage weights. All aggregate arrays
    have shape ``(n_center, n_center)``; ``S``, ``O``, and ``E`` are in
    ``[0,1]`` and uncovered entries are zero with ``pair_is_covered=False``.

    Only the aligned barycenter inputs in ``bundle['Gs']``/``bundle['Ts']`` are
    used. A held-out ``bundle['seed']`` is never included.
    """
    if input_node_orders is None:
        raise RuntimeError(
            "Transported adjacency requires full-support mode with trustworthy "
            "input node ordering."
        )
    if epsilon <= 0 or tiny_overshoot_tolerance < 0:
        raise ValueError("epsilon must be positive and overshoot tolerance nonnegative.")

    p = np.asarray(bundle.get("p"), dtype=float)
    Cs = list(bundle.get("Cs") or [])
    graphs = list(bundle.get("Gs") or [])
    Ts = bundle.get("Ts")
    lambdas = np.asarray(bundle.get("lambdas"), dtype=float)
    n_inputs, n_center = len(Cs), p.shape[0]
    if Ts is None:
        raise RuntimeError("Saved couplings Ts are required for transported support.")
    if not (
        len(graphs)
        == len(Ts)
        == len(input_node_orders)
        == lambdas.shape[0]
        == n_inputs
    ):
        raise ValueError(
            "Cs, Gs, Ts, input_node_orders, and lambdas must have equal input counts."
        )
    if np.any(~np.isfinite(lambdas)) or np.any(lambdas < 0):
        raise ValueError("lambdas must contain finite nonnegative values.")
    lambda_sum = float(np.sum(lambdas))
    if lambda_sum <= epsilon:
        raise ValueError("lambdas must have a positive finite sum.")

    numerator = np.zeros((n_center, n_center), dtype=float)
    second_numerator = np.zeros_like(numerator)
    coverage_weight_sum = np.zeros_like(numerator)
    contributing = np.zeros((n_center, n_center), dtype=np.int32)
    matched_fractions: List[np.ndarray] = []
    adjacencies: List[np.ndarray] = []
    input_supports: List[np.ndarray] = []
    input_coverages: List[np.ndarray] = []
    warnings: List[str] = []
    substantial_overshoots = 0
    maximum_overshoot = 0.0

    for input_index, (C_s, graph, T_s, node_order, weight) in enumerate(
        zip(Cs, graphs, Ts, input_node_orders, lambdas)
    ):
        n_s = np.asarray(C_s).shape[0]
        if len(node_order) != n_s:
            raise ValueError(
                f"input_node_orders[{input_index}] has {len(node_order)} nodes; "
                f"C_{input_index} has {n_s} rows."
            )
        T = np.asarray(T_s, dtype=float)
        if T.shape != (n_center, n_s):
            raise ValueError(
                f"T_{input_index} must have shape {(n_center, n_s)}; got {T.shape}."
            )
        if np.any(~np.isfinite(T)):
            raise ValueError(f"T_{input_index} contains nonfinite values.")
        if np.any(T < -tiny_overshoot_tolerance):
            raise ValueError(f"T_{input_index} contains substantial negative mass.")
        T = np.where((T < 0) & (T >= -tiny_overshoot_tolerance), 0.0, T)

        adjacency = build_aligned_binary_adjacency(graph, node_order)
        if adjacency.shape != (n_s, n_s):
            raise AssertionError("Aligned input adjacency has an unexpected shape.")
        Q, matched_mass = row_normalize_coupling(T, epsilon=epsilon)
        fractions = matched_mass / np.maximum(p, epsilon)
        tiny = (fractions > 1.0) & (
            fractions <= 1.0 + tiny_overshoot_tolerance
        )
        fractions[tiny] = 1.0
        substantial = fractions > 1.0 + tiny_overshoot_tolerance
        if np.any(substantial):
            substantial_overshoots += int(np.count_nonzero(substantial))
            maximum_overshoot = max(
                maximum_overshoot, float(np.max(fractions[substantial] - 1.0))
            )
        fractions = np.clip(fractions, 0.0, 1.0)
        support_s = transport_adjacency_probability(adjacency, Q)
        coverage_s = np.outer(fractions, fractions)
        weight_s = float(weight) * coverage_s
        np.fill_diagonal(weight_s, 0.0)
        numerator += weight_s * support_s
        second_numerator += weight_s * support_s * support_s
        coverage_weight_sum += weight_s
        contributing += (weight_s > epsilon).astype(np.int32)
        matched_fractions.append(fractions)
        adjacencies.append(adjacency)
        input_supports.append(support_s)
        input_coverages.append(coverage_s)

    pair_is_covered = coverage_weight_sum > epsilon
    support = np.zeros_like(coverage_weight_sum)
    support_variance = np.zeros_like(coverage_weight_sum)
    support[pair_is_covered] = (
        numerator[pair_is_covered] / coverage_weight_sum[pair_is_covered]
    )
    second_moment = np.zeros_like(coverage_weight_sum)
    second_moment[pair_is_covered] = (
        second_numerator[pair_is_covered] / coverage_weight_sum[pair_is_covered]
    )
    variance_values = (
        second_moment[pair_is_covered] - support[pair_is_covered] ** 2
    )
    if np.any(variance_values < -1e-10):
        warnings.append(
            "Weighted support variance had negative values beyond roundoff."
        )
    support_variance[pair_is_covered] = np.maximum(variance_values, 0.0)
    support_std = np.sqrt(support_variance)
    pair_coverage = coverage_weight_sum / lambda_sum
    adjacency_evidence = numerator / lambda_sum
    for matrix in (
        support,
        support_variance,
        support_std,
        pair_coverage,
        adjacency_evidence,
        pair_is_covered,
        contributing,
    ):
        np.fill_diagonal(matrix, 0)

    if substantial_overshoots:
        warnings.append(
            f"Found {substantial_overshoots} matched fractions above 1 beyond "
            f"roundoff; maximum overshoot={maximum_overshoot:.3g}. Values were reported "
            "and clipped to preserve canonical [0, 1] coverage ranges."
        )
    if not np.isclose(lambda_sum, 1.0, rtol=1e-8, atol=1e-10):
        warnings.append(
            f"lambdas sum to {lambda_sum:.12g}, not 1; pair coverage and "
            "adjacency evidence were normalized by this sum."
        )

    fractions_array = np.vstack(matched_fractions)
    upper = np.triu(np.ones_like(pair_coverage, dtype=bool), k=1)
    range_tolerance = tiny_overshoot_tolerance
    for name, matrix in {
        "conditional_adjacency_support": support,
        "conditional_adjacency_support_std": support_std,
        "pair_coverage": pair_coverage,
        "adjacency_evidence": adjacency_evidence,
    }.items():
        if np.any(matrix < -range_tolerance) or np.any(matrix > 1 + range_tolerance):
            raise ValueError(f"{name} must lie in [0, 1] within tolerance.")
    if not np.allclose(
        adjacency_evidence,
        pair_coverage * support,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError("Canonical identity adjacency_evidence = pair_coverage * conditional_adjacency_support failed.")
    diagnostics: Dict[str, Any] = {
        "n_inputs": n_inputs,
        "n_center": n_center,
        "lambda_sum": lambda_sum,
        "covered_pair_count": int(np.count_nonzero(pair_is_covered & upper)),
        "uncovered_pair_count": int(np.count_nonzero(~pair_is_covered & upper)),
        "minimum_pair_coverage": (
            float(np.min(pair_coverage[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "median_pair_coverage": (
            float(np.median(pair_coverage[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "maximum_pair_coverage": (
            float(np.max(pair_coverage[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "mean_conditional_adjacency_support": (
            float(np.mean(support[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "mean_conditional_adjacency_support_std": (
            float(np.mean(support_std[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "mean_adjacency_evidence": (
            float(np.mean(adjacency_evidence[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else None
        ),
        "minimum_pair_contributing_inputs": (
            int(np.min(contributing[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else 0
        ),
        "maximum_pair_contributing_inputs": (
            int(np.max(contributing[pair_is_covered & upper]))
            if np.any(pair_is_covered & upper)
            else 0
        ),
        "minimum_matched_fraction": float(np.min(fractions_array)),
        "median_matched_fraction": float(np.median(fractions_array)),
        "mean_matched_fraction": float(np.mean(fractions_array)),
        "maximum_matched_fraction": float(np.max(fractions_array)),
        "substantial_matched_fraction_overshoots": substantial_overshoots,
        "maximum_matched_fraction_overshoot": maximum_overshoot,
        "maximum_conditional_adjacency_support_asymmetry": float(
            np.max(np.abs(support - support.T))
        ),
        "maximum_evidence_identity_error": float(
            np.max(np.abs(adjacency_evidence - pair_coverage * support))
        ),
    }
    return ConditionalAdjacencySupportResult(
        conditional_adjacency_support=support,
        conditional_adjacency_support_variance=support_variance,
        conditional_adjacency_support_std=support_std,
        pair_coverage=pair_coverage,
        pair_is_covered=pair_is_covered,
        adjacency_evidence=adjacency_evidence,
        pair_contributing_inputs=contributing,
        matched_fractions=fractions_array,
        input_adjacencies=adjacencies,
        input_conditional_adjacency_support=input_supports,
        input_pair_coverage=input_coverages,
        warnings=warnings,
        diagnostics=diagnostics,
    )


def compute_metric_decomposition(
    distances: np.ndarray,
    finite_pair_mask: Optional[np.ndarray] = None,
    *,
    epsilon: float = 1e-15,
) -> MetricDecompositionResult:
    """Compute exact two-hop metric decomposition with O(n²) working memory.

    For square distance matrix ``D``, the decomposition distance is
    ``B_ij=min_{k not in {i,j}}(D_ik+D_kj)``. The raw gap is ``G=B-D`` and may
    be negative. ``metric_relative_gap=max(0,G)/(D+epsilon)`` and
    ``metric_directness=max(0,G)/(B+epsilon)``. Unavailable pairs are ``NaN``;
    diagonal entries are zero. Directness lies in ``[0,1]``.
    """
    D = np.asarray(distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distances must be square; got shape {D.shape}.")
    if D.shape[0] < 3:
        raise ValueError("At least three nodes are required for a two-hop test.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if finite_pair_mask is None:
        mask = np.isfinite(D) & (D > 0)
        mask &= mask.T
        np.fill_diagonal(mask, False)
    else:
        mask = np.asarray(finite_pair_mask, dtype=bool)
        if mask.shape != D.shape:
            raise ValueError(
                f"finite_pair_mask shape {mask.shape} does not match {D.shape}."
            )
        mask = mask & mask.T & np.isfinite(D) & (D > 0)
        np.fill_diagonal(mask, False)

    finite_distances = np.where(np.isfinite(D), D, np.inf)
    best = np.full_like(D, np.inf, dtype=float)
    for intermediate in range(D.shape[0]):
        candidate = np.add.outer(
            finite_distances[:, intermediate],
            finite_distances[intermediate, :],
        )
        candidate[intermediate, :] = np.inf
        candidate[:, intermediate] = np.inf
        np.minimum(best, candidate, out=best)

    available = mask & np.isfinite(best)
    best_output = np.full_like(D, np.nan, dtype=float)
    best_output[available] = best[available]
    np.fill_diagonal(best_output, 0.0)
    gap = np.full_like(D, np.nan, dtype=float)
    gap[available] = best[available] - D[available]
    relative_gap = np.full_like(D, np.nan, dtype=float)
    relative_gap[available] = np.maximum(0.0, gap[available]) / (
        D[available] + epsilon
    )
    directness = np.full_like(D, np.nan, dtype=float)
    directness[available] = np.maximum(0.0, gap[available]) / (
        best[available] + epsilon
    )
    for matrix in (gap, relative_gap, directness):
        matrix[:] = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 0.0)

    upper = np.triu(available, k=1)
    values = relative_gap[upper]
    gap_values = gap[upper]
    positive = values > np.finfo(float).eps
    materially_negative = gap_values < (
        -1e-10 * np.maximum(1.0, np.abs(D[upper]))
    )
    if np.any(directness[available] < -1e-10) or np.any(
        directness[available] > 1.0 + 1e-10
    ):
        raise AssertionError("metric_directness must lie in [0, 1].")
    if not np.allclose(
        relative_gap[available],
        np.maximum(0.0, gap[available]) / (D[available] + epsilon),
    ):
        raise AssertionError("metric_relative_gap identity failed.")
    if not np.allclose(
        directness[available],
        np.maximum(0.0, gap[available]) / (best_output[available] + epsilon),
    ):
        raise AssertionError("metric_directness identity failed.")
    diagnostics: Dict[str, Any] = {
        "finite_pair_count": int(values.size),
        "positive_metric_relative_gap_pair_count": int(np.count_nonzero(positive)),
        "positive_metric_relative_gap_fraction": (
            float(np.mean(positive)) if values.size else 0.0
        ),
        "minimum_metric_relative_gap": float(np.min(values)) if values.size else None,
        "median_metric_relative_gap": float(np.median(values)) if values.size else None,
        "mean_metric_relative_gap": float(np.mean(values)) if values.size else None,
        "maximum_metric_relative_gap": float(np.max(values)) if values.size else None,
        "materially_negative_metric_decomposition_gap_count": int(
            np.count_nonzero(materially_negative)
        ),
        "minimum_metric_decomposition_gap": (
            float(np.min(gap_values)) if gap_values.size else None
        ),
        "maximum_asymmetry": (
            float(np.nanmax(np.abs(relative_gap - relative_gap.T)))
            if values.size
            else 0.0
        ),
        "algorithm": "iterative min-plus exact-two-hop update",
        "working_memory_order": "O(n^2)",
    }
    warnings: List[str] = []
    negative_count = int(np.count_nonzero(materially_negative))
    if negative_count:
        warnings.append(
            f"Found {negative_count} materially negative metric decomposition "
            "gaps; inspect metric-repair diagnostics."
        )
    return MetricDecompositionResult(
        metric_decomposition_distance=best_output,
        metric_decomposition_gap=gap,
        metric_relative_gap=relative_gap,
        metric_directness=directness,
        finite_pair_mask=available,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def _constraint_statuses(
    type_u: Any,
    type_v: Any,
    confidence_u: float,
    confidence_v: float,
    scalar_u: float,
    scalar_v: float,
    confidence_threshold: float,
) -> Tuple[str, str]:
    """Return Morse-type and scalar-order compatibility statuses."""
    if pd.isna(type_u) or pd.isna(type_v):
        return "unavailable", "unavailable"
    type_u, type_v = int(type_u), int(type_v)
    if type_u == -1 or type_v == -1:
        return "regular_allowed", "not_applicable_regular"
    if type_u not in {0, 1, 2} or type_v not in {0, 1, 2}:
        return "unknown_type", "unavailable"
    high_confidence = (
        np.isfinite(confidence_u)
        and np.isfinite(confidence_v)
        and confidence_u >= confidence_threshold
        and confidence_v >= confidence_threshold
    )
    if not high_confidence:
        return "uncertain", "uncertain"

    pair = frozenset((type_u, type_v))
    type_status = (
        "compatible" if pair in {frozenset((0, 1)), frozenset((1, 2))} else "violation"
    )
    if not np.isfinite(scalar_u) or not np.isfinite(scalar_v):
        scalar_status = "unavailable"
    elif type_u == type_v:
        scalar_status = "not_applicable_equal_type"
    else:
        lower_scalar, upper_scalar = (
            (scalar_u, scalar_v) if type_u < type_v else (scalar_v, scalar_u)
        )
        scalar_status = (
            "compatible" if lower_scalar <= upper_scalar + 1e-12 else "violation"
        )
    return type_status, scalar_status


def construct_candidate_edge_table(
    nodes: pd.DataFrame,
    structural_distances: np.ndarray,
    metric_relative_gap: np.ndarray,
    config: ReconstructionConfig,
    *,
    conditional_adjacency_support: Optional[np.ndarray] = None,
    conditional_adjacency_support_std: Optional[np.ndarray] = None,
    pair_coverage: Optional[np.ndarray] = None,
    pair_is_covered: Optional[np.ndarray] = None,
    adjacency_evidence: Optional[np.ndarray] = None,
    pair_contributing_inputs: Optional[np.ndarray] = None,
    metric_decomposition_distance: Optional[np.ndarray] = None,
    metric_decomposition_gap: Optional[np.ndarray] = None,
    metric_directness: Optional[np.ndarray] = None,
) -> CandidateEdgeResult:
    """Build the canonical finite-pair candidate table and candidate union.

    Every matrix has shape ``(n_center, n_center)``. Conditional adjacency
    support, its standard deviation, pair coverage, and adjacency evidence are
    in ``[0,1]``; uncovered transport entries are zero and identified by
    ``pair_is_covered``. Metric columns expose ``B``, raw ``G=B-D``,
    ``max(0,G)/(D+epsilon)``, and ``max(0,G)/(B+epsilon)`` separately.
    """
    required_columns = {
        "node_id",
        "x",
        "y",
        "z",
        "scalar_value",
        "morse_type",
        "type_confidence",
        "mass",
    }
    missing_columns = sorted(required_columns - set(nodes.columns))
    if missing_columns:
        raise ValueError(f"nodes dataframe is missing columns {missing_columns}.")
    D = np.asarray(structural_distances, dtype=float)
    delta = np.asarray(metric_relative_gap, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or delta.shape != D.shape:
        raise ValueError(
            "structural_distances and metric_relative_gap must be aligned square matrices."
        )
    n = D.shape[0]
    if set(nodes["node_id"].astype(int)) != set(range(n)) or len(nodes) != n:
        raise ValueError("nodes must contain exactly one row for every node id 0..n-1.")
    indexed_nodes = nodes.set_index("node_id").loc[range(n)]

    support_mode = conditional_adjacency_support is not None
    support = (
        np.asarray(conditional_adjacency_support, dtype=float)
        if support_mode
        else np.zeros_like(D)
    )
    support_std = (
        np.asarray(conditional_adjacency_support_std, dtype=float)
        if conditional_adjacency_support_std is not None
        else np.zeros_like(D)
    )
    coverage = (
        np.asarray(pair_coverage, dtype=float)
        if pair_coverage is not None
        else np.zeros_like(D)
    )
    covered = (
        np.asarray(pair_is_covered, dtype=bool)
        if pair_is_covered is not None
        else coverage > 0
    )
    evidence = (
        np.asarray(adjacency_evidence, dtype=float)
        if adjacency_evidence is not None
        else coverage * support
    )
    contributing = (
        np.asarray(pair_contributing_inputs)
        if pair_contributing_inputs is not None
        else np.zeros_like(D, dtype=int)
    )
    for name, matrix in {
        "conditional_adjacency_support": support,
        "conditional_adjacency_support_std": support_std,
        "pair_coverage": coverage,
        "pair_is_covered": covered,
        "adjacency_evidence": evidence,
        "pair_contributing_inputs": contributing,
    }.items():
        if matrix.shape != D.shape:
            raise ValueError(f"{name} shape {matrix.shape} does not match {D.shape}.")
    for name, matrix in {
        "metric_decomposition_distance": metric_decomposition_distance,
        "metric_decomposition_gap": metric_decomposition_gap,
        "metric_directness": metric_directness,
    }.items():
        if matrix is not None and np.asarray(matrix).shape != D.shape:
            raise ValueError(
                f"{name} shape {np.asarray(matrix).shape} does not match {D.shape}."
            )
    range_tolerance = 1e-10
    for name, matrix in {
        "conditional_adjacency_support": support,
        "conditional_adjacency_support_std": support_std,
        "pair_coverage": coverage,
        "adjacency_evidence": evidence,
    }.items():
        if np.any(matrix < -range_tolerance) or np.any(
            matrix > 1.0 + range_tolerance
        ):
            raise ValueError(f"{name} must lie in [0, 1] within tolerance.")
    if support_mode and not np.allclose(
        evidence, coverage * support, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "adjacency_evidence must equal pair_coverage * "
            "conditional_adjacency_support."
        )

    finite_upper = np.triu(np.isfinite(D) & (D > 0), k=1)
    source, target = np.where(finite_upper)
    coords = indexed_nodes[["x", "y", "z"]].to_numpy(dtype=float)
    euclidean = np.linalg.norm(coords[source] - coords[target], axis=1)
    masses = indexed_nodes["mass"].to_numpy(dtype=float)
    types = indexed_nodes["morse_type"].to_numpy()
    confidences = indexed_nodes["type_confidence"].to_numpy(dtype=float)
    scalars = indexed_nodes["scalar_value"].to_numpy(dtype=float)
    statuses = [
        _constraint_statuses(
            types[i],
            types[j],
            confidences[i],
            confidences[j],
            scalars[i],
            scalars[j],
            config.type_confidence_threshold,
        )
        for i, j in zip(source, target)
    ]
    type_status = np.asarray([status[0] for status in statuses], dtype=object)
    scalar_status = np.asarray([status[1] for status in statuses], dtype=object)

    table = pd.DataFrame(
        {
            "source": source,
            "target": target,
            "structural_distance": D[source, target],
            "weight": D[source, target],
            "euclidean_distance": euclidean,
            "conditional_adjacency_support": support[source, target],
            "conditional_adjacency_support_std": support_std[source, target],
            "pair_coverage": coverage[source, target],
            "pair_is_covered": covered[source, target],
            "adjacency_evidence": evidence[source, target],
            "n_contributing_inputs": contributing[source, target].astype(int),
            "metric_decomposition_distance": (
                np.asarray(metric_decomposition_distance)[source, target]
                if metric_decomposition_distance is not None
                else np.full(source.shape, np.nan)
            ),
            "metric_decomposition_gap": (
                np.asarray(metric_decomposition_gap)[source, target]
                if metric_decomposition_gap is not None
                else np.full(source.shape, np.nan)
            ),
            "metric_relative_gap": delta[source, target],
            "metric_directness": (
                np.asarray(metric_directness)[source, target]
                if metric_directness is not None
                else np.full(source.shape, np.nan)
            ),
            "source_mass": masses[source],
            "target_mass": masses[target],
            "source_type": types[source],
            "target_type": types[target],
            "source_type_confidence": confidences[source],
            "target_type_confidence": confidences[target],
            "source_scalar": scalars[source],
            "target_scalar": scalars[target],
            "scalar_difference": np.abs(scalars[source] - scalars[target]),
            "type_compatibility_status": type_status,
            "scalar_order_status": scalar_status,
        }
    )
    table["type_pair"] = [
        "unavailable"
        if pd.isna(type_u) or pd.isna(type_v)
        else f"{min(int(type_u), int(type_v))}–{max(int(type_u), int(type_v))}"
        for type_u, type_v in zip(types[source], types[target])
    ]

    support_component = np.clip(
        table["conditional_adjacency_support"].to_numpy(dtype=float), 0.0, 1.0
    )
    metric_relative_gap_component = (
        table["metric_relative_gap"]
        .rank(method="average", pct=True)
        .fillna(0.0)
        .to_numpy()
    )
    pair_coverage_component = np.clip(
        table["pair_coverage"].to_numpy(dtype=float), 0.0, 1.0
    )
    uncertainty_component = np.clip(
        table["conditional_adjacency_support_std"].to_numpy(dtype=float) / 0.5,
        0.0,
        1.0,
    )
    type_penalty = (
        config.type_violation_penalty
        * (type_status == "violation")
        * (config.type_constraint_mode == "soft")
    )
    scalar_penalty = (
        config.scalar_order_violation_penalty
        * (scalar_status == "violation")
        * (config.scalar_order_constraint_mode == "soft")
    )
    table["conditional_support_component"] = support_component
    table["metric_relative_gap_percentile_component"] = metric_relative_gap_component
    table["pair_coverage_component"] = pair_coverage_component
    table["uncertainty_penalty_component"] = uncertainty_component
    table["type_violation_penalty"] = type_penalty
    table["scalar_order_violation_penalty"] = scalar_penalty
    table["evidence_score"] = (
        config.evidence_conditional_support_weight * support_component
        + config.evidence_metric_relative_gap_weight * metric_relative_gap_component
        + config.evidence_pair_coverage_weight * pair_coverage_component
        - config.evidence_uncertainty_penalty * uncertainty_component
        - type_penalty
        - scalar_penalty
    )

    pair_index = np.full((n, n), -1, dtype=int)
    row_ids = np.arange(len(table))
    pair_index[source, target] = row_ids
    pair_index[target, source] = row_ids
    reasons: List[set] = [set() for _ in range(len(table))]

    def mark_top_k(matrix: np.ndarray, k: int, reason: str) -> None:
        for node in range(n):
            values = np.asarray(matrix[node], dtype=float).copy()
            valid = np.isfinite(values) & np.isfinite(D[node]) & (D[node] > 0)
            valid[node] = False
            candidates = np.where(valid)[0]
            if candidates.size == 0:
                continue
            ranked = candidates[np.argsort(values[candidates], kind="stable")[::-1]]
            for neighbor in ranked[: min(k, ranked.size)]:
                reasons[pair_index[node, neighbor]].add(reason)

    if support_mode:
        mark_top_k(
            support,
            config.top_k_conditional_support,
            "top_k_conditional_adjacency_support",
        )
        covered_values = table.loc[
            (
                table["pair_coverage"]
                >= config.minimum_pair_coverage
            )
            & (table["conditional_adjacency_support"] > 0),
            "conditional_adjacency_support",
        ].dropna()
        global_support_threshold = (
            float(
                covered_values.quantile(config.global_conditional_support_quantile)
            )
            if len(covered_values)
            else None
        )
        if global_support_threshold is not None:
            global_rows = table.index[
                (
                    table["conditional_adjacency_support"]
                    >= global_support_threshold
                )
                & (table["conditional_adjacency_support"] > 0)
                & (
                    table["pair_coverage"]
                    >= config.minimum_pair_coverage
                )
            ]
            for row in global_rows:
                reasons[row].add("global_strong_conditional_adjacency_support")
    else:
        global_support_threshold = None
        mark_top_k(
            -D,
            config.top_k_conditional_support,
            "nearest_structural_metric_only",
        )
    mark_top_k(
        delta,
        config.top_k_metric_relative_gap,
        "top_k_metric_relative_gap",
    )

    selected = np.asarray([bool(reason) for reason in reasons], dtype=bool)
    hard_type = (
        (type_status == "violation") & (config.type_constraint_mode == "hard")
    )
    hard_scalar = (
        (scalar_status == "violation")
        & (config.scalar_order_constraint_mode == "hard")
    )
    hard_excluded = hard_type | hard_scalar
    selected &= ~hard_excluded
    for row in np.where(hard_type)[0]:
        reasons[row].add("excluded_hard_type")
    for row in np.where(hard_scalar)[0]:
        reasons[row].add("excluded_hard_scalar_order")

    degree = np.bincount(
        np.concatenate(
            [source[selected], target[selected]]
        ),
        minlength=n,
    )
    for node in np.where(degree == 0)[0]:
        incident = np.where(((source == node) | (target == node)) & ~hard_excluded)[0]
        if incident.size == 0:
            raise ValueError(
                f"Hard constraints leave node {node} without any eligible candidate pair."
            )
        ranking = table.loc[
            incident, ["evidence_score", "structural_distance"]
        ].sort_values(
            ["evidence_score", "structural_distance"],
            ascending=[False, True],
        )
        chosen = int(ranking.index[0])
        selected[chosen] = True
        reasons[chosen].add("isolation_repair")
        other = target[chosen] if source[chosen] == node else source[chosen]
        degree[node] += 1
        degree[other] += 1

    table["candidate_selected"] = selected
    table["candidate_selection_reason"] = [
        "|".join(sorted(reason)) if reason else "not_selected" for reason in reasons
    ]
    selected_table = table.loc[selected].copy().reset_index(drop=True)
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(
        (int(node_id), dict(row))
        for node_id, row in indexed_nodes.to_dict(orient="index").items()
    )
    for row in selected_table.to_dict(orient="records"):
        candidate_graph.add_edge(
            int(row["source"]),
            int(row["target"]),
            **_spanner_edge_attributes(row, "candidate"),
            candidate_selection_reason=row["candidate_selection_reason"],
        )
    if set(candidate_graph.nodes()) != set(range(n)):
        raise AssertionError("Candidate graph does not preserve every barycenter node.")
    if any(candidate_graph.degree(node) == 0 for node in candidate_graph.nodes()):
        raise AssertionError("Every barycenter node must have at least one candidate edge.")

    degrees = np.asarray([candidate_graph.degree(node) for node in range(n)], dtype=int)
    evidence_formula = (
        f"{config.evidence_conditional_support_weight:g}·conditional_adjacency_support + "
        f"{config.evidence_metric_relative_gap_weight:g}·metric_relative_gap_percentile + "
        f"{config.evidence_pair_coverage_weight:g}·pair_coverage − "
        f"{config.evidence_uncertainty_penalty:g}·"
        "min(conditional_adjacency_support_std/0.5, 1)"
        " − active soft constraint penalties"
    )
    warnings: List[str] = []
    if not support_mode:
        warnings.append(
            "Metric-only mode: conditional adjacency components are zero and nearest "
            "structural neighbors supplement metric-relative-gap candidates."
        )
    diagnostics: Dict[str, Any] = {
        "n_all_finite_pairs": int(len(table)),
        "n_selected_candidates": int(len(selected_table)),
        "candidate_fraction": float(len(selected_table) / len(table)),
        "minimum_candidates_per_node": int(np.min(degrees)),
        "median_candidates_per_node": float(np.median(degrees)),
        "mean_candidates_per_node": float(np.mean(degrees)),
        "maximum_candidates_per_node": int(np.max(degrees)),
        "nodes_without_candidates": int(np.count_nonzero(degrees == 0)),
        "global_conditional_adjacency_support_threshold": global_support_threshold,
        "hard_excluded_pair_count": int(np.count_nonzero(hard_excluded)),
        "conditional_adjacency_mode": bool(support_mode),
    }
    return CandidateEdgeResult(
        all_pairs=table,
        selected_edges=selected_table,
        graph=candidate_graph,
        diagnostics=diagnostics,
        warnings=warnings,
        evidence_formula=evidence_formula,
    )


def infer_target_component_count(
    bundle: Mapping[str, Any], configured_count: Optional[int] = None
) -> ComponentTarget:
    """Infer the desired component count from aligned input graphs."""
    graphs = bundle.get("Gs")
    input_counts: List[int] = []
    if graphs is not None:
        for graph in graphs:
            undirected = graph.to_undirected() if graph.is_directed() else graph
            input_counts.append(int(nx.number_connected_components(undirected)))
    if input_counts and len(set(input_counts)) == 1:
        return ComponentTarget(
            count=input_counts[0],
            source="unanimous input graph component count",
            input_component_counts=input_counts,
            assumption_used=False,
        )
    if configured_count is not None:
        if configured_count < 1:
            raise ValueError("configured_count must be a positive integer.")
        return ComponentTarget(
            count=int(configured_count),
            source="configured target_component_count",
            input_component_counts=input_counts,
            assumption_used=False,
        )
    return ComponentTarget(
        count=1,
        source="default connected-component assumption",
        input_component_counts=input_counts,
        assumption_used=True,
    )


def _spanner_edge_attributes(row: Mapping[str, Any], source_stage: str) -> Dict[str, Any]:
    """Select stable edge attributes from a candidate-table row."""
    return {
        "weight": float(row["structural_distance"]),
        "conditional_adjacency_support": float(
            row["conditional_adjacency_support"]
        ),
        "conditional_adjacency_support_std": float(
            row["conditional_adjacency_support_std"]
        ),
        "pair_coverage": float(row["pair_coverage"]),
        "pair_is_covered": bool(row["pair_is_covered"]),
        "adjacency_evidence": float(row["adjacency_evidence"]),
        "n_contributing_inputs": int(row["n_contributing_inputs"]),
        "metric_decomposition_distance": float(
            row["metric_decomposition_distance"]
        ),
        "metric_decomposition_gap": float(row["metric_decomposition_gap"]),
        "metric_relative_gap": float(row["metric_relative_gap"]),
        "metric_directness": float(row["metric_directness"]),
        "evidence_score": float(row["evidence_score"]),
        "source_stage": source_stage,
        "type_compatibility_status": row["type_compatibility_status"],
        "scalar_order_status": row["scalar_order_status"],
        "type_compatible": row["type_compatibility_status"] != "violation",
        "scalar_order_compatible": row["scalar_order_status"] != "violation",
    }


def _straight_line_crossing_count(graph: nx.Graph, nodes: pd.DataFrame) -> Optional[int]:
    """Count proper crossings in the fixed-coordinate ``(x, y)`` drawing."""
    indexed = nodes.set_index("node_id")
    node_order = list(graph.nodes())
    xy = indexed.loc[node_order, ["x", "y"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(xy)):
        return None
    positions = {
        int(node): xy[position] for position, node in enumerate(node_order)
    }
    edges = list(graph.edges())
    total = 0
    tolerance = 1e-12
    edge_u = np.asarray([int(edge[0]) for edge in edges], dtype=int)
    edge_v = np.asarray([int(edge[1]) for edge in edges], dtype=int)
    starts = np.asarray([positions[int(u)] for u in edge_u])
    ends = np.asarray([positions[int(v)] for v in edge_v])
    for index in range(len(edges) - 1):
        u, v = edge_u[index], edge_v[index]
        later_u, later_v = edge_u[index + 1 :], edge_v[index + 1 :]
        not_shared = (
            (later_u != u)
            & (later_v != u)
            & (later_u != v)
            & (later_v != v)
        )
        if not np.any(not_shared):
            continue
        a, b = starts[index], ends[index]
        c = starts[index + 1 :][not_shared]
        d = ends[index + 1 :][not_shared]
        ab = b - a
        cd = d - c
        orient_abc = ab[0] * (c[:, 1] - a[1]) - ab[1] * (c[:, 0] - a[0])
        orient_abd = ab[0] * (d[:, 1] - a[1]) - ab[1] * (d[:, 0] - a[0])
        orient_cda = cd[:, 0] * (a[1] - c[:, 1]) - cd[:, 1] * (a[0] - c[:, 0])
        orient_cdb = cd[:, 0] * (b[1] - c[:, 1]) - cd[:, 1] * (b[0] - c[:, 0])
        non_collinear = (
            (np.abs(orient_abc) >= tolerance)
            & (np.abs(orient_abd) >= tolerance)
            & (np.abs(orient_cda) >= tolerance)
            & (np.abs(orient_cdb) >= tolerance)
        )
        crosses = (
            ((orient_abc > 0) != (orient_abd > 0))
            & ((orient_cda > 0) != (orient_cdb > 0))
            & non_collinear
        )
        total += int(np.count_nonzero(crosses))
    return total


def _spanner_metric_diagnostics(
    graph: nx.Graph,
    structural_distances: np.ndarray,
    quantile: float,
) -> Dict[str, Any]:
    """Compute stress and stretch against finite structural distances."""
    D = np.asarray(structural_distances, dtype=float)
    n = D.shape[0]
    graph_distances = np.full((n, n), np.inf, dtype=float)
    for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight="weight"):
        for target, distance in lengths.items():
            graph_distances[int(source), int(target)] = float(distance)
    np.fill_diagonal(graph_distances, 0.0)
    target_pairs = np.triu(np.isfinite(D) & (D > 0), k=1)
    evaluated = target_pairs & np.isfinite(graph_distances)
    target_values = D[evaluated]
    graph_values = graph_distances[evaluated]
    if target_values.size:
        residual = graph_values - target_values
        normalized_stress = float(
            np.sqrt(np.sum(residual * residual) / np.sum(target_values * target_values))
        )
        stretch_values = graph_values / target_values
        maximum_stretch = float(np.max(stretch_values))
        high_quantile_stretch = float(np.quantile(stretch_values, quantile))
    else:
        normalized_stress = np.nan
        maximum_stretch = np.nan
        high_quantile_stretch = np.nan
    target_count = int(np.count_nonzero(target_pairs))
    return {
        "normalized_metric_stress": normalized_stress,
        "maximum_stretch": maximum_stretch,
        "high_quantile_stretch": high_quantile_stretch,
        "metric_pair_coverage": (
            float(np.count_nonzero(evaluated) / target_count) if target_count else 1.0
        ),
    }


def build_conditional_adjacency_aware_spanner(
    nodes: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    all_finite_pairs: pd.DataFrame,
    structural_distances: np.ndarray,
    *,
    stretch: float,
    target_component_count: int,
    planarity_mode: ConstraintMode = "off",
    force_planarity_in_3d: bool = False,
    is_genuinely_3d: bool = False,
    stretch_diagnostic_quantile: float = 0.95,
    type_constraint_mode: ConstraintMode = "soft",
    scalar_order_constraint_mode: ConstraintMode = "soft",
) -> SpannerResult:
    """Build a constrained greedy metric spanner, preserving every node."""
    if stretch < 1.0:
        raise ValueError("stretch must be at least 1.")
    n = len(nodes)
    if target_component_count < 1 or target_component_count > max(n, 1):
        raise ValueError("target_component_count must lie between 1 and node count.")
    if planarity_mode not in {"off", "soft", "hard"}:
        raise ValueError("planarity_mode must be 'off', 'soft', or 'hard'.")
    effective_planarity: ConstraintMode = (
        "off"
        if is_genuinely_3d and not force_planarity_in_3d
        else planarity_mode
    )

    graph = nx.Graph()
    indexed_nodes = nodes.set_index("node_id")
    graph.add_nodes_from(
        (int(node_id), dict(attributes))
        for node_id, attributes in indexed_nodes.to_dict(orient="index").items()
    )
    ordered = candidate_edges.sort_values(
        ["structural_distance", "evidence_score"],
        ascending=[True, False],
        kind="stable",
    )
    union_find = nx.utils.UnionFind(graph.nodes())
    component_count = n
    rejected_nonplanar = 0
    for row in ordered.to_dict(orient="records"):
        u, v = int(row["source"]), int(row["target"])
        different_components = union_find[u] != union_find[v]
        if different_components and component_count <= target_component_count:
            continue
        threshold = stretch * float(row["structural_distance"])
        reachable = nx.single_source_dijkstra_path_length(
            graph, u, cutoff=threshold, weight="weight"
        )
        if reachable.get(v, np.inf) <= threshold:
            continue
        if effective_planarity == "hard" and not geometric_planarity_allows_edge(
            graph, u, v
        ):
            rejected_nonplanar += 1
            continue
        graph.add_edge(u, v, **_spanner_edge_attributes(row, "spanner"))
        if different_components:
            union_find.union(u, v)
            component_count -= 1

    fallback_records: List[Dict[str, Any]] = []
    while nx.number_connected_components(graph) > target_component_count:
        component_id: Dict[int, int] = {}
        for index, component in enumerate(nx.connected_components(graph)):
            for node in component:
                component_id[int(node)] = index
        pairs = all_finite_pairs
        intercomponent = pairs[
            [
                component_id[int(u)] != component_id[int(v)]
                for u, v in zip(pairs["source"], pairs["target"])
            ]
        ].copy()
        if type_constraint_mode == "hard":
            intercomponent = intercomponent[
                intercomponent["type_compatibility_status"] != "violation"
            ]
        if scalar_order_constraint_mode == "hard":
            intercomponent = intercomponent[
                intercomponent["scalar_order_status"] != "violation"
            ]
        intercomponent["bridge_evidence"] = (
            intercomponent["evidence_score"]
            + intercomponent["type_violation_penalty"]
            + intercomponent["scalar_order_violation_penalty"]
        )
        ranked_bridges = intercomponent.sort_values(
            ["bridge_evidence", "structural_distance"],
            ascending=[False, True],
            kind="stable",
        )
        chosen: Optional[Dict[str, Any]] = None
        for row in ranked_bridges.to_dict(orient="records"):
            u, v = int(row["source"]), int(row["target"])
            if graph.has_edge(u, v):
                continue
            if effective_planarity == "hard" and not geometric_planarity_allows_edge(
                graph, u, v
            ):
                rejected_nonplanar += 1
                continue
            chosen = row
            break
        if chosen is None:
            raise RuntimeError(
                "No feasible finite-pair bridge can reach the target component count "
                "without violating hard constraints."
            )
        u, v = int(chosen["source"]), int(chosen["target"])
        graph.add_edge(u, v, **_spanner_edge_attributes(chosen, "fallback_bridge"))
        fallback_records.append(
            {
                "source": u,
                "target": v,
                "weight": float(chosen["structural_distance"]),
                "evidence_score": float(chosen["evidence_score"]),
                "type_compatibility_status": chosen["type_compatibility_status"],
                "scalar_order_status": chosen["scalar_order_status"],
            }
        )

    if set(graph.nodes()) != set(range(n)):
        raise AssertionError("Spanner must preserve every barycenter node.")
    components = int(nx.number_connected_components(graph))
    if components != target_component_count:
        raise AssertionError(
            f"Spanner has {components} components, expected {target_component_count}."
        )
    metric_diagnostics = _spanner_metric_diagnostics(
        graph, structural_distances, stretch_diagnostic_quantile
    )
    conditional_support_values = np.asarray(
        [
            data.get("conditional_adjacency_support", np.nan)
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )
    pair_coverage_values = np.asarray(
        [data.get("pair_coverage", np.nan) for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    crossing_count = _straight_line_crossing_count(graph, nodes)
    diagnostics: Dict[str, Any] = {
        "stretch_parameter": float(stretch),
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "connected_components": components,
        "cycle_rank": int(graph.number_of_edges() - graph.number_of_nodes() + components),
        "mean_degree": float(
            2.0 * graph.number_of_edges() / max(graph.number_of_nodes(), 1)
        ),
        **metric_diagnostics,
        "mean_selected_conditional_adjacency_support": (
            float(np.nanmean(conditional_support_values))
            if np.any(np.isfinite(conditional_support_values))
            else None
        ),
        "mean_pair_coverage": (
            float(np.nanmean(pair_coverage_values))
            if np.any(np.isfinite(pair_coverage_values))
            else None
        ),
        "type_violations": int(
            sum(
                data.get("type_compatibility_status") == "violation"
                for _, _, data in graph.edges(data=True)
            )
        ),
        "scalar_order_violations": int(
            sum(
                data.get("scalar_order_status") == "violation"
                for _, _, data in graph.edges(data=True)
            )
        ),
        "geometric_planar": None if crossing_count is None else bool(crossing_count == 0),
        "straight_line_crossing_count": crossing_count,
        "fallback_bridge_count": int(len(fallback_records)),
        "effective_planarity_mode": effective_planarity,
        "rejected_nonplanar_edges": int(rejected_nonplanar),
    }
    return SpannerResult(
        graph=graph,
        stretch=float(stretch),
        target_component_count=target_component_count,
        effective_planarity_mode=effective_planarity,
        fallback_bridges=pd.DataFrame.from_records(
            fallback_records,
            columns=[
                "source",
                "target",
                "weight",
                "evidence_score",
                "type_compatibility_status",
                "scalar_order_status",
            ],
        ),
        diagnostics=diagnostics,
        rejected_nonplanar_edges=rejected_nonplanar,
    )


def save_graph_pickle(graph: nx.Graph, path: PathLike) -> Path:
    """Save a NetworkX graph with the repository's pickle convention."""
    import pickle

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def input_graph_topology_statistics(
    bundle: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-input topology statistics and weighted distribution summaries."""
    graphs = bundle.get("Gs")
    if graphs is None:
        return pd.DataFrame(), pd.DataFrame()
    weights = np.asarray(bundle.get("lambdas"), dtype=float)
    if weights.shape != (len(graphs),):
        raise ValueError("lambdas must align with input graphs.")
    rows: List[Dict[str, Any]] = []
    for index, (graph, weight) in enumerate(zip(graphs, weights)):
        undirected = graph.to_undirected() if graph.is_directed() else graph
        n = undirected.number_of_nodes()
        m = undirected.number_of_edges()
        components = nx.number_connected_components(undirected)
        cycle_rank = m - n + components
        rows.append(
            {
                "input_index": index,
                "lambda": float(weight),
                "node_count": n,
                "edge_count": m,
                "mean_degree": 2.0 * m / max(n, 1),
                "edge_to_node_ratio": m / max(n, 1),
                "component_count": components,
                "cycle_rank": cycle_rank,
                "normalized_cycle_rank": cycle_rank / max(n, 1),
            }
        )
    table = pd.DataFrame(rows)
    normalized_weights = weights / np.sum(weights)
    summary_rows: List[Dict[str, Any]] = []
    metrics = [
        "mean_degree",
        "edge_to_node_ratio",
        "component_count",
        "cycle_rank",
        "normalized_cycle_rank",
    ]
    for metric in metrics:
        values = table[metric].to_numpy(dtype=float)
        weighted_mean = float(np.sum(normalized_weights * values))
        weighted_std = float(
            np.sqrt(np.sum(normalized_weights * (values - weighted_mean) ** 2))
        )
        summary_rows.append(
            {
                "metric": metric,
                "weighted_mean": weighted_mean,
                "weighted_std": weighted_std,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return table, pd.DataFrame(summary_rows)


def graph_objective_components(
    graph: nx.Graph,
    structural_distances: np.ndarray,
    config: ReconstructionConfig,
) -> Dict[str, float]:
    """Evaluate every explicit term of the bounded local-repair objective."""
    metric_error = _spanner_metric_diagnostics(
        graph, structural_distances, config.stretch_diagnostic_quantile
    )["normalized_metric_stress"]
    supports = np.asarray(
        [
            data.get("conditional_adjacency_support", np.nan)
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )
    uncertainties = np.asarray(
        [
            data.get("conditional_adjacency_support_std", np.nan)
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )
    mean_support = float(np.nanmean(supports)) if np.any(np.isfinite(supports)) else 0.0
    mean_uncertainty = (
        float(np.nanmean(uncertainties))
        if np.any(np.isfinite(uncertainties))
        else 0.0
    )
    violation_count = sum(
        (data.get("type_compatibility_status") == "violation")
        + (data.get("scalar_order_status") == "violation")
        for _, _, data in graph.edges(data=True)
    )
    violation_rate = violation_count / max(graph.number_of_edges(), 1)
    terms = {
        "metric_error": float(metric_error),
        "edge_penalty": float(
            config.objective_edge_penalty_weight * graph.number_of_edges()
        ),
        "conditional_support_reward": float(
            -config.objective_conditional_support_reward_weight * mean_support
        ),
        "uncertainty_penalty": float(
            config.objective_uncertainty_penalty_weight * mean_uncertainty
        ),
        "constraint_penalty": float(
            config.objective_constraint_penalty_weight * violation_rate
        ),
    }
    terms["total_objective"] = float(sum(terms.values()))
    return terms


def augment_consensus_edges(
    spanner: nx.Graph,
    candidate_edges: pd.DataFrame,
    *,
    conditional_support_quantile: float,
    minimum_pair_coverage: float,
    planarity_mode: ConstraintMode = "off",
    type_constraint_mode: ConstraintMode = "soft",
    scalar_order_constraint_mode: ConstraintMode = "soft",
) -> ConsensusAugmentationResult:
    """Add omitted edges above one conditional-adjacency-support quantile."""
    if not 0.0 <= conditional_support_quantile <= 1.0:
        raise ValueError("conditional_support_quantile must lie in [0, 1].")
    graph = spanner.copy()
    omitted = candidate_edges[
        [
            not graph.has_edge(int(u), int(v))
            for u, v in zip(candidate_edges["source"], candidate_edges["target"])
        ]
    ].copy()
    eligible_support = omitted.loc[
        (omitted["pair_coverage"] >= minimum_pair_coverage)
        & (omitted["conditional_adjacency_support"] > 0),
        "conditional_adjacency_support",
    ].dropna()
    threshold = (
        float(eligible_support.quantile(conditional_support_quantile))
        if len(eligible_support)
        else None
    )
    if threshold is None:
        return ConsensusAugmentationResult(
            graph=graph,
            conditional_support_quantile=conditional_support_quantile,
            conditional_support_threshold=None,
            added_edges=pd.DataFrame(),
            diagnostics={"added_edge_count": 0, "rejected_nonplanar": 0},
        )
    eligible = omitted[
        (omitted["conditional_adjacency_support"] >= threshold)
        & (omitted["pair_coverage"] >= minimum_pair_coverage)
    ].copy()
    if type_constraint_mode == "hard":
        eligible = eligible[
            eligible["type_compatibility_status"] != "violation"
        ]
    if scalar_order_constraint_mode == "hard":
        eligible = eligible[eligible["scalar_order_status"] != "violation"]
    eligible["uncertainty_adjusted_evidence"] = (
        eligible["evidence_score"]
        - eligible["uncertainty_penalty_component"]
    )
    eligible = eligible.sort_values(
        [
            "conditional_adjacency_support",
            "pair_coverage",
            "uncertainty_adjusted_evidence",
            "structural_distance",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    )

    added: List[Dict[str, Any]] = []
    rejected_nonplanar = 0
    for row in eligible.to_dict(orient="records"):
        u, v = int(row["source"]), int(row["target"])
        if graph.has_edge(u, v):
            continue
        if planarity_mode == "hard" and not geometric_planarity_allows_edge(
            graph, u, v
        ):
            rejected_nonplanar += 1
            continue
        graph.add_edge(
            u, v, **_spanner_edge_attributes(row, "consensus_augmentation")
        )
        added.append(
            {
                "source": u,
                "target": v,
                "weight": float(row["structural_distance"]),
                "conditional_adjacency_support": float(
                    row["conditional_adjacency_support"]
                ),
                "pair_coverage": float(row["pair_coverage"]),
                "evidence_score": float(row["evidence_score"]),
            }
        )
    return ConsensusAugmentationResult(
        graph=graph,
        conditional_support_quantile=conditional_support_quantile,
        conditional_support_threshold=threshold,
        added_edges=pd.DataFrame(added),
        diagnostics={
            "added_edge_count": len(added),
            "rejected_nonplanar": rejected_nonplanar,
        },
    )


def summarize_topology_graph(
    graph: nx.Graph,
    structural_distances: np.ndarray,
    nodes: pd.DataFrame,
    *,
    low_conditional_support_threshold: float,
    stretch_quantile: float = 0.95,
) -> Dict[str, Any]:
    """Summarize topology, metric quality, evidence, and drawing crossings."""
    metric = _spanner_metric_diagnostics(
        graph, structural_distances, stretch_quantile
    )
    supports = np.asarray(
        [
            data.get("conditional_adjacency_support", np.nan)
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )
    finite_support = supports[np.isfinite(supports)]
    components = nx.number_connected_components(graph)
    return {
        "edge_count": int(graph.number_of_edges()),
        "cycle_rank": int(graph.number_of_edges() - graph.number_of_nodes() + components),
        "metric_stress": metric["normalized_metric_stress"],
        "maximum_stretch": metric["maximum_stretch"],
        "mean_conditional_adjacency_support": (
            float(np.mean(finite_support)) if finite_support.size else None
        ),
        "median_conditional_adjacency_support": (
            float(np.median(finite_support)) if finite_support.size else None
        ),
        "low_conditional_adjacency_support_edge_count": int(
            np.count_nonzero(finite_support <= low_conditional_support_threshold)
        ),
        "type_violations": int(
            sum(
                data.get("type_compatibility_status") == "violation"
                for _, _, data in graph.edges(data=True)
            )
        ),
        "crossing_count": _straight_line_crossing_count(graph, nodes),
    }


def local_repair_topology(
    graph: nx.Graph,
    candidate_edges: pd.DataFrame,
    structural_distances: np.ndarray,
    config: ReconstructionConfig,
    *,
    target_component_count: int,
    planarity_mode: ConstraintMode = "off",
) -> LocalRepairResult:
    """Apply deterministic bounded remove/add/swap/fallback-replacement moves."""
    current = graph.copy()
    objective_before = graph_objective_components(
        current, structural_distances, config
    )
    current_objective = objective_before
    history: List[Dict[str, Any]] = []
    edited_pairs: set = set()
    support_values = candidate_edges["conditional_adjacency_support"].dropna()
    low_support_threshold = (
        float(
            support_values.quantile(
                config.local_repair_low_conditional_support_quantile
            )
        )
        if len(support_values)
        else 0.0
    )

    def pair_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def eligible_additions(base: nx.Graph) -> pd.DataFrame:
        omitted = candidate_edges[
            [
                not base.has_edge(int(u), int(v))
                and pair_key(int(u), int(v)) not in edited_pairs
                for u, v in zip(candidate_edges["source"], candidate_edges["target"])
            ]
        ].copy()
        if config.type_constraint_mode == "hard":
            omitted = omitted[
                omitted["type_compatibility_status"] != "violation"
            ]
        if config.scalar_order_constraint_mode == "hard":
            omitted = omitted[omitted["scalar_order_status"] != "violation"]
        return omitted.sort_values(
            [
                "evidence_score",
                "conditional_adjacency_support",
                "structural_distance",
            ],
            ascending=[False, False, True],
            kind="stable",
        )

    iterations_completed = 0
    for iteration in range(config.local_repair_max_iterations):
        iterations_completed = iteration + 1
        proposals: List[Tuple[str, nx.Graph, List[Tuple[str, int, int]]]] = []
        bridges = {pair_key(int(u), int(v)) for u, v in nx.bridges(current)}
        weak_edges = sorted(
            [
                (u, v, data)
                for u, v, data in current.edges(data=True)
                if pair_key(int(u), int(v)) not in bridges
                and pair_key(int(u), int(v)) not in edited_pairs
                and (
                    not np.isfinite(
                        data.get("conditional_adjacency_support", np.nan)
                    )
                    or data.get("conditional_adjacency_support", 0.0)
                    <= low_support_threshold
                )
            ],
            key=lambda edge: (
                edge[2].get("evidence_score", -np.inf),
                edge[2].get("support", -np.inf),
                edge[2].get("weight", np.inf),
                edge[0],
                edge[1],
            ),
        )
        additions = eligible_additions(current)

        if weak_edges:
            u, v, _ = weak_edges[0]
            proposal = current.copy()
            proposal.remove_edge(u, v)
            if nx.number_connected_components(proposal) == target_component_count:
                proposals.append(("remove", proposal, [("remove", int(u), int(v))]))

        if not additions.empty:
            row = additions.iloc[0].to_dict()
            u, v = int(row["source"]), int(row["target"])
            if planarity_mode != "hard" or geometric_planarity_allows_edge(
                current, u, v
            ):
                proposal = current.copy()
                proposal.add_edge(
                    u, v, **_spanner_edge_attributes(row, "local_repair_add")
                )
                proposals.append(("add", proposal, [("add", u, v)]))

        if weak_edges and not additions.empty:
            remove_u, remove_v, _ = weak_edges[0]
            row = additions.iloc[0].to_dict()
            add_u, add_v = int(row["source"]), int(row["target"])
            proposal = current.copy()
            proposal.remove_edge(remove_u, remove_v)
            if (
                nx.number_connected_components(proposal) == target_component_count
                and (
                    planarity_mode != "hard"
                    or geometric_planarity_allows_edge(proposal, add_u, add_v)
                )
            ):
                proposal.add_edge(
                    add_u, add_v, **_spanner_edge_attributes(row, "local_repair_swap")
                )
                proposals.append(
                    (
                        "swap",
                        proposal,
                        [
                            ("swap_remove", int(remove_u), int(remove_v)),
                            ("swap_add", add_u, add_v),
                        ],
                    )
                )

        fallback_edges = [
            (u, v, data)
            for u, v, data in current.edges(data=True)
            if data.get("source_stage") == "fallback_bridge"
        ]
        if fallback_edges:
            old_u, old_v, old_data = fallback_edges[0]
            without = current.copy()
            without.remove_edge(old_u, old_v)
            component_map = {
                node: index
                for index, component in enumerate(nx.connected_components(without))
                for node in component
            }
            replacements = eligible_additions(without)
            replacements = replacements[
                [
                    component_map[int(u)] != component_map[int(v)]
                    and float(support)
                    > float(
                        old_data.get("conditional_adjacency_support", -np.inf)
                    )
                    for u, v, support in zip(
                        replacements["source"],
                        replacements["target"],
                        replacements["conditional_adjacency_support"].fillna(
                            -np.inf
                        ),
                    )
                ]
            ]
            if not replacements.empty:
                row = replacements.iloc[0].to_dict()
                add_u, add_v = int(row["source"]), int(row["target"])
                proposal = without.copy()
                if (
                    planarity_mode != "hard"
                    or geometric_planarity_allows_edge(proposal, add_u, add_v)
                ):
                    proposal.add_edge(
                        add_u,
                        add_v,
                        **_spanner_edge_attributes(row, "fallback_bridge_replacement"),
                    )
                    proposals.append(
                        (
                            "replace_fallback",
                            proposal,
                            [
                                ("fallback_remove", int(old_u), int(old_v)),
                                ("fallback_replacement_add", add_u, add_v),
                            ],
                        )
                    )

        evaluated: List[
            Tuple[float, str, nx.Graph, List[Tuple[str, int, int]], Dict[str, float]]
        ] = []
        for operation, proposal, edits in proposals:
            proposal_objective = graph_objective_components(
                proposal, structural_distances, config
            )
            metric_change = (
                proposal_objective["metric_error"]
                - current_objective["metric_error"]
            )
            metric_allowed = (
                metric_change <= config.local_repair_metric_tolerance
                if operation == "remove"
                else metric_change <= 1e-12
            )
            if operation == "swap":
                metric_allowed = metric_change < -1e-12
            if (
                metric_allowed
                and proposal_objective["total_objective"]
                < current_objective["total_objective"] - 1e-12
            ):
                evaluated.append(
                    (
                        proposal_objective["total_objective"],
                        operation,
                        proposal,
                        edits,
                        proposal_objective,
                    )
                )
        if not evaluated:
            iterations_completed = iteration
            break
        _, operation, accepted, edits, accepted_objective = min(
            evaluated, key=lambda item: (item[0], item[1])
        )
        for edit_kind, u, v in edits:
            record: Dict[str, Any] = {
                "iteration": iteration,
                "operation": operation,
                "edit": edit_kind,
                "source": u,
                "target": v,
            }
            for name, value in current_objective.items():
                record[f"before_{name}"] = value
            for name, value in accepted_objective.items():
                record[f"after_{name}"] = value
            history.append(record)
            edited_pairs.add(pair_key(u, v))
        current = accepted
        current_objective = accepted_objective

    return LocalRepairResult(
        graph=current,
        edit_history=pd.DataFrame(history),
        objective_before=objective_before,
        objective_after=current_objective,
        iterations_completed=iterations_completed,
    )


def validate_morse_like_reconstruction(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    structural_distances: np.ndarray,
    masses: np.ndarray,
    all_candidate_pairs: pd.DataFrame,
    *,
    type_confidence_threshold: float = 0.75,
    weak_conditional_support_quantile: float = 0.10,
    strong_conditional_support_quantile: float = 0.95,
    epsilon: float = 1e-15,
    numerical_tolerance: float = 1e-10,
) -> ReconstructionValidationResult:
    """Validate metric fidelity, conditional adjacency, topology, and Morse-like cues."""
    D = np.asarray(structural_distances, dtype=float)
    p = np.asarray(masses, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"structural_distances must be square; got {D.shape}.")
    n = D.shape[0]
    if p.shape != (n,):
        raise ValueError(f"masses must have shape ({n},); got {p.shape}.")
    original_nodes = set(range(n))
    reconstructed_nodes = set(int(node) for node in graph.nodes())
    missing_nodes = sorted(original_nodes - reconstructed_nodes)
    extra_nodes = sorted(reconstructed_nodes - original_nodes)
    retention = (n - len(missing_nodes)) / max(n, 1)
    if retention != 1.0 or extra_nodes:
        raise AssertionError(
            f"Node retention must equal 1; missing={missing_nodes}, extra={extra_nodes}."
        )

    graph_distances = np.full((n, n), np.inf, dtype=float)
    for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight="weight"):
        for target, distance in lengths.items():
            graph_distances[int(source), int(target)] = float(distance)
    np.fill_diagonal(graph_distances, 0.0)
    pair_mask = np.triu(np.isfinite(D) & (D > 0), k=1)
    pair_i, pair_j = np.where(pair_mask)
    target_values = D[pair_i, pair_j]
    reconstructed_values = graph_distances[pair_i, pair_j]
    pair_weights = p[pair_i] * p[pair_j]
    connected = np.isfinite(reconstructed_values)
    disconnected_count = int(np.count_nonzero(~connected))
    warnings: List[str] = []
    if disconnected_count:
        metric_error = np.inf
        warnings.append(
            f"{disconnected_count} finite target pairs are disconnected in reconstruction."
        )
    else:
        relative_error = (
            reconstructed_values - target_values
        ) / (target_values + epsilon)
        metric_error = float(
            np.sqrt(
                np.sum(pair_weights * relative_error * relative_error)
                / np.sum(pair_weights)
            )
        )
    finite_stretch = reconstructed_values[connected] / target_values[connected]
    below = connected & (
        reconstructed_values
        < target_values - numerical_tolerance * np.maximum(1.0, target_values)
    )
    below_count = int(np.count_nonzero(below))
    below_magnitude = (
        float(np.max(target_values[below] - reconstructed_values[below]))
        if below_count
        else 0.0
    )
    if below_count:
        warnings.append(
            f"{below_count} reconstructed distances fall unexpectedly below D_metric."
        )

    indexed_nodes = nodes.set_index("node_id").loc[range(n)]
    degree = dict(graph.degree())
    edge_detour_ratios: Dict[Tuple[int, int], float] = {}
    for edge_u, edge_v in list(graph.edges()):
        edge_u, edge_v = int(edge_u), int(edge_v)
        edge_attributes = dict(graph.edges[edge_u, edge_v])
        edge_weight = float(edge_attributes.get("weight", np.nan))
        graph.remove_edge(edge_u, edge_v)
        try:
            detour_distance = float(
                nx.shortest_path_length(
                    graph, source=edge_u, target=edge_v, weight="weight"
                )
            )
        except nx.NetworkXNoPath:
            detour_distance = np.inf
        finally:
            graph.add_edge(edge_u, edge_v, **edge_attributes)
        edge_key = (min(edge_u, edge_v), max(edge_u, edge_v))
        edge_detour_ratios[edge_key] = (
            detour_distance / edge_weight
            if np.isfinite(edge_weight) and edge_weight > 0
            else np.nan
        )

    edge_rows: List[Dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        u, v = int(u), int(v)
        edge_detour_ratio = edge_detour_ratios[(min(u, v), max(u, v))]
        data["edge_detour_ratio"] = edge_detour_ratio
        type_u, type_v = indexed_nodes.loc[u, "morse_type"], indexed_nodes.loc[v, "morse_type"]
        confidence_u = float(indexed_nodes.loc[u, "type_confidence"])
        confidence_v = float(indexed_nodes.loc[v, "type_confidence"])
        scalar_u = float(indexed_nodes.loc[u, "scalar_value"])
        scalar_v = float(indexed_nodes.loc[v, "scalar_value"])
        critical = (
            pd.notna(type_u)
            and pd.notna(type_v)
            and int(type_u) in {0, 1, 2}
            and int(type_v) in {0, 1, 2}
        )
        apparent_type_violation = bool(
            critical
            and (
                int(type_u) == int(type_v)
                or frozenset((int(type_u), int(type_v))) == frozenset((0, 2))
            )
        )
        apparent_scalar_violation = bool(
            critical
            and int(type_u) != int(type_v)
            and (
                (type_u < type_v and scalar_u > scalar_v + numerical_tolerance)
                or (type_v < type_u and scalar_v > scalar_u + numerical_tolerance)
            )
        )
        low_confidence = bool(
            confidence_u < type_confidence_threshold
            or confidence_v < type_confidence_threshold
        )
        edge_rows.append(
            {
                "source": u,
                "target": v,
                "weight": float(data.get("weight", np.nan)),
                "conditional_adjacency_support": float(
                    data.get("conditional_adjacency_support", np.nan)
                ),
                "conditional_adjacency_support_std": float(
                    data.get("conditional_adjacency_support_std", np.nan)
                ),
                "pair_coverage": float(data.get("pair_coverage", np.nan)),
                "pair_is_covered": bool(data.get("pair_is_covered", False)),
                "adjacency_evidence": float(
                    data.get("adjacency_evidence", np.nan)
                ),
                "n_contributing_inputs": int(
                    data.get("n_contributing_inputs", 0)
                ),
                "metric_decomposition_distance": float(
                    data.get("metric_decomposition_distance", np.nan)
                ),
                "metric_decomposition_gap": float(
                    data.get("metric_decomposition_gap", np.nan)
                ),
                "metric_relative_gap": float(
                    data.get("metric_relative_gap", np.nan)
                ),
                "metric_directness": float(
                    data.get("metric_directness", np.nan)
                ),
                "edge_detour_ratio": edge_detour_ratio,
                "evidence_score": float(data.get("evidence_score", np.nan)),
                "source_stage": data.get("source_stage"),
                "type_compatibility_status": data.get("type_compatibility_status"),
                "scalar_order_status": data.get("scalar_order_status"),
                "source_type": type_u,
                "target_type": type_v,
                "source_type_confidence": confidence_u,
                "target_type_confidence": confidence_v,
                "source_scalar": scalar_u,
                "target_scalar": scalar_v,
                "source_degree": int(degree[u]),
                "target_degree": int(degree[v]),
                "apparent_type_violation": apparent_type_violation,
                "apparent_scalar_order_violation": apparent_scalar_violation,
                "low_confidence_endpoint": low_confidence,
            }
        )
    selected_edges = pd.DataFrame(edge_rows)
    finite_support = selected_edges["conditional_adjacency_support"].dropna()
    positive_candidate_support = all_candidate_pairs.loc[
        all_candidate_pairs["conditional_adjacency_support"] > 0,
        "conditional_adjacency_support",
    ].dropna()
    weak_threshold = (
        float(
            positive_candidate_support.quantile(
                weak_conditional_support_quantile
            )
        )
        if len(positive_candidate_support)
        else 0.0
    )
    strong_threshold = (
        float(
            positive_candidate_support.quantile(
                strong_conditional_support_quantile
            )
        )
        if len(positive_candidate_support)
        else np.inf
    )
    weak_edges = selected_edges[
        selected_edges["conditional_adjacency_support"] <= weak_threshold
    ]
    candidate_selection_mask = (
        all_candidate_pairs["candidate_selected"].astype(bool)
        if "candidate_selected" in all_candidate_pairs
        else np.ones(len(all_candidate_pairs), dtype=bool)
    )
    candidate_pool = all_candidate_pairs[candidate_selection_mask].copy()
    omitted_mask = [
        not graph.has_edge(int(u), int(v))
        for u, v in zip(candidate_pool["source"], candidate_pool["target"])
    ]
    strong_omitted = candidate_pool[
        np.asarray(omitted_mask)
        & (
            candidate_pool["conditional_adjacency_support"]
            >= strong_threshold
        )
    ].sort_values(
        [
            "conditional_adjacency_support",
            "pair_coverage",
            "evidence_score",
        ],
        ascending=False,
    )
    support_quantiles = {
        str(quantile): float(finite_support.quantile(quantile))
        for quantile in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    } if len(finite_support) else {}
    pair_coverage_values = selected_edges["pair_coverage"].to_numpy(dtype=float)
    support_values = selected_edges["conditional_adjacency_support"].to_numpy(
        dtype=float
    )
    support_covered = np.isfinite(pair_coverage_values) & np.isfinite(
        support_values
    )
    coverage_weighted_conditional_support = (
        float(
            np.sum(
                pair_coverage_values[support_covered]
                * support_values[support_covered]
            )
            / np.sum(pair_coverage_values[support_covered])
        )
        if np.any(support_covered)
        and np.sum(pair_coverage_values[support_covered]) > 0
        else None
    )
    finite_pair_coverage = selected_edges["pair_coverage"].dropna()
    finite_adjacency_evidence = selected_edges["adjacency_evidence"].dropna()
    finite_metric_relative_gap = selected_edges["metric_relative_gap"].dropna()
    finite_metric_directness = selected_edges["metric_directness"].dropna()

    node_types = indexed_nodes["morse_type"]
    degree_table = pd.DataFrame(
        {
            "node_id": range(n),
            "morse_type": node_types.to_numpy(),
            "degree": [degree[node] for node in range(n)],
            "type_confidence": indexed_nodes["type_confidence"].to_numpy(),
        }
    )
    degree_by_type = (
        degree_table.groupby("morse_type", dropna=False)["degree"]
        .agg(["count", "min", "mean", "median", "max"])
        .reset_index()
    )
    unexpected = degree_table[
        ((degree_table["morse_type"] == -1) & (degree_table["degree"] != 2))
        | (degree_table["morse_type"].isin([0, 2]) & (degree_table["degree"] < 1))
        | ((degree_table["morse_type"] == 1) & (degree_table["degree"] < 2))
    ].copy()
    unexpected["diagnostic_rule"] = np.select(
        [
            unexpected["morse_type"] == -1,
            unexpected["morse_type"] == 1,
        ],
        ["regular degree != 2", "saddle degree < 2"],
        default="minimum/maximum degree < 1",
    )
    low_confidence_violations = selected_edges[
        selected_edges["low_confidence_endpoint"]
        & (
            selected_edges["apparent_type_violation"]
            | selected_edges["apparent_scalar_order_violation"]
        )
    ].copy()
    selected_edges["questionable_score"] = (
        (1.0 - selected_edges["conditional_adjacency_support"].fillna(0.0))
        * selected_edges["pair_coverage"].fillna(0.0)
        + selected_edges["conditional_adjacency_support_std"].fillna(0.0)
        + selected_edges["apparent_type_violation"].astype(float)
        + selected_edges["apparent_scalar_order_violation"].astype(float)
        + 0.25 * selected_edges["low_confidence_endpoint"].astype(float)
    )
    questionable = selected_edges.sort_values(
        [
            "questionable_score",
            "conditional_adjacency_support",
            "pair_coverage",
        ],
        ascending=[False, True, False],
    ).head(25)

    component_count = int(nx.number_connected_components(graph))
    degrees = np.asarray([degree[node] for node in range(n)], dtype=float)
    crossing_count = _straight_line_crossing_count(graph, nodes)
    summary: Dict[str, Any] = {
        "original_barycenter_node_count": n,
        "reconstructed_node_count": graph.number_of_nodes(),
        "missing_nodes": missing_nodes,
        "extra_nodes": extra_nodes,
        "node_retention_fraction": retention,
        "weighted_relative_metric_error": metric_error,
        "median_stretch": float(np.median(finite_stretch)),
        "stretch_p90": float(np.quantile(finite_stretch, 0.90)),
        "stretch_p95": float(np.quantile(finite_stretch, 0.95)),
        "stretch_p99": float(np.quantile(finite_stretch, 0.99)),
        "maximum_stretch": float(np.max(finite_stretch)),
        "disconnected_finite_pair_count": disconnected_count,
        "unexpected_distance_below_target_count": below_count,
        "maximum_unexpected_distance_deficit": below_magnitude,
        "mean_selected_conditional_adjacency_support": (
            float(finite_support.mean()) if len(finite_support) else None
        ),
        "median_selected_conditional_adjacency_support": (
            float(finite_support.median()) if len(finite_support) else None
        ),
        "mean_selected_pair_coverage": (
            float(finite_pair_coverage.mean()) if len(finite_pair_coverage) else None
        ),
        "mean_selected_adjacency_evidence": (
            float(finite_adjacency_evidence.mean())
            if len(finite_adjacency_evidence)
            else None
        ),
        "mean_selected_metric_relative_gap": (
            float(finite_metric_relative_gap.mean())
            if len(finite_metric_relative_gap)
            else None
        ),
        "mean_selected_metric_directness": (
            float(finite_metric_directness.mean())
            if len(finite_metric_directness)
            else None
        ),
        "coverage_weighted_conditional_adjacency_support": (
            coverage_weighted_conditional_support
        ),
        "selected_edge_conditional_adjacency_support_quantiles": support_quantiles,
        "weak_conditional_adjacency_support_threshold": weak_threshold,
        "weak_conditional_support_threshold": weak_threshold,
        "weak_conditional_adjacency_support_selected_edge_count": int(
            len(weak_edges)
        ),
        "strong_conditional_adjacency_support_threshold": strong_threshold,
        "strong_conditional_adjacency_supported_omitted_candidate_count": int(
            len(strong_omitted)
        ),
        "edge_count": int(graph.number_of_edges()),
        "component_count": component_count,
        "cycle_rank": int(graph.number_of_edges() - graph.number_of_nodes() + component_count),
        "edge_to_node_ratio": float(graph.number_of_edges() / max(n, 1)),
        "degree_minimum": int(np.min(degrees)),
        "degree_median": float(np.median(degrees)),
        "degree_mean": float(np.mean(degrees)),
        "degree_maximum": int(np.max(degrees)),
        "geometric_planar": None if crossing_count is None else bool(crossing_count == 0),
        "fixed_coordinate_crossing_count": crossing_count,
        "critical_to_critical_incompatible_edge_count": int(
            np.count_nonzero(selected_edges["apparent_type_violation"])
        ),
        "scalar_order_violation_count": int(
            np.count_nonzero(selected_edges["apparent_scalar_order_violation"])
        ),
        "regular_nodes_degree_not_two": int(
            np.count_nonzero(
                (degree_table["morse_type"] == -1) & (degree_table["degree"] != 2)
            )
        ),
        "unexpected_critical_degree_node_count": int(
            np.count_nonzero(unexpected["morse_type"] != -1)
        ),
        "low_confidence_nodes_in_apparent_violation_edges": int(
            len(
                set(low_confidence_violations["source"])
                | set(low_confidence_violations["target"])
            )
        ),
    }
    return ReconstructionValidationResult(
        summary=summary,
        stretch_values=finite_stretch,
        selected_edges=selected_edges,
        degree_by_type=degree_by_type,
        unexpected_degree_nodes=unexpected,
        low_confidence_violation_nodes=low_confidence_violations,
        strong_omitted_candidates=strong_omitted,
        questionable_edges=questionable,
        warnings=warnings,
    )


def build_structural_distance_baseline(
    nodes: pd.DataFrame,
    structural_distances: np.ndarray,
    *,
    edge_count: Optional[int] = None,
    edge_fraction: Optional[float] = None,
    source_stage: str = "distance_baseline",
) -> nx.Graph:
    """Build a node-preserving graph from globally shortest structural pairs."""
    if (edge_count is None) == (edge_fraction is None):
        raise ValueError("Specify exactly one of edge_count or edge_fraction.")
    D = np.asarray(structural_distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"structural_distances must be square; got {D.shape}.")
    n = D.shape[0]
    indexed = nodes.set_index("node_id")
    if set(indexed.index.astype(int)) != set(range(n)):
        raise ValueError("nodes must align exactly with structural distance indices.")
    graph = nx.Graph()
    graph.add_nodes_from(
        (int(node_id), dict(attributes))
        for node_id, attributes in indexed.loc[range(n)].to_dict(orient="index").items()
    )
    upper = np.triu(np.isfinite(D) & (D > 0), k=1)
    source, target = np.where(upper)
    order = np.lexsort((target, source, D[source, target]))
    if edge_fraction is not None:
        if not 0.0 < edge_fraction <= 1.0:
            raise ValueError("edge_fraction must lie in (0, 1].")
        keep = int(np.ceil(len(order) * edge_fraction))
    else:
        keep = min(max(int(edge_count), 0), len(order))
    for index in order[:keep]:
        u, v = int(source[index]), int(target[index])
        graph.add_edge(
            u,
            v,
            weight=float(D[u, v]),
            source_stage=source_stage,
        )
    return graph


def annotate_graph_with_candidate_evidence(
    graph: nx.Graph,
    all_candidate_pairs: pd.DataFrame,
    *,
    source_stage: Optional[str] = None,
) -> nx.Graph:
    """Copy a graph and attach canonical adjacency/constraint evidence to edges."""
    annotated = graph.copy()
    lookup = all_candidate_pairs.set_index(["source", "target"], drop=False)
    for u, v, data in annotated.edges(data=True):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        original_weight = float(data.get("weight", row["structural_distance"]))
        evidence = _spanner_edge_attributes(
            row,
            source_stage or str(data.get("source_stage", "baseline")),
        )
        evidence["weight"] = original_weight
        data.update(evidence)
    return annotated


def compare_reconstruction_graphs(
    graphs: Mapping[str, nx.Graph],
    nodes: pd.DataFrame,
    structural_distances: np.ndarray,
    masses: np.ndarray,
    all_candidate_pairs: pd.DataFrame,
    config: ReconstructionConfig,
) -> Tuple[pd.DataFrame, Dict[str, ReconstructionValidationResult]]:
    """Apply identical reconstruction validation to every comparison graph."""
    rows: List[Dict[str, Any]] = []
    validations: Dict[str, ReconstructionValidationResult] = {}
    n_original = np.asarray(structural_distances).shape[0]
    for label, graph in graphs.items():
        result = validate_morse_like_reconstruction(
            graph,
            nodes,
            structural_distances,
            masses,
            all_candidate_pairs,
            type_confidence_threshold=config.type_confidence_threshold,
            weak_conditional_support_quantile=(
                config.local_repair_low_conditional_support_quantile
            ),
            strong_conditional_support_quantile=(
                config.global_conditional_support_quantile
            ),
        )
        validations[label] = result
        summary = result.summary
        support = result.selected_edges[
            "conditional_adjacency_support"
        ].to_numpy(dtype=float)
        pair_coverage = result.selected_edges["pair_coverage"].to_numpy(
            dtype=float
        )
        weighted_mask = (
            np.isfinite(support) & np.isfinite(pair_coverage) & (support > 0)
        )
        support_weighted_pair_coverage = (
            float(
                np.sum(support[weighted_mask] * pair_coverage[weighted_mask])
                / np.sum(support[weighted_mask])
            )
            if np.any(weighted_mask) and np.sum(support[weighted_mask]) > 0
            else None
        )
        rows.append(
            {
                "reconstruction": label,
                "nodes_retained": int(
                    n_original - len(summary["missing_nodes"])
                ),
                "node_retention_fraction": summary["node_retention_fraction"],
                "edges": int(graph.number_of_edges()),
                "components": summary["component_count"],
                "cycle_rank": summary["cycle_rank"],
                "metric_stress": summary["weighted_relative_metric_error"],
                "maximum_stretch": summary["maximum_stretch"],
                "mean_conditional_adjacency_support": summary[
                    "mean_selected_conditional_adjacency_support"
                ],
                "conditional_support_weighted_pair_coverage": (
                    support_weighted_pair_coverage
                ),
                "type_violations": summary[
                    "critical_to_critical_incompatible_edge_count"
                ],
                "geometric_planar": summary["geometric_planar"],
                "straight_line_crossings": summary[
                    "fixed_coordinate_crossing_count"
                ],
            }
        )
    return pd.DataFrame(rows), validations


def build_candidate_edge_density(
    nodes: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    config: ReconstructionConfig,
    *,
    resolution: int,
    bandwidth: float,
) -> DensityFieldResult:
    """Rasterize canonical candidate evidence mass and smooth it."""
    from scipy.ndimage import gaussian_filter

    if resolution < 8:
        raise ValueError("resolution must be at least 8.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    indexed = nodes.set_index("node_id")
    xy = indexed.loc[range(len(nodes)), ["x", "y"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(xy)):
        raise ValueError("Finite x-y coordinates are required for density routing.")
    selected = (
        candidate_edges[candidate_edges["candidate_selected"].astype(bool)].copy()
        if "candidate_selected" in candidate_edges
        else candidate_edges.copy()
    )
    support = np.clip(
        selected["conditional_adjacency_support"]
        .fillna(0.0)
        .to_numpy(dtype=float),
        0.0,
        1.0,
    )
    pair_coverage = np.clip(
        selected["pair_coverage"].fillna(0.0).to_numpy(dtype=float), 0.0, 1.0
    )
    metric_relative_gap = selected["metric_relative_gap"].fillna(0.0).to_numpy(
        dtype=float
    )
    bounded_metric_relative_gap = metric_relative_gap / (
        1.0 + np.maximum(metric_relative_gap, 0.0)
    )
    uncertainty = selected["conditional_adjacency_support_std"].fillna(
        0.0
    ).to_numpy(dtype=float)
    evidence_mass = np.maximum(
        0.0,
        pair_coverage
        * (
            config.density_conditional_support_weight * support
            + config.density_metric_relative_gap_weight
            * bounded_metric_relative_gap
            + config.density_pair_coverage_weight
        )
        - config.density_uncertainty_penalty * uncertainty,
    )

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
    for row, mass in zip(selected.to_dict(orient="records"), evidence_mass):
        if mass <= 0:
            continue
        start = xy[int(row["source"])]
        end = xy[int(row["target"])]
        length = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(length / sample_spacing)) + 1)
        parameter = np.linspace(0.0, 1.0, sample_count)
        samples = start[None, :] + parameter[:, None] * (end - start)[None, :]
        x_index = np.clip(
            np.rint((samples[:, 0] - lower[0]) / dx).astype(int),
            0,
            resolution - 1,
        )
        y_index = np.clip(
            np.rint((samples[:, 1] - lower[1]) / dy).astype(int),
            0,
            resolution - 1,
        )
        np.add.at(mass_grid, (y_index, x_index), mass / sample_count)

    sigma_physical = bandwidth * float(np.max(span))
    sigma_pixels = (sigma_physical / dy, sigma_physical / dx)
    smoothed = gaussian_filter(mass_grid, sigma=sigma_pixels, mode="constant")
    maximum = float(np.max(smoothed))
    density = smoothed / maximum if maximum > 0 else smoothed
    metadata = {
        "resolution": int(resolution),
        "bandwidth_fraction": float(bandwidth),
        "x_min": float(lower[0]),
        "x_max": float(upper[0]),
        "y_min": float(lower[1]),
        "y_max": float(upper[1]),
        "dx": dx,
        "dy": dy,
        "sample_spacing": sample_spacing,
        "evidence_mass_total": float(np.sum(evidence_mass)),
        "rasterized_mass_total": float(np.sum(mass_grid)),
        "evidence_formula": (
            "max(0, pair_coverage*(w_conditional_support*"
            "conditional_adjacency_support + w_metric_relative_gap*"
            "metric_relative_gap/(1+metric_relative_gap) + "
            "w_pair_coverage) - w_uncertainty*"
            "conditional_adjacency_support_std)"
        ),
    }
    return DensityFieldResult(
        density=density,
        raw_mass_grid=mass_grid,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        bandwidth=float(bandwidth),
        resolution=int(resolution),
        evidence_mass_total=float(np.sum(evidence_mass)),
        metadata=metadata,
    )


def _astar_density_route(
    traversal_cost: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    dx: float,
    dy: float,
    length_penalty: float,
    occupancy: np.ndarray,
    occupancy_penalty: float,
) -> List[Tuple[int, int]]:
    """A* route on an eight-neighbor raster."""
    import heapq

    shape = traversal_cost.shape
    frontier: List[Tuple[float, float, int, int]] = [(0.0, 0.0, start[0], start[1])]
    best = {start: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    while frontier:
        _, cost_so_far, row, column = heapq.heappop(frontier)
        current = (row, column)
        if current == goal:
            break
        if cost_so_far > best.get(current, np.inf):
            continue
        for d_row, d_column in neighbors:
            next_row, next_column = row + d_row, column + d_column
            if not (0 <= next_row < shape[0] and 0 <= next_column < shape[1]):
                continue
            step_length = float(np.hypot(d_column * dx, d_row * dy))
            density_cost = 0.5 * (
                traversal_cost[row, column]
                + traversal_cost[next_row, next_column]
            )
            transition = (
                density_cost * step_length
                + length_penalty * step_length
                + occupancy_penalty * occupancy[next_row, next_column]
            )
            next_cost = cost_so_far + transition
            next_cell = (next_row, next_column)
            if next_cost >= best.get(next_cell, np.inf):
                continue
            best[next_cell] = next_cost
            parent[next_cell] = current
            heuristic = length_penalty * float(
                np.hypot(
                    (goal[1] - next_column) * dx,
                    (goal[0] - next_row) * dy,
                )
            )
            heapq.heappush(
                frontier,
                (next_cost + heuristic, next_cost, next_row, next_column),
            )
    if goal not in best:
        raise RuntimeError(f"No raster route found from {start} to {goal}.")
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def route_graph_through_density(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    density_field: DensityFieldResult,
    config: ReconstructionConfig,
) -> GeometricRoutingResult:
    """Route existing edges geometrically without changing nodes or adjacency."""
    indexed = nodes.set_index("node_id")
    xy = indexed.loc[list(graph.nodes()), ["x", "y"]].to_numpy(dtype=float)
    node_position = {
        int(node): xy[position] for position, node in enumerate(graph.nodes())
    }
    x_coordinates = density_field.x_coordinates
    y_coordinates = density_field.y_coordinates
    dx = float(x_coordinates[1] - x_coordinates[0])
    dy = float(y_coordinates[1] - y_coordinates[0])
    density = density_field.density
    traversal_cost = -np.log(density + 1e-12)
    occupancy = np.zeros_like(density, dtype=float)
    routed = graph.copy()
    original_nodes = set(routed.nodes())
    original_edges = {tuple(sorted(edge)) for edge in routed.edges()}
    positive_density = density[density > 0]
    high_density_threshold = (
        float(np.quantile(positive_density, config.density_high_region_quantile))
        if positive_density.size
        else 0.0
    )
    route_rows: List[Dict[str, Any]] = []
    intersection_events = 0
    ordered_edges = sorted(
        routed.edges(data=True),
        key=lambda edge: (
            -float(edge[2].get("evidence_score", 0.0)),
            min(int(edge[0]), int(edge[1])),
            max(int(edge[0]), int(edge[1])),
        ),
    )
    for u, v, data in ordered_edges:
        start_xy, goal_xy = node_position[int(u)], node_position[int(v)]
        start = (
            int(np.argmin(np.abs(y_coordinates - start_xy[1]))),
            int(np.argmin(np.abs(x_coordinates - start_xy[0]))),
        )
        goal = (
            int(np.argmin(np.abs(y_coordinates - goal_xy[1]))),
            int(np.argmin(np.abs(x_coordinates - goal_xy[0]))),
        )
        cells = _astar_density_route(
            traversal_cost,
            start,
            goal,
            dx,
            dy,
            config.density_length_penalty,
            occupancy,
            config.density_occupancy_penalty,
        )
        if start == goal:
            cells = [start, goal]
        interior = cells[1:-1]
        intersection_events += int(
            np.count_nonzero([occupancy[row, column] > 0 for row, column in interior])
        )
        for row, column in cells:
            occupancy[row, column] += 1.0
        polyline = np.asarray(
            [[x_coordinates[column], y_coordinates[row]] for row, column in cells],
            dtype=float,
        )
        polyline[0] = start_xy
        polyline[-1] = goal_xy
        route_length = float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))
        straight_length = float(np.linalg.norm(goal_xy - start_xy))
        inflation = route_length / max(straight_length, 1e-15)
        route_density = np.asarray([density[row, column] for row, column in cells])
        leaves_high_density = bool(
            np.any(route_density < high_density_threshold)
        )
        routed.edges[u, v]["route_polyline"] = polyline
        routed.edges[u, v]["geometric_route_length"] = route_length
        routed.edges[u, v]["route_length_inflation"] = inflation
        routed.edges[u, v]["route_leaves_high_density"] = leaves_high_density
        route_rows.append(
            {
                "source": int(u),
                "target": int(v),
                "straight_length": straight_length,
                "route_length": route_length,
                "route_length_inflation": inflation,
                "leaves_high_density": leaves_high_density,
                "endpoint_anchoring_error": float(
                    max(
                        np.linalg.norm(polyline[0] - start_xy),
                        np.linalg.norm(polyline[-1] - goal_xy),
                    )
                ),
            }
        )
    if set(routed.nodes()) != original_nodes:
        raise AssertionError("Density routing changed the node set.")
    if {tuple(sorted(edge)) for edge in routed.edges()} != original_edges:
        raise AssertionError("Density routing changed graph adjacency.")
    route_statistics = pd.DataFrame(route_rows)
    diagnostics = {
        "resolution": density_field.resolution,
        "bandwidth": density_field.bandwidth,
        "edge_count": routed.number_of_edges(),
        "mean_route_length_inflation": float(
            route_statistics["route_length_inflation"].mean()
        ),
        "median_route_length_inflation": float(
            route_statistics["route_length_inflation"].median()
        ),
        "p95_route_length_inflation": float(
            route_statistics["route_length_inflation"].quantile(0.95)
        ),
        "maximum_route_length_inflation": float(
            route_statistics["route_length_inflation"].max()
        ),
        "route_intersection_events": int(intersection_events),
        "routes_leaving_high_density_region": int(
            route_statistics["leaves_high_density"].sum()
        ),
        "maximum_endpoint_anchoring_error": float(
            route_statistics["endpoint_anchoring_error"].max()
        ),
        "occupancy_penalty": float(config.density_occupancy_penalty),
        "high_density_quantile": float(config.density_high_region_quantile),
    }
    return GeometricRoutingResult(
        graph=routed,
        route_statistics=route_statistics,
        diagnostics=diagnostics,
        density_field=density_field,
    )


def select_density_routing_result(
    results: List[GeometricRoutingResult],
    rule: DensitySelectionRule,
) -> GeometricRoutingResult:
    """Select one explicitly ranked density-routing result."""
    if not results:
        raise ValueError("At least one routing result is required.")
    if rule == "minimum_intersections_then_inflation":
        key = lambda result: (
            result.diagnostics["route_intersection_events"],
            result.diagnostics["routes_leaving_high_density_region"],
            result.diagnostics["median_route_length_inflation"],
            result.diagnostics["resolution"],
            result.diagnostics["bandwidth"],
        )
    elif rule == "minimum_inflation_then_intersections":
        key = lambda result: (
            result.diagnostics["median_route_length_inflation"],
            result.diagnostics["route_intersection_events"],
            result.diagnostics["routes_leaving_high_density_region"],
            result.diagnostics["resolution"],
            result.diagnostics["bandwidth"],
        )
    else:
        raise ValueError(f"Unsupported density selection rule {rule!r}.")
    return min(results, key=key)


def draw_density_routing_diagnostics(
    result: GeometricRoutingResult,
    nodes: pd.DataFrame,
):
    """Draw density, straight topology overlay, and routed polylines."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    density = result.density_field.density
    extent = [
        result.density_field.x_coordinates[0],
        result.density_field.x_coordinates[-1],
        result.density_field.y_coordinates[0],
        result.density_field.y_coordinates[-1],
    ]
    indexed_nodes = nodes.set_index("node_id")
    position = {
        int(node): indexed_nodes.loc[node, ["x", "y"]].to_numpy(dtype=float)
        for node in result.graph.nodes()
    }
    xy = np.asarray([position[int(node)] for node in result.graph.nodes()])
    straight = np.asarray(
        [[position[int(u)], position[int(v)]] for u, v in result.graph.edges()]
    )
    routed_lines = [
        data["route_polyline"] for _, _, data in result.graph.edges(data=True)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax in axes:
        image = ax.imshow(density, origin="lower", extent=extent, cmap="magma")
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color="white", zorder=4)
        ax.set(aspect="equal", xlabel="x", ylabel="y")
    axes[0].set_title("Weighted candidate-edge density")
    axes[1].add_collection(
        LineCollection(straight, colors="#35b779", linewidths=0.6, alpha=0.6)
    )
    axes[1].set_title("Final topology over density")
    axes[2].add_collection(
        LineCollection(routed_lines, colors="#4cc9f0", linewidths=0.7, alpha=0.7)
    )
    axes[2].set_title("Density-routed edge geometry")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="normalized density")
    fig.suptitle(
        f"resolution={result.diagnostics['resolution']}, "
        f"bandwidth={result.diagnostics['bandwidth']:g}, "
        f"median inflation={result.diagnostics['median_route_length_inflation']:.3f}"
    )
    return fig, axes


def run_synthetic_formula_checks(*, tolerance: float = 1e-12) -> Dict[str, bool]:
    """Check canonical transport identities and metric-decomposition examples."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    graph = nx.Graph()
    graph.add_edge(0, 1)
    common = {
        "Cs": [np.zeros((2, 2))],
        "Gs": [graph],
        "lambdas": np.ones(1),
        "p": np.ones(2),
    }
    full = compute_conditional_adjacency_support(
        {**common, "Ts": [np.eye(2)]},
        [[0, 1]],
    )
    partial = compute_conditional_adjacency_support(
        {**common, "Ts": [np.diag([0.2, 0.3])]},
        [[0, 1]],
    )
    if not np.isclose(full.pair_coverage[0, 1], 1.0, atol=tolerance):
        raise AssertionError("Full-coverage synthetic pair did not have coverage one.")
    if not np.isclose(
        full.adjacency_evidence[0, 1],
        full.pair_coverage[0, 1]
        * full.conditional_adjacency_support[0, 1],
        atol=tolerance,
    ):
        raise AssertionError("Full-coverage evidence identity failed.")
    if not np.isclose(
        partial.conditional_adjacency_support[0, 1], 1.0, atol=tolerance
    ):
        raise AssertionError("Partial-coverage conditional support must remain one.")
    if not np.isclose(partial.pair_coverage[0, 1], 0.06, atol=tolerance):
        raise AssertionError("Partial-coverage synthetic pair must have coverage 0.06.")
    if not np.isclose(
        partial.adjacency_evidence[0, 1],
        partial.pair_coverage[0, 1]
        * partial.conditional_adjacency_support[0, 1],
        atol=tolerance,
    ):
        raise AssertionError("Partial-coverage evidence identity failed.")

    path_distances = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    metric = compute_metric_decomposition(path_distances)
    if not np.isclose(metric.metric_decomposition_gap[0, 2], 0.0, atol=tolerance):
        raise AssertionError("Path pair AC must have zero metric decomposition gap.")
    if not np.isclose(metric.metric_relative_gap[0, 2], 0.0, atol=tolerance):
        raise AssertionError("Path pair AC must have zero metric relative gap.")
    if not np.isclose(metric.metric_directness[0, 2], 0.0, atol=tolerance):
        raise AssertionError("Path pair AC must have zero metric directness.")
    if not np.isclose(metric.metric_decomposition_gap[0, 1], 2.0, atol=tolerance):
        raise AssertionError("Path pair AB must have metric decomposition gap two.")
    if not np.isclose(metric.metric_relative_gap[0, 1], 2.0, atol=tolerance):
        raise AssertionError("Path pair AB must have metric relative gap two.")
    if not np.isclose(
        metric.metric_directness[0, 1], 2.0 / 3.0, atol=tolerance
    ):
        raise AssertionError("Path pair AB must have metric directness two-thirds.")
    return {
        "full_coverage_identity": True,
        "partial_coverage_identity": True,
        "path_metric_decomposition": True,
    }


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/path values to JSON-compatible objects."""
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
    return value


def save_canonical_metric_arrays(
    output_dir: PathLike,
    metric_decomposition: MetricDecompositionResult,
    *,
    conditional_adjacency: Optional[ConditionalAdjacencySupportResult] = None,
    write_legacy_name_aliases: bool = False,
) -> Dict[str, Path]:
    """Save canonical terminology-v2 arrays and optional deprecated aliases."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {
        "metric_decomposition_distance.npy": (
            metric_decomposition.metric_decomposition_distance
        ),
        "metric_decomposition_gap.npy": metric_decomposition.metric_decomposition_gap,
        "metric_relative_gap.npy": metric_decomposition.metric_relative_gap,
        "metric_directness.npy": metric_decomposition.metric_directness,
    }
    if conditional_adjacency is not None:
        arrays.update(
            {
                "conditional_adjacency_support.npy": (
                    conditional_adjacency.conditional_adjacency_support
                ),
                "conditional_adjacency_support_std.npy": (
                    conditional_adjacency.conditional_adjacency_support_std
                ),
                "pair_coverage.npy": conditional_adjacency.pair_coverage,
                "pair_is_covered.npy": conditional_adjacency.pair_is_covered,
                "adjacency_evidence.npy": conditional_adjacency.adjacency_evidence,
                "pair_contributing_inputs.npy": (
                    conditional_adjacency.pair_contributing_inputs
                ),
            }
        )
    if write_legacy_name_aliases:
        arrays.update(
            {
                "irreducibility.npy": metric_decomposition.metric_relative_gap,
                "best_two_hop_distance.npy": (
                    metric_decomposition.metric_decomposition_distance
                ),
            }
        )
        if conditional_adjacency is not None:
            arrays.update(
                {
                    "edge_support.npy": (
                        conditional_adjacency.conditional_adjacency_support
                    ),
                    "edge_support_std.npy": (
                        conditional_adjacency.conditional_adjacency_support_std
                    ),
                    "pair_observability.npy": conditional_adjacency.pair_coverage,
                }
            )
    saved: Dict[str, Path] = {}
    for name, array in arrays.items():
        path = output / name
        np.save(path, array)
        saved[name] = path
    return saved


def save_reconstruction_manifest(
    output_dir: PathLike,
    *,
    source_barycenter_dir: PathLike,
    source_metadata: Mapping[str, Any],
    config: ReconstructionConfig,
    reconstruction_mode: ReconstructionMode,
    validation_result: Mapping[str, Any],
    warnings_and_fallbacks: List[str],
    terminology_version: int = 2,
) -> Dict[str, Any]:
    """Save resolved configuration and a complete reconstruction manifest."""
    import importlib.metadata
    import json
    import platform
    from datetime import datetime, timezone

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Reconstruction output directory does not exist: {output}")
    resolved_config = config.resolved_dict(Path.cwd())
    config_path = output / "reconstruction_config.json"
    config_path.write_text(
        json.dumps(_json_safe(resolved_config), indent=2),
        encoding="utf-8",
    )
    required_files = [
        "reconstruction_config.json",
        "decoded_nodes.csv",
        "D_raw.npy",
        "D_metric.npy",
        "metric_diagnostics.json",
        "metric_decomposition_distance.npy",
        "metric_decomposition_gap.npy",
        "metric_relative_gap.npy",
        "metric_directness.npy",
        "candidate_edges.csv",
        "selected_edges.csv",
        "edge_edit_history.csv",
        "G_topology.gpickle",
        "reconstruction_metrics.json",
        "baseline_comparison.csv",
    ]
    missing = [name for name in required_files if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot finalize reconstruction manifest; missing required artifacts: "
            + ", ".join(missing)
        )
    if not list(output.glob("spanner_stretch_*.gpickle")):
        raise FileNotFoundError("No intermediate spanner graphs were saved.")
    figures_dir = output / "figures"
    if not figures_dir.is_dir() or not list(figures_dir.glob("*")):
        raise FileNotFoundError("No reconstruction figures were saved.")

    optional_names = [
        "conditional_adjacency_support.npy",
        "conditional_adjacency_support_std.npy",
        "pair_coverage.npy",
        "pair_is_covered.npy",
        "adjacency_evidence.npy",
        "pair_contributing_inputs.npy",
        "density_grid.npy",
        "density_grid_metadata.json",
        "route_statistics.csv",
        "G_routed.gpickle",
    ]
    optional_status = {
        name: {
            "created": bool((output / name).is_file()),
            "path": str((output / name).resolve()) if (output / name).is_file() else None,
        }
        for name in optional_names
    }
    package_names = {
        "python": None,
        "numpy": "numpy",
        "pandas": "pandas",
        "networkx": "networkx",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "jupyter": "jupyter",
        "nbformat": "nbformat",
    }
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for label, package in package_names.items():
        if package is None:
            continue
        try:
            versions[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None

    created_files = sorted(
        {
            str(path.relative_to(output).as_posix())
            for path in output.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        | {"manifest.json"}
    )
    if int(terminology_version) != 2:
        raise ValueError(
            "Only terminology_version=2 is supported for canonical artifacts."
        )
    manifest: Dict[str, Any] = {
        "terminology_version": int(terminology_version),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_barycenter_directory": str(Path(source_barycenter_dir).resolve()),
        "reconstruction_directory": str(output),
        "source_metadata": _json_safe(dict(source_metadata)),
        "configuration": _json_safe(resolved_config),
        "reconstruction_mode": reconstruction_mode,
        "files_created": created_files,
        "optional_artifacts": optional_status,
        "library_versions": versions,
        "validation_result": _json_safe(dict(validation_result)),
        "warnings_and_fallbacks": sorted(set(warnings_and_fallbacks)),
        "source_arrays_copied_or_modified": False,
        "deprecated_aliases": (
            {
                name: {"deprecated": True, "created": (output / name).is_file()}
                for name in (
                    "edge_support.npy",
                    "edge_support_std.npy",
                    "pair_observability.npy",
                    "irreducibility.npy",
                    "best_two_hop_distance.npy",
                )
            }
            if config.write_legacy_name_aliases
            else {}
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_graph_pickle(path: PathLike) -> nx.Graph:
    """Load a graph using the repository's pickle serialization convention."""
    import pickle

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Graph pickle does not exist: {source}")
    with source.open("rb") as stream:
        graph = pickle.load(stream)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Pickle did not contain a NetworkX graph: {type(graph)!r}")
    return graph


def verify_reloaded_topology(
    graph: nx.Graph,
    *,
    expected_nodes: set,
    expected_edge_count: int,
    saved_metrics: Mapping[str, Any],
    numerical_tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Verify saved topology structure, attributes, and a metric subset."""
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
        "conditional_adjacency_support",
        "conditional_adjacency_support_std",
        "pair_coverage",
        "pair_is_covered",
        "adjacency_evidence",
        "n_contributing_inputs",
        "metric_decomposition_distance",
        "metric_decomposition_gap",
        "metric_relative_gap",
        "metric_directness",
        "evidence_score",
        "source_stage",
        "type_compatibility_status",
        "scalar_order_status",
    }
    actual_nodes = set(graph.nodes())
    if actual_nodes != expected_nodes:
        raise AssertionError(
            f"Reloaded node set mismatch: missing={expected_nodes - actual_nodes}, "
            f"extra={actual_nodes - expected_nodes}."
        )
    if graph.number_of_edges() != expected_edge_count:
        raise AssertionError(
            f"Reloaded edge count {graph.number_of_edges()} != {expected_edge_count}."
        )
    missing_node_attributes = {
        node: sorted(required_node_attributes - set(attributes))
        for node, attributes in graph.nodes(data=True)
        if required_node_attributes - set(attributes)
    }
    missing_edge_attributes = {
        f"{u}-{v}": sorted(required_edge_attributes - set(attributes))
        for u, v, attributes in graph.edges(data=True)
        if required_edge_attributes - set(attributes)
    }
    if missing_node_attributes:
        raise AssertionError(
            f"Reloaded graph has missing node attributes: {missing_node_attributes}"
        )
    if missing_edge_attributes:
        raise AssertionError(
            f"Reloaded graph has missing edge attributes: {missing_edge_attributes}"
        )
    components = int(nx.number_connected_components(graph))
    cycle_rank = int(graph.number_of_edges() - graph.number_of_nodes() + components)
    supports = np.asarray(
        [
            data["conditional_adjacency_support"]
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )
    mean_support = float(np.nanmean(supports))
    recomputed = {
        "reconstructed_node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "component_count": components,
        "cycle_rank": cycle_rank,
        "edge_to_node_ratio": float(
            graph.number_of_edges() / max(graph.number_of_nodes(), 1)
        ),
        "mean_selected_conditional_adjacency_support": mean_support,
    }
    comparisons = {
        "reconstructed_node_count": saved_metrics["reconstructed_node_count"],
        "component_count": saved_metrics["component_count"],
        "cycle_rank": saved_metrics["cycle_rank"],
        "edge_to_node_ratio": saved_metrics["edge_to_node_ratio"],
        "mean_selected_conditional_adjacency_support": saved_metrics[
            "mean_selected_conditional_adjacency_support"
        ],
    }
    for key, expected in comparisons.items():
        actual = recomputed[key]
        if isinstance(expected, (int, np.integer)):
            matches = actual == expected
        else:
            matches = np.isclose(
                float(actual),
                float(expected),
                rtol=numerical_tolerance,
                atol=numerical_tolerance,
            )
        if not matches:
            raise AssertionError(
                f"Reloaded metric {key}={actual!r} does not match saved {expected!r}."
            )
    recomputed["required_node_attributes_verified"] = True
    recomputed["required_edge_attributes_verified"] = True
    recomputed["node_set_verified"] = True
    return recomputed
