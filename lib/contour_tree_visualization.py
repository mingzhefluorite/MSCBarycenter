"""
Contour tree visualization using branch decomposition layout.

Based on the Fuzzy Contour Trees layout algorithm (Lohfink et al., CGF 2020).
Each node is visualized individually (no saddle bundling).

Input: NetworkX graph(s) with:
  - nodes: attribute "height" (scalar function value)
  - edges: attribute "weight" (scalar difference between nodes)

Layout is **deterministic**: for fixed structure, ``height``, ``weight``, and
optional ``root``, the result does not depend on NetworkX node/edge iteration
order or on isomorphic relabeling of node ids.

Persistence for a branch = |height(saddle) - height(leaf)| = sum of edge weights along path.

Layout debug (INFO logs; configure logging in the caller, e.g. ``logging.basicConfig(level=logging.INFO)``)::

    export CONTOUR_TREE_LAYOUT_DEBUG=1
    # optional:
    export CONTOUR_TREE_LAYOUT_DEBUG_CROSSINGS=1   # log each accepted swap + branch x% order
    export CONTOUR_TREE_LAYOUT_DEBUG_SEP_PASSES=1  # log branch x% after each separation pass
"""

import heapq
import itertools
import logging
import math
import os
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def contour_layout_debug_enabled() -> bool:
    """
    If True, ``contour_tree_layout`` / ``_assign_layout`` emit INFO logs for layout steps.

    Enable with environment variable ``CONTOUR_TREE_LAYOUT_DEBUG=1`` (or ``true`` / ``yes``).

    Extra verbosity (can be large on big trees)::

        CONTOUR_TREE_LAYOUT_DEBUG_CROSSINGS=1   # each accepted swap in crossing minimization
        CONTOUR_TREE_LAYOUT_DEBUG_SEP_PASSES=1  # x% order after each subtree-separation pass
    """
    return _env_truthy("CONTOUR_TREE_LAYOUT_DEBUG")


def _layout_log_branch_x_order(
    G: nx.Graph,
    branches: List[Dict],
    x_percent: List[float],
    step: str,
) -> None:
    """Log all branches sorted by normalized x (x_percent), then branch index."""
    n = len(branches)
    order = sorted(range(n), key=lambda bi: (x_percent[bi], bi))
    logger.info("[contour_tree_layout] %s: branches by x%% (low→high), count=%s", step, n)
    for b in order:
        br = branches[b]
        nodes = br["nodes"]
        saddle, leaf = nodes[0], nodes[-1]
        logger.info(
            "  b=%s x%%=%.8f depth=%s parent_branch=%s mode=%s saddle=%s(h=%.6f) leaf=%s(h=%.6f)",
            b,
            x_percent[b],
            br.get("depth"),
            br.get("parent_branch"),
            br.get("mode"),
            saddle,
            _get_height(G, saddle),
            leaf,
            _get_height(G, leaf),
        )


def _layout_log_sibling_orders(
    branches: List[Dict],
    order_by_parent: Dict[int, List[int]],
    step: str,
) -> None:
    """Log child branch index order per parent (before / after crossing minimization)."""
    logger.info("[contour_tree_layout] %s: sibling branch order by parent", step)
    for p in sorted(order_by_parent.keys()):
        parts = []
        for b in order_by_parent[p]:
            br = branches[b]
            s, lf = br["nodes"][0], br["nodes"][-1]
            parts.append(f"b{b}[{s}→{lf}]")
        logger.info("  parent_branch=%s: %s", p, " ".join(parts))


def _layout_log_final_positions(G: nx.Graph, pos_dict: Dict, step: str) -> None:
    """Log graph nodes sorted by layout x."""
    items = sorted(pos_dict.items(), key=lambda kv: (kv[1][0], kv[0]))
    logger.info("[contour_tree_layout] %s: nodes by layout x (low→high)", step)
    for nid, (x, y) in items:
        logger.info(
            "  node=%s x=%.8f y=%.8f height=%.8f",
            nid,
            x,
            y,
            _get_height(G, nid),
        )


def _layout_log_branch_decomposition(G: nx.Graph, branches: List[Dict]) -> None:
    logger.info(
        "[contour_tree_layout] branch decomposition: n_branches=%s n_nodes=%s",
        len(branches),
        G.number_of_nodes(),
    )
    for bi, br in enumerate(branches):
        logger.info(
            "  branch[%s] parent_branch=%s depth=%s mode=%s nodes=%s",
            bi,
            br.get("parent_branch"),
            br.get("depth"),
            br.get("mode"),
            br.get("nodes"),
        )


def _validate_contour_tree(G: nx.Graph) -> None:
    """Validate that G is a tree with required node/edge attributes."""
    if not nx.is_tree(G):
        raise ValueError("Graph must be a tree")
    for n in G.nodes():
        if "height" not in G.nodes[n]:
            raise ValueError(f"Node {n} missing 'height' attribute")
    for u, v in G.edges():
        if "weight" not in G.edges[u, v]:
            raise ValueError(f"Edge ({u},{v}) missing 'weight' attribute")


def _get_height(G: nx.Graph, node) -> float:
    return G.nodes[node]["height"]


def _get_weight(G: nx.Graph, u, v) -> float:
    return G.edges[u, v].get("weight", abs(_get_height(G, u) - _get_height(G, v)))


def _node_lex_key(n) -> Tuple:
    """
    Stable tie-break for node ids across types (layout must not depend on
    NetworkX insertion / iteration order or relabeling convention).
    """
    if isinstance(n, bool):
        return (3, int(n))
    if isinstance(n, (np.integer, int)):
        return (0, int(n))
    if isinstance(n, (np.floating,)):
        return (1, float(n))
    if isinstance(n, float):
        return (1, float(n))
    return (2, str(n))


def _min_height_node(G: nx.Graph):
    """Global minimum (root) with deterministic tie-break when heights coincide."""
    return min(G.nodes(), key=lambda n: (_get_height(G, n), _node_lex_key(n)))


def _sorted_neighbors(
    G: nx.Graph,
    u,
    excluded: Optional[set] = None,
) -> List:
    """Neighbors of ``u`` sorted by (height, id) for deterministic walks."""
    ex = excluded or set()
    return sorted(
        (nb for nb in G.neighbors(u) if nb not in ex),
        key=lambda nb: (_get_height(G, nb), _node_lex_key(nb)),
    )


def _sorted_nodes(G: nx.Graph) -> List:
    """All nodes sorted by (height, id)."""
    return sorted(G.nodes(), key=lambda n: (_get_height(G, n), _node_lex_key(n)))


def _find_path(
    G: nx.Graph,
    start,
    stop,
    mode: Optional[str],
    visited: Optional[set] = None,
    excluded_from_start: Optional[set] = None,
) -> Optional[List]:
    """
    Find path from start to stop in G.
    mode: 'up' = only move to higher height, 'down' = only to lower, None = any direction.
    excluded_from_start: neighbors of start that must not be used (e.g. parent branch edges).
    """
    if visited is None:
        visited = set()
    if excluded_from_start is None:
        excluded_from_start = set()
    path = [start]
    visited.add(start)
    if start == stop:
        return path

    for nb in _sorted_neighbors(G, start, excluded_from_start):
        if nb in visited:
            continue
        if mode == "up" and _get_height(G, nb) <= _get_height(G, start):
            continue
        if mode == "down" and _get_height(G, nb) >= _get_height(G, start):
            continue
        found = _find_path(G, nb, stop, mode, visited.copy(), excluded_from_start=None)
        if found is not None:
            return path + found
    return None


def _path_persistence(G: nx.Graph, path: List) -> float:
    """Persistence of a path = |height(first) - height(last)|."""
    if len(path) < 2:
        return 0.0
    return abs(_get_height(G, path[0]) - _get_height(G, path[-1]))


def _path_edges(path: List) -> set:
    """Return set of edges (as frozenset) in path."""
    if len(path) < 2:
        return set()
    return {frozenset({path[i], path[i + 1]}) for i in range(len(path) - 1)}


def _find_path_to_leaf_avoiding_edges(
    G: nx.Graph,
    start,
    first_step,
    used_edges: set,
    excluded_from_start: Optional[set] = None,
) -> Optional[List]:
    """
    Find path from start through first_step to a leaf, using only edges not in used_edges.
    Returns path [start, first_step, ..., leaf] or None.
    """
    if excluded_from_start is None:
        excluded_from_start = set()
    if frozenset({start, first_step}) in used_edges:
        return None
    if first_step in excluded_from_start:
        return None

    def dfs(u, visited, path):
        if G.degree(u) == 1 and u != start:
            return path
        for v in _sorted_neighbors(G, u):
            if v in visited:
                continue
            if frozenset({u, v}) in used_edges:
                continue
            new_visited = visited | {v}
            new_path = path + [v]
            result = dfs(v, new_visited, new_path)
            if result is not None:
                return result
        return None

    return dfs(first_step, {start, first_step}, [start, first_step])


def _find_main_branch(
    G: nx.Graph,
    root,
    fixed: set,
    min_scalar: float,
    max_scalar: float,
    excluded_from_start: Optional[set] = None,
) -> Optional[Tuple[List, str, float]]:
    """
    Find the main branch from root to a leaf with maximum persistence.
    Returns (path, mode, persistence) or None.
    excluded_from_start: neighbors of root to exclude (parent branch edges).
    """
    if excluded_from_start is None:
        excluded_from_start = set()
    leaves = sorted(
        (
            n
            for n in G.nodes()
            if G.degree(n) == 1 and n != root and n not in fixed
        ),
        key=lambda n: (_get_height(G, n), _node_lex_key(n)),
    )
    if not leaves:
        return None

    best_path = None
    best_persistence = -1.0
    best_mode = "up"

    for leaf in leaves:
        for mode in ["up", "down", None]:
            visited = set()
            path = _find_path(
                G, root, leaf, mode, visited, excluded_from_start=excluded_from_start
            )
            if path is None:
                continue
            pers = _path_persistence(G, path)
            if pers > best_persistence:
                best_persistence = pers
                best_path = path
                best_mode = mode if mode is not None else "up"

    if best_path is None:
        return None
    return (best_path, best_mode, best_persistence)


def _find_edge_disjoint_branches_from(
    G: nx.Graph,
    start,
    used_edges: set,
    excluded_from_start: Optional[set] = None,
) -> List[Tuple[List, str, float]]:
    """
    Find branches from start to leaves. Each branch uses only edges not in used_edges.
    One branch per unclaimed edge from start. Ensures no edge reuse.
    Returns list of (path, mode, persistence).
    """
    if excluded_from_start is None:
        excluded_from_start = set()
    results = []
    for nb in _sorted_neighbors(G, start, excluded_from_start):
        if frozenset({start, nb}) in used_edges:
            continue
        path = _find_path_to_leaf_avoiding_edges(
            G, start, nb, used_edges, excluded_from_start=excluded_from_start
        )
        if path is not None:
            pers = _path_persistence(G, path)
            mode = "up" if _get_height(G, path[-1]) > _get_height(G, path[0]) else "down"
            results.append((path, mode, pers))
    return sorted(
        results,
        key=lambda item: (
            -item[2],
            tuple((_get_height(G, x), _node_lex_key(x)) for x in item[0]),
            item[1],
        ),
    )


def _get_branch_decomposition(
    G: nx.Graph,
    root=None,
) -> List[Dict]:
    """
    Compute branch decomposition of contour tree G.
    Each edge belongs to exactly one branch; nodes may appear in multiple branches.
    Returns list of branches, each with:
      nodes: list of node ids (path from saddle to leaf)
      mode: 'up' or 'down'
      parent_branch: index of parent (-1 for main branch)
      depth: recursion depth
      persistence: branch persistence
      isorange: [min_height, max_height] of branch
    """
    _validate_contour_tree(G)
    if root is None:
        root = _min_height_node(G)

    min_scalar = min(_get_height(G, n) for n in G.nodes())
    max_scalar = max(_get_height(G, n) for n in G.nodes())

    branches = []
    used_edges = set()
    heap: List[Tuple] = []
    counter = 0

    heapq.heappush(
        heap,
        (0, _get_height(G, root), _node_lex_key(root), counter, root, -1),
    )
    counter += 1

    while heap:
        _, _, _, _, start, parent_idx = heapq.heappop(heap)
        # Exclude parent branch edges: from saddle, cannot use edges to parent path
        excluded = set()
        if parent_idx >= 0:
            parent_path = branches[parent_idx]["nodes"]
            for i, n in enumerate(parent_path):
                if n == start:
                    if i > 0:
                        excluded.add(parent_path[i - 1])
                    if i < len(parent_path) - 1:
                        excluded.add(parent_path[i + 1])
                    break

        if parent_idx < 0:
            # Trunk: single main branch
            result = _find_main_branch(
                G, start, set(), min_scalar, max_scalar, excluded_from_start=excluded
            )
            found = [result] if result is not None else []
        else:
            # Child saddles: one branch per unclaimed edge (no edge reuse)
            found = _find_edge_disjoint_branches_from(
                G, start, used_edges, excluded_from_start=excluded
            )

        if not found:
            if start == root:
                break
            continue

        depth = 0 if parent_idx < 0 else branches[parent_idx]["depth"] + 1
        for path, mode, persistence in found:
            heights = [_get_height(G, n) for n in path]
            isorange = [min(heights), max(heights)]
            branch = {
                "nodes": path,
                "mode": mode,
                "parent_branch": parent_idx,
                "depth": depth,
                "persistence": persistence,
                "isorange": isorange,
            }
            branches.append(branch)
            b_new = len(branches) - 1
            used_edges.update(_path_edges(path))
            d_heap = branches[b_new]["depth"]
            for i in range(1, len(path) - 1):
                heapq.heappush(
                    heap,
                    (
                        d_heap,
                        _get_height(G, path[i]),
                        _node_lex_key(path[i]),
                        counter,
                        path[i],
                        b_new,
                    ),
                )
                counter += 1

    return branches


def get_branch_decomposition(
    G: nx.Graph,
    root=None,
) -> Tuple[List[Dict], Optional[int]]:
    """
    Public API: compute branch decomposition of contour tree G.

    The trunk is the path from root (min height) to a leaf with maximum persistence.
    Child branches attach at saddles (internal nodes) on the trunk or parent branch.
    Edges along the trunk/parent path are excluded from child branches.

    Parameters
    ----------
    G : nx.Graph
        Contour tree. Must be a tree with node attribute "height" and edge "weight".
    root : node, optional
        Root node (global min). Default: node with minimum height.

    Returns
    -------
    branches : list of dict
        Each branch has:
        - nodes: path from saddle to leaf (saddle is nodes[0], leaf is nodes[-1])
        - mode: 'up' or 'down'
        - parent_branch: index of parent (-1 for trunk)
        - depth: recursion depth
        - persistence: |height(saddle) - height(leaf)|
        - isorange: [min_height, max_height]
    root : node
        The root node used.
    """
    _validate_contour_tree(G)
    if root is None:
        root = _min_height_node(G)
    branches = _get_branch_decomposition(G, root=root)
    return branches, root


def format_branch_decomposition(branches: List[Dict], root=None) -> str:
    """
    Format branch decomposition for debugging/inspection.

    Returns a human-readable string, e.g.:
      [0] trunk: 1-0-5-7-19-21-15-17-13-10-9
      [1] saddle 0 -> leaf 3: 0-2-3
      [2] saddle 5 -> leaf 6: 5-6
    """
    lines = []
    for i, b in enumerate(branches):
        path_str = "-".join(str(n) for n in b["nodes"])
        if b["parent_branch"] < 0:
            lines.append(f"  [{i}] trunk: {path_str}")
        else:
            saddle = b["nodes"][0]
            leaf = b["nodes"][-1]
            lines.append(f"  [{i}] saddle {saddle} -> leaf {leaf}: {path_str}")
    return "Branch decomposition:\n" + "\n".join(lines)


def diagnose_branch_decomposition(
    G: nx.Graph,
    branches: List[Dict],
) -> str:
    """
    Diagnose which graph nodes are missing from the branch decomposition.

    The decomposition includes all branches (nodes may appear in multiple branches).
    This diagnostic helps identify any nodes still missing and their structure.

    Returns
    -------
    str
        Human-readable report: node counts, missing nodes with degree/height/type.
    """
    nodes_in_branches = set()
    for b in branches:
        nodes_in_branches.update(b["nodes"])

    all_nodes = set(G.nodes())
    missing = sorted(all_nodes - nodes_in_branches, key=lambda n: (_get_height(G, n), n))

    lines = [
        "Branch decomposition diagnostic:",
        f"  Graph nodes: {len(all_nodes)}",
        f"  Nodes in decomposition: {len(nodes_in_branches)}",
        f"  Missing nodes: {len(missing)}",
    ]
    if missing:
        lines.append("")
        lines.append("Missing nodes (height, degree, type, neighbors):")
        for n in missing:
            h = G.nodes[n].get("height", float("nan"))
            d = G.degree(n)
            t = G.nodes[n].get("type", "?")
            nbs = list(G.neighbors(n))
            nbs_in = [nb for nb in nbs if nb in nodes_in_branches]
            nbs_out = [nb for nb in nbs if nb not in nodes_in_branches]
            lines.append(f"  {n}: height={h:.4f} deg={d} type={t} neighbors={nbs} (in_decomp={nbs_in}, not_in={nbs_out})")
        lines.append("")
        lines.append("Note: All branches are included; nodes may appear in multiple branches.")
    return "\n".join(lines)


def _branch_is_proper_descendant(
    children_by_parent: Dict[int, List[int]],
    ancestor_b: int,
    b: int,
) -> bool:
    """True if branch index ``b`` is a strict descendant of ``ancestor_b`` in the branch tree."""
    stack = list(children_by_parent.get(ancestor_b, []))
    seen = set()
    while stack:
        cur = stack.pop()
        if cur == b:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(children_by_parent.get(cur, []))
    return False


def _count_x_percent_duplicate_pairs(x_percent: List[float], eps: float = 1e-7) -> int:
    """Count unordered pairs of distinct branches with (nearly) identical x_percent."""
    n = len(x_percent)
    c = 0
    for a in range(n):
        xa = x_percent[a]
        for b in range(a + 1, n):
            if abs(xa - x_percent[b]) < eps:
                c += 1
    return c


def _resolve_duplicate_x_percent_pairs(
    children_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    eps: float = 1e-7,
    min_shift: float = 5e-6,
    max_rounds: int = 4096,
) -> None:
    """
    Break coincident x_percent between branches by translating whole subtrees.

    - If one branch is a proper descendant of the other, shift the descendant subtree
      (so the parent column stays fixed).
    - Otherwise shift the higher-index subtree (deterministic tie-break).

    Uses small ``min_shift`` so layout stays visually similar while guaranteeing
    pairwise separation after enough rounds (trunk branch 0 is never shifted).
    """
    n = len(x_percent)
    if n < 2:
        return
    for _ in range(max_rounds):
        found = None
        for a in range(n):
            xa = x_percent[a]
            for bb in range(a + 1, n):
                if abs(xa - x_percent[bb]) >= eps:
                    continue
                found = (a, bb)
                break
            if found:
                break
        if not found:
            break
        a, b = found
        if _branch_is_proper_descendant(children_by_parent, a, b):
            root = b
        elif _branch_is_proper_descendant(children_by_parent, b, a):
            root = a
        else:
            root = b
        _apply_subtree_shift_x_percent(
            root, min_shift, children_by_parent, x_percent, trunk_idx=0
        )


def _layout_objective(
    G: nx.Graph,
    branches: List[Dict],
    x_percent: List[float],
) -> int:
    """
    Scalar to minimize in sibling-order search: geometric crossings plus a large
    penalty for coincident branch x_percent (treated as degenerate crossings).
    """
    n = len(branches)
    dup = _count_x_percent_duplicate_pairs(x_percent)
    cross = _compute_global_crossings(G, branches, x_percent)
    penalty = max(n * n, 64) * 1000
    return cross + penalty * dup


def _compute_x_percent_from_order(
    G: nx.Graph,
    branches: List[Dict],
    order_by_parent: Dict[int, List[int]],
    children_by_parent: Dict[int, List[int]],
) -> List[float]:
    """
    Compute x_percent for each branch. Children alternate left/right from parent;
    ``order_by_parent`` determines sibling order (used by crossing minimization).

    After the geometric spread, coincident x_percent values are resolved so distinct
    branches do not share the same column (see ``_resolve_duplicate_x_percent_pairs``).
    """
    n_branches = len(branches)
    x_percent = [0.5] * n_branches
    for parent_idx in sorted(order_by_parent.keys()):
        child_indices = order_by_parent[parent_idx]
        if not child_indices:
            continue
        parent_x = x_percent[parent_idx]
        depth = branches[child_indices[0]]["depth"]
        base_offset = 0.4 / (2 ** (depth - 1)) if depth >= 1 else 0.2

        left_count = right_count = 0
        for b in child_indices:
            n_sibs = len(child_indices)
            if n_sibs == 1:
                side = -1 if (depth % 2 == 0) else 1
                slot = 1
            elif (left_count + right_count) % 2 == 0:
                side = -1
                left_count += 1
                slot = left_count
            else:
                side = 1
                right_count += 1
                slot = right_count
            x_percent[b] = parent_x + side * base_offset * slot
    _resolve_duplicate_x_percent_pairs(children_by_parent, x_percent)
    return x_percent


def _compute_global_crossings(
    G: nx.Graph,
    branches: List[Dict],
    x_percent: List[float],
) -> int:
    """
    Count geometric crossings for all branch pairs (layout search also adds a
    large penalty for coincident ``x_percent`` via ``_layout_objective``).

    A crossing occurs when:
    - A's vertical segment passes through B's saddle: B should be closer. Count if B further.
    - B's vertical segment passes through A's saddle: A should be closer. Count if A further.
    - A's vertical segment passes through B's leaf: B should be further. Count if B closer.
    - B's vertical segment passes through A's leaf: A should be further. Count if A closer.
    - Siblings (same saddle): inner leaf should be further. Count if inner leaf closer.
    Handles saddle-maximum vs saddle-minimum (different directions).
    Only count when both branches are on the same side of trunk (0.5).
    """
    n_branches = len(branches)
    crossings = 0
    for a in range(n_branches):
        y_sa = _get_height(G, branches[a]["nodes"][0])
        y_la = _get_height(G, branches[a]["nodes"][-1])
        x_a = x_percent[a]
        side_a = -1 if x_a < 0.5 else 1
        y_min_a, y_max_a = min(y_sa, y_la), max(y_sa, y_la)
        for b in range(a + 1, n_branches):
            y_sb = _get_height(G, branches[b]["nodes"][0])
            y_lb = _get_height(G, branches[b]["nodes"][-1])
            x_b = x_percent[b]
            side_b = -1 if x_b < 0.5 else 1
            if side_a != side_b:
                continue
            dist_a = abs(x_a - 0.5)
            dist_b = abs(x_b - 0.5)
            y_min_b, y_max_b = min(y_sb, y_lb), max(y_sb, y_lb)

            # A's segment passes through B's saddle (B should be closer)
            if y_min_a < y_sb < y_max_a and dist_b >= dist_a:
                crossings += 1
            # B's segment passes through A's saddle (A should be closer)
            if y_min_b < y_sa < y_max_b and dist_a >= dist_b:
                crossings += 1
            # A's segment passes through B's leaf (B should be further; saddle-minimum vertical crosses)
            if y_sa != y_sb and y_min_a < y_lb < y_max_a and dist_b <= dist_a:
                crossings += 1
            # B's segment passes through A's leaf (A should be further)
            if y_sa != y_sb and y_min_b < y_la < y_max_b and dist_a <= dist_b:
                crossings += 1
            # Siblings: inner leaf should be further
            if y_sa == y_sb:
                if y_min_a <= y_lb <= y_max_a and dist_b < dist_a:
                    crossings += 1
                if y_min_b <= y_la <= y_max_b and dist_a < dist_b:
                    crossings += 1
    return crossings


def _subtree_x_percent_bounds(
    b: int,
    children_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    memo_lo: Dict[int, float],
    memo_hi: Dict[int, float],
) -> Tuple[float, float]:
    """Min/max x_percent over branch ``b`` and all descendant branches (inclusive)."""
    if b in memo_lo:
        return memo_lo[b], memo_hi[b]
    lo = hi = x_percent[b]
    for ch in children_by_parent.get(b, []):
        clo, chi = _subtree_x_percent_bounds(
            ch, children_by_parent, x_percent, memo_lo, memo_hi
        )
        lo = min(lo, clo)
        hi = max(hi, chi)
    memo_lo[b] = lo
    memo_hi[b] = hi
    return lo, hi


def _refresh_all_subtree_bounds(
    n_branches: int,
    children_by_parent: Dict[int, List[int]],
    x_percent: List[float],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    memo_lo: Dict[int, float] = {}
    memo_hi: Dict[int, float] = {}
    for b in range(n_branches):
        _subtree_x_percent_bounds(b, children_by_parent, x_percent, memo_lo, memo_hi)
    return memo_lo, memo_hi


def _subtree_branch_indices(
    children_by_parent: Dict[int, List[int]],
    root_b: int,
) -> List[int]:
    """Branch indices in the subtree rooted at ``root_b`` (including ``root_b``)."""
    out: List[int] = []
    stack = [root_b]
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        stack.extend(sorted(children_by_parent.get(u, [])))
    return out


def _subtree_height_span_for_branch_indices(
    G: nx.Graph,
    branches: List[Dict],
    branch_indices: List[int],
) -> Tuple[float, float]:
    """Min / max node ``height`` over all nodes appearing on any of ``branch_indices``."""
    y_lo = float("inf")
    y_hi = float("-inf")
    for b in branch_indices:
        for nid in branches[b]["nodes"]:
            h = _get_height(G, nid)
            y_lo = min(y_lo, h)
            y_hi = max(y_hi, h)
    if not branch_indices or y_lo == float("inf"):
        return 0.0, 0.0
    return y_lo, y_hi


def _trunk_child_band_overlap_score(
    G: nx.Graph,
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    trunk_child_branches: List[int],
) -> float:
    """
    Heuristic: sum of (x-interval overlap) * (height-interval overlap) over pairs of
    direct trunk subtrees (children of branch 0). Larger = worse.
    """
    if len(trunk_child_branches) < 2:
        return 0.0
    n = len(branches)
    memo_lo, memo_hi = _refresh_all_subtree_bounds(
        n, children_by_parent, x_percent
    )
    y_spans = {
        c: _subtree_height_span_for_branch_indices(
            G, branches, _subtree_branch_indices(children_by_parent, c)
        )
        for c in trunk_child_branches
    }
    score = 0.0
    lst = list(trunk_child_branches)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            a, b = lst[i], lst[j]
            ox = max(
                0.0,
                min(memo_hi[a], memo_hi[b]) - max(memo_lo[a], memo_lo[b]),
            )
            ya0, ya1 = y_spans[a]
            yb0, yb1 = y_spans[b]
            oy = max(0.0, min(ya1, yb1) - max(ya0, yb0))
            score += ox * oy + 1e-9 * ox
    return score


def _squeeze_master_trunk_subtrees_toward_trunk(
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    depth_scale: float = 0.12,
) -> None:
    """
    Compress each direct subtree of the master branch (0) toward ``x_percent[0]`` so
    wide nested structures do not expand too far horizontally. Uses the max branch
    depth inside each trunk-attached subtree to set a scale factor in (0, 1].
    """
    tx = x_percent[0]
    kids = sorted(children_by_parent.get(0, []))
    if not kids:
        return
    for c in kids:
        tset = _subtree_branch_indices(children_by_parent, c)
        dmax = max(branches[t]["depth"] for t in tset)
        scale = min(1.0, 1.0 / (1.0 + depth_scale * max(0, dmax - 1)))
        if scale >= 1.0 - 1e-15:
            continue
        for t in tset:
            x_percent[t] = tx + (x_percent[t] - tx) * scale


def _order_by_parent_shallow_copy(order_by_parent: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {p: list(v) for p, v in order_by_parent.items()}


def _optimize_trunk_child_order_for_band_overlap(
    G: nx.Graph,
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    order_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    max_perm: int = 8,
) -> bool:
    """
    Reorder only ``order_by_parent[0]`` (subtrees hanging from master branch) to
    reduce overlap of (x%, height) bands. Updates ``order_by_parent`` and
    ``x_percent`` in place when an improvement is found.

    Exhaustive search for ``<= max_perm`` trunk children; cheap hill-climbing otherwise.
    """
    kids = order_by_parent.get(0)
    if not kids or len(kids) < 2:
        return False
    base_ob = _order_by_parent_shallow_copy(order_by_parent)

    def _eval_for_trunk_order(trunk_order: List[int]) -> Tuple[float, List[float]]:
        ob = dict(base_ob)
        ob[0] = list(trunk_order)
        xt = _compute_x_percent_from_order(G, branches, ob, children_by_parent)
        _resolve_duplicate_x_percent_pairs(children_by_parent, xt)
        sc = _trunk_child_band_overlap_score(
            G, branches, children_by_parent, xt, trunk_order
        )
        return sc, xt

    cur_order = list(kids)
    best_score, _ = _eval_for_trunk_order(cur_order)
    best_order = tuple(cur_order)

    if len(kids) <= max_perm:
        for perm in itertools.permutations(kids):
            sc, xt = _eval_for_trunk_order(list(perm))
            if sc < best_score - 1e-15:
                best_score = sc
                best_order = perm
    else:
        cur = list(kids)
        for _ in range(len(kids) * 8):
            improved = False
            for i in range(len(cur) - 1):
                trial = cur[:]
                trial[i], trial[i + 1] = trial[i + 1], trial[i]
                sc, _ = _eval_for_trunk_order(trial)
                if sc < best_score - 1e-15:
                    best_score = sc
                    best_order = tuple(trial)
                    cur = trial
                    improved = True
                    break
            if not improved:
                break

    if best_order == tuple(kids):
        return False
    order_by_parent[0] = list(best_order)
    x_percent[:] = _compute_x_percent_from_order(
        G, branches, order_by_parent, children_by_parent
    )
    _resolve_duplicate_x_percent_pairs(children_by_parent, x_percent)
    return True


def _apply_subtree_shift_x_percent(
    root_b: int,
    dx: float,
    children_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    trunk_idx: int = 0,
) -> None:
    """Add ``dx`` to ``x_percent`` for ``root_b`` and every strict descendant branch."""
    if root_b == trunk_idx or abs(dx) < 1e-15:
        return
    stack = [root_b]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur == trunk_idx:
            continue
        x_percent[cur] += dx
        for ch in sorted(children_by_parent.get(cur, [])):
            stack.append(ch)


def _separate_overlapping_subtree_intervals(
    G: nx.Graph,
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    order_by_parent: Dict[int, List[int]],
    x_percent: List[float],
    margin: float,
    max_passes: int,
) -> None:
    """
    After initial x_percent assignment, push sibling subtrees apart using each
    subtree's horizontal bounding interval in x_percent space.

    Swaps in crossing minimization only reorder children; they do not reserve
    width for nested branches, so distinct subtrees can share nearly the same x.
    This pass translates whole subtrees (same delta on every descendant branch)
    so consecutive siblings (left-to-right by current x) do not overlap, with
    at least ``margin`` gap between their intervals.

    Packing is **recursive along the master branch hierarchy**: at each branch
    node we first reserve space inside each child subtree (post-order), then pack
    that node's siblings so their x% bounding intervals (including descendants) do
    not overlap—so a fixed subtree zone is not invaded by siblings or outer layout
    once children have been placed.
    """
    n_branches = len(branches)
    if n_branches < 2 or margin <= 0:
        return

    for pass_i in range(max_passes):
        changed = False

        def dfs_pack(parent_idx: int) -> None:
            nonlocal changed
            children = order_by_parent.get(parent_idx, [])
            for c in children:
                dfs_pack(c)
            if len(children) < 2:
                return
            memo_lo, memo_hi = _refresh_all_subtree_bounds(
                n_branches, children_by_parent, x_percent
            )
            ordered = sorted(children, key=lambda c: (x_percent[c], c))
            right_edge = float("-inf")
            for c in ordered:
                lo, hi = memo_lo[c], memo_hi[c]
                if lo < right_edge + margin - 1e-15:
                    delta = right_edge + margin - lo
                    _apply_subtree_shift_x_percent(
                        c, delta, children_by_parent, x_percent
                    )
                    changed = True
                memo_lo, memo_hi = _refresh_all_subtree_bounds(
                    n_branches, children_by_parent, x_percent
                )
                lo, hi = memo_lo[c], memo_hi[c]
                right_edge = max(right_edge, hi)

        dfs_pack(0)
        if contour_layout_debug_enabled() and _env_truthy(
            "CONTOUR_TREE_LAYOUT_DEBUG_SEP_PASSES"
        ):
            _layout_log_branch_x_order(
                G,
                branches,
                x_percent,
                f"subtree_separate pass={pass_i} changed={changed}",
            )
        if not changed:
            break


def _minimize_crossings_global(
    G: nx.Graph,
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    order_by_parent: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    """
    Minimize crossings by trying all swaps, applying the one that reduces the most.
    Stops when no swap reduces crossings.
    """
    order_by_parent = {p: list(children) for p, children in order_by_parent.items()}
    x_percent = _compute_x_percent_from_order(
        G, branches, order_by_parent, children_by_parent
    )
    best_count = _layout_objective(G, branches, x_percent)
    if contour_layout_debug_enabled():
        d0 = _count_x_percent_duplicate_pairs(x_percent)
        c0 = _compute_global_crossings(G, branches, x_percent)
        logger.info(
            "[contour_tree_layout] crossing minimization: initial objective=%s "
            "(crossings=%s duplicate_x_pairs=%s)",
            best_count,
            c0,
            d0,
        )
        _layout_log_branch_x_order(
            G, branches, x_percent, "crossing_min (initial x% from sibling order)"
        )

    swap_k = 0
    while True:
        best_delta = 0
        best_swap = None
        for parent_idx, children in sorted(order_by_parent.items()):
            if len(children) < 2:
                continue
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    new_order = dict(order_by_parent)
                    new_children = list(children)
                    new_children[i], new_children[j] = new_children[j], new_children[i]
                    new_order[parent_idx] = new_children
                    new_x = _compute_x_percent_from_order(
                        G, branches, new_order, children_by_parent
                    )
                    new_count = _layout_objective(G, branches, new_x)
                    delta = best_count - new_count
                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (parent_idx, i, j)
        if best_delta <= 0:
            break
        parent_idx, i, j = best_swap
        children = order_by_parent[parent_idx]
        ci, cj = children[i], children[j]
        children[i], children[j] = children[j], children[i]
        x_percent = _compute_x_percent_from_order(
            G, branches, order_by_parent, children_by_parent
        )
        best_count = _layout_objective(G, branches, x_percent)
        if contour_layout_debug_enabled() and _env_truthy(
            "CONTOUR_TREE_LAYOUT_DEBUG_CROSSINGS"
        ):
            swap_k += 1
            logger.info(
                "[contour_tree_layout] crossing_min swap #%s: parent_branch=%s "
                "swap branches b%s <-> b%s | objective now=%s (delta=%s)",
                swap_k,
                parent_idx,
                ci,
                cj,
                best_count,
                best_delta,
            )
            _layout_log_branch_x_order(
                G, branches, x_percent, f"crossing_min after swap #{swap_k}"
            )

    if contour_layout_debug_enabled():
        df = _count_x_percent_duplicate_pairs(x_percent)
        cf = _compute_global_crossings(G, branches, x_percent)
        logger.info(
            "[contour_tree_layout] crossing minimization: done, final objective=%s "
            "(crossings=%s duplicate_x_pairs=%s)",
            best_count,
            cf,
            df,
        )

    return order_by_parent


def _fallback_layout(
    G: nx.Graph,
    width: float = 1.0,
    height_scale: float = 1.0,
) -> Dict:
    """
    Fallback layout when branch decomposition fails (e.g. non-monotonic heights).
    Y = height. X = spread along a path from root to leaf.
    """
    root = _min_height_node(G)
    leaves = sorted(
        (n for n in G.nodes() if G.degree(n) == 1 and n != root),
        key=lambda n: (_get_height(G, n), _node_lex_key(n)),
    )
    if not leaves:
        # Single node or no leaves (e.g. 2-node graph)
        nodes = _sorted_nodes(G)
        if len(nodes) == 1:
            return {nodes[0]: (width / 2, _get_height(G, nodes[0]) * height_scale)}
        # Use any path
        path = [root] + [n for n in nodes if n != root]
    else:
        path = _find_path(G, root, leaves[0], None, set())
        if path is None:
            path = [root] + [
                n
                for n in nx.dfs_preorder_nodes(
                    G, root, sort_neighbors=lambda n: _sorted_neighbors(G, n)
                )
                if n != root
            ]
    n_path = len(path)
    pos_dict = {}
    for i, node in enumerate(path):
        x = width * (i / (n_path - 1)) if n_path > 1 else width / 2
        pos_dict[node] = (x, _get_height(G, node) * height_scale)
    for n in _sorted_nodes(G):
        if n not in pos_dict:
            pos_dict[n] = (width / 2, _get_height(G, n) * height_scale)
    if contour_layout_debug_enabled():
        _layout_log_final_positions(G, pos_dict, "_fallback_layout result")
    return pos_dict


def _branch_span_covers_level(branch: Dict, level: float, eps: float = 1e-12) -> bool:
    lo, hi = branch["isorange"]
    return (lo - eps) <= level <= (hi + eps)


def _build_children_by_parent(branches: List[Dict]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for b in range(1, len(branches)):
        p = branches[b]["parent_branch"]
        out.setdefault(p, []).append(b)
    for p in list(out.keys()):
        out[p] = sorted(out[p])
    return out


def _order_by_parent_from_group_order(
    children_by_parent: Dict[int, List[int]],
    groups_data: Dict[str, Any],
    group_order: List[int],
) -> Dict[int, List[int]]:
    """Sibling list per parent sorted by global group permutation (legacy helper)."""
    b2g = groups_data["branch_to_group"]
    gpos = {g: i for i, g in enumerate(group_order)}
    out: Dict[int, List[int]] = {}
    for p, chs in children_by_parent.items():
        out[p] = sorted(chs, key=lambda c: (gpos.get(b2g[c], 10**9), 0.0, c))
    return out


def _senior_child_branch(branches: List[Dict], child_ids: List[int]) -> int:
    """Most senior side branch: max persistence, then smallest index (stable)."""
    if not child_ids:
        return -1
    if len(child_ids) == 1:
        return child_ids[0]
    return min(
        child_ids,
        key=lambda c: (-float(branches[c].get("persistence", 0.0)), c),
    )


def _derive_group_order_from_nested(
    branches: List[Dict],
    groups_data: Dict[str, Any],
    children_by_parent: Dict[int, List[int]],
    order_by_parent: Dict[int, List[int]],
    group_order_fallback: List[int],
    x_raw: List[float],
) -> List[int]:
    """
    Sort branch groups by left-to-right anchor from nested normalized layout,
    for ORDER / DAG after hierarchical sibling optimization.
    """
    gids = sorted(groups_data["groups"].keys())
    n = len(branches)
    if n <= 1:
        return list(gids)
    xn = _branch_x_normalized_nested_layout(
        n,
        children_by_parent,
        branches,
        groups_data,
        group_order_fallback,
        order_by_parent,
        x_raw,
    )
    anchor: Dict[int, float] = {}
    for g in gids:
        mem = groups_data["groups"][g]["members"]
        anchor[g] = min(float(xn[b]) for b in mem)
    return sorted(gids, key=lambda gg: (anchor[gg], gg))


def _hierarchical_sibling_orders_sa(
    G: nx.Graph,
    branches: List[Dict],
    groups_data: Dict[str, Any],
    children_by_parent: Dict[int, List[int]],
    group_order: List[int],
    seed: int = 0,
    outer_rounds: int = 10,
    inner_swaps_per_parent: int = 120,
    x_tie: Optional[List[float]] = None,
    warm_start: Optional[Dict[int, List[int]]] = None,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """
    Stage 2 (PERMUTE): recursively refine sibling order at each branch-tree parent.

    Simulated annealing swaps children within ``order_by_parent[p]``; scoring uses
    the same nested x layout as the final pipeline (trunk at 0; subtrees stay on
    one side of the global trunk). ``group_order`` is fixed during this pass
    (initial global group order) so root left/right vs trunk stays stable.

    If ``warm_start`` is set, SA starts from that sibling order (e.g. polish after
    POSITION). If ``x_tie`` is set, nested sibling tie-break uses those values.
    """
    rng = np.random.default_rng(seed)
    if warm_start is not None:
        order_by_parent = {int(k): v[:] for k, v in warm_start.items()}
        for p, chs in children_by_parent.items():
            if p not in order_by_parent:
                gp = {g: i for i, g in enumerate(group_order)}
                b2g = groups_data["branch_to_group"]
                order_by_parent[p] = sorted(
                    chs, key=lambda c: (gp.get(b2g[c], 10**9), 0.0, c)
                )
    else:
        order_by_parent = _order_by_parent_from_group_order(
            children_by_parent, groups_data, group_order
        )
    n_b = len(branches)
    x_ph = list(x_tie) if x_tie is not None else [0.0] * n_b
    if len(x_ph) != n_b:
        x_ph = [0.0] * n_b

    def score_state(obp: Dict[int, List[int]]) -> float:
        xn = _branch_x_normalized_nested_layout(
            n_b,
            children_by_parent,
            branches,
            groups_data,
            group_order,
            obp,
            x_ph,
        )
        x_pct = [
            min(0.98, max(0.02, (float(xv) + 1.0) * 0.5)) for xv in xn
        ]
        return float(_layout_objective(G, branches, x_pct))

    best = {k: v[:] for k, v in order_by_parent.items()}
    cur = {k: v[:] for k, v in order_by_parent.items()}
    best_sc = cur_sc = score_state(cur)
    init_sc = cur_sc
    parents = [p for p, chs in cur.items() if len(chs) >= 2]
    if not parents:
        return best, {
            "initial_score": float(init_sc),
            "score": float(best_sc),
        }

    T0, T1 = 0.85, 0.02
    total_steps = max(1, outer_rounds * inner_swaps_per_parent * len(parents))
    for t in range(total_steps):
        frac = float(t) / float(max(total_steps - 1, 1))
        T = T0 * (T1 / T0) ** frac
        p = int(rng.choice(parents))
        chs = cur[p]
        if len(chs) < 2:
            continue
        trial = {k: v[:] for k, v in cur.items()}
        i, j = sorted(rng.choice(len(chs), size=2, replace=False).tolist())
        trial[p][i], trial[p][j] = trial[p][j], trial[p][i]
        sc = score_state(trial)
        if sc <= cur_sc or rng.random() < math.exp((cur_sc - sc) / max(T, 1e-9)):
            cur, cur_sc = trial, sc
            if sc < best_sc:
                best, best_sc = {k: v[:] for k, v in trial.items()}, sc

    return best, {
        "initial_score": float(init_sc),
        "score": float(best_sc),
    }


def _build_branch_groups(
    G: nx.Graph,
    branches: List[Dict],
) -> Dict[str, Any]:
    """
    Stage 1 (SIMPLIFY): construct branch groups and coarse contour-tree edges.
    """
    n = len(branches)
    if n == 0:
        return {
            "branch_to_group": {},
            "groups": {},
            "coarse_edges": [],
        }

    children_by_parent = _build_children_by_parent(branches)
    parent = {b: branches[b]["parent_branch"] for b in range(n)}

    group_of = {b: b for b in range(n)}
    members = {b: {b} for b in range(n)}

    def external_levels(mset: set) -> List[float]:
        lvls: List[float] = []
        for b in sorted(mset):
            pb = parent[b]
            if pb >= 0 and pb not in mset:
                lvls.append(float(_get_height(G, branches[b]["nodes"][0])))
            for ch in children_by_parent.get(b, []):
                if ch not in mset:
                    lvls.append(float(_get_height(G, branches[ch]["nodes"][0])))
        return lvls

    def crossing_criterion_ok(mset: set) -> bool:
        lvls = external_levels(mset)
        if not lvls:
            return True
        for h in lvls:
            c = sum(1 for b in mset if _branch_span_covers_level(branches[b], h))
            if c != 1:
                return False
        return True

    changed = True
    while changed:
        changed = False
        for b in range(1, n):
            gb = group_of[b]
            pb = parent[b]
            if pb < 0:
                continue
            gp = group_of[pb]
            if gb == gp:
                continue
            candidate = members[gb] | members[gp]
            if not crossing_criterion_ok(candidate):
                continue
            # Merge child group into parent group (deterministic representative: min id).
            rep = min(gb, gp)
            oth = max(gb, gp)
            merged = members[rep] | members[oth]
            members[rep] = merged
            del members[oth]
            for bb in merged:
                group_of[bb] = rep
            changed = True

    group_ids = sorted(members.keys())
    groups: Dict[int, Dict[str, Any]] = {}
    coarse_edge_set = set()
    for gid in group_ids:
        g_members = sorted(members[gid])
        span_lo = min(branches[b]["isorange"][0] for b in g_members)
        span_hi = max(branches[b]["isorange"][1] for b in g_members)
        parent_groups = set()
        child_groups = set()
        for b in g_members:
            pb = parent[b]
            if pb >= 0:
                gp = group_of[pb]
                if gp != gid:
                    parent_groups.add(gp)
                    coarse_edge_set.add((gp, gid))
            for ch in children_by_parent.get(b, []):
                gc = group_of[ch]
                if gc != gid:
                    child_groups.add(gc)
        groups[gid] = {
            "id": gid,
            "members": g_members,
            "span": (float(span_lo), float(span_hi)),
            "parents": sorted(parent_groups),
            "children": sorted(child_groups),
        }

    coarse_edges = sorted(coarse_edge_set)
    branch_to_group = {b: group_of[b] for b in range(n)}
    return {
        "branch_to_group": branch_to_group,
        "groups": groups,
        "coarse_edges": coarse_edges,
    }


def _group_subtree_branch_count(
    members: List[int],
    children_by_parent: Dict[int, List[int]],
) -> int:
    """All branch indices in the subtree of the branch tree rooted at ``members``."""
    seen = set(int(b) for b in members)
    stack = list(seen)
    while stack:
        u = stack.pop()
        for v in children_by_parent.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen)


def _group_intersection_penalty_for_order(
    branches: List[Dict],
    groups_data: Dict[str, Any],
    group_order: List[int],
    children_by_parent: Dict[int, List[int]],
    x_slot: float = 1.12,
    width_scale: float = 0.78,
) -> float:
    """
    PERMUTE objective: approximate intersection mass between branch-group rectangles.

    Each group has a fixed y-interval (scalar span of its branches). Its x-interval
    is centered at the group's slot in the permutation, with half-width growing with
    the number of branches in that group's subtree (subbranches), so thicker subtrees
    penalize overlaps more and SA can separate them in the order.
    """
    groups = groups_data["groups"]
    n_g = len(group_order)
    if n_g <= 1:
        return 0.0

    y_lo: Dict[int, float] = {}
    y_hi: Dict[int, float] = {}
    half_w: Dict[int, float] = {}
    for gid in group_order:
        info = groups[gid]
        lo, hi = info["span"]
        y_lo[gid], y_hi[gid] = float(lo), float(hi)
        n_sub = _group_subtree_branch_count(info["members"], children_by_parent)
        n_mem = max(1, len(info["members"]))
        # Horizontal extent grows with subtree size (subbranches).
        w = 1.0 + 0.55 * math.sqrt(float(n_sub)) + 0.22 * float(n_mem)
        half_w[gid] = width_scale * w

    penalty = 0.0
    for i in range(n_g):
        gi = group_order[i]
        xi = float(i) * x_slot
        li, ri = xi - half_w[gi], xi + half_w[gi]
        y0a, y1a = y_lo[gi], y_hi[gi]
        for j in range(i + 1, n_g):
            gj = group_order[j]
            xj = float(j) * x_slot
            lj, rj = xj - half_w[gj], xj + half_w[gj]
            y0b, y1b = y_lo[gj], y_hi[gj]
            oy = max(0.0, min(y1a, y1b) - max(y0a, y0b))
            if oy <= 0.0:
                continue
            ox = max(0.0, min(ri, rj) - max(li, lj))
            if ox <= 0.0:
                continue
            penalty += ox * oy
    return float(penalty)


def _preview_x_percent_from_group_order(
    branches: List[Dict],
    groups_data: Dict[str, Any],
    group_order: List[int],
) -> List[float]:
    """
    Map a global ``group_order`` to branch ``x_percent`` (nested layout, trunk 0.5).

    Builds a default sibling order per parent from ``group_order``; used by the
    legacy flat group-order SA only.
    """
    n = len(branches)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    children_by_parent = _build_children_by_parent(branches)
    obp = _order_by_parent_from_group_order(
        children_by_parent, groups_data, group_order
    )
    x_placeholder = [0.0] * n
    xn = _branch_x_normalized_nested_layout(
        n,
        children_by_parent,
        branches,
        groups_data,
        group_order,
        obp,
        x_placeholder,
    )
    return [min(0.98, max(0.02, (float(xv) + 1.0) * 0.5)) for xv in xn]


def _weighted_crossing_objective_for_group_order(
    G: nx.Graph,
    branches: List[Dict],
    groups_data: Dict[str, Any],
    group_order: List[int],
) -> float:
    """
    PERMUTE objective: group-rectangle overlap plus branch-level layout score
    from a preview placement induced by the same permutation (crossings + dup penalty).
    """
    children_by_parent = _build_children_by_parent(branches)
    rect = _group_intersection_penalty_for_order(
        branches, groups_data, group_order, children_by_parent
    )
    x_prev = _preview_x_percent_from_group_order(branches, groups_data, group_order)
    branch_score = float(_layout_objective(G, branches, x_prev))
    return rect + 1.25 * branch_score


def _permute_groups_simulated_annealing(
    G: nx.Graph,
    branches: List[Dict],
    groups_data: Dict[str, Any],
    seed: int = 0,
    steps: int = 7000,
    t0: float = 1.0,
    t1: float = 1e-3,
) -> Dict[str, Any]:
    """
    Stage 2 (PERMUTE): simulated annealing for branch-group order.
    """
    group_ids = sorted(groups_data["groups"].keys())
    if len(group_ids) <= 1:
        return {
            "order": group_ids,
            "score": 0.0,
            "initial_score": 0.0,
        }

    rng = np.random.default_rng(seed)
    cur = list(group_ids)
    cur_score = _weighted_crossing_objective_for_group_order(
        G, branches, groups_data, cur
    )
    best = list(cur)
    best_score = cur_score
    init_score = cur_score

    for k in range(max(steps, 1)):
        trial = cur[:]
        if len(cur) > 2 and rng.random() < 0.65:
            a, b = sorted(rng.choice(len(cur), size=2, replace=False).tolist())
            trial[a], trial[b] = trial[b], trial[a]
        else:
            a, b = sorted(rng.choice(len(cur), size=2, replace=False).tolist())
            item = trial.pop(b)
            trial.insert(a, item)
        trial_score = _weighted_crossing_objective_for_group_order(
            G, branches, groups_data, trial
        )
        delta = trial_score - cur_score
        alpha = k / max(steps - 1, 1)
        temp = (t0 ** (1.0 - alpha)) * (t1 ** alpha)
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12))
        if accept:
            cur, cur_score = trial, trial_score
            if cur_score < best_score:
                best, best_score = list(cur), cur_score

    return {
        "order": best,
        "score": float(best_score),
        "initial_score": float(init_score),
    }


def _spans_overlap(a: Tuple[float, float], b: Tuple[float, float], eps: float = 1e-12) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1]) + eps


def _build_order_constraints_dag(
    branches: List[Dict],
    groups_data: Dict[str, Any],
    group_order: List[int],
) -> Dict[str, Any]:
    """
    Stage 3 (ORDER): build partial-order DAG + minimum separation constraints.
    """
    n = len(branches)
    if n == 0:
        return {"dag": nx.DiGraph(), "delta": {}}

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    delta: Dict[Tuple[int, int], float] = {}
    b2g = groups_data["branch_to_group"]
    gpos = {g: i for i, g in enumerate(group_order)}

    # Intra-group hierarchy constraints.
    for b in range(n):
        p = branches[b]["parent_branch"]
        if p >= 0 and b2g[b] == b2g[p]:
            if _spans_overlap(tuple(branches[b]["isorange"]), tuple(branches[p]["isorange"])):
                dag.add_edge(p, b)
                delta[(p, b)] = max(delta.get((p, b), 0.0), 0.35)

    # Inter-group order constraints.
    for a in range(n):
        ga = b2g[a]
        for b in range(a + 1, n):
            gb = b2g[b]
            if ga == gb:
                continue
            if not _spans_overlap(tuple(branches[a]["isorange"]), tuple(branches[b]["isorange"])):
                continue
            if gpos[ga] < gpos[gb]:
                u, v = a, b
            else:
                u, v = b, a
            dag.add_edge(u, v)
            sep = 0.45 + 0.1 * min(
                float(branches[u].get("persistence", 1.0)),
                float(branches[v].get("persistence", 1.0)),
            )
            delta[(u, v)] = max(delta.get((u, v), 0.0), sep)

    # Break accidental cycles deterministically.
    while True:
        try:
            nx.find_cycle(dag, orientation="original")
        except nx.NetworkXNoCycle:
            break
        cyc = nx.find_cycle(dag, orientation="original")
        # Remove weakest edge in the discovered cycle.
        weakest = min(cyc, key=lambda e: (delta.get((e[0], e[1]), 0.0), e[0], e[1]))
        u, v = weakest[0], weakest[1]
        if dag.has_edge(u, v):
            dag.remove_edge(u, v)
            delta.pop((u, v), None)

    return {"dag": dag, "delta": delta}


def _solve_branch_positions_from_dag(
    branches: List[Dict],
    dag: nx.DiGraph,
    delta: Dict[Tuple[int, int], float],
    groups_data: Dict[str, Any],
    group_order: List[int],
) -> List[float]:
    """
    Stage 4 (POSITION): longest-path style placement under x[v] >= x[u] + delta(u,v).
    """
    n = len(branches)
    if n == 0:
        return []
    x = [0.0] * n
    topo = list(nx.topological_sort(dag))
    for v in topo:
        best = x[v]
        for u in dag.predecessors(v):
            best = max(best, x[u] + delta.get((u, v), 0.4))
        # Deterministic tie-break so branches rarely share identical POSITION x.
        x[v] = best + 1e-6 * float(v)

    # Center around trunk group to keep geometry stable.
    b2g = groups_data["branch_to_group"]
    gpos = {g: i for i, g in enumerate(group_order)}
    trunk_group = b2g.get(0, group_order[0] if group_order else 0)
    center_target = float(gpos.get(trunk_group, 0))
    trunk_x = x[0] if n > 0 else 0.0
    shift = center_target - trunk_x
    x = [xi + shift for xi in x]
    return x


def contour_tree_layout_diagnostics(
    G: nx.Graph,
    root=None,
) -> Dict[str, Any]:
    """
    Compute paper-pipeline diagnostics without generating full node coordinates.
    """
    branches = _get_branch_decomposition(G, root)
    groups_data = _build_branch_groups(G, branches)
    children_by_parent = _build_children_by_parent(branches)
    gseed = sorted(groups_data["groups"].keys())
    hier = _hierarchical_sibling_orders_sa(
        G, branches, groups_data, children_by_parent, gseed, seed=0
    )
    order_by_parent = hier[0]
    x_ph = [0.0] * len(branches)
    group_order = _derive_group_order_from_nested(
        branches,
        groups_data,
        children_by_parent,
        order_by_parent,
        gseed,
        x_ph,
    )
    order_data = _build_order_constraints_dag(branches, groups_data, group_order)
    x_raw = _solve_branch_positions_from_dag(
        branches,
        order_data["dag"],
        order_data["delta"],
        groups_data,
        group_order,
    )
    unique_x = len({round(float(x), 8) for x in x_raw})
    return {
        "n_branches": len(branches),
        "n_groups": len(groups_data["groups"]),
        "n_coarse_edges": len(groups_data["coarse_edges"]),
        "group_intersection_initial": float(hier[1]["initial_score"]),
        "group_intersection_final": float(hier[1]["score"]),
        "crossing_initial": float(hier[1]["initial_score"]),
        "crossing_final": float(hier[1]["score"]),
        "n_dag_edges": int(order_data["dag"].number_of_edges()),
        "n_constraints": len(order_data["delta"]),
        "n_unique_branch_x": int(unique_x),
        "group_order": list(group_order),
    }


def _branch_x_normalized_nested_layout(
    n_branches: int,
    children_by_parent: Dict[int, List[int]],
    branches: List[Dict],
    groups_data: Dict[str, Any],
    group_order: List[int],
    order_by_parent: Dict[int, List[int]],
    x_raw: List[float],
) -> List[float]:
    """
    Normalized x in [-1, 1] for each branch index.

    - Global trunk (branch 0) is fixed at 0.
    - Children of the trunk follow ``order_by_parent[0]`` (sibling permutation).
      A **virtual trunk** is inserted in that list at index ``k`` (default
      ``len(children)//2``): branches before ``k`` use ``[-1, -δ]``, branches from
      ``k`` onward use ``[δ, 1]``, so both sides are used whenever there are two
      or more side branches.
    - At internal parents, the **senior** child (max persistence) is the subtree
      trunk; siblings split left/right of it by group order / ``x_raw`` ties, then
      strips tile ``[L, R]``; the parent branch column is the interval midpoint
      ``0.5*(L+R)``. Pixel columns are then **explicitly** separated by
      ``_enforce_explicit_branch_column_gaps`` using a **minimal** epsilon-scale
      gap so sibling / parent-child columns do not coincide without large
      horizontal moves.

    ``order_by_parent[p]`` is the recursive sibling permutation; ``group_order``
    still affects internal left/right vs the senior child via ``gpos``.

    When ``len(children)==1``, the child always gets a **leafward half-interval** so
    its branch column differs from the parent's (avoids collapsing e.g. parent
    ``[41,19,13]`` and child ``[19,51,21]`` to the same ``x_coords``). Saddle
    junctions remain on the parent column via ``_assign_layout``'s
    ``saddle_to_parent_x``.
    """
    if n_branches <= 0:
        return []
    if n_branches == 1:
        return [0.0]
    b2g = groups_data["branch_to_group"]
    gpos = {g: i for i, g in enumerate(group_order)}
    x_norm = [0.0] * n_branches

    def kids_ordered(b: int) -> List[int]:
        chs = list(children_by_parent.get(b, []))
        if not chs:
            return []
        pref = order_by_parent.get(b)
        if not pref:
            pref = sorted(chs)
        pos_map = {c: i for i, c in enumerate(pref)}
        return sorted(
            chs,
            key=lambda c: (pos_map.get(c, 10**9), float(x_raw[c]), c),
        )

    def cmp_ref(c: int, ref: int) -> int:
        """Negative: c left of ref subtree trunk; positive: right."""
        pr, pc = gpos.get(b2g[ref], 10**9), gpos.get(b2g[c], 10**9)
        if pc < pr:
            return -1
        if pc > pr:
            return 1
        dv = float(x_raw[c]) - float(x_raw[ref])
        if dv < 0.0:
            return -1
        if dv > 0.0:
            return 1
        if c < ref:
            return -1
        if c > ref:
            return 1
        return 0

    def partition_strips(L: float, R: float, seq: List[int]) -> None:
        m = len(seq)
        if m == 0:
            return
        span = float(R - L)
        for i, c in enumerate(seq):
            lo = L + span * (float(i) / float(m))
            hi = L + span * (float(i + 1) / float(m))
            dfs(c, lo, hi)

    def dfs(b: int, L: float, R: float) -> None:
        chs = kids_ordered(b)
        if not chs:
            x_norm[b] = 0.5 * (L + R)
            return

        if b == 0:
            # Virtual trunk in sibling order: prefix → left of x=0, suffix → right.
            k = len(chs) // 2
            left_c, right_c = chs[:k], chs[k:]

            span = float(R - L)
            gap = min(1e-4 * max(abs(span), 1.0), 0.02 * max(abs(span), 1e-12))
            nL, nR = len(left_c), len(right_c)

            if nL == 0 and nR == 0:
                x_norm[0] = 0.0
                return
            if nL == 0:
                # Entirely on the right of the global trunk (x_norm > 0).
                partition_strips(gap, R, right_c)
            elif nR == 0:
                # Entirely on the left (x_norm < 0).
                partition_strips(L, -gap, left_c)
            else:
                partition_strips(L, -gap, left_c)
                partition_strips(gap, R, right_c)
            x_norm[0] = 0.0
            return

        if len(chs) == 1:
            c = chs[0]
            # One child branch under ``b``: parent column stays at the interval
            # center; the child subtree is laid out in a **leafward half** of ``[L,R]``
            # so ``x_norm[c] != x_norm[b]`` (pixel ``x_coords`` differ). This is
            # required when ``b`` has both a path to its own leaf and a subbranch
            # (e.g. ``[41,19,13]`` with child ``[19,51,21]``): full ``[L,R]`` for the
            # child would collapse parent and child branch columns. Saddles still
            # use ``x_coords[parent_branch]`` in ``_assign_layout``, so interior
            # junctions stay on the parent column while the subbranch leaf column
            # moves outward.
            span = float(R - L)
            m = 0.5 * (L + R)
            if span <= 1e-15:
                dfs(c, L, R)
                x_norm[b] = m
                return
            min_child_span = max(1e-9, 0.07 * span)
            if abs(m) <= 1e-14:
                use_right_half = R >= abs(L)
            else:
                use_right_half = m > 0.0
            if use_right_half:
                c_lo, c_hi = m, R
                if c_hi - c_lo < min_child_span:
                    c_lo = max(L, c_hi - min_child_span)
            else:
                c_lo, c_hi = L, m
                if c_hi - c_lo < min_child_span:
                    c_hi = min(R, c_lo + min_child_span)
            if c_hi - c_lo <= 1e-12:
                dfs(c, L, R)
                x_norm[b] = m
                return
            dfs(c, c_lo, c_hi)
            x_norm[b] = m
            return

        sen = _senior_child_branch(branches, chs)
        pref = order_by_parent.get(b) or sorted(chs)
        pos_map = {c: i for i, c in enumerate(pref)}

        left_o: List[int] = []
        right_o: List[int] = []
        for c in chs:
            if c == sen:
                continue
            t = cmp_ref(c, sen)
            if t < 0:
                left_o.append(c)
            elif t > 0:
                right_o.append(c)
            else:
                (left_o if pos_map.get(c, 0) < pos_map.get(sen, 0) else right_o).append(
                    c
                )

        ord_ch = kids_ordered(b)
        left_o = [c for c in ord_ch if c in set(left_o)]
        right_o = [c for c in ord_ch if c in set(right_o)]
        layout_order = left_o + [sen] + right_o
        m = len(layout_order)
        span = float(R - L)
        for i, c in enumerate(layout_order):
            lo = L + span * (float(i) / float(m))
            hi = L + span * (float(i + 1) / float(m))
            dfs(c, lo, hi)
        x_norm[b] = 0.5 * (L + R)

    dfs(0, -1.0, 1.0)
    return x_norm


def _bump_subtree_x_coords(
    children_by_parent: Dict[int, List[int]],
    x_coords: List[float],
    root_b: int,
    delta: float,
    width: float,
) -> None:
    """Add ``delta`` to ``x_coords`` for ``root_b`` and every descendant branch."""
    lo = 1e-12
    hi = max(lo + 1e-12, float(width) - 1e-12)
    for bi in _subtree_branch_indices(children_by_parent, root_b):
        x_coords[bi] = min(hi, max(lo, float(x_coords[bi]) + float(delta)))


def _enforce_explicit_branch_column_gaps(
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    x_coords: List[float],
    width: float,
    min_frac: float = 5e-7,
    max_rounds: int = 32,
) -> None:
    """
    After nested layout, break **degenerate** equal branch columns with a **tiny**
    minimum separation (default ≈ ``5e-7`` of ``width``, plus a float-epsilon
    floor). This is only large enough to distinguish sibling / parent–child
    ``x_coords`` that would otherwise coincide; it is **not** meant to reshape
    the tree horizontally (avoids new edge crossings from large nudges).

    - **Sibling** branch columns (same ``parent_branch``)
    - **Parent** vs **child** branch columns (branch ``0`` / trunk column never moves)

    Whole subtrees are shifted together so interior structure is preserved.
    """
    n = len(branches)
    if n <= 1:
        return
    w = float(width)
    # Sub-permille of width, but never smaller than a few hundred ULPs of ``w``
    # so coincident float columns still separate.
    gap = max(
        float(min_frac) * w,
        512.0 * float(np.finfo(float).eps) * max(1.0, abs(w)),
        1e-15 * max(1.0, abs(w)),
    )
    trunk_x = float(x_coords[0])

    for _ in range(max(1, int(max_rounds))):
        changed = False
        # A) siblings under the same parent, left-to-right by current x
        for p in range(n):
            chs = list(children_by_parent.get(p, []))
            if len(chs) < 2:
                continue
            chs.sort(key=lambda c: (float(x_coords[c]), int(c)))
            for i in range(1, len(chs)):
                d = float(x_coords[chs[i]]) - float(x_coords[chs[i - 1]])
                if d >= gap - 1e-14:
                    continue
                need = gap - d
                _bump_subtree_x_coords(
                    children_by_parent, x_coords, chs[i], need, width
                )
                changed = True
        # B) parent vs child: separate child subtree from parent column
        for c in range(1, n):
            p = int(branches[c]["parent_branch"])
            if p < 0:
                continue
            d = abs(float(x_coords[c]) - float(x_coords[p]))
            if d >= gap - 1e-14:
                continue
            need = gap - d
            if p == 0:
                if float(x_coords[c]) >= trunk_x - 1e-14:
                    _bump_subtree_x_coords(
                        children_by_parent, x_coords, c, need, width
                    )
                else:
                    _bump_subtree_x_coords(
                        children_by_parent, x_coords, c, -need, width
                    )
            else:
                if float(x_coords[c]) >= float(x_coords[p]):
                    _bump_subtree_x_coords(
                        children_by_parent, x_coords, c, need, width
                    )
                else:
                    _bump_subtree_x_coords(
                        children_by_parent, x_coords, c, -need, width
                    )
            changed = True
        if not changed:
            break


def _branch_x_coords_nested_intervals(
    branches: List[Dict],
    children_by_parent: Dict[int, List[int]],
    groups_data: Dict[str, Any],
    group_order: List[int],
    order_by_parent: Dict[int, List[int]],
    x_raw: List[float],
    width: float,
    margin_frac: float = 0.03,
) -> List[float]:
    """
    Map ``_branch_x_normalized_nested_layout`` to ``[0, width]`` (trunk at width/2).
    """
    n = len(branches)
    if n == 0:
        return []
    if n == 1:
        return [float(width) / 2.0]
    x_norm = _branch_x_normalized_nested_layout(
        n, children_by_parent, branches, groups_data, group_order, order_by_parent, x_raw
    )
    margin = float(margin_frac) * float(width)
    w_inner = max(float(width) - 2.0 * margin, 1e-12)
    out: List[float] = []
    for b in range(n):
        t = (float(x_norm[b]) + 1.0) * 0.5
        xpix = margin + t * w_inner
        out.append(min(float(width) - 1e-12, max(1e-12, xpix)))
    return out


def _assign_layout(
    G: nx.Graph,
    branches: List[Dict],
    width: float = 1.0,
    height_scale: float = 1.0,
) -> Dict:
    """Assign positions using paper-style 4-stage contour-tree pipeline."""
    if not branches:
        if contour_layout_debug_enabled():
            logger.info(
                "[contour_tree_layout] _assign_layout: empty branch list → _fallback_layout"
            )
        return _fallback_layout(G, width=width, height_scale=height_scale)

    n_branches = len(branches)

    # Stage 1: simplify to branch groups.
    groups_data = _build_branch_groups(G, branches)
    children_by_parent = _build_children_by_parent(branches)

    # Stage 2: hierarchical PERMUTE — sibling order per parent (recursive SA),
    # with nested x preview that never assigns a subtree interval across the trunk.
    group_order_seed = sorted(groups_data["groups"].keys())
    hier = _hierarchical_sibling_orders_sa(
        G,
        branches,
        groups_data,
        children_by_parent,
        group_order_seed,
        seed=0,
    )
    order_by_parent = hier[0]
    x_ph = [0.0] * n_branches
    group_order = _derive_group_order_from_nested(
        branches,
        groups_data,
        children_by_parent,
        order_by_parent,
        group_order_seed,
        x_ph,
    )

    # Stage 3: build partial-order DAG + min-separation constraints.
    order_data = _build_order_constraints_dag(branches, groups_data, group_order)

    # Stage 4: solve x positions from DAG constraints.
    x_raw = _solve_branch_positions_from_dag(
        branches,
        order_data["dag"],
        order_data["delta"],
        groups_data,
        group_order,
    )

    # Polish hierarchical sibling order using POSITION ``x_raw`` tie-breaks
    # (same DAG / ``group_order``; nested layout only becomes more consistent).
    polish = _hierarchical_sibling_orders_sa(
        G,
        branches,
        groups_data,
        children_by_parent,
        group_order,
        seed=11,
        outer_rounds=4,
        inner_swaps_per_parent=100,
        x_tie=x_raw,
        warm_start=order_by_parent,
    )
    order_by_parent = polish[0]

    if contour_layout_debug_enabled():
        logger.info(
            "[contour_tree_layout] pipeline stats: groups=%s coarse_edges=%s "
            "hierarchical_perm_initial=%.6f hierarchical_perm_final=%.6f "
            "dag_edges=%s constraints=%s",
            len(groups_data["groups"]),
            len(groups_data["coarse_edges"]),
            hier[1]["initial_score"],
            hier[1]["score"],
            order_data["dag"].number_of_edges(),
            len(order_data["delta"]),
        )
        logger.info(
            "[contour_tree_layout] hierarchical polish (x_raw tie): initial=%.6f final=%.6f",
            polish[1]["initial_score"],
            polish[1]["score"],
        )
        for gid in group_order:
            info = groups_data["groups"][gid]
            logger.info(
                "  group=%s members=%s span=(%.6f, %.6f)",
                gid,
                info["members"],
                info["span"][0],
                info["span"][1],
            )

    # Single-branch layout: spread nodes horizontally along the path
    if n_branches == 1:
        pos_dict = {}
        branch = branches[0]
        nodes = branch["nodes"]
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = width * (i / (n_nodes - 1)) if n_nodes > 1 else width / 2
            y = _get_height(G, node) * height_scale
            pos_dict[node] = (x, y)
        for n in _sorted_nodes(G):
            if n not in pos_dict:
                pos_dict[n] = (width / 2, _get_height(G, n) * height_scale)
        if contour_layout_debug_enabled():
            _layout_log_final_positions(G, pos_dict, "single-branch path layout")
        return pos_dict

    # Map PERMUTE + branch tree to layout x: nested disjoint intervals (trunk at
    # width/2); POSITION ``x_raw`` orders siblings; ``order_by_parent`` is the
    # hierarchical permutation per parent.
    x_coords = _branch_x_coords_nested_intervals(
        branches,
        children_by_parent,
        groups_data,
        group_order,
        order_by_parent,
        x_raw,
        width,
    )
    _enforce_explicit_branch_column_gaps(
        branches, children_by_parent, x_coords, width
    )

    if contour_layout_debug_enabled():
        logger.info(
            "[contour_tree_layout] _assign_layout: branch layout_x nested intervals "
            "(hierarchical sibling order + group anchors + x_raw tie-break, "
            "n_branches=%s)",
            n_branches,
        )
        for b in sorted(range(n_branches), key=lambda bi: (x_coords[bi], bi)):
            logger.info(
                "  b=%s layout_x=%.8f x_raw=%.8f",
                b,
                x_coords[b],
                x_raw[b],
            )

    # Precompute saddle -> parent x: each saddle must sit exactly on its parent branch
    saddle_to_parent_x: Dict[int, float] = {}
    for b in range(1, n_branches):
        parent_idx = branches[b]["parent_branch"]
        saddle_node = branches[b]["nodes"][0]
        saddle_to_parent_x[saddle_node] = x_coords[parent_idx]

    pos_dict = {}
    for b, branch in enumerate(branches):
        x = x_coords[b]
        for i, node in enumerate(branch["nodes"]):
            y = _get_height(G, node) * height_scale
            # Saddle nodes always use parent branch x; others use this branch's x
            if node in saddle_to_parent_x:
                node_x = saddle_to_parent_x[node]
            else:
                node_x = x
            pos_dict[node] = (node_x, y)

    for n in _sorted_nodes(G):
        if n not in pos_dict:
            pos_dict[n] = (width / 2, _get_height(G, n) * height_scale)

    if contour_layout_debug_enabled():
        _layout_log_final_positions(
            G,
            pos_dict,
            "final pos_dict (saddle nodes use parent branch x)",
        )

    return pos_dict


def _get_edge_segments(
    G: nx.Graph,
    branches: List[Dict],
    pos: Dict,
) -> List[List[Tuple[float, float]]]:
    """
    Build edge segments for drawing. Each branch becomes an L-shaped edge:
    saddle -> horizontal at saddle y -> leaf.
    For single-branch (path) layout, draw straight segments between consecutive nodes.
    """
    segments = []
    if len(branches) == 1:
        # Path layout: draw edges between consecutive nodes
        nodes = branches[0]["nodes"]
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            segments.append([pos[u], pos[v]])
        return segments
    if not branches:
        # Fallback: draw edges from graph structure
        for u, v in sorted(G.edges(), key=lambda e: (_node_lex_key(e[0]), _node_lex_key(e[1]))):
            segments.append([pos[u], pos[v]])
        return segments
    for branch in branches:
        nodes = branch["nodes"]
        if len(nodes) < 2:
            continue
        saddle = nodes[0]
        leaf = nodes[-1]
        xs, ys = pos[saddle]
        xl, yl = pos[leaf]

        # L-shaped: (saddle) -> (leaf_x, saddle_y) -> (leaf)
        segments.append([(xs, ys), (xl, ys), (xl, yl)])
    return segments


def edge_segment_node_ids(
    G: nx.Graph,
    branches: List[Dict],
) -> List[List]:
    """
    For each polyline from _get_edge_segments (same order), node ids used for
    styling. Path layout: two endpoints per segment; multi-branch: all nodes on
    the branch; empty-branch fallback: edge endpoints.
    """
    out: List[List] = []
    if len(branches) == 1:
        nodes = branches[0]["nodes"]
        for i in range(len(nodes) - 1):
            out.append([nodes[i], nodes[i + 1]])
        return out
    if not branches:
        for u, v in sorted(G.edges(), key=lambda e: (_node_lex_key(e[0]), _node_lex_key(e[1]))):
            out.append([u, v])
        return out
    for branch in branches:
        nodes = branch["nodes"]
        if len(nodes) < 2:
            continue
        out.append(list(nodes))
    return out


def _apply_density_y_resolution_improvement(
    pos: Dict,
    edge_segments: List[List[Tuple[float, float]]],
    sigma_scale: float = 1.0,
) -> Tuple[Dict, List[List[Tuple[float, float]]]]:
    """
    Apply a monotone, density-aware remap to y coordinates.

    This follows the idea from "Drawing Contour Trees in the Plane":
    dense isovalue regions are spread while preserving vertical order.
    """
    if not pos:
        return pos, edge_segments

    ys = np.array([float(v[1]) for v in pos.values()], dtype=float)
    if ys.size < 2:
        return pos, edge_segments
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    span = max(y_max - y_min, 1e-12)
    sigma = max((span / math.sqrt(max(ys.size, 1))) * float(sigma_scale), 1e-12)

    def _cdf_like(y: float) -> float:
        # Logistic approximation to Gaussian-kernel integral.
        z = np.clip((ys - float(y)) / sigma, -60.0, 60.0)
        return float(np.mean(1.0 / (1.0 + np.exp(z))))

    c0 = _cdf_like(y_min)
    c1 = _cdf_like(y_max)
    den = max(c1 - c0, 1e-12)

    def _map_y(y: float) -> float:
        t = (_cdf_like(float(y)) - c0) / den
        return y_min + span * t

    pos_out = {n: (float(x), float(_map_y(y))) for n, (x, y) in pos.items()}
    seg_out: List[List[Tuple[float, float]]] = []
    for seg in edge_segments:
        seg_out.append([(float(x), float(_map_y(y))) for (x, y) in seg])
    return pos_out, seg_out


def contour_tree_layout(
    G: nx.Graph,
    root=None,
    width: float = 1.0,
    height_scale: float = 1.0,
    apply_resolution_improvement: bool = False,
    resolution_sigma_scale: float = 1.0,
) -> Tuple[Dict, List[Dict], List[List[Tuple[float, float]]]]:
    """
    Compute layout for a single contour tree.

    Parameters
    ----------
    G : nx.Graph
        Contour tree. Must be a tree with node attribute "height" and edge "weight".
    root : node, optional
        Root node (global min). Default: node with minimum height.
    width : float
        Horizontal span for layout (default 1.0).
    height_scale : float
        Scale factor for y-axis (default 1.0).
    apply_resolution_improvement : bool
        If True, apply density-aware monotone remap to y-coordinates.
    resolution_sigma_scale : float
        Bandwidth factor for the remap (lower = stronger spread in dense regions).

    Returns
    -------
    pos : dict
        Node positions {node: (x, y)}.
    branches : list
        Branch decomposition.
    edge_segments : list
        List of polyline segments for drawing edges.
    """
    branches = _get_branch_decomposition(G, root)
    if contour_layout_debug_enabled():
        _layout_log_branch_decomposition(G, branches)
    pos = _assign_layout(G, branches, width=width, height_scale=height_scale)
    edge_segments = _get_edge_segments(G, branches, pos)
    if apply_resolution_improvement:
        pos, edge_segments = _apply_density_y_resolution_improvement(
            pos, edge_segments, sigma_scale=resolution_sigma_scale
        )
    return pos, branches, edge_segments


# Default type-to-color mapping for Mode 1 (node_color_mode="type")
_TYPE_COLOR_MAP = {0: "blue", 1: "white", 2: "red"}


def plot_contour_tree(
    G: Union[nx.Graph, List[nx.Graph]],
    ax=None,
    root=None,
    width: float = 1.0,
    height_scale: float = 1.0,
    apply_resolution_improvement: bool = False,
    resolution_sigma_scale: float = 1.0,
    node_size: float = 120,
    node_color: str = "steelblue",
    node_color_mode: Optional[str] = None,
    node_colors: Optional[Union[Dict, List]] = None,
    use_pie: Optional[Union[Dict, List]] = None,
    pie_size: float = 0.03,
    type_color_map: Optional[Dict[int, str]] = None,
    edge_color: str = "gray",
    edge_linewidth: float = 1.5,
    show_labels: bool = True,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Plot one or more contour trees.

    Parameters
    ----------
    G : nx.Graph or list of nx.Graph
        Contour tree(s). Each must be a tree with node "height" and edge "weight".
    ax : matplotlib Axes, optional
        Axes to draw on. If None, creates new figure.
    root : node, optional
        Root for layout (node with min height by default).
    width : float
        Horizontal layout width.
    height_scale : float
        Scale for y-axis.
    node_size : float
        Node marker size.
    node_color : str
        Node color (used when node_color_mode is None and node_colors is None).
    node_color_mode : str, optional
        "type" = color nodes by "type" attribute (Mode 1). None = use node_color for all.
    node_colors : dict or list, optional
        Custom per-node colors. Dict: node -> color (str or list of (color, frac) for pie).
        List: when G is a list, node_colors[i] is dict/list for trees[i].
    use_pie : list of bool, optional
        When node_colors is used, use_pie[i]=True means draw pie at node i. Index by nodes_order.
    pie_size : float
        Radius for pie charts when use_pie is True.
    type_color_map : dict, optional
        Map node type -> color. Default: {0: "blue", 1: "gray", 2: "white", 3: "red"}.
    edge_color : str
        Edge color.
    edge_linewidth : float
        Edge line width.
    show_labels : bool
        Whether to show node labels.
    title : str, optional
        Plot title.
    **kwargs
        Passed to plt.subplots when ax is None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if isinstance(G, (list, tuple)):
        trees = list(G)
    else:
        trees = [G]

    n_trees = len(trees)
    if ax is None:
        # Max 3 trees per row; use multiple rows when needed
        ncols = min(n_trees, 3)
        nrows = (n_trees + ncols - 1) // ncols
        default_figsize = (20 * ncols, 16 * nrows)
        figsize = kwargs.pop("figsize", default_figsize)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        axes = np.atleast_2d(axes)
        axes_flat = axes.ravel()
        for i in range(n_trees, len(axes_flat)):
            axes_flat[i].set_visible(False)
        axes = [axes_flat[i] for i in range(n_trees)]
    else:
        axes = np.atleast_1d(ax).ravel()
        if len(axes) < n_trees:
            raise ValueError(f"Need {n_trees} axes for {n_trees} trees, got {len(axes)}")
        axes = [axes[i] for i in range(n_trees)]

    def _draw_pie_at(pos_xy, color_fracs: List[Tuple[str, float]], size: float, ax_obj):
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
            w = Wedge(pos_xy, size, start_deg, start_deg + span_deg, facecolor=color, edgecolor="black", linewidth=0.5)
            ax_obj.add_patch(w)
            start_deg += span_deg

    for idx, T in enumerate(trees):
        ax = axes[idx]
        pos, branches, edge_segments = contour_tree_layout(
            T,
            root=root,
            width=width,
            height_scale=height_scale,
            apply_resolution_improvement=apply_resolution_improvement,
            resolution_sigma_scale=resolution_sigma_scale,
        )

        # Draw edges
        lc = LineCollection(
            edge_segments,
            colors=edge_color,
            linewidths=edge_linewidth,
            zorder=0,
        )
        ax.add_collection(lc)

        # Resolve node_colors for this tree
        nc_this = None
        up_this = None
        if node_colors is not None:
            nc_this = node_colors[idx] if isinstance(node_colors, (list, tuple)) else node_colors
            up_this = use_pie[idx] if use_pie is not None and isinstance(use_pie, (list, tuple)) else None
            if isinstance(up_this, dict):
                up_this = [up_this.get(n, False) for n in T.nodes()]

        nodes_list_T = list(T.nodes())
        node_to_idx_T = {n: i for i, n in enumerate(nodes_list_T)}

        # Draw nodes
        xs = [pos[n][0] for n in T.nodes()]
        ys = [pos[n][1] for n in T.nodes()]
        if nc_this is not None:
            # Custom node colors (dict or list indexed by nodes_list_T)
            xs_plot, ys_plot, colors_plot = [], [], []
            for n in T.nodes():
                pos_n = pos[n]
                idx_n = node_to_idx_T.get(n, -1)
                if isinstance(nc_this, dict):
                    col = nc_this.get(n, "#808080")
                else:
                    col = nc_this[idx_n] if 0 <= idx_n < len(nc_this) else "#808080"
                use_pie_n = up_this[idx_n] if up_this is not None and 0 <= idx_n < len(up_this) else False
                if use_pie_n and isinstance(col, (list, tuple)):
                    _draw_pie_at(pos_n, col, pie_size, ax)
                else:
                    c = col if isinstance(col, str) else "#808080"
                    xs_plot.append(pos_n[0])
                    ys_plot.append(pos_n[1])
                    colors_plot.append(c)
            if xs_plot:
                ax.scatter(xs_plot, ys_plot, s=node_size, c=colors_plot, zorder=1, edgecolors="black", linewidths=0.5)
        elif node_color_mode == "type":
            tcm = type_color_map if type_color_map is not None else _TYPE_COLOR_MAP

            def _type_to_color(t):
                if t is None:
                    return "gray"
                try:
                    t_int = int(round(float(t)))
                    return tcm.get(t_int, "gray")
                except (TypeError, ValueError):
                    return "gray"

            colors_by_type = [_type_to_color(T.nodes[n].get("type", None)) for n in T.nodes()]
            ax.scatter(xs, ys, s=node_size, c=colors_by_type, zorder=1, edgecolors="black", linewidths=0.5)
        else:
            ax.scatter(xs, ys, s=node_size, c=node_color, zorder=1)

        if show_labels:
            for n in T.nodes():
                if nc_this is not None:
                    idx_n = node_to_idx_T.get(n, -1)
                    if isinstance(nc_this, dict):
                        col = nc_this.get(n, "#808080")
                    else:
                        col = nc_this[idx_n] if 0 <= idx_n < len(nc_this) else "#808080"
                    c_str = col if isinstance(col, str) else (col[0][0] if col else "#808080")
                elif node_color_mode == "type":
                    col = _type_to_color(T.nodes[n].get("type", None))
                    c_str = col
                else:
                    c_str = node_color
                is_light = c_str.lower() in ("#ffffff", "#fff", "white") or c_str.lower().startswith("#fff")
                lbl_color = "black" if is_light else "white"
                ax.annotate(
                    str(n),
                    pos[n],
                    fontsize=7,
                    ha="center",
                    va="center",
                    color=lbl_color,
                    weight="bold",
                    zorder=2,
                )

        ax.set_aspect("equal")
        ax.autoscale()
        if title is not None:
            t = title[idx] if isinstance(title, (list, tuple)) and idx < len(title) else (title if n_trees == 1 else f"{title} {idx + 1}")
            ax.set_title(t)

    plt.tight_layout()
    return axes[0] if n_trees == 1 else axes


def plot_input_trees(
    trees: List[nx.Graph],
    dataset_name: Optional[str] = None,
    time_step: Optional[int] = None,
    node_color_mode: str = "type",
    width: float = 1.0,
    height_scale: float = 1.0,
    apply_resolution_improvement: bool = False,
    resolution_sigma_scale: float = 1.0,
    show_labels: bool = True,
    **kwargs,
):
    """
    Convenience API for plotting all input contour trees as a grid.

    Parameters mirror ``plot_contour_tree`` but default to input-tree usage:
    - colors by node ``type``
    - optional title from dataset/time-step context
    """
    if not isinstance(trees, (list, tuple)) or len(trees) == 0:
        raise ValueError("trees must be a non-empty list of contour-tree graphs")

    if dataset_name is None:
        title = "Input contour trees"
    elif time_step is None:
        title = f"Input contour trees ({dataset_name})"
    else:
        title = f"Input contour trees ({dataset_name}, time_step={time_step})"

    return plot_contour_tree(
        list(trees),
        width=width,
        height_scale=height_scale,
        node_color_mode=node_color_mode,
        apply_resolution_improvement=apply_resolution_improvement,
        resolution_sigma_scale=resolution_sigma_scale,
        show_labels=show_labels,
        title=title,
        **kwargs,
    )


def get_persistence(G: nx.Graph, branch: Dict) -> float:
    """
    Persistence of a branch = |height(saddle) - height(leaf)|.
    For contour trees this equals the sum of edge weights along the path
    (when weight = |height difference|).
    """
    nodes = branch["nodes"]
    if len(nodes) < 2:
        return 0.0
    return abs(_get_height(G, nodes[0]) - _get_height(G, nodes[-1]))


def relabel_graph_to_contiguous(
    G: nx.Graph,
    nodes_order: Optional[List] = None,
) -> Tuple[nx.Graph, List, List]:
    """
    Relabel graph nodes to contiguous 0..N-1 so coupling matrix rows match node IDs.

    Parameters
    ----------
    G : nx.Graph
        Graph with possibly non-contiguous node IDs.
    nodes_order : list, optional
        Order of nodes for relabeling. new_id = index in nodes_order.
        If None, use sorted(G.nodes(), key=(height, id)).

    Returns
    -------
    G_relabeled : nx.Graph
        Copy of G with nodes 0, 1, ..., N-1.
    nodes_order_new : list
        [0, 1, ..., N-1] - the new node order (same as matrix row order).
    nodes_order_old : list
        Original node IDs in the same order. nodes_order_old[i] = original ID for new node i.
        Use for indexing into pre-relabel matrices (e.g. C_center[np.ix_(nodes_order_old, nodes_order_old)]).
    """
    if nodes_order is None:
        nodes_order = sorted(G.nodes(), key=lambda n: (_get_height(G, n), n))
    nodes_order_old = list(nodes_order)
    old_to_new = {old: i for i, old in enumerate(nodes_order_old)}
    G_relabeled = nx.relabel_nodes(G, old_to_new, copy=True)
    nodes_order_new = list(range(len(nodes_order_old)))
    return G_relabeled, nodes_order_new, nodes_order_old


def _align_oh_vector(oh, n_types: int) -> np.ndarray:
    """Pad/trim a type block to length n_types (FGW feature slice order = unique_types)."""
    v = np.asarray(oh, dtype=float).reshape(-1)
    out = np.zeros(int(n_types), dtype=float)
    if n_types <= 0:
        return out
    m = int(min(n_types, v.size))
    if m > 0:
        out[:m] = v[:m]
    return out


def matrices_from_contour_tree(
    G: nx.Graph,
    relation_kind: str,
    attribute_mode: str,
    unique_types: List,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    h_min: float,
    h_range: float,
    domain_size: float,
    relation_rescale: float = 1.0,
    relation_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Build C (relation matrix) and X (feature matrix) from a contour tree graph.
    Used to derive C_center_tree, X_center_tree from postprocessed G_center for color transfer.

    Parameters
    ----------
    G : nx.Graph
        Contour tree with node attributes height, type, x, y, z.
    relation_kind : str
        "weighted_adjacency" | "shortestpath".
    attribute_mode, unique_types, min_x, max_x, ... : normalization params.
    relation_rescale, relation_offset : scaling for relation matrix (match source Cs).

    Returns
    -------
    C_tree : np.ndarray
        Relation matrix (n x n).
    X_tree : np.ndarray
        Feature matrix (n x d) in same layout as barycenter X.
    nodes_order : list
        Node IDs in matrix row order.
    """
    nodes_order = sorted(G.nodes(), key=lambda n: (_get_height(G, n), n))
    n = len(nodes_order)
    node_to_idx = {n: i for i, n in enumerate(nodes_order)}

    # Relation matrix
    if relation_kind == "weighted_adjacency":
        C_tree = nx.to_numpy_array(G, nodelist=nodes_order, dtype=float, weight="weight")
        edge_mask = C_tree > 0
        # C_tree[edge_mask] = (C_tree[edge_mask] - relation_offset) * relation_rescale
        # C_tree[edge_mask] += 1
    elif relation_kind == "shortestpath":
        C_tree = nx.floyd_warshall_numpy(G, nodelist=nodes_order, weight="weight")
        C_tree = np.nan_to_num(C_tree, nan=np.inf, posinf=np.inf)
        np.fill_diagonal(C_tree, 0.0)
        finite = np.isfinite(C_tree) & (C_tree > 0)
        # if np.any(finite):
        #     C_tree[finite] = C_tree[finite] * relation_rescale
    else:
        raise ValueError(f"relation_kind must be 'weighted_adjacency' or 'shortestpath', got {relation_kind}")

    # Feature matrix X (same layout as barycenter)
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    type_scale = 1.0 / np.sqrt(2)

    parts = []
    if attribute_mode in ("full", "euclidean_type"):
        coords = np.array([
            [
                (G.nodes[n].get("x", 0) - min_x) / domain_size,
                (G.nodes[n].get("y", 0) - min_y) / domain_size,
                (G.nodes[n].get("z", 0) - min_z) / domain_size,
            ]
            for n in nodes_order
        ], dtype=float)
        parts.append(coords)
    if attribute_mode in ("full", "height_type"):
        heights = np.array([
            (G.nodes[n].get("height", 0) - h_min) / h_range
            for n in nodes_order
        ], dtype=float).reshape(-1, 1)
        parts.append(heights)
    onehot = np.zeros((n, n_types), dtype=float)
    for i, n in enumerate(nodes_order):
        tf = G.nodes[n].get("type_features", None)
        if tf is not None:
            onehot[i, :] = _align_oh_vector(tf, n_types)
            continue
        t = G.nodes[n].get("type", 1)
        try:
            t = int(round(float(t)))
        except (TypeError, ValueError):
            t = 1
        idx = type_to_idx.get(t, 0)
        onehot[i, idx] = type_scale
    parts.append(onehot)
    X_tree = np.column_stack(parts)

    return C_tree, X_tree, nodes_order


def reconstruct_graph_from_fgw_barycenter(
    C_center: np.ndarray,
    X_center: np.ndarray,
    attribute_mode: str,
    unique_types: List,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    h_min: float,
    h_range: float,
    domain_size: float,
    threshold: float = 0.5,
    relation_kind: str = "weighted_adjacency",
) -> nx.Graph:
    """
    Reconstruct a graph from FGW barycenter matrices C_center and X_center.

    Node attributes x, y, z, height, type are recovered from X_center columns
    based on attribute_mode. The type block may be soft (several positive entries);
    ``type`` is the argmax class for contour-tree logic, while ``type_features`` keeps
    the full FGW center slice and ``type_mass`` is a nonnegative simplex (same order
    as ``unique_types``). Edge selection depends on relation_kind:
    - adjacency/weighted_adjacency: add edge when C_center >= threshold
    - shortestpath: add edge when C_center is finite and positive (for MST)

    Parameters
    ----------
    C_center : np.ndarray
        Relation/adjacency matrix (N x N).
    X_center : np.ndarray
        Node feature matrix. Layout depends on attribute_mode.
    attribute_mode, unique_types, min_x, max_x, min_y, max_y, min_z, max_z,
    h_min, h_range, domain_size : normalization params from barycenter cell.
    threshold : float
        For adjacency: min value to add edge. For shortestpath: max distance (unused if adding all).
    relation_kind : str
        "weighted_adjacency" | "shortestpath". Affects edge selection and MST type.

    Returns
    -------
    G : nx.Graph
        Graph with node attributes x, y, z, height, type and edge attribute weight.
    """
    C_center = np.asarray(C_center, dtype=float)
    X_center = np.asarray(X_center, dtype=float)
    if relation_kind == "shortestpath":
        C_center = np.nan_to_num(C_center, nan=np.inf, posinf=np.inf)
    else:
        C_center = np.nan_to_num(C_center, nan=0.0)
    np.fill_diagonal(C_center, 0.0)

    n = C_center.shape[0]
    n_types = len(unique_types)
    n_coord_cols = 0
    height_col = -1
    if attribute_mode == "full":
        n_coord_cols = 4  # x, y, z, height
        height_col = 3
    elif attribute_mode == "height_type":
        n_coord_cols = 1  # height only
        height_col = 0
    elif attribute_mode == "euclidean_type":
        n_coord_cols = 3  # x, y, z
        height_col = -1  # no height in euclidean_type
    else:
        raise ValueError(f"Unknown attribute_mode: {attribute_mode}")

    onehot_start = n_coord_cols
    onehot_end = n_coord_cols + n_types

    G = nx.Graph()
    for i in range(n):
        # Recover height
        if height_col >= 0:
            h_norm = float(X_center[i, height_col])
            height = h_min + h_norm * h_range
        else:
            height = h_min  # fallback when no height column

        # Recover x, y, z
        if attribute_mode in ("full", "euclidean_type"):
            x = min_x + X_center[i, 0] * domain_size
            y = min_y + X_center[i, 1] * domain_size
            z = min_z + X_center[i, 2] * domain_size
        else:
            x = y = z = 0.0

        # Type block from FGW center: may be soft (multi-type), not a true one-hot.
        oh = _align_oh_vector(X_center[i, onehot_start:onehot_end], n_types)
        type_idx = int(np.argmax(oh)) if np.any(np.isfinite(oh)) else 0
        node_type = unique_types[type_idx]
        pos = np.maximum(oh, 0.0)
        ssum = float(pos.sum())
        if ssum > 1e-20:
            type_mass = (pos / ssum).tolist()
        else:
            type_mass = [1.0 / max(n_types, 1)] * max(n_types, 1)

        G.add_node(
            i,
            x=x,
            y=y,
            z=z,
            height=height,
            type=node_type,
            type_features=oh.astype(float).tolist(),
            type_mass=type_mass,
        )

    for i in range(n):
        for j in range(i + 1, n):
            w = C_center[i, j]
            if relation_kind == "shortestpath":
                add_edge = np.isfinite(w) and w > 0
                # Use distance for MST; postprocess will set |height diff| for contour tree
                edge_weight = float(w)
            else:
                add_edge = w >= threshold
                hi, hj = G.nodes[i]["height"], G.nodes[j]["height"]
                edge_weight = float(abs(hi - hj))
            if add_edge:
                G.add_edge(i, j, weight=edge_weight)

    return G


def _node_type(G: nx.Graph, n, min_type: int = 0, saddle_type: int = 1, max_type: int = 2) -> int:
    """Return node type as int (min=0, saddle=1, max=2)."""
    t = G.nodes[n].get("type", None)
    if t is None:
        return saddle_type
    try:
        return int(round(float(t)))
    except (TypeError, ValueError):
        return saddle_type


def _shortestpath_reconstruction(
    G: nx.Graph,
    min_type: int = 0,
    saddle_type: int = 1,
    max_type: int = 2,
    verbose: bool = True,
) -> nx.Graph:
    """
    Two-step reconstruction for shortestpath mode:

    1. Remove edges (a,b) where there exists c with W(a,c)+W(c,b) < W(a,b).
       Output nodes and edges after this step.
    2. Customized MST: no min-max edges; each min and max has degree 1.
    """
    G = G.copy()

    # Step 1: Remove edges shortcut by a path
    # to_remove = []
    # for u, v in list(G.edges()):
    #     w_uv = G.edges[u, v]["weight"]
    #     G_temp = G.copy()
    #     G_temp.remove_edge(u, v)
    #     try:
    #         sp_len = nx.shortest_path_length(G_temp, u, v, weight="weight")
    #         if sp_len < w_uv:
    #             to_remove.append((u, v))
    #     except nx.NetworkXNoPath:
    #         pass
    # for u, v in to_remove:
    #     G.remove_edge(u, v)

    # if verbose:
    #     print(
    #         f"After step 1 (remove shortcut edges): nodes = {G.number_of_nodes()}, "
    #         f"edges = {G.number_of_edges()}, "
    #         f"G is connected: {nx.is_connected(G)}"
    #     )

    # Step 2: Customized MST - no min-max edges, extrema have degree 1
    mins = [n for n in G.nodes() if _node_type(G, n, min_type, saddle_type, max_type) == min_type]
    saddles = [n for n in G.nodes() if _node_type(G, n, min_type, saddle_type, max_type) == saddle_type]
    maxs = [n for n in G.nodes() if _node_type(G, n, min_type, saddle_type, max_type) == max_type]

    def allowed(u, v):
        tu = _node_type(G, u, min_type, saddle_type, max_type)
        tv = _node_type(G, v, min_type, saddle_type, max_type)
        return not ((tu == min_type or tu == max_type) and (tv == max_type or tv == min_type))

    # Collect allowed edges with weights, sorted by weight (Kruskal-style)
    edges_with_w = []
    for u, v in G.edges():
        if allowed(u, v):
            edges_with_w.append((G.edges[u, v]["weight"], u, v))
    edges_with_w.sort(key=lambda x: x[0])

    T = nx.Graph()
    for n in G.nodes():
        T.add_node(n, **dict(G.nodes[n]))
    deg = {n: 0 for n in G.nodes()}
    extrema = set(mins) | set(maxs)

    for w, u, v in edges_with_w:
        if T.number_of_edges() >= T.number_of_nodes() - 1:
            break
        # Skip if would give extrema degree > 1
        if u in extrema and deg[u] >= 1:
            continue
        if v in extrema and deg[v] >= 1:
            continue
        if u in mins and v in saddles and T.nodes[u]["height"] > T.nodes[v]["height"]:
            continue
        if u in maxs and v in saddles and T.nodes[u]["height"] < T.nodes[v]["height"]:
            continue
        if u in saddles and v in mins and T.nodes[u]["height"] < T.nodes[v]["height"]:
            continue
        if u in saddles and v in maxs and T.nodes[u]["height"] > T.nodes[v]["height"]:
            continue
        # Skip if creates cycle
        if T.has_node(u) and T.has_node(v) and nx.has_path(T, u, v):
            continue
        T.add_edge(u, v, weight=w)
        deg[u] += 1
        deg[v] += 1
    
    print("Is connected before adding not legit edges:", nx.is_connected(T))
    print("Number of edges before adding not legit edges:", T.number_of_edges())
    print("Number of nodes before adding not legit edges:", T.number_of_nodes())

    # If disconnected (e.g. no saddle-saddle edges), add min-weight edges to connect
    while not nx.is_connected(T) and T.number_of_edges() < T.number_of_nodes() - 1:
        comps = list(nx.connected_components(T))
        best = None
        for w, u, v in edges_with_w:
            cu = next((c for c in comps if u in c), None)
            cv = next((c for c in comps if v in c), None)
            if cu is not None and cv is not None and cu != cv:
                # if u in extrema and deg[u] >= 1:
                #     continue
                # if v in extrema and deg[v] >= 1:
                #     continue
                if best is None or w < best[0]:
                    best = (w, u, v)
        if best is None:
            break
        w, u, v = best
        T.add_edge(u, v, weight=w)
        deg[u] += 1
        deg[v] += 1

    # Set edge weights to |height diff| for contour tree
    # for u, v in T.edges():
    #     h_u = T.nodes[u].get("height", 0)
    #     h_v = T.nodes[v].get("height", 0)
    #     T.edges[u, v]["weight"] = abs(h_u - h_v)

    return T


def _suppress_degree2_saddles(
    G: nx.Graph,
    min_type: int = 0,
    saddle_type: int = 1,
    max_type: int = 2,
) -> nx.Graph:
    """
    Collapse degree-2 saddles only. Do not suppress degree-2 extrema (keep for sanity check).
    For each degree-2 saddle: remove it and connect its two neighbors.
    New edge weight = sum of the two original edge weights.
    """
    G = G.copy()
    while True:
        deg2_saddles = sorted(
            (
                n
                for n in G.nodes()
                if G.degree(n) == 2
                and _node_type(G, n, min_type, saddle_type, max_type) == saddle_type
            ),
            key=lambda n: (_get_height(G, n), _node_lex_key(n)),
        )
        if not deg2_saddles:
            break
        for n in deg2_saddles:
            if n not in G or G.degree(n) != 2:
                continue
            if _node_type(G, n, min_type, saddle_type, max_type) != saddle_type:
                continue
            neighbors = _sorted_neighbors(G, n)
            if len(neighbors) != 2:
                continue
            u, v = neighbors[0], neighbors[1]
            w1 = G.edges[n, u].get("weight", abs(_get_height(G, n) - _get_height(G, u)))
            w2 = G.edges[n, v].get("weight", abs(_get_height(G, n) - _get_height(G, v)))
            G.remove_node(n)
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=w1 + w2)
    return G


def postprocess_pfgw_barycenter(
    G: nx.Graph,
    relation_kind: str = "weighted_adjacency",
    min_type: int = 0,
    saddle_type: int = 1,
    max_type: int = 2,
    verbose: bool = True,
    suppress_degree2_saddles: bool = True,
) -> nx.Graph:
    """
    Postprocess PFGW barycenter graph: compute spanning tree, then suppress degree-2 saddles.

    - weighted_adjacency/adjacency: maximum spanning tree (keep strongest edges)
    - shortestpath: two-step reconstruction (remove shortcut edges, then customized MST
      with no min-max edges and extrema degree 1)
    - suppress_degree2_saddles: if True, collapse degree-2 saddles only (extrema kept for sanity check)

    Parameters
    ----------
    G : nx.Graph
        Graph from reconstruct_graph_from_fgw_barycenter.
    relation_kind : str
        "weighted_adjacency" | "shortestpath".
    min_type, saddle_type, max_type : int
        Type IDs for min, saddle, max.
    verbose : bool
        If True, print node/edge counts after step 1 for shortestpath.
    suppress_degree2_saddles : bool
        If True, collapse degree-2 saddles after MST step. Degree-2 extrema are never suppressed.

    Returns
    -------
    G : nx.Graph
        Graph as spanning tree (with degree-2 saddles suppressed if requested).
    """
    G = G.copy()

    if relation_kind == "shortestpath":
        # G = nx.minimum_spanning_tree(G, weight="weight")
        G = _shortestpath_reconstruction(
            G, min_type=min_type, saddle_type=saddle_type, max_type=max_type, verbose=verbose
        )
    else:
        if not nx.is_tree(G):
            G = nx.maximum_spanning_tree(G, weight="weight")

    if suppress_degree2_saddles:
        G = _suppress_degree2_saddles(G, min_type=min_type, saddle_type=saddle_type, max_type=max_type)

    return G


def validate_contour_tree_structure(
    G: nx.Graph,
    valid_types: Optional[set] = None,
    height_weight_tol_rel: float = 0.1,
    min_type: int = 0,
    max_type: int = 2,
) -> Dict[str, int]:
    """
    Count violations of contour tree validity criteria.

    Criteria:
    (1) Each node type is in {0, 1, 2, 3} (min, saddle, max) or in valid_types.
    (2) Each extrema (type min/max) must be connected to exactly one saddle. Violations:
        - extrema with degree != 1 (connects to multiple nodes),
        - extrema whose neighbor is not a saddle (degree <= 1).
    (3) Edge weight is close to |height(u) - height(v)|.
    (4) Two-degree nodes: count of nodes with degree 2 (contour trees have only leaves and saddles).
    (5) Type-value inconsistency: for leaves, local max must have height > neighbor,
        local min must have height < neighbor.

    Returns
    -------
    dict with keys "type_violations", "extrema_saddle_violations", "weight_violations",
    "two_degree_nodes", "type_value_inconsistencies".
    """
    if valid_types is None:
        valid_types = {0, 1, 2, 3}

    v1 = 0  # type violations
    v2 = 0  # extrema-saddle violations
    v3 = 0  # weight violations
    v4 = 0  # two-degree nodes
    v5 = 0  # type-value inconsistencies

    for n in G.nodes():
        t = G.nodes[n].get("type", None)
        if t is None or (t not in valid_types and t not in {0, 1, 2, 3}):
            v1 += 1
        if G.degree(n) == 2:
            print(f"{n}: two-degree node")
            v4 += 1

    for u in G.nodes():
        t = G.nodes[u].get("type", None)
        try:
            t_int = int(t) if t is not None else None
        except (TypeError, ValueError):
            t_int = None
        if t_int not in (min_type, max_type):
            if G.degree(u) <= 1:
                v2 += 1
                print(f"{u}: saddle with degree = 1")
        # Extrema must have degree 1 and connect to exactly one saddle (degree > 1)
        else:
            if G.degree(u) != 1:
                v2 += 1
                print(f"{u}: extrema with degree != 1")
            else:
                nb = next(G.neighbors(u), None)
                if nb is not None and (G.nodes[nb].get("type", None) in {0, 2}):
                    v2 += 1
                    print(f"{u}: extrema with a non-saddle neighbor")

    for u in G.nodes():
        if G.degree(u) != 1:
            continue
        # Type-value consistency: max must have height > neighbor, min must have height < neighbor
        nb = next(G.neighbors(u), None)
        if nb is not None:
            t = G.nodes[u].get("type", None)
            try:
                t_int = int(round(float(t))) if t is not None else None
            except (TypeError, ValueError):
                t_int = None
            hu = G.nodes[u].get("height", 0)
            hnb = G.nodes[nb].get("height", 0)
            if t_int == max_type and hu <= hnb:
                v5 += 1
            elif t_int == min_type and hu >= hnb:
                v5 += 1

    h_range = 1.0
    heights = [G.nodes[n].get("height", 0) for n in G.nodes()]
    if heights:
        h_range = max(max(heights) - min(heights), 1e-12)

    for u, v in G.edges():
        w = G.edges[u, v].get("weight", 0)
        h_diff = abs(G.nodes[u].get("height", 0) - G.nodes[v].get("height", 0))
        if abs(w - h_diff) > height_weight_tol_rel * h_range:
            print(f"({u}, {v}): weight = {w}, height diff = {h_diff}")
            v3 += 1

    return {
        "type_violations": v1,
        "extrema_saddle_violations": v2,
        "weight_violations": v3,
        "two_degree_nodes": v4,
        "type_value_inconsistencies": v5,
    }


def contour_tree_positions(
    graphs: List[nx.Graph],
    roots: Optional[List] = None,
    idx: Optional[int] = None,
    width: float = 1.0,
    height_scale: float = 1.0,
) -> List[Dict]:
    """
    Compute layout positions for a list of contour trees.

    Parameters
    ----------
    graphs : list of nx.Graph
        Contour trees (each with height, weight).
    roots : list, optional
        Root node per graph. If None, use min-height node.
    idx : int, optional
        If given, return positions for graph[idx] only (as single-element list).
    width, height_scale : float
        Layout parameters.

    Returns
    -------
    list of dict
        Each dict maps node -> (x, y).
    """
    if roots is None:
        roots = [None] * len(graphs)
    result = []
    for g_idx, G in enumerate(graphs):
        if idx is not None and g_idx != idx:
            continue
        root = roots[g_idx] if g_idx < len(roots) else None
        pos, _, _ = contour_tree_layout(G, root=root, width=width, height_scale=height_scale)
        result.append(pos)
    return result
