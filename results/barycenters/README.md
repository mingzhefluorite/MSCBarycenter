# MSC FGW barycenter save bundle

This document describes the on-disk **barycenter bundle** written by
`lib/barycenter_io.save_barycenter_bundle` (typically under
`results/barycenters/<run_name>/`, e.g. `latest/`). The bundle is designed to be
fully reconstitutable for evaluation, visualization, sparsification, and color
transfer in notebooks such as `msc_barycenter_evaluation.ipynb`.

---

## Context: Morse(-Smale) complexes and FGW barycenters

A **Morse complex** (and the related **Morse–Smale complex**, MSC) summarizes the
gradient topology of a scalar field on a domain. Critical points
(minima / saddles / maxima) and the ascending / descending manifolds that
connect them form a graph whose nodes carry:

| Attribute | Meaning |
|-----------|---------|
| `(x, y, z)` | Spatial position in the mesh / volume |
| `scalar_value` / `height` | Field value at the node |
| `type` | Morse type: typically `0` min, `1` saddle, `2` max, `-1` regular (sampled along integral lines) |

In this project, each MSC snapshot is loaded from
`treeNodes_*.txt` / `treeEdges_*.txt` (via `lib/graph_reader.py`), optionally
subsampled along Morse paths (`prepare_graph_for_barycenter`), and encoded as:

1. **Relation matrix** `C` — pairwise structure on nodes (usually normalized
   **shortest-path** distances on the MSC graph; alternatives include adjacency /
   Laplacian / heat-kernel style relations).
2. **Feature matrix** `Y` — per-node attributes used by the *fused* part of FGW:
   normalized `(x,y,z)`, normalized scalar, and a scaled one-hot Morse-type block.
3. **Mass vector** `p` — node probability weights (uniform when
   `balanced_distribution=True`).

A **(partial) Fused Gromov–Wasserstein (PFGW) barycenter** then finds a
consensus structure `(C_center, X_center, p)` that is close, in a
Gromov–Wasserstein sense, to a family of input MSCs `{ (C_s, Y_s, p_s) }`, with
barycentric weights `lambdas`. Partial (unbalanced) transport allows mass to be
discarded under a penalty (`Lambda_list` / `lambda_penalty`), which is useful
when noise creates spurious critical points.

The saved bundle stores the **pre-thresholding** barycenter (dense relation and
complete graph), optional **MST reconstruction**, the **input family** used in
the PFGW solve, final **couplings** `T_s`, and enough **domain / solver metadata**
to decode coordinates, scalars, and Morse types without re-running the notebook.

**Pipeline (producer):** `msc_barycenter_shortestpath.ipynb`  
**Pipeline (consumer):** `msc_barycenter_evaluation.ipynb`  
**I/O API:** `lib/barycenter_io.py`

---

## Directory layout

```
results/barycenters/<run_name>/
├── meta.json                 # domain, relation, feature, PFGW params, flags
├── C_center.npy              # dense barycenter relation (pre-thresholding)
├── X_center.npy              # barycenter node features
├── p.npy                     # barycenter mass
├── lambdas.npy               # barycentric weights over inputs
├── Ts.npz                    # final couplings T_0 .. T_{S-1}  (optional)
├── Cs.npz / Ys.npz / ps.npz  # input relations / features / masses
├── nodes_list.json           # per-input node order used for matrices (optional)
├── G_complete.gpickle        # complete (unthresholded) barycenter graph
├── G_mst.gpickle             # MST (+ planar non-crossing) reconstruction (optional)
├── inputs/
│   └── G_{i}.gpickle         # barycenter-input MSC graphs (optional)
└── seed/                     # only when exclude_seed=True (out-of-set seed)
    ├── C.npy / Y.npy / p.npy
    ├── G.gpickle             # optional
    └── nodes.json            # optional
```

Default path helper: `default_barycenter_outdir(root="results/barycenters", run_name="latest")`.

---

## File reference

### `meta.json`

JSON metadata required to interpret matrices and rebuild graphs. Fields fall into
several groups.

#### Domain / scalar normalization

| Key | Type | Role |
|-----|------|------|
| `min_x`, `max_x`, `min_y`, `max_y`, `min_z`, `max_z` | float | Bounding box of the spatial domain |
| `domain_size` | float | Scale used to normalize coordinates in features (`coord / domain_size`) |
| `min_scalar`, `max_scalar`, `scalar_range` | float | Scalar field range; features store `(s - min_scalar) / scalar_range` |

Decoding a barycenter node from `X_center[i]`:

```text
x = min_x + X[i,0] * domain_size
y = min_y + X[i,1] * domain_size
z = min_z + X[i,2] * domain_size
scalar = min_scalar + X[i,3] * scalar_range   # when scalar columns are present
```

#### Morse-type encoding

| Key | Type | Role |
|-----|------|------|
| `MSC_TYPE_ORDER` | list[int] | Column order of the type one-hot block (e.g. `[0, 1, 2, -1]`) |
| `type_scale` | float | Amplitude of each one-hot entry (often `1/√2` so type mismatch cost is 1 under squared Euclidean) |

Feature layout for each node (columns of `X_center` / `Y_s`):

| Columns | Content |
|---------|---------|
| `0:3` | Normalized `(x, y, z)` |
| `3` | Normalized scalar |
| `4:4+len(MSC_TYPE_ORDER)` | Scaled one-hot Morse type (`type_scale` on the matching type, else 0) |

Example with `MSC_TYPE_ORDER = [0, 1, 2, -1]` → feature dimension **8**.

#### Relation matrix construction

| Key | Type | Role |
|-----|------|------|
| `relation_kind` | str | How `C` was built: typically `"shortestpath"`; also `"adjacency"`, `"laplacian"`, etc. |
| `relation_rescale` | float | Scaling applied before FGW (`C_scaled = C_raw * rescale` for shortest-path) |
| `relation_offset` | float | Additive offset used in some relation encodings |
| `sp_weighted` | bool | Whether shortest paths use edge lengths (vs hop count) |
| `laplacian_base` | str | Base for Laplacian relations (e.g. `"adjacency"`) |
| `laplacian_normalization` | str | Laplacian normalization mode |
| `tau` | float | Heat-kernel / diffusion time scale when applicable |
| `sp_affinity` | str | Affinity transform of SP distances (e.g. `"exp"`) |
| `sp_sigma` | float \| null | Bandwidth for affinity; `null` if unused |

#### PFGW solver settings

| Key | Type | Role |
|-----|------|------|
| `alpha_fgw` | float | FGW trade-off between feature (Wasserstein) and structure (Gromov) terms |
| `lambda_penalty` | float \| null | Partial-transport discard penalty (scalar form) |
| `Lambda_list` | list[float] \| null | Per-input discard penalties |
| `gw_restarts` | int | Number of random restarts for the GW/FGW solve |
| `gw_seed_start` | int | RNG seed offset for restarts |
| `max_iter` | int | Max solver iterations |
| `balanced_distribution` | bool | Uniform node masses when true |
| `N_center` | int | Requested / resulting barycenter support size |

#### Data provenance and seed policy

| Key | Type | Role |
|-----|------|------|
| `MSC_DIR` | str | Directory of MSC `treeNodes` / `treeEdges` files used for this run |
| `mesh_ids` | list | Mesh / snapshot ids for **barycenter inputs only** (aligned with `Cs`) |
| `exclude_seed` | bool | If true, the initialization seed was **held out** of the PFGW input set |
| `seed_graph_id` | int \| null | Original index of the seed in the full loaded sequence |
| `seed_mesh_id` | any \| null | Mesh id of the seed (when held out, also mirrored under `seed/`) |

#### Bundle completeness flags (written by the saver)

| Key | Type | Role |
|-----|------|------|
| `n_center` | int | `C_center.shape[0]` |
| `n_inputs` | int | Number of PFGW inputs (`len(Cs)`) |
| `has_Ts` | bool | Whether `Ts.npz` was written |
| `has_G_complete` | bool | Whether `G_complete.gpickle` was written |
| `has_G_mst` | bool | Whether `G_mst.gpickle` was written |
| `has_input_graphs` | bool | Whether `inputs/G_*.gpickle` were written |
| `has_seed_bundle` | bool | Whether `seed/` was written (`exclude_seed`) |

---

### `C_center.npy`

- **Shape:** `(n_center, n_center)`
- **Dtype:** float
- Dense barycenter **relation** matrix in the same encoding as `relation_kind`
  (after `relation_rescale` / `relation_offset`), **before** edge thresholding.
- For `shortestpath`, entries are scaled path distances; non-edges / unreachable
  pairs may appear as large / infinite values that are sanitized when building graphs.

### `X_center.npy`

- **Shape:** `(n_center, d)` — typically `d = 3 + 1 + len(MSC_TYPE_ORDER)`
- Barycenter **node features** (see layout above). Used both as the FGW feature
  marginal and to decode spatial / scalar / type attributes onto reconstructed graphs.

### `p.npy`

- **Shape:** `(n_center,)`
- Barycenter node masses (probability vector; sums to 1 under balanced setups).

### `lambdas.npy`

- **Shape:** `(n_inputs,)`
- Barycentric weights over the PFGW input family. Often uniform (`1/n_inputs`).

### `Cs.npz` / `Ys.npz` / `ps.npz`

Compressed NumPy archives of the **barycenter input set** (seed excluded when
`exclude_seed=True`):

| Archive | Keys | Content |
|---------|------|---------|
| `Cs.npz` | `C_0` … `C_{S-1}` | Relation matrices |
| `Ys.npz` | `Y_0` … `Y_{S-1}` | Feature matrices (same column layout as `X_center`) |
| `ps.npz` | `p_0` … `p_{S-1}` | Mass vectors |

Index `i` is aligned across `Cs` / `Ys` / `ps` / `lambdas` / `Ts` / `mesh_ids` /
`nodes_list` / `inputs/G_i.gpickle`.

### `Ts.npz` (optional)

- Keys: `T_0` … `T_{S-1}`
- Final PFGW couplings between the barycenter and each input.
- **Shape of `T_i`:** `(n_center, n_i)` where `n_i = Cs[i].shape[0]`.
- Used for color transfer and soft correspondence analysis.

### `nodes_list.json` (optional)

- JSON list of length `n_inputs`; each entry is the ordered list of MSC node ids
  corresponding to rows/columns of that input’s `C` / `Y` / `p`.
- Needed to map matrix indices back onto `inputs/G_i` (or original file ids).

### `G_complete.gpickle`

- NetworkX graph pickle of the **complete (unthresholded)** barycenter graph:
  one node per barycenter support point, with an edge for every finite positive
  off-diagonal entry of `C_center` (weights unscaled via
  `relation_rescale` / `relation_offset` when applicable).
- Typical node attributes: `x`, `y`, `z`, `height`, `scalar_value`, `type`.
- Typical edge attribute: `weight`.
- Downstream evaluation / sparsification starts from this dense graph (or from
  `C_center` itself) rather than from the MST alone.

### `G_mst.gpickle` (optional)

- Kruskal-style **MST reconstruction** from `C_center` with planar
  non-crossing constraints in `(x, y)` (see
  `reconstruct_msc_barycenter_mst_graph` in `lib/msc_barycenter_reconstruction.py`).
- Same node attribute decoding as `G_complete`; far fewer edges (tree / forest).

### `inputs/G_{i}.gpickle` (optional)

- The processed MSC graphs that entered the PFGW solve (after path sampling /
  component selection), index-aligned with `Cs[i]`.

### `seed/` (only if `exclude_seed=True`)

When the seed network initializes the barycenter but is **not** included as a
PFGW input (out-of-set baseline for evaluation):

| File | Content |
|------|---------|
| `C.npy`, `Y.npy`, `p.npy` | Seed relation / features / mass |
| `G.gpickle` | Optional seed MSC graph |
| `nodes.json` | Optional seed node-id order |
| (`meta.seed_mesh_id`, `meta.seed_graph_id`) | Provenance of the held-out seed |

If `exclude_seed=False`, there is no `seed/` directory; the seed is one of the
inputs and is indexed by `meta.seed_graph_id` into `Cs` / `Ys` / `ps`.

---

## Loading the bundle

```python
from lib.barycenter_io import (
    load_barycenter_bundle,
    resolve_eval_seed,
    default_barycenter_outdir,
)

bundle = load_barycenter_bundle(default_barycenter_outdir(run_name="latest"))
# bundle keys include:
#   meta, C_center, X_center, p, lambdas, Cs, Ys, ps, Ts,
#   nodes_list, G_complete, G_mst, Gs, mesh_ids, seed,
#   exclude_seed, seed_graph_id, outdir

seed_info = resolve_eval_seed(bundle)
# C_seed, Y_seed, p_seed, seed_label, exclude_seed, ...
```

`load_barycenter_bundle` also contains a **legacy repair** path: older saves
sometimes wrote seed-inclusive `Cs` while `Ts` / `lambdas` matched only the
barycenter inputs; the loader splits the seed out into `bundle["seed"]` when it
detects that inconsistency.

---

## Example: `latest/` snapshot

Values below reflect a typical saved run (synthetic descending Morse complex
family). Exact numbers change with data and solver settings; consult
`latest/meta.json` for the authoritative record of a given save.

| Item | Example |
|------|---------|
| Data | `MSC_DIR = data/synthetic/DESC-MC_0.0` |
| Relation | `relation_kind = "shortestpath"`, `sp_weighted = true` |
| Features | `X_center.shape = (407, 8)`; `MSC_TYPE_ORDER = [0,1,2,-1]`; `type_scale ≈ 0.707` |
| Inputs | `n_inputs = 9` (`mesh_ids` 1…9); seed mesh `0` held out |
| Couplings | `Ts` present; each `T_i` is `(407, n_i)` |
| Graphs | `G_complete` dense; `G_mst` ≈ tree (`n−1` edges); `inputs/G_0`…`G_8` + `seed/` |
| FGW | `alpha_fgw = 0.5`, `Lambda_list` all `0.02`, `gw_restarts = 10`, `max_iter = 200` |

---

## What is intentionally *not* in the save

- Raw `treeNodes_*.txt` / `treeEdges_*.txt` source files (referenced via `MSC_DIR`
  / `mesh_ids` instead).
- Intermediate solver logs beyond the final couplings in `Ts.npz`.
- Post-threshold / sparsified evaluation graphs (those are produced later in
  `msc_barycenter_evaluation.ipynb` from `G_complete` / `C_center`).
- Color-transfer images or plot artifacts.

---

## Related code

| Module | Role |
|--------|------|
| `lib/barycenter_io.py` | Save / load / seed resolution |
| `lib/graph_reader.py` | MSC file I/O and path sampling |
| `lib/msc_barycenter_reconstruction.py` | MST reconstruction from `C_center`, `X_center` |
| `lib/msc_barycenter_threshold.py` | Complete / thresholded barycenter graphs |
| `lib/msc_barycenter_eval.py` | PFGW cost and evaluation helpers |
| `lib/msc_color_transfer.py` | Color transfer via couplings |
| `lib/graph_sparsify.py` | Downstream sparsification of saved barycenters |
