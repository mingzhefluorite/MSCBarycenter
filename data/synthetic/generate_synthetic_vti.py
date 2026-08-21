#!/usr/bin/env python
"""
Generate synthetic 2D height fields as VTI image data for Morse-Smale experiments.

The base field is a sinusoidal egg-carton on [0, 1]^2: nine local maxima on a 3x3
grid with the eight outer peaks on the domain boundary (corners and edge midpoints)
and a taller central peak (~2x the boundary height). A small offset inside the
periodic factors avoids flat valley plateaus.

Noisy instances add a controllable number of localized Gaussian bumps *or* dips
on top of that regular pattern (see --n-bumps). Each feature is centered on a
random allowed grid pixel, uses a random anisotropic Gaussian shape, and a fair
coin flip chooses bump (+) vs dip (-). Magnitude and footprint stay smaller than
the regular egg-carton / center peaks. Placement can be restricted to the
lower-left quadrant (x <= 0.5 and y <= 0.5) when --restrict-noise-region is
enabled (default on; use --no-restrict-noise-region to disable).

Run with ParaView's pvpython, e.g.:

    "C:\\Program Files\\ParaView 5.13.3\\bin\\pvpython.exe" generate_synthetic_vti.py

Plain CPython with vtk + numpy also works if VTK is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from vtk.util import numpy_support
import vtk


# Grid size: 301 = 1 + 300 places peaks at x, y in {0, 0.5, 1} on grid nodes.
DEFAULT_NX = 301
DEFAULT_NY = 301

# Height range matches the reference surface plot (peaks near -1, valleys near -4).
HEIGHT_MIN = 0.0
HEIGHT_MAX = 1.0

# cos^2(2 pi x) has maxima at x in {0, 0.5, 1} (boundary-aligned 3x3 grid).
# Center Gaussian boost: amplitude 1.0 gives ~2x boundary height after normalization.
BUMP_AMPLITUDE = 1.0
BUMP_SIGMA = 0.12

# Keeps valley level sets off exact zero (no flat plateaus along cos^2 nodal lines).
SURFACE_OFFSET = 0.05

# Inset of outer peaks from the domain edge, in [0, 1) coordinates.
# 0 = peaks on boundary (corners at 0/1); 0.05 moves them to x,y in {0.05, 0.95}.
PEAK_INSET = 0.02

# Localized noisy bumps/dips on top of the regular base field.
# Regular center bump uses BUMP_SIGMA=0.12 and spans ~full [HEIGHT_MIN, HEIGHT_MAX]
# after normalization; noisy features stay strictly smaller in footprint and |height|.
DEFAULT_N_NOISE_BUMPS = 1
NOISE_BUMP_HEIGHT_MIN = 0.08  # |peak| magnitude (normalized units)
NOISE_BUMP_HEIGHT_MAX = 0.18
NOISE_BUMP_SIGMA_MIN = 0.025  # domain units; must stay < BUMP_SIGMA
NOISE_BUMP_SIGMA_MAX = 0.055
NOISE_BUMP_MARGIN = 0.05  # keep feature centers away from domain edge
NOISE_BUMP_MIN_SEPARATION = 0.06
NOISE_BUMP_DIP_PROB = 0.5  # P(dip); 1 - this is P(bump)
# If True, noisy features are placed only where x <= MAX_X/2 and y <= MAX_Y/2.
NOISE_REGION_RESTRICT = True


def _periodic_coord(
    coord: np.ndarray,
    peak_inset: float,
) -> np.ndarray:
    """Map domain coord in [0, 1] so cos^2 peaks lie at inset, 0.5, 1-inset."""
    if peak_inset <= 0.0:
        return coord
    if peak_inset >= 0.5:
        raise ValueError(f"peak_inset must be < 0.5, got {peak_inset}")
    return (coord - peak_inset) / (1.0 - 2.0 * peak_inset)


def boundary_aligned_grid_peaks(
    x: np.ndarray,
    y: np.ndarray,
    surface_offset: float = SURFACE_OFFSET,
    peak_inset: float = PEAK_INSET,
) -> np.ndarray:
    """Nine peaks on a 3x3 grid; outer ring inset from the boundary when peak_inset > 0.

    Uses cos^2(2 pi x) cos^2(2 pi y) on warped coordinates. With peak_inset=0, peaks
    sit at x, y in {0, 0.5, 1}. With peak_inset=delta, outer peaks move to
    {delta, 1-delta} while the center stays at 0.5.
    """
    tx = _periodic_coord(x, peak_inset)
    ty = _periodic_coord(y, peak_inset)
    cx = np.cos(2.0 * np.pi * tx) ** 2
    cy = np.cos(2.0 * np.pi * ty) ** 2
    return (surface_offset + cx) * (surface_offset + cy)


def center_bump(x: np.ndarray, y: np.ndarray, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2.0 * sigma**2))


def normalize_height(raw: np.ndarray) -> np.ndarray:
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    if raw_max <= raw_min:
        return np.full_like(raw, HEIGHT_MAX, dtype=np.float32)
    unit = (raw - raw_min) / (raw_max - raw_min)
    return (HEIGHT_MIN + (HEIGHT_MAX - HEIGHT_MIN) * unit).astype(np.float32)


def base_height_field(
    x: np.ndarray,
    y: np.ndarray,
    bump_amplitude: float = BUMP_AMPLITUDE,
    bump_sigma: float = BUMP_SIGMA,
    surface_offset: float = SURFACE_OFFSET,
    peak_inset: float = PEAK_INSET,
) -> np.ndarray:
    periodic = boundary_aligned_grid_peaks(
        x, y, surface_offset=surface_offset, peak_inset=peak_inset
    )
    bump = center_bump(x, y, bump_amplitude, bump_sigma)
    return normalize_height(periodic + bump)


def noise_region_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where noise is allowed: lower-left quadrant x <= MAX_X/2, y <= MAX_Y/2."""
    max_x = float(np.max(x))
    max_y = float(np.max(y))
    return ((x <= max_x / 2.0) & (y <= max_y / 2.0))


def _allowed_pixel_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    restrict_noise_region: bool,
    margin: float,
) -> np.ndarray:
    """Boolean mask of grid pixels eligible as noisy-bump centers."""
    max_x = float(np.max(x))
    max_y = float(np.max(y))
    min_x = float(np.min(x))
    min_y = float(np.min(y))
    mask = (
        (x >= min_x + margin)
        & (x <= max_x - margin)
        & (y >= min_y + margin)
        & (y <= max_y - margin)
    )
    if restrict_noise_region:
        mask &= noise_region_mask(x, y)
    return mask


def _anisotropic_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    amplitude: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
) -> np.ndarray:
    """Anisotropic Gaussian feature, rotated by ``theta`` radians.

    ``amplitude`` may be positive (bump) or negative (dip).
    """
    dx = x - cx
    dy = y - cy
    ct = np.cos(theta)
    st = np.sin(theta)
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    sx2 = max(sigma_x, 1e-6) ** 2
    sy2 = max(sigma_y, 1e-6) ** 2
    return amplitude * np.exp(-0.5 * (xr * xr / sx2 + yr * yr / sy2))


def noisy_bumps_field(
    x: np.ndarray,
    y: np.ndarray,
    n_bumps: int,
    rng: np.random.Generator,
    *,
    height_min: float = NOISE_BUMP_HEIGHT_MIN,
    height_max: float = NOISE_BUMP_HEIGHT_MAX,
    sigma_min: float = NOISE_BUMP_SIGMA_MIN,
    sigma_max: float = NOISE_BUMP_SIGMA_MAX,
    margin: float = NOISE_BUMP_MARGIN,
    min_separation: float = NOISE_BUMP_MIN_SEPARATION,
    dip_prob: float = NOISE_BUMP_DIP_PROB,
    restrict_noise_region: bool = NOISE_REGION_RESTRICT,
    max_tries_per_bump: int = 200,
) -> tuple[np.ndarray, list[dict]]:
    """
    Add ``n_bumps`` Gaussian bumps/dips by sampling random pixels.

    Each feature:
      - center = random allowed grid pixel (optional quadrant restriction)
      - shape = random anisotropic Gaussian (``sigma_x``, ``sigma_y``, ``theta``)
      - magnitude = uniform in ``[height_min, height_max]``
      - sign = coin flip: bump (+) with probability ``1 - dip_prob``, else dip (-)
      - footprint and |height| stay below the regular peaks
        (``BUMP_SIGMA``, full normalized height span)
    """
    if n_bumps < 0:
        raise ValueError(f"n_bumps must be >= 0, got {n_bumps}")
    if height_min <= 0.0 or height_max < height_min:
        raise ValueError(
            f"Need 0 < height_min <= height_max, got [{height_min}, {height_max}]"
        )
    if not (0.0 <= dip_prob <= 1.0):
        raise ValueError(f"dip_prob must be in [0, 1], got {dip_prob}")
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError(
            f"Need 0 < sigma_min <= sigma_max, got [{sigma_min}, {sigma_max}]"
        )
    if sigma_max >= BUMP_SIGMA:
        raise ValueError(
            f"noise bump sigma_max ({sigma_max}) must be < regular BUMP_SIGMA ({BUMP_SIGMA})"
        )

    field = np.zeros_like(x, dtype=np.float64)
    if n_bumps == 0:
        return field.astype(np.float32), []

    allowed = _allowed_pixel_mask(
        x, y, restrict_noise_region=restrict_noise_region, margin=margin
    )
    flat = np.flatnonzero(allowed.ravel(order="C"))
    if flat.size == 0:
        raise ValueError(
            "No eligible pixels for noisy features; reduce margin or disable "
            "--restrict-noise-region."
        )

    min_sep2 = float(min_separation) ** 2
    centers: list[tuple[float, float]] = []
    records: list[dict] = []

    for _ in range(n_bumps):
        chosen = None
        for _try in range(max_tries_per_bump):
            idx = int(rng.choice(flat))
            i, j = np.unravel_index(idx, x.shape)
            cx = float(x[i, j])
            cy = float(y[i, j])
            if all((cx - px) ** 2 + (cy - py) ** 2 >= min_sep2 for px, py in centers):
                chosen = (i, j, cx, cy)
                break
        if chosen is None:
            idx = int(rng.choice(flat))
            i, j = np.unravel_index(idx, x.shape)
            chosen = (i, j, float(x[i, j]), float(y[i, j]))

        i, j, cx, cy = chosen
        centers.append((cx, cy))
        magnitude = float(rng.uniform(height_min, height_max))
        is_dip = bool(rng.random() < dip_prob)
        kind = "dip" if is_dip else "bump"
        amplitude = -magnitude if is_dip else magnitude
        sigma_x = float(rng.uniform(sigma_min, sigma_max))
        sigma_y = float(rng.uniform(sigma_min, sigma_max))
        theta = float(rng.uniform(0.0, np.pi))
        field += _anisotropic_gaussian(
            x, y, cx, cy, amplitude, sigma_x, sigma_y, theta
        )
        records.append(
            {
                "i": int(i),
                "j": int(j),
                "cx": cx,
                "cy": cy,
                "kind": kind,
                "amplitude": amplitude,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "theta": theta,
            }
        )

    return field.astype(np.float32), records


def build_coordinate_arrays(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    return np.meshgrid(xs, ys, indexing="ij")


def write_vti(
    filepath: Path,
    height: np.ndarray,
    scalar_name: str = "height",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] | None = None,
) -> None:
    if height.ndim != 2:
        raise ValueError(f"height must be 2D, got shape {height.shape}")

    nx, ny = height.shape
    if spacing is None:
        sx = 1.0 / max(nx - 1, 1)
        sy = 1.0 / max(ny - 1, 1)
        spacing = (sx, sy, 1.0)

    vtk_array = numpy_support.numpy_to_vtk(
        height.ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_array.SetName(scalar_name)

    image = vtk.vtkImageData()
    image.SetOrigin(*origin)
    image.SetSpacing(*spacing)
    image.SetDimensions(nx, ny, 1)
    image.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(filepath))
    writer.SetInputData(image)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {filepath}")


def generate_instances(
    output_dir: Path,
    nx: int,
    ny: int,
    n_noisy: int,
    n_bumps: int,
    noise_bump_height_min: float,
    noise_bump_height_max: float,
    noise_bump_sigma_min: float,
    noise_bump_sigma_max: float,
    noise_bump_margin: float,
    noise_bump_min_separation: float,
    noise_bump_dip_prob: float,
    peak_inset: float,
    seed: int,
    scalar_name: str,
    restrict_noise_region: bool = NOISE_REGION_RESTRICT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = build_coordinate_arrays(nx, ny)
    base = base_height_field(x, y, peak_inset=peak_inset)

    written: list[Path] = []
    base_path = output_dir / "synthetic_base.vti"
    write_vti(base_path, base, scalar_name=scalar_name)
    written.append(base_path)

    rng = np.random.default_rng(seed)
    for idx in range(n_noisy):
        noise, records = noisy_bumps_field(
            x,
            y,
            n_bumps,
            rng,
            height_min=noise_bump_height_min,
            height_max=noise_bump_height_max,
            sigma_min=noise_bump_sigma_min,
            sigma_max=noise_bump_sigma_max,
            margin=noise_bump_margin,
            min_separation=noise_bump_min_separation,
            dip_prob=noise_bump_dip_prob,
            restrict_noise_region=restrict_noise_region,
        )
        noisy = (base + noise).astype(np.float32)
        out_path = output_dir / f"synthetic_noise_{idx + 1:02d}.vti"
        write_vti(out_path, noisy, scalar_name=scalar_name)
        written.append(out_path)

        centers_path = output_dir / f"synthetic_noise_{idx + 1:02d}_bumps.txt"
        with centers_path.open("w", encoding="utf-8") as f:
            f.write("# index  i  j  cx  cy  kind  amplitude  sigma_x  sigma_y  theta\n")
            for k, rec in enumerate(records):
                f.write(
                    f"{k}  {rec['i']}  {rec['j']}  "
                    f"{rec['cx']:.8f}  {rec['cy']:.8f}  {rec['kind']}  "
                    f"{rec['amplitude']:.8f}  {rec['sigma_x']:.8f}  "
                    f"{rec['sigma_y']:.8f}  {rec['theta']:.8f}\n"
                )
        n_bump = sum(1 for r in records if r["kind"] == "bump")
        n_dip = len(records) - n_bump
        print(
            f"{out_path.name}: {len(records)} feature(s) "
            f"({n_bump} bump, {n_dip} dip; centers -> {centers_path.name})"
        )

    return written


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Write synthetic height-field VTI files (base + noisy variants)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Directory for .vti output (default: script directory)",
    )
    parser.add_argument("--nx", type=int, default=DEFAULT_NX, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=DEFAULT_NY, help="Grid points in y")
    parser.add_argument(
        "--n-noisy",
        type=int,
        default=4,
        help="Number of noisy instances to generate",
    )
    parser.add_argument(
        "--n-bumps",
        type=int,
        default=DEFAULT_N_NOISE_BUMPS,
        help=(
            "Number of noisy Gaussian bumps/dips per noisy instance "
            f"(default: {DEFAULT_N_NOISE_BUMPS})"
        ),
    )
    parser.add_argument(
        "--noise-bump-height-min",
        type=float,
        default=NOISE_BUMP_HEIGHT_MIN,
        help="Minimum |peak| magnitude of each noisy bump/dip (normalized units)",
    )
    parser.add_argument(
        "--noise-bump-height-max",
        type=float,
        default=NOISE_BUMP_HEIGHT_MAX,
        help="Maximum |peak| magnitude of each noisy bump/dip",
    )
    parser.add_argument(
        "--noise-bump-dip-prob",
        type=float,
        default=NOISE_BUMP_DIP_PROB,
        help="Probability that a feature is a dip rather than a bump (default: 0.5)",
    )
    parser.add_argument(
        "--noise-bump-sigma-min",
        type=float,
        default=NOISE_BUMP_SIGMA_MIN,
        help=f"Minimum spatial sigma (domain units; regular bump sigma={BUMP_SIGMA})",
    )
    parser.add_argument(
        "--noise-bump-sigma-max",
        type=float,
        default=NOISE_BUMP_SIGMA_MAX,
        help=f"Maximum spatial sigma (must be < regular BUMP_SIGMA={BUMP_SIGMA})",
    )
    parser.add_argument(
        "--noise-bump-margin",
        type=float,
        default=NOISE_BUMP_MARGIN,
        help="Keep noisy feature centers at least this far from the domain edge",
    )
    parser.add_argument(
        "--noise-bump-min-separation",
        type=float,
        default=NOISE_BUMP_MIN_SEPARATION,
        help="Minimum distance between noisy feature centers",
    )
    parser.add_argument(
        "--peak-inset",
        type=float,
        default=PEAK_INSET,
        help=(
            "Distance of outer peaks from domain edge in [0, 1] coords. "
            "0 = on boundary; try 0.03--0.08 for slightly interior peaks."
        ),
    )
    parser.add_argument(
        "--restrict-noise-region",
        action=argparse.BooleanOptionalAction,
        default=NOISE_REGION_RESTRICT,
        help=(
            "Place noisy features only in x <= MAX_X/2 and y <= MAX_Y/2 "
            "(default: on; use --no-restrict-noise-region to disable)"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed for noisy runs")
    parser.add_argument(
        "--scalar-name",
        default="height",
        help="VTK point-data array name (default: height)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bumps < 0:
        raise SystemExit(f"--n-bumps must be >= 0, got {args.n_bumps}")
    if args.noise_bump_height_min <= 0.0:
        raise SystemExit("--noise-bump-height-min must be > 0 (feature magnitude)")
    if not (0.0 <= args.noise_bump_dip_prob <= 1.0):
        raise SystemExit("--noise-bump-dip-prob must be in [0, 1]")
    if args.noise_bump_sigma_max >= BUMP_SIGMA:
        raise SystemExit(
            f"--noise-bump-sigma-max must be < regular BUMP_SIGMA={BUMP_SIGMA}"
        )

    paths = generate_instances(
        output_dir=args.output_dir,
        nx=args.nx,
        ny=args.ny,
        n_noisy=args.n_noisy,
        n_bumps=args.n_bumps,
        noise_bump_height_min=args.noise_bump_height_min,
        noise_bump_height_max=args.noise_bump_height_max,
        noise_bump_sigma_min=args.noise_bump_sigma_min,
        noise_bump_sigma_max=args.noise_bump_sigma_max,
        noise_bump_margin=args.noise_bump_margin,
        noise_bump_min_separation=args.noise_bump_min_separation,
        noise_bump_dip_prob=args.noise_bump_dip_prob,
        peak_inset=args.peak_inset,
        seed=args.seed,
        scalar_name=args.scalar_name,
        restrict_noise_region=args.restrict_noise_region,
    )

    print(f"Wrote {len(paths)} VTI file(s) to {args.output_dir.resolve()}:")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
