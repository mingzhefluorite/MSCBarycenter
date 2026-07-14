#!/usr/bin/env python
"""
Generate synthetic 2D height fields as VTI image data for Morse-Smale experiments.

The base field is a sinusoidal egg-carton on [0, 1]^2: nine local maxima on a 3x3
grid with the eight outer peaks on the domain boundary (corners and edge midpoints)
and a taller central peak (~2x the boundary height). A small offset inside the
periodic factors avoids flat valley plateaus.

Four additional instances add smoothed random Gaussian noise on top of the base field.
Noise amplitude is weighted by (1 - height): small near the tall center peak (topology
preserved) and larger on lower/saddle-rich periphery (irregular MSC boundaries).
Noise is further restricted to the lower-left quadrant (x <= 0.5 and y <= 0.5)
when --restrict-noise-region is enabled (default on; use --no-restrict-noise-region
to disable).

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

# Smoothed random noise (std in height units). Light blur avoids single-pixel speckle.
NOISE_STD = 0.03
NOISE_SMOOTH_SIGMA_CELLS = 4.0
# Weight ~ (1 - height)^power: 1 = linear, >1 suppresses noise on mid-height ridges.
NOISE_HEIGHT_POWER = 5.0
# If True, noise is applied only where x <= MAX_X/2 and y <= MAX_Y/2.
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


def _gaussian_kernel_1d(sigma_cells: float) -> np.ndarray:
    radius = max(1, int(3.0 * sigma_cells))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coords / sigma_cells) ** 2)
    kernel /= kernel.sum()
    return kernel


def _blur_separable(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    kernel = _gaussian_kernel_1d(sigma_cells)

    def blur_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        out = np.empty_like(arr)
        if axis == 0:
            for i in range(arr.shape[0]):
                out[i, :] = np.convolve(arr[i, :], kernel, mode="same")
        else:
            for j in range(arr.shape[1]):
                out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
        return out

    return blur_axis(blur_axis(field, axis=0), axis=1)


def smoothed_gaussian_noise(
    shape: tuple[int, int],
    std: float,
    smooth_sigma_cells: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Irregular random field: i.i.d. Gaussian draws with light separable smoothing."""
    white = rng.normal(0.0, 1.0, shape).astype(np.float64)
    if smooth_sigma_cells <= 0.0:
        return (white * std).astype(np.float32)
    smooth = _blur_separable(white, smooth_sigma_cells)
    smooth *= std / (float(smooth.std()) + 1e-12)
    return smooth.astype(np.float32)


def noise_sensitivity_weight(
    base: np.ndarray,
    height_power: float = NOISE_HEIGHT_POWER,
) -> np.ndarray:
    """Higher weight where the base field is lower (periphery / saddles vs center peak)."""
    h_min = float(base.min())
    h_max = float(base.max())
    h_norm = (base.astype(np.float64) - h_min) / (h_max - h_min + 1e-12)
    weight = np.clip(1.0 - h_norm, 0.0, 1.0)
    if height_power != 1.0:
        weight = weight**height_power
    return weight.astype(np.float32)


def noise_region_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where noise is allowed: lower-left quadrant x <= MAX_X/2, y <= MAX_Y/2."""
    max_x = float(np.max(x))
    max_y = float(np.max(y))
    return ((x <= max_x / 2.0) & (y <= max_y / 2.0)).astype(np.float32)


def height_weighted_noise(
    base: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = NOISE_STD,
    smooth_sigma_cells: float = NOISE_SMOOTH_SIGMA_CELLS,
    height_power: float = NOISE_HEIGHT_POWER,
    restrict_noise_region: bool = NOISE_REGION_RESTRICT,
) -> np.ndarray:
    """Smoothed Gaussian noise, amplitude modulated by base height and optional region."""
    raw = smoothed_gaussian_noise(
        base.shape, noise_std, smooth_sigma_cells, rng
    )
    weight = noise_sensitivity_weight(base, height_power=height_power)
    if restrict_noise_region:
        weight = weight * noise_region_mask(x, y)
    return (raw * weight).astype(np.float32)


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
    noise_std: float,
    noise_smooth_sigma_cells: float,
    noise_height_power: float,
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
        noise = height_weighted_noise(
            base,
            x,
            y,
            rng,
            noise_std=noise_std,
            smooth_sigma_cells=noise_smooth_sigma_cells,
            height_power=noise_height_power,
            restrict_noise_region=restrict_noise_region,
        )
        noisy = (base + noise).astype(np.float32)
        out_path = output_dir / f"synthetic_noise_{idx + 1:02d}.vti"
        write_vti(out_path, noisy, scalar_name=scalar_name)
        written.append(out_path)

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
        "--noise-std",
        type=float,
        default=NOISE_STD,
        help="Overall scale of smoothed Gaussian noise",
    )
    parser.add_argument(
        "--noise-smooth-sigma-cells",
        type=float,
        default=NOISE_SMOOTH_SIGMA_CELLS,
        help=(
            "Gaussian blur sigma for noise in grid cells (0 = white noise; "
            "3--6 = mildly smooth irregular bumps)"
        ),
    )
    parser.add_argument(
        "--noise-height-power",
        type=float,
        default=NOISE_HEIGHT_POWER,
        help=(
            "Weight noise by (1 - height)^power so the center peak is barely "
            "perturbed; increase for stronger periphery emphasis"
        ),
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
            "Limit noise to x <= MAX_X/2 and y <= MAX_Y/2 "
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
    paths = generate_instances(
        output_dir=args.output_dir,
        nx=args.nx,
        ny=args.ny,
        n_noisy=args.n_noisy,
        noise_std=args.noise_std,
        noise_smooth_sigma_cells=args.noise_smooth_sigma_cells,
        noise_height_power=args.noise_height_power,
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
