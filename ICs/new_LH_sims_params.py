"""Generate new Quijote LH cosmological parameters with a Sobol sequence.

The generated columns match ``checkpoints/quijote_params.txt``:

    Omega_m, Omega_b, h, n_s, sigma_8

Examples
--------
Create 256 new cosmologies:

    python ICs/new_LH_sims_params.py --n-samples 256

Append the next 128 Sobol points to the same output file:

    python ICs/new_LH_sims_params.py --n-samples 128 --append
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import qmc


PARAM_NAMES = ("Omega_m", "Omega_b", "h", "n_s", "sigma_8")

# Base Quijote LH priors.
PRIORS = np.array(
    [
        [0.10, 0.50],  # Omega_m
        [0.03, 0.07],  # Omega_b
        [0.50, 0.90],  # h
        [0.80, 1.20],  # n_s
        [0.60, 1.00],  # sigma_8
    ],
    dtype=np.float64,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "checkpoints" / "new_LH_sims_params.txt"
HEADER = "Omega_m                 Omega_b                  h                        n_s                      sigma_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create new Quijote LH cosmological parameters using Sobol sampling."
    )
    parser.add_argument(
        "-N",
        "--n-samples",
        type=int,
        required=True,
        help="Number of new cosmologies to generate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output txt file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "Sobol index to start from. Use this to continue a previous Sobol run "
            "without appending to the same file."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append to the output file and automatically start after the number "
            "of rows already saved there."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed used for scrambled Sobol sampling. Keep fixed when extending.",
    )
    parser.add_argument(
        "--no-scramble",
        dest="scramble",
        action="store_false",
        help="Use the deterministic unscrambled Sobol sequence.",
    )
    parser.set_defaults(scramble=True)
    return parser.parse_args()


def generate_sobol_cosmologies(
    n_samples: int,
    *,
    start_index: int = 0,
    scramble: bool = True,
    seed: int | None = 12345,
) -> np.ndarray:
    """Return Sobol samples scaled to the Quijote LH prior ranges."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")

    sampler = qmc.Sobol(
        d=len(PARAM_NAMES),
        scramble=scramble,
        seed=seed if scramble else None,
    )
    if start_index:
        sampler.fast_forward(start_index)

    # SciPy warns for arbitrary N because Sobol balance is strongest for powers of 2.
    # We still allow arbitrary N so simulations can be added in whatever batch size is needed.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol' points require n to be a power of 2.*",
        )
        unit_samples = sampler.random(n_samples)

    prior_min = PRIORS[:, 0]
    prior_max = PRIORS[:, 1]
    return prior_min + unit_samples * (prior_max - prior_min)


def load_existing_params(path: Path) -> np.ndarray:
    """Load an existing five-column parameter file, returning an empty array if absent."""
    if not path.exists():
        return np.empty((0, len(PARAM_NAMES)), dtype=np.float64)

    params = np.loadtxt(path)
    if params.size == 0:
        return np.empty((0, len(PARAM_NAMES)), dtype=np.float64)

    params = np.atleast_2d(params)
    if params.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"{path} has {params.shape[1]} columns; expected {len(PARAM_NAMES)}."
        )
    return params


def save_params(path: Path, params: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, params, fmt="%.18e", header=HEADER, comments="#")


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()

    existing = load_existing_params(output) if args.append else np.empty((0, len(PARAM_NAMES)))
    start_index = existing.shape[0] if args.append else args.start_index

    new_params = generate_sobol_cosmologies(
        args.n_samples,
        start_index=start_index,
        scramble=args.scramble,
        seed=args.seed,
    )
    all_params = np.vstack([existing, new_params]) if existing.size else new_params
    save_params(output, all_params)

    mode = "Appended" if args.append and existing.size else "Saved"
    print(f"{mode} {args.n_samples} new Sobol cosmologies to {output}")
    print(f"Rows in output: {all_params.shape[0]}")
    print(f"Sobol start index for new rows: {start_index}")
    print(f"Scramble: {args.scramble}; seed: {args.seed if args.scramble else 'unused'}")


if __name__ == "__main__":
    main()
