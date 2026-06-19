"""Create CAMB spectra and 2LPT parameter files for new Quijote LH sims.

This script reads the five-column cosmology file produced by
``new_LH_sims_params.py`` and creates, for each row, the files needed by the
2LPT initial-condition code:

    <output-root>/<sim-id>/ICs/Cosmo_params.dat
    <output-root>/<sim-id>/ICs/Pk_mm_z=0.000.txt
    <output-root>/<sim-id>/ICs/2LPT_512.param

The output power-spectrum format follows the Quijote archive files used by the
existing IC scripts: 400 logarithmic k values from 2e-5 to 10 and linear
z = 0 matter power, generated with CAMB for a flat LCDM cosmology with no
massive neutrinos.

Example
-------
Generate the first ten new sims as LH/2000 ... LH/2009:

    python ICs/create_2lpt_params_new_LHsims.py --n-sims 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import camb
import numpy as np
from camb import model
from tqdm import tqdm


PARAM_NAMES = ("Omega_m", "Omega_b", "h", "n_s", "sigma_8")
REPO_ROOT = Path(__file__).resolve().parents[1]
IC_DIR = Path(__file__).resolve().parent

DEFAULT_COSMO_FILE = REPO_ROOT / "checkpoints" / "new_LH_sims_params.txt"
DEFAULT_TEMPLATE = IC_DIR / "2LPT_base_512.param"
DEFAULT_OUTPUT_ROOT = Path("/mnt/ceph/users/spandey/discodj_runs/LH")

PK_FILENAME = "Pk_mm_z=0.000.txt"
COSMO_FILENAME = "Cosmo_params.dat"
K_MIN = 2.0e-5
K_MAX = 10.0
N_K = 400


@dataclass(frozen=True)
class Cosmology:
    omega_m: float
    omega_b: float
    h: float
    n_s: float
    sigma_8: float

    @classmethod
    def from_row(cls, row: np.ndarray) -> "Cosmology":
        if row.shape[0] != len(PARAM_NAMES):
            raise ValueError(f"Expected {len(PARAM_NAMES)} cosmological parameters, got {row.shape[0]}.")
        omega_m, omega_b, h, n_s, sigma_8 = row
        if omega_b >= omega_m:
            raise ValueError(f"Omega_b must be smaller than Omega_m; got {omega_b} >= {omega_m}.")
        return cls(
            omega_m=float(omega_m),
            omega_b=float(omega_b),
            h=float(h),
            n_s=float(n_s),
            sigma_8=float(sigma_8),
        )

    @property
    def omega_cdm(self) -> float:
        return self.omega_m - self.omega_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAMB z=0 linear power spectra and 2LPT params for new Quijote LH sims."
    )
    parser.add_argument(
        "--cosmo-file",
        type=Path,
        default=DEFAULT_COSMO_FILE,
        help=f"Five-column cosmology txt file. Default: {DEFAULT_COSMO_FILE}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root for LH/<sim-id>/ICs directories. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"2LPT template file. Default: {DEFAULT_TEMPLATE}",
    )
    parser.add_argument(
        "--first-row",
        type=int,
        default=0,
        help="First row in the cosmology file to process.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Number of rows to process. Default: all rows from --first-row onward.",
    )
    parser.add_argument(
        "--start-sim-id",
        type=int,
        default=2000,
        help="Simulation id assigned to row 0 of --cosmo-file. Default: 2000.",
    )
    parser.add_argument(
        "--redshift-ic",
        type=float,
        default=127.0,
        help="Starting redshift written to the 2LPT parameter file. Default: 127.",
    )
    parser.add_argument(
        "--param-output-name",
        default=None,
        help="Name of the written 2LPT parameter file. Default is derived from the template.",
    )
    parser.add_argument(
        "--nnu",
        type=float,
        default=3.044,
        help="Effective number of massless neutrino species in CAMB. Default: 3.044.",
    )
    parser.add_argument(
        "--accuracy-boost",
        type=float,
        default=1.0,
        help="CAMB AccuracyBoost/lAccuracyBoost/lSampleBoost. Default: 1.0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if all expected outputs already exist.",
    )
    return parser.parse_args()


def load_cosmologies(path: Path, first_row: int, n_sims: int | None) -> np.ndarray:
    params = np.loadtxt(path)
    params = np.atleast_2d(params)
    if params.shape[1] != len(PARAM_NAMES):
        raise ValueError(f"{path} has {params.shape[1]} columns; expected {len(PARAM_NAMES)}.")
    if first_row < 0:
        raise ValueError("--first-row must be non-negative.")
    if n_sims is not None and n_sims <= 0:
        raise ValueError("--n-sims must be positive when provided.")

    last_row = None if n_sims is None else first_row + n_sims
    selected = params[first_row:last_row]
    if selected.size == 0:
        raise ValueError(f"No rows selected from {path}.")
    return selected


def output_param_name(template: Path, requested_name: str | None) -> str:
    if requested_name:
        return requested_name
    name = template.name
    if name.startswith("2LPT_base_"):
        return name.replace("2LPT_base_", "2LPT_", 1)
    if name == "2LPT_base.param":
        return "2LPT.param"
    return "2LPT.param"


def camb_linear_power_z0(
    cosmo: Cosmology,
    *,
    nnu: float,
    accuracy_boost: float,
) -> np.ndarray:
    """Return Quijote-format k and z=0 linear P(k) for this cosmology."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * cosmo.h,
        ombh2=cosmo.omega_b * cosmo.h**2,
        omch2=cosmo.omega_cdm * cosmo.h**2,
        omk=0.0,
        mnu=0.0,
        num_massive_neutrinos=0,
        nnu=nnu,
        TCMB=2.7255,
    )
    pars.set_dark_energy(w=-1.0, wa=0.0)
    pars.InitPower.set_params(As=2.1e-9, ns=cosmo.n_s)
    pars.set_matter_power(redshifts=[0.0], kmax=K_MAX, nonlinear=False)
    pars.NonLinear = model.NonLinear_none
    pars.WantTransfer = True
    pars.set_accuracy(
        AccuracyBoost=accuracy_boost,
        lAccuracyBoost=accuracy_boost,
        lSampleBoost=accuracy_boost,
    )

    results = camb.get_results(pars)
    sigma8_at_As = results.get_sigma8()[0]
    kh, _, pk = results.get_matter_power_spectrum(minkh=K_MIN, maxkh=K_MAX, npoints=N_K)

    # CAMB returns P(k) in (Mpc/h)^3 here. The archived Quijote 2LPT input
    # spectra use the same k grid with P(k) multiplied by h^3.
    sigma8_rescale = (cosmo.sigma_8 / sigma8_at_As) ** 2
    pk_quijote_units = pk[0] * sigma8_rescale * cosmo.h**3
    return np.column_stack([kh, pk_quijote_units])


def save_cosmo_params(path: Path, cosmo: Cosmology) -> None:
    values = np.array([cosmo.omega_m, cosmo.omega_b, cosmo.h, cosmo.n_s, cosmo.sigma_8])
    np.savetxt(path, values[None, :], fmt="%.5f")


def format_2lpt_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def replace_2lpt_line(line: str, replacements: dict[str, float | int]) -> str:
    parts = line.split(maxsplit=2)
    if not parts or parts[0] not in replacements:
        return line

    leading = line[: len(line) - len(line.lstrip())]
    key = parts[0]
    rest = "" if len(parts) < 3 else f" {parts[2].rstrip()}"
    return f"{leading}{key:<24}{format_2lpt_value(replacements[key])}{rest}\n"


def write_2lpt_param(
    template: Path,
    output_path: Path,
    cosmo: Cosmology,
    *,
    seed: int,
    redshift_ic: float,
) -> None:
    replacements = {
        "Omega": cosmo.omega_m,
        "OmegaLambda": 1.0 - cosmo.omega_m,
        "OmegaBaryon": 0.0,
        "OmegaDM_2ndSpecies": 0.0,
        "HubbleParam": cosmo.h,
        "Redshift": redshift_ic,
        "Sigma8": cosmo.sigma_8,
        "Seed": seed,
    }
    lines = template.read_text().splitlines(keepends=True)
    output_path.write_text("".join(replace_2lpt_line(line, replacements) for line in lines))


def expected_outputs(ic_dir: Path, param_name: str) -> tuple[Path, Path, Path]:
    return (
        ic_dir / COSMO_FILENAME,
        ic_dir / PK_FILENAME,
        ic_dir / param_name,
    )


def main() -> None:
    args = parse_args()
    cosmo_file = args.cosmo_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    template = args.template.expanduser().resolve()
    param_name = output_param_name(template, args.param_output_name)

    rows = load_cosmologies(cosmo_file, args.first_row, args.n_sims)
    if not template.exists():
        raise FileNotFoundError(f"2LPT template not found: {template}")

    print(f"Reading cosmologies from {cosmo_file}")
    print(f"Writing IC inputs under {output_root}")
    print(f"2LPT template: {template}")
    print(f"2LPT output name: {param_name}")

    n_written = 0
    n_skipped = 0
    for rel_idx, row in enumerate(tqdm(rows, desc="new LH IC inputs")):
        file_row = args.first_row + rel_idx
        sim_id = args.start_sim_id + file_row
        seed = sim_id
        cosmo = Cosmology.from_row(row)

        ic_dir = output_root / f"{sim_id}" / "ICs"
        cosmo_path, pk_path, param_path = expected_outputs(ic_dir, param_name)
        if not args.overwrite and all(path.exists() for path in (cosmo_path, pk_path, param_path)):
            n_skipped += 1
            continue

        ic_dir.mkdir(parents=True, exist_ok=True)
        pk = camb_linear_power_z0(
            cosmo,
            nnu=args.nnu,
            accuracy_boost=args.accuracy_boost,
        )
        np.savetxt(pk_path, pk, fmt="%.18e")
        save_cosmo_params(cosmo_path, cosmo)
        write_2lpt_param(
            template,
            param_path,
            cosmo,
            seed=seed,
            redshift_ic=args.redshift_ic,
        )
        n_written += 1

    print(f"Generated files for {n_written} simulations.")
    print(f"Skipped {n_skipped} simulations with existing complete outputs.")


if __name__ == "__main__":
    main()
