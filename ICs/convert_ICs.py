import os
import sys
from pathlib import Path

import numpy as np


def get_rank_info():
    """Use SLURM task ids when launched with srun; fall back to serial."""
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    nprocs = int(os.environ.get("SLURM_NTASKS", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    return rank, max(nprocs, 1)


def file_index(path):
    return int(path.name.rsplit(".", 1)[-1])


def get_file_indices(path):
    coord_indices = {file_index(p) for p in path.glob("Coordinates_ptype_1.*")}
    amp_indices = {file_index(p) for p in path.glob("Amplitudes_ptype_1.*")}
    phase_indices = {file_index(p) for p in path.glob("Phases_ptype_1.*")}

    common = coord_indices & amp_indices & phase_indices
    if not common:
        raise FileNotFoundError(f"No complete Coordinates/Amplitudes/Phases file triplets in {path}")

    missing = (coord_indices | amp_indices | phase_indices) - common
    if missing:
        raise FileNotFoundError(f"Incomplete file triplets in {path}; bad indices: {sorted(missing)}")

    expected = set(range(max(common) + 1))
    if common != expected:
        raise FileNotFoundError(f"Missing file indices in {path}: {sorted(expected - common)}")

    return sorted(common)


def read_array_file(path, dtype):
    with path.open("rb") as f:
        nfiles = np.fromfile(f, dtype=np.int32, count=1)[0]
        nmesh = np.fromfile(f, dtype=np.int32, count=1)[0]
        nx = np.fromfile(f, dtype=np.int32, count=1)[0]
        values = np.fromfile(f, dtype=dtype, count=-1)
    return nfiles, nmesh, nx, values


def convert_one(sim_id):
    path = Path(f"/mnt/ceph/users/spandey/discodj_runs/LH/{sim_id}/ICs")
    savefname = path / "IC_delta512.npy"

    if savefname.exists():
        print(f"Skipping sim_id={sim_id}; {savefname} already exists", flush=True)
        return

    file_indices = get_file_indices(path)
    idx = []
    amp = []
    phase = []
    nmesh_values = set()

    print(f"\nConvert for path: {path}/ using {len(file_indices)} file chunks", flush=True)

    for i in file_indices:
        _, nmesh, _, coordinates = read_array_file(path / f"Coordinates_ptype_1.{i}", np.int64)
        nmesh_values.add(int(nmesh))

        kx = (coordinates // (nmesh // 2 + 1)) // nmesh
        ky = (coordinates // (nmesh // 2 + 1)) % nmesh
        kz = (coordinates % (nmesh // 2 + 1)) % nmesh
        idx.append(np.array([kx, ky, kz]).T)

        _, nmesh_amp, _, aa = read_array_file(path / f"Amplitudes_ptype_1.{i}", np.float32)
        _, nmesh_phase, _, ph = read_array_file(path / f"Phases_ptype_1.{i}", np.float32)
        nmesh_values.update([int(nmesh_amp), int(nmesh_phase)])
        amp.append(aa)
        phase.append(ph)

    if len(nmesh_values) != 1:
        raise ValueError(f"Inconsistent Nmesh values for sim_id={sim_id}: {sorted(nmesh_values)}")

    nmesh = nmesh_values.pop()
    idx = np.concatenate(idx)
    amp = np.concatenate(amp)
    phase = np.concatenate(phase)

    val = amp * np.exp(1j * phase)
    cmesh = val.reshape(nmesh, nmesh, idx[:, 2].max() + 1)
    mesh = np.fft.irfftn(cmesh, s=(nmesh, nmesh, nmesh), norm="ortho") * nmesh**1.5
    np.save(savefname, mesh.astype(np.float32))

    for i in file_indices:
        for prefix in ("Coordinates_ptype_1", "Amplitudes_ptype_1", "Phases_ptype_1"):
            (path / f"{prefix}.{i}").unlink(missing_ok=True)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python convert_ICs.py <sim_id_start> <sim_id_end>")

    imin = int(sys.argv[1])
    imax = int(sys.argv[2])
    points = np.arange(imin, imax, dtype=int)
    rank, nprocs = get_rank_info()
    assigned_points = points[rank::nprocs]

    print(
        f"Rank {rank}/{nprocs}: assigned {len(assigned_points)} sims "
        f"from [{imin}, {imax})",
        flush=True,
    )

    for sim_id in assigned_points:
        convert_one(int(sim_id))


if __name__ == "__main__":
    main()
