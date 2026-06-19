"""
Multi-GPU halo sampling for the 3 Gpc DISCO-DJ PM fields.

This mirrors GOTHAM/src/generate_catalog.py but is written for the 3 Gpc
field layout produced by IC_3gpc_test/pm_multigpu.py:

    dmo/disco/dmo_fields_subvols_grid_8_CV_<sim_id>_3gpc_multigpu.npy

Launch one process per GPU.  Each process reads a disjoint slice of the
memory-mapped PM field, samples halo-token sequences on its GPU, decodes the
tokens into halo rows, and saves a partial catalog.  Rank 0 then concatenates
the partial catalogs into a single npy file with columns:

    x, y, z, M, vx, vy, vz, concentration
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle as pk
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path("/mnt/ceph/users/spandey/quijote_v2_gotham")
IC3_ROOT = ROOT / "IC_3gpc_test"
GOTHAM_SRC = ROOT / "GOTHAM" / "src"
if str(GOTHAM_SRC) not in sys.path:
    sys.path.insert(0, str(GOTHAM_SRC))

from model_enc_dec_cos import HaloDecoderModel  # noqa: E402


FIDUCIAL_COSMO_PARAMS = np.array([0.3175, 0.0490, 0.6711, 0.9624, 0.8340], dtype=np.float32)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def get_rank_info() -> tuple[int, int, int]:
    rank = int(
        os.environ.get(
            "RANK",
            os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")),
        )
    )
    world = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")),
        )
    )
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")),
        )
    )
    return rank, world, local_rank


def log(message: str, rank: int | None = None, *, all_ranks: bool = True) -> None:
    if rank is None:
        print(message, flush=True)
    elif all_ranks or rank == 0:
        print(f"[rank {rank}] {message}", flush=True)


def parse_cosmo_params(value: str) -> np.ndarray:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 5:
        raise ValueError(
            "--cosmo-params must contain 5 comma-separated values: "
            "Omega_m,Omega_b,h,n_s,sigma_8"
        )
    return np.asarray(parts, dtype=np.float32)


def cosmo_params_from_meta(meta: dict) -> np.ndarray | None:
    cosmo = meta.get("cosmo")
    if not isinstance(cosmo, dict):
        return None
    omega_b = float(cosmo["Omega_b"])
    omega_m = float(cosmo["Omega_c"]) + omega_b
    return np.array(
        [omega_m, omega_b, float(cosmo["h"]), float(cosmo["n_s"]), float(cosmo["sigma8"])],
        dtype=np.float32,
    )


def resolve_cosmo_params(args: argparse.Namespace, meta: dict, sim_id: int, rank: int) -> np.ndarray:
    if args.cosmo_params:
        params = parse_cosmo_params(args.cosmo_params)
        log(f"Using explicit cosmology params {params.tolist()}", rank=rank, all_ranks=False)
        return params

    if args.use_param_file:
        table = np.loadtxt(args.param_file)
        row = sim_id - args.param_sim_offset
        if row < 0 or row >= len(table):
            raise IndexError(
                f"sim_id={sim_id} with --param-sim-offset={args.param_sim_offset} "
                f"selects row {row}, outside parameter table with {len(table)} rows."
            )
        params = np.asarray(table[row], dtype=np.float32)
        log(f"Using parameter-table row {row}: {params.tolist()}", rank=rank, all_ranks=False)
        return params

    params = cosmo_params_from_meta(meta)
    if params is not None:
        log(f"Using cosmology from PM metadata: {params.tolist()}", rank=rank, all_ranks=False)
        return params

    log(
        "PM metadata did not contain cosmology; falling back to fiducial Quijote CV params.",
        rank=rank,
        all_ranks=False,
    )
    return FIDUCIAL_COSMO_PARAMS.copy()


def load_model(checkpoint_path: Path, device: torch.device, compile_blocks: bool) -> HaloDecoderModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    config = checkpoint["config"]

    model = HaloDecoderModel(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if compile_blocks:
        for i in range(len(model.transformer.h)):
            model.transformer.h[i] = torch.compile(model.transformer.h[i], mode="default")

    return model


class TokenDecoder:
    def __init__(
        self,
        *,
        boxsize: float,
        grid: int,
        seed: int,
        rank: int,
        pad_as_end: bool = False,
    ) -> None:
        self.boxsize = float(boxsize)
        self.grid = int(grid)
        self.rng = np.random.default_rng(seed + 1009 * rank)
        self.pad_as_end = pad_as_end

        self.nvocab = 131
        self.start_token = self.nvocab + 1
        self.space_token = self.nvocab + 2
        self.pad_token = self.nvocab + 3
        self.end_token = self.nvocab + 4
        self.pos_vocab = 40
        self.add_space_token = False
        self.dim_tot = 8
        self.dim_prop = 5
        self.ntokens_per_halo = self.dim_tot + 1 if self.add_space_token else self.dim_tot

        self.xarray = np.arange(self.pos_vocab, dtype=np.float32) * (
            self.boxsize / self.pos_vocab / self.grid
        )
        self.xarray = np.concatenate(
            (self.xarray, np.array([self.boxsize / self.grid], dtype=np.float32))
        )
        self.dx = self.boxsize / (self.pos_vocab * self.grid)

        self.bins_digitize = np.zeros((self.dim_prop, self.nvocab + 1), dtype=np.float32)
        self.bins_digitize[0, :-1] = np.linspace(12.7, 15.0, self.nvocab)
        self.bins_digitize[0, -1] = 15.0
        for i in range(1, 4):
            self.bins_digitize[i, :-1] = np.linspace(-1250.0, 1250.0, self.nvocab)
            self.bins_digitize[i, -1] = 1250.0
        self.bins_digitize[4, :-1] = np.linspace(1.0, 16.0, self.nvocab)
        self.bins_digitize[4, -1] = 16.0

        self.bins_step = np.zeros(self.dim_prop, dtype=np.float32)
        self.bins_step[0] = (15.0 - 12.7) / (self.nvocab - 1)
        for i in range(1, 4):
            self.bins_step[i] = (1250.0 - (-1250.0)) / (self.nvocab - 1)
        self.bins_step[4] = (16.0 - 1.0) / (self.nvocab - 1)

        self.invalid_no_end = 0
        self.invalid_length = 0
        self.invalid_token = 0

    def decode_batch(self, sentences: np.ndarray, global_indices: np.ndarray) -> np.ndarray:
        rows: list[list[float]] = []
        for sentence, global_index in zip(sentences, global_indices):
            decoded = self.decode_sentence(sentence, int(global_index))
            if decoded:
                rows.extend(decoded)

        if not rows:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def decode_sentence(self, sentence: np.ndarray, global_index: int) -> list[list[float]]:
        start_hits = np.flatnonzero(sentence == self.start_token)
        if len(start_hits) == 0:
            self.invalid_token += 1
            return []
        ind_start = int(start_hits[0])

        end_hits = np.flatnonzero(sentence[ind_start + 1 :] == self.end_token)
        if len(end_hits) == 0 and self.pad_as_end:
            end_hits = np.flatnonzero(sentence[ind_start + 1 :] == self.pad_token)
        if len(end_hits) == 0:
            self.invalid_no_end += 1
            return []
        ind_end = ind_start + 1 + int(end_hits[0])

        payload_len = ind_end - ind_start - 1
        if payload_len < 0 or payload_len % self.ntokens_per_halo != 0:
            self.invalid_length += 1
            return []

        nhalos = payload_len // self.ntokens_per_halo
        if nhalos == 0:
            return []

        jx, jy, jz = np.unravel_index(global_index, (self.grid, self.grid, self.grid))
        cell = self.boxsize / self.grid
        out: list[list[float]] = []

        for halo_id in range(nhalos):
            base = ind_start + halo_id * self.ntokens_per_halo + 1
            pos_tokens = sentence[base : base + 3].astype(np.int64, copy=False)
            prop_tokens = sentence[base + 3 : base + 3 + self.dim_prop].astype(np.int64, copy=False)

            if (
                pos_tokens.shape[0] != 3
                or prop_tokens.shape[0] != self.dim_prop
                or np.any(pos_tokens < 0)
                or np.any(pos_tokens >= len(self.xarray))
                or np.any(prop_tokens < 0)
                or np.any(prop_tokens >= self.bins_digitize.shape[1])
            ):
                self.invalid_token += 1
                continue

            prop = self.bins_digitize[np.arange(self.dim_prop), prop_tokens]
            prop = prop + self.rng.uniform(-0.5, 0.5, size=self.dim_prop).astype(np.float32) * self.bins_step
            prop = np.clip(prop, self.bins_digitize[:, 0], self.bins_digitize[:, -1])

            coord_x = (self.xarray[pos_tokens[0]] + cell * jx + self.rng.uniform(-0.5, 0.5) * self.dx) % self.boxsize
            coord_y = (self.xarray[pos_tokens[1]] + cell * jy + self.rng.uniform(-0.5, 0.5) * self.dx) % self.boxsize
            coord_z = (self.xarray[pos_tokens[2]] + cell * jz + self.rng.uniform(-0.5, 0.5) * self.dx) % self.boxsize

            prop[0] = np.power(10.0, prop[0])
            out.append(
                [
                    float(coord_x),
                    float(coord_y),
                    float(coord_z),
                    float(prop[0]),
                    float(prop[1]),
                    float(prop[2]),
                    float(prop[3]),
                    float(prop[4]),
                ]
            )

        return out

    def stats(self) -> dict[str, int]:
        return {
            "invalid_no_end": self.invalid_no_end,
            "invalid_length": self.invalid_length,
            "invalid_token": self.invalid_token,
        }


def rank_bounds(n_items: int, rank: int, world: int) -> tuple[int, int]:
    start = (n_items * rank) // world
    end = (n_items * (rank + 1)) // world
    return start, end


def wait_for_files(paths: Iterable[Path], *, timeout_s: float, rank: int, label: str) -> None:
    t0 = time.time()
    paths = list(paths)
    last_report = 0.0
    while True:
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for {len(missing)} {label} files. "
                f"First missing: {missing[0]}"
            )
        if elapsed - last_report > 60.0:
            log(
                f"Waiting for {len(missing)} / {len(paths)} {label} files after {elapsed:.0f}s",
                rank=rank,
                all_ranks=False,
            )
            last_report = elapsed
        time.sleep(5.0)


def wait_for_any_file(paths: Iterable[Path], *, timeout_s: float, rank: int, label: str) -> Path:
    t0 = time.time()
    paths = list(paths)
    last_report = 0.0
    while True:
        for path in paths:
            if path.exists():
                return path
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f}s waiting for any {label} file. "
                f"Candidates: {[str(path) for path in paths]}"
            )
        if elapsed - last_report > 60.0:
            log(f"Waiting for {label} after {elapsed:.0f}s", rank=rank, all_ranks=False)
            last_report = elapsed
        time.sleep(5.0)


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def combine_partials(
    *,
    partial_dir: Path,
    output_path: Path,
    meta_path: Path,
    world: int,
    args: argparse.Namespace,
    cosmo_params: np.ndarray,
    decoder_stats: dict[str, int],
) -> None:
    partial_paths = [partial_dir / f"partial_rank{rank:04d}.npy" for rank in range(world)]
    shapes = []
    dtype = None
    total_rows = 0
    for path in partial_paths:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != 8:
            raise ValueError(f"Partial catalog {path} has shape {arr.shape}; expected (N, 8).")
        if dtype is None:
            dtype = arr.dtype
        elif arr.dtype != dtype:
            raise ValueError(f"Partial catalog {path} has dtype {arr.dtype}; expected {dtype}.")
        shapes.append(arr.shape)
        total_rows += arr.shape[0]
        del arr

    tmp = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    out = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float32 if dtype is None else dtype,
        shape=(total_rows, 8),
    )
    offset = 0
    for path, shape in zip(partial_paths, shapes):
        arr = np.load(path, mmap_mode="r")
        n = shape[0]
        if n:
            out[offset : offset + n] = arr
            offset += n
        del arr
    out.flush()
    del out
    os.replace(tmp, output_path)

    save_json(
        meta_path,
        {
            "output_path": str(output_path),
            "partial_dir": str(partial_dir),
            "sim_id": args.sim_id,
            "boxsize": args.boxsize,
            "grid": args.grid,
            "grid_sbox": args.grid_sbox,
            "cosmo_params_order": ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"],
            "cosmo_params": [float(x) for x in cosmo_params],
            "world_size": world,
            "batch_size": args.batch_size,
            "checkpoint": str(args.checkpoint),
            "pm_field_path": str(args.pm_field_path),
            "pm_meta_path": str(args.pm_meta_path),
            "n_halos": int(total_rows),
            "decoder_stats_rank0": decoder_stats,
            "catalog_columns": ["x", "y", "z", "M", "vx", "vy", "vz", "concentration"],
        },
    )


def sample_rank(
    *,
    args: argparse.Namespace,
    rank: int,
    world: int,
    local_rank: int,
    partial_dir: Path,
    pm_fields: np.memmap,
    rand_sel: np.ndarray,
    cosmo_params: np.ndarray,
) -> dict[str, int]:
    if torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        device_index = 0 if visible_count == 1 else local_rank % visible_count
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    log(
        f"Using device={device}, local_rank={local_rank}, "
        f"visible_cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        rank=rank,
    )
    if device.type != "cuda":
        raise RuntimeError("This sampler is intended for GPU inference, but CUDA is not available.")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    model = load_model(args.checkpoint, device, args.compile)
    decoder = TokenDecoder(
        boxsize=args.boxsize,
        grid=args.grid,
        seed=args.seed,
        rank=rank,
        pad_as_end=args.pad_as_end,
    )

    start, end = rank_bounds(len(pm_fields), rank, world)
    log(
        f"Processing subbox rows [{start}, {end}) out of {len(pm_fields)} "
        f"with batch_size={args.batch_size}",
        rank=rank,
    )

    param_one = torch.tensor(cosmo_params, device=device, dtype=torch.bfloat16).reshape(1, 5)
    partial_catalogs: list[np.ndarray] = []
    n_sequences = 0
    n_halos = 0
    t0 = time.time()
    use_autocast = device.type == "cuda"

    with torch.inference_mode():
        for s in tqdm(range(start, end, args.batch_size), disable=not args.progress, desc=f"rank {rank}"):
            e = min(s + args.batch_size, end)
            host_chunk = np.asarray(pm_fields[s:e])
            dmo_chunk = torch.from_numpy(host_chunk).to(device=device, dtype=torch.bfloat16)
            dmo_chunk = dmo_chunk.movedim(-1, 1).contiguous()
            params_chunk = param_one.expand(e - s, -1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                out = model.generate(
                    dmo_chunk,
                    params=params_chunk,
                    max_new_tokens=args.max_new_tokens,
                    start_token=decoder.start_token,
                    end_token=decoder.end_token,
                    pad_token=decoder.pad_token,
                )

            if out.shape[1] < args.target_len:
                pad_len = args.target_len - out.shape[1]
                pad = torch.full(
                    (out.shape[0], pad_len),
                    decoder.pad_token,
                    dtype=out.dtype,
                    device=out.device,
                )
                out = torch.cat([out, pad], dim=1)
            elif out.shape[1] > args.target_len:
                out = out[:, : args.target_len]

            tokens = out.cpu().numpy().astype(np.int16, copy=False)
            # Match generate_catalog.py: np.delete(data, [1,2,3,4,5,6], axis=1)
            sentences = np.concatenate((tokens[:, :1], tokens[:, 7:]), axis=1)
            catalog = decoder.decode_batch(sentences, rand_sel[s:e])
            if catalog.shape[0]:
                partial_catalogs.append(catalog)
                n_halos += catalog.shape[0]
            n_sequences += e - s

            del host_chunk, dmo_chunk, params_chunk, out, tokens, sentences, catalog
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            if args.log_every > 0:
                batches_done = (s - start) // args.batch_size + 1
                if batches_done % args.log_every == 0:
                    elapsed = time.time() - t0
                    log(
                        f"progress: {n_sequences:,} sequences, {n_halos:,} halos, "
                        f"{elapsed:.1f}s elapsed",
                        rank=rank,
                    )

    if partial_catalogs:
        rank_catalog = np.concatenate(partial_catalogs, axis=0)
    else:
        rank_catalog = np.empty((0, 8), dtype=np.float32)

    partial_path = partial_dir / f"partial_rank{rank:04d}.npy"
    np.save(partial_path, rank_catalog)
    stats = decoder.stats()
    stats.update(
        {
            "rank": rank,
            "n_sequences": int(n_sequences),
            "n_halos": int(rank_catalog.shape[0]),
            "elapsed_s": float(time.time() - t0),
        }
    )
    save_json(partial_dir / f"partial_rank{rank:04d}.json", stats)
    (partial_dir / f"rank{rank:04d}.done").write_text("done\n")
    log(f"Saved partial catalog {partial_path} with shape={rank_catalog.shape}; stats={stats}", rank=rank)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_id", nargs="?", type=int, default=int(os.environ.get("SIM_ID", "0")))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("DISCO_GEN_CHECKPOINT", ROOT / "GOTHAM/checkpoints/checkpoint_new.pt")),
    )
    parser.add_argument(
        "--pm-field-path",
        type=Path,
        default=None,
        help="Path to dmo_fields_subvols_grid_8_CV_<sim_id>_3gpc_multigpu.npy.",
    )
    parser.add_argument("--pm-meta-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("DISCO_GEN_OUTPUT_DIR", IC3_ROOT / "generated_halo_cats")),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--param-file",
        type=Path,
        default=Path(os.environ.get("DISCO_GEN_PARAM_FILE", ROOT / "GOTHAM/checkpoints/quijote_params.txt")),
    )
    parser.add_argument("--use-param-file", action="store_true", default=env_bool("DISCO_GEN_USE_PARAM_FILE", False))
    parser.add_argument("--param-sim-offset", type=int, default=int(os.environ.get("DISCO_GEN_PARAM_SIM_OFFSET", "0")))
    parser.add_argument("--cosmo-params", default=os.environ.get("DISCO_GEN_COSMO_PARAMS", ""))
    parser.add_argument("--boxsize", type=float, default=float(os.environ.get("DISCO_GEN_BOXSIZE", "3000.0")))
    parser.add_argument("--grid", type=int, default=int(os.environ.get("DISCO_GEN_GRID", "192")))
    parser.add_argument("--grid-sbox", type=int, default=int(os.environ.get("DISCO_GEN_GRID_SBOX", "8")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DISCO_GEN_BATCH_SIZE", "8192")))
    parser.add_argument("--target-len", type=int, default=int(os.environ.get("DISCO_GEN_TARGET_LEN", "296")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("DISCO_GEN_MAX_NEW_TOKENS", "289")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DISCO_GEN_SEED", "12345")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("DISCO_GEN_LOG_EVERY", "5")))
    parser.add_argument("--barrier-timeout-s", type=float, default=float(os.environ.get("DISCO_GEN_BARRIER_TIMEOUT_S", "86400")))
    parser.add_argument("--run-id", default=os.environ.get("DISCO_GEN_RUN_ID", os.environ.get("SLURM_JOB_ID", "manual")))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=env_bool("DISCO_GEN_COMPILE", True))
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=env_bool("DISCO_GEN_PROGRESS", True))
    parser.add_argument("--pad-as-end", action=argparse.BooleanOptionalAction, default=env_bool("DISCO_GEN_PAD_AS_END", False))
    parser.add_argument("--overwrite", action="store_true", default=env_bool("DISCO_GEN_OVERWRITE", False))
    parser.add_argument("--keep-partials", action="store_true", default=env_bool("DISCO_GEN_KEEP_PARTIALS", False))
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/shapes and rank slicing without sampling.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    rank, world, local_rank = get_rank_info()
    args.sim_id = int(args.sim_id)

    if args.pm_field_path is None:
        args.pm_field_path = (
            IC3_ROOT
            / "dmo/disco"
            / f"dmo_fields_subvols_grid_{args.grid_sbox}_CV_{args.sim_id}_3gpc_multigpu.npy"
        )
    if args.pm_meta_path is None:
        args.pm_meta_path = (
            IC3_ROOT
            / "dmo/disco"
            / f"meta_dmo_fields_subvols_grid_{args.grid_sbox}_CV_{args.sim_id}_3gpc_multigpu.pkl"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_path is None:
        args.output_path = args.output_dir / f"generated_halo_catalog_CV_{args.sim_id}_3gpc_multigpu.npy"
    output_meta_path = args.output_path.with_suffix(".json")

    partial_dir = args.output_dir / ".partials" / f"{args.output_path.stem}.run_{args.run_id}"

    if rank == 0:
        if args.output_path.exists() and not args.overwrite:
            log(f"Output already exists, skipping: {args.output_path}", rank=rank)
            (partial_dir / "skip").parent.mkdir(parents=True, exist_ok=True)
            (partial_dir / "skip").write_text("skip\n")
        else:
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            partial_dir.mkdir(parents=True, exist_ok=True)
            (partial_dir / "init.done").write_text("ready\n")
    else:
        wait_for_any_file(
            [partial_dir / "init.done", partial_dir / "skip"],
            timeout_s=args.barrier_timeout_s,
            rank=rank,
            label="initialization marker",
        )

    if (partial_dir / "skip").exists():
        return
    if rank == 0:
        (partial_dir / "init.done").write_text("ready\n")
    else:
        wait_for_files([partial_dir / "init.done"], timeout_s=args.barrier_timeout_s, rank=rank, label="init marker")

    if not args.pm_field_path.exists():
        raise FileNotFoundError(f"PM field file not found: {args.pm_field_path}")
    if not args.pm_meta_path.exists():
        raise FileNotFoundError(f"PM metadata file not found: {args.pm_meta_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    with args.pm_meta_path.open("rb") as f:
        meta = pk.load(f)
    pm_fields = np.load(args.pm_field_path, mmap_mode="r")
    if pm_fields.ndim != 5 or pm_fields.shape[1:4] != (args.grid_sbox, args.grid_sbox, args.grid_sbox):
        raise ValueError(
            f"Expected PM field shape (N,{args.grid_sbox},{args.grid_sbox},{args.grid_sbox},C), "
            f"got {pm_fields.shape}."
        )
    if pm_fields.shape[-1] != 18:
        raise ValueError(f"Expected 18 PM field channels for this checkpoint, got {pm_fields.shape[-1]}.")

    n_expected_full = args.grid**3
    rand_sel = np.asarray(meta.get("rand_sel", np.arange(n_expected_full)), dtype=np.int64)
    if len(rand_sel) != len(pm_fields):
        if len(pm_fields) == n_expected_full:
            log(
                f"Metadata rand_sel has length {len(rand_sel)}, but PM field has full length {len(pm_fields)}; "
                "using natural full-grid ordering.",
                rank=rank,
            )
            rand_sel = np.arange(n_expected_full, dtype=np.int64)
        else:
            raise ValueError(
                f"PM field has {len(pm_fields)} rows but metadata rand_sel has {len(rand_sel)} entries."
            )
    if np.any(rand_sel < 0) or np.any(rand_sel >= n_expected_full):
        raise ValueError("rand_sel contains indices outside the 3 Gpc grid.")

    cosmo_params = resolve_cosmo_params(args, meta, args.sim_id, rank)
    start, end = rank_bounds(len(pm_fields), rank, world)
    log(
        f"Config: sim_id={args.sim_id}, world={world}, rank_slice=[{start},{end}), "
        f"pm_shape={pm_fields.shape}, pm_dtype={pm_fields.dtype}, output={args.output_path}",
        rank=rank,
    )

    if args.dry_run:
        if rank == 0:
            log("Dry run complete; no model loaded and no catalog written.", rank=rank)
        return

    stats = sample_rank(
        args=args,
        rank=rank,
        world=world,
        local_rank=local_rank,
        partial_dir=partial_dir,
        pm_fields=pm_fields,
        rand_sel=rand_sel,
        cosmo_params=cosmo_params,
    )

    if rank == 0:
        done_files = [partial_dir / f"rank{r:04d}.done" for r in range(world)]
        wait_for_files(done_files, timeout_s=args.barrier_timeout_s, rank=rank, label="rank done")
        log("All rank partials are present; combining final halo catalog", rank=rank)
        combine_partials(
            partial_dir=partial_dir,
            output_path=args.output_path,
            meta_path=output_meta_path,
            world=world,
            args=args,
            cosmo_params=cosmo_params,
            decoder_stats=stats,
        )
        log(f"Saved final halo catalog: {args.output_path}", rank=rank)
        if not args.keep_partials:
            shutil.rmtree(partial_dir)
            log(f"Removed partial directory: {partial_dir}", rank=rank)


if __name__ == "__main__":
    main()
