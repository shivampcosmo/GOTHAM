# GOTHAM

GOTHAM is a research codebase for generating Quijote-like dark-matter-only fields and training a transformer to generate halo catalogs from those fields. The workflow covers initial-condition conversion, Disco-DJ/FastPM evolution, DMO subvolume preprocessing, Rockstar halo tokenization, DMO-conditioned autoregressive training, and checkpoint-based catalog generation.

For a detailed file-by-file digest, see [LIVE_CODE_SUMMARY.md](LIVE_CODE_SUMMARY.md).

## Main Workflow

1. Build/copy 2LPT inputs in `ICs/`, then run generated Slurm jobs from `run_scripts_IC/`.
2. Convert raw 2LPT Fourier outputs with `ICs/convert_ICs.py`.
3. Run `prepare/pm.py` to evolve ICs and save DMO subvolume tensors plus metadata.
4. Run `prepare/process_halo_props.py` to convert Quijote Rockstar halos into aligned token sentences.
5. Train with `src/train.py`.
6. Generate catalogs with `src/generate_catalog.py`.

## Directory Map

- `src/`: PyTorch models, DDP training, 3D CNN/ViT encoders, CBAM blocks, and catalog generation.
- `prepare/`: Disco-DJ PM preprocessing, older DMO-field preprocessing, halo tokenization, and Cython NGP assignment helpers.
- `ICs/`: 2LPT templates, parameter-file helpers, IC conversion, older PM runner, and simple CIC density-grid generation.
- `run_scripts_*`: Master scripts that generate and submit Slurm batches; generated `.slurm`, `.out`, and `.err` files are historical job artifacts.
- `checkpoints/`: Cosmology table and model checkpoint artifacts used by inference.
- `temp/`: Exploratory notebooks plus the 3Gpc worker/combine generation prototype.

## Important Notes

The code is path- and cluster-specific: many files hard-code `/mnt/ceph` or `/mnt/home` paths, module loads, and conda environments. There is no dependency lock file. Before production training, check the caveats in `LIVE_CODE_SUMMARY.md`, especially the `src/train.py` runtime issues and the distinction between `model_enc_dec.py` and `model_enc_dec_cos.py`.
