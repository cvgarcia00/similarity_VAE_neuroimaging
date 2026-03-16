# Beta-VAE for ADNI PET

Clean, public version of the Beta-VAE pipeline used in our publication.

## What is included

- Core Beta-VAE training/testing pipeline.
- Latent-space analyses and downstream prediction/classification utilities.
- Config-driven execution (`config.yaml`).

## Repository structure

- `src/beta_vae_model/`: model, dataloader, training, analysis utilities.
- `scripts/run_training.py`: main entrypoint for training/testing.
- `config.yaml`: experiment, data, and model configuration.
- `config.py`: config loader utility.

## Data requirements

This repository does **not** include ADNI data.
You need to provide your own authorized ADNI-derived files.

Expected paths by default (editable in `config.yaml`):

- `ADNI_BIDS/` (BIDS-converted PET data)
- `DATA/ADNI_BIDS.csv`
- `DATA/ADNIMERGE.csv`
- Optional template files under `DATA/vae_model_data/`

## Quick start

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Verify and edit `config.yaml` paths.

3. Run training/testing:

```bash
python scripts/run_training.py \
  --config config.yaml \
  --results RESULTS/beta_vae_results \
  --adnimerge DATA/ADNIMERGE.csv
```

## Notes

- The code falls back to CPU if CUDA is unavailable. (but CPU is NOT recommended)
- Outputs are written under the `--results` folder.
- Large files, model checkpoints, and local run artifacts are ignored by `.gitignore`.

## Citation

If you use this code, please cite the associated paper. Any inquiries can be asked to the corresponding author.
# similarity_VAE_neuroimaging
