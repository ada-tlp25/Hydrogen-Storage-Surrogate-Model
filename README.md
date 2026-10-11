# Developing a Surrogate Model for Underground Hydrogen Storage using Deep Learning

## Overview

This project develops a surrogate model for Underground Hydrogen Storage (UHS). Surrogate models allow reservoir simulations to be completed rapidly. A synthetic dataset is generated from natural gas storage data (AGSI) and a numerical reservoir simulator (ECLIPSE E300). The synthetic dataset is then used to train a three-dimensional surrogate model that predicts both spatial (hydrogen saturation plume and pressure) and scalar (injection rate, withdrawal rate and hydrogen stored) features.

The full write-up is in [`docs/final-report.pdf`](docs/final-report.pdf).

MSc Individual Research Project, Geo-Energy with Machine Learning and Data Science, Department of Earth Science and Engineering, Imperial College London (2025–26).

## Repository structure

```
├── data_generation/        # Builds synthetic cycles and ECLIPSE .DATA / schedule files from AGSI data
│   ├── eclipse_ng_gen.py
│   └── synthetic_script.py
├── preprocessing/          # Loads AGSI natural gas data and computes cycle statistics
│   ├── ng_loader.py
│   └── ng_cycles_stats.py
├── training/               # Files uploaded to Google Colab / the HPC for dataset building and training
│   ├── unet.py             # Multi-head 3D U-Net
│   ├── dataset.py          # PyTorch dataset and transforms
│   ├── unrst_loader.py     # Converts ECLIPSE UNRST output to pkl (run on the HPC)
│   ├── get_normal_rate.py
│   ├── get_norm_stats.py
│   ├── npy_create.py
│   ├── dataset_stats.py
│   ├── check_split.py
│   ├── add_normal_rates.ipynb
│   ├── train_unet.ipynb
│   ├── normalization_stats.ipynb
│   └── test_set_stats.ipynb
├── hpc/                    # PBS job scripts for the Imperial CX3 cluster
│   ├── run_simulations.pbs
│   ├── unrst_to_pkl.pbs
│   └── static_grid_pkl.pbs
├── notebooks/
│   ├── 01_reproduce_results.ipynb   # Start here: step-by-step reproduction guide
│   ├── methods_figures.ipynb
│   ├── natural_gas_figures.ipynb
│   └── results_figures.ipynb
├── data/
│   ├── agsi_depleted/      # AGSI storage data for depleted-field sites (JSON)
│   └── synthetic_cycles_full.pkl
├── docs/                   # Final report, project plan and their LaTeX sources
├── archive/                # Exploratory / superseded work, not needed to reproduce results
├── environment.yml         # Local environment
├── requirements-hpc.txt
└── requirements-colab.txt
```

## Environments

The project runs in three environments:

| Environment | Used for | Setup |
|---|---|---|
| Local | Data generation, preprocessing, report figures | `conda env create -f environment.yml` then `conda activate IRP_env` |
| Imperial HPC (CX3) | ECLIPSE simulations, UNRST to pkl conversion | `pip install -r requirements-hpc.txt` |
| Google Colab | Dataset building, model training and evaluation | `pip install -r requirements-colab.txt` |

## How to run

1. Run `notebooks/01_reproduce_results.ipynb` and follow its directions for changing directory paths. This generates the synthetic cycles and the ECLIPSE input files.
2. Run the generated scripts on ECLIPSE E300 v2024.2. `hpc/run_simulations.pbs` is an example for the Imperial CX3 cluster; it expects a `data_files.txt` listing the `.DATA` file names.
3. Convert the simulation outputs to pkl with `convert_UNRST_to_pkl` in `training/unrst_loader.py` (example job: `hpc/unrst_to_pkl.pbs`).
4. Upload the resulting pkl files to Google Drive, along with these files from `training/`:
   - `unet.py`
   - `dataset.py`
   - `get_normal_rate.py`
   - `get_norm_stats.py`
   - `add_normal_rates.ipynb`
   - `train_unet.ipynb`

   Also upload `data/synthetic_cycles_full.pkl` (or the version created in step 1).
5. Run `add_normal_rates.ipynb`, then zip the resulting directory.
6. In `train_unet.ipynb`, set the unzip path and the dataset directory, and update the output paths at the bottom of the notebook.
7. Run `train_unet.ipynb`. In this study it was run on an NVIDIA A100 (40 GB); reaching epoch 90 (the checkpoint used) took about 20 hours.
8. Reproduce the report figures and tables with the notebooks in `notebooks/` (`results_figures.ipynb`, `methods_figures.ipynb`, `natural_gas_figures.ipynb`).

## License

Released under the MIT License. See [LICENSE](LICENSE).
