## Project Overview

ML system to predict ATP tennis match winners using historical match data and engineered player features. The driving goal is to beat online betting houses by predicting match outcomes more accurately than the bookmakers' implied probabilities.

## Environment Setup

```bash
export PYTHONPATH=/path/to/tennispredictor/
python -m pip install -r requirements.txt
```

Requires the external [Jeff Sackmann tennis_atp dataset](https://github.com/JeffSackmann/tennis_atp) repo cloned separately.

## Commands

**Data pipeline (run first):**
```bash
python preprocessing/clean_data.py --tennisdir "/path/to/tennis_atp" --datadir "/path/to/data/dir"
```

**Train neural network:**
```bash
python main/train.py --csv "/path/to/data/dir/atp_database.csv"
```

**Train traditional classifiers:**
```bash
python main/classifier.py --csv "/path/to/data/dir/atp_database.csv" --rootdir "/path/to/data/dir"
# To run inference with a saved classifier, add --timestring and --classifier_name
```

**Hyperparameter search:**
```bash
python main/gridsearch.py --csv /path/to/csv
```

**Deploy on new matches:**
```bash
python preprocessing/generate_deploy.py   # produces deploy.csv
python main/deploy.py --csv /path/to/deploy.csv --ckpt_path /path/to/model --rootdir /path/to/output
```

There are no automated tests; validation is done by checking test set accuracy metrics.

## Architecture

### Data Flow
```
tennis_atp CSVs → clean_data.py → atp_database.csv → pipeline.py → normalized 36-dim features → model training/inference
```

### Key Components

**`preprocessing/`**
- `clean_data.py` (`ATP` class): Consolidates raw ATP CSV files, drops nulls, converts strings to numeric. Entry point for the entire pipeline.
- `pipeline.py`: Feature engineering classes consumed by both training and deployment:
  - `Elo`: Computes per-player ELO ratings across the match history (more predictive than ATP ranking points).
  - `TimePeriod`: Encodes dates as sinusoidal features capturing year and seasonal periodicity.
  - `RecentMatches`: Win/loss streaks, weeks inactive, head-to-head records.
  - `Dataspring`: Orchestrates train/val/test splits (60/20/20) and feature normalization.

**`models/model.py`**
PyTorch Lightning MLP: 36-dim input → 256 → 256 → 256 → 128 → 2-class output. Adam optimizer, binary cross-entropy. Logged via wandb.

**`main/`**
- `train.py`: Trains the MLP with early stopping and checkpointing.
- `classifier.py`: Trains and serializes 8 scikit-learn classifiers; includes permutation importance analysis.
- `deploy.py`: Loads a Lightning checkpoint and runs inference.
- `gridsearch.py`: Sweeps learning rates, batch sizes, and optimizers.

**`param_tennis.py`**
Central `Params` class — all hyperparameters (batch size 128, LR 1e-6, 100 epochs, dataset sizes) and directory setup live here. Always update this file rather than scattering magic numbers in scripts.

### Feature Space (36 features)
Symmetric player1/player2 representations of: ELO score, win/loss streak, weeks inactive, head-to-head record, handedness, seed, sinusoidal time encoding.

## Known Gaps
- `Glicko` class in `pipeline.py` is partially implemented — ELO is the active rating system.
- Betting strategy (Kelly Criterion) is referenced in the README and literature but not implemented.
- Odds data integration from tennis-data.co.uk is planned but absent.
- Some paths in scripts are hardcoded; prefer passing via CLI args.