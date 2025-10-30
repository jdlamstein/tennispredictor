# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a tennis match prediction system that uses machine learning to predict ATP tennis match winners. The project leverages ELO ratings (calculated relative to opponents with historical memory) and various player statistics to train both deep learning models (PyTorch Lightning MLP) and traditional classifiers. The ultimate goal is to predict match outcomes with sufficient accuracy for sports betting strategies.

Data source: Jeff Sackmann's tennis_atp repository (https://github.com/JeffSackmann/tennis_atp)

## Setup and Environment

Set PYTHONPATH to the project directory:
```bash
export PYTHONPATH=/path/to/tennispredictor/
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Common Development Commands

### Data Preprocessing
Clean and consolidate raw ATP data (must run first):
```bash
python preprocessing/clean_data.py --tennisdir "/path/to/tennis_atp" --savedir "/path/to/data/directory"
```

This script:
- Consolidates multiple CSV files from the tennis_atp repository
- Removes null values and converts strings to numeric
- Calculates ELO scores for all players (more predictive than ATP rankings)
- Computes temporal features (sine/cosine of day-of-year)
- Tracks win/loss streaks over 2-week windows
- Randomly assigns winners/losers to player1/player2 columns to prevent label leakage

### Training

Train neural network (PyTorch Lightning MLP):
```bash
python main/train.py --csv "/path/to/data/atp_database.csv" --rootdir "/path/to/data/directory"
```
- Uses Wandb for experiment tracking
- Adam optimizer with learning rate 1e-6
- Batch size 128
- Early stopping on validation accuracy
- GPU acceleration enabled

Train traditional classifiers:
```bash
python main/classifier.py --csv "/path/to/data/atp_database.csv" --rootdir "/path/to/data/directory"
```
- Trains: Nearest Neighbors, Linear SVM, Decision Tree, Random Forest, Neural Net, AdaBoost, Naive Bayes, QDA
- Saves classifiers as .joblib files timestamped
- Generates feature importance plots using permutation importance

### Prediction

Predict with trained classifiers:
```bash
python main/classifier.py --csv "/path/to/data/atp_database.csv" \
    --rootdir "/path/to/data/directory" \
    --timestring "timestring_of_trained_classifier" \
    --classifier_name "AdaBoost"
```

Deploy neural network on new data:
```bash
python main/deploy.py --csv /path/to/deploy.csv \
    --ckpt_path /path/to/model.ckpt \
    --rootdir /path/to/analysis/dir
```

### Generating Deployment Data

Create deploy.csv for upcoming matches:
```bash
python preprocessing/generate_deploy.py --csv "/path/to/data/atp_database.csv" --rootdir "/path/to/data/directory"
```

Edit the `pairings` list in generate_deploy.py to specify matchups using the Player_Pairings namedtuple:
```python
Player_Pairings(player1='Name1', player2='Name2', tourney_date='YYYYMMDD', surface=3, best_of=5, tourney_name="Tournament")
```

## Architecture

### Core Components

**param_tennis.py**: Central configuration object (Param class)
- Manages all paths (model_dir, data_dir, resources_dir, fig_dir, atp_dir)
- Stores hyperparameters (batch_size, epochs, learning_rate, optimizer)
- Auto-creates timestamped directories for experiment tracking
- Defines dataset splits: train (60%), validation (20%), test (20%)

**preprocessing/pipeline.py**: Data transformation pipeline with several key classes:
- `Elo`: Calculates and updates ELO ratings chronologically through all matches (K-factor varies by games played and rating level)
- `TimePeriod`: Converts dates to year + sinusoidal day-of-year features (sine_day, cosine_day)
- `RecentMatches`: Tracks win/loss streaks, weeks inactive, games in last 2 weeks, and head-to-head records
- `Dataspring`: Main data pipeline class that normalizes features using training set statistics (stored in meta.csv for deployment)

**preprocessing/clean_data.py**: ATP data consolidation
- Loads all CSVs from tennis_atp repo (matches, futures, quals)
- Converts winner/loser format to player1/player2 with game_winner label
- Parses categorical features (surface, tourney_level, hand, ioc, round, entry)
- Cleans scores (removes withdrawals, retirements, defaults)
- Randomly scrambles player1/player2 assignments to prevent label leakage

**models/model.py**: PyTorch Lightning MLP
- Input: 36 features per match
- Architecture: 256→256→256→128→2 (all ReLU activations)
- Output: 2-class log softmax (player1 wins vs player2 wins)
- Uses binary cross-entropy loss
- Tracks validation and test accuracy

**main/train.py**: Neural network training orchestration
- Integrates with Wandb for experiment tracking
- Uses early stopping (patience=4, monitors val_acc)
- Saves top 3 checkpoints based on val_loss
- GPU training enabled

**main/classifier.py**: Traditional ML classifiers
- Trains 8 different classifiers in one run
- Generates permutation importance plots for each classifier
- Saves models with timestamped directories
- Supports prediction mode when timestring provided

**preprocessing/generate_deploy.py**: Live match prediction data generator
- Extracts latest statistics for specified players from historical database
- Calculates days since last match, updated streaks
- Computes head-to-head records
- Outputs deploy.csv with identical feature format for inference

### Feature Engineering Strategy

The pipeline creates 36 features per match, focusing on:
1. **Player attributes**: hand, age, height, IOC (country), ELO
2. **Recent form**: winning streak, losing streak, weeks inactive, games in last 2 weeks
3. **Head-to-head**: player1_v_player2_wins, player2_v_player1_wins
4. **Match context**: surface, best_of, tourney_level, round, year, sine_day, cosine_day

Notably excluded (to prevent label leakage):
- Match statistics (aces, double faults, service points, break points)
- ATP rankings (ELO preferred as more predictive)
- Scores

### Data Flow

1. Raw CSV files → clean_data.py → consolidated atp_database.csv
2. atp_database.csv → Elo.populate_elo() → ELO columns added
3. Updated CSV → TimePeriod.run() → temporal features added
4. Updated CSV → RecentMatches.run() → streak/activity features added
5. Final CSV → Dataspring → normalized tensors → PyTorch DataLoader
6. For deployment: atp_database.csv → generate_deploy.py → deploy.csv → trained model

### Model Performance

Achieved test accuracies:
- Neural Network (PyTorch Lightning): 93.1%
- Naive Bayes: 93.0%
- Linear SVM: 92.7%
- AdaBoost: 92.6%
- Decision Tree: 92.0%
- QDA: 91.7%
- Random Forest: 87.1%
- Nearest Neighbors: 71.5%

Feature importance analysis shows winning/losing streaks are most predictive across all classifiers.

## Important Notes

- Always set PYTHONPATH before running any scripts
- The tennis_atp data directory must be downloaded separately from Jeff Sackmann's repo
- Data cleaning must run before training (it's a prerequisite)
- The meta.csv file (containing training set mean/std) is required for deployment predictions
- GPU training is enabled by default in train.py (set accelerator='cpu' if needed)
- Wandb account required for training with experiment tracking
- Timestrings format: YYYY_MM_DD_HH_MM_SS (used to organize model checkpoints)
- Surface codes: numeric mapping from original string values (Hard, Clay, Grass, Carpet)
