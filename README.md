# Rigorous Detection of Shortcut Learning in Chest X-Ray Classifiers

CS229 project template for detecting shortcut learning in chest X-ray classifiers.

The project studies whether chest radiograph classifiers rely on non-medical cues such as
hospital markings, metadata overlays, laterality markers, support devices, and acquisition
artifacts instead of anatomical disease evidence.

## Project Structure

```text
.
├── configs/              # Small experiment configs
├── data/                 # Local datasets and generated metadata (gitignored)
├── notebooks/            # Exploration
├── src/
│   ├── data.py
│   ├── interpretability.py
│   ├── models.py
│   ├── shortcuts.py
│   ├── train.py
│   └── utils.py
└── tests/
```

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate cs229-shortcut-detection
```

If you update dependencies later, refresh the environment with:

```bash
conda env update -f environment.yml --prune
```

## Usage

Run the placeholder training entrypoint:

```bash
python -m src.train --config configs/default.yaml
```

Run tests:

```bash
pytest
```

## Project Ideas

- Baseline evaluation: train a chest X-ray classifier and report per-condition AUC.
- Saliency audit: compare Grad-CAM, Integrated Gradients, SHAP, or LIME attribution maps.
- Shortcut ablation: mask known shortcut regions and measure per-disease prediction shifts.
- Hospital stratification: compare hospital-specific and hospital-balanced training splits.
- Saliency shift analysis: compare pre/post stratification attention patterns.
- Shortcut leaderboard: rank diseases by shortcut reliance score with significance tests.
