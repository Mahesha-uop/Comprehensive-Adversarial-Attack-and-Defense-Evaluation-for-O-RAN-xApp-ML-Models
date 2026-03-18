# Adversarial Attacks and Robust Defenses for ML Intelligence in O-RAN Cellular Networks

Extends [Chiejina et al. (ACM WiSec 2024)](https://dl.acm.org/doi/abs/10.1145/3643833.3656119) with 7 adversarial attacks and 2 defenses on O-RAN interference classification xApps (CNN for spectrograms, DNN for KPMs).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy matplotlib scikit-learn pillow adversarial-robustness-toolbox
```

### Dataset

Download from [nextgwirelesslab.org/datasets](https://www.nextgwirelesslab.org/datasets) and place files as:

```
newdataset/{soi,cwi,ci}/*.png
kpm-data2/{clean1..4,jammer1..6}.json
```

## Execution

Run in order:

```bash
# 1. Preprocess data → generates .npy files
python load_dataset.py

# 2. Base paper replication (FGSM, PGD, distillation, adversarial training)
python replicate_base_paper.py 

# 3. Extended attacks 
python extended_attacks.py
```

Models and results checkpoint to `checkpoints/` and `extended_results/`. Re-running skips completed steps.

## Files

| File | Description |
|------|-------------|
| `load_dataset.py` | Data preprocessing and train/val/test splitting |
| `replicate_base_paper.py` | Baseline replication with checkpointing |
| `extended_attacks.py` | 7-attack evaluation via IBM ART |
| `test_representations.py` | Input representation comparison (grayscale vs RGB) |
| `generate_figures.py` | All report figures |

## Reference

O. Chiejina et al., "System-level Analysis of Adversarial Attacks and Defenses on Intelligence in O-RAN based Cellular Networks," ACM WiSec, 2024.
