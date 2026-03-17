# Adversarial Attack & Defense Evaluation for O-RAN xApps

Extends [Chiejina et al. (ACM WiSec 2024)](https://doi.org/10.1145/3643833.3656127) with 7 adversarial attacks and 2 defenses on O-RAN interference classification xApps (CNN for spectrograms, DNN for KPMs).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy matplotlib scikit-learn pillow adversarial-robustness-toolbox
```

### Dataset

Download from [nextgwirelesslab.org/datasets](https://www.nextgwirelesslab.org/datasets) and place files as:

```
data/spectrograms/{soi,cwi,ci}/*.png
data/kpms/{clean1..4,jammer1..6}.json
```

## Execution

Run in order:

```bash
# 1. Preprocess data → generates .npy files
python load_dataset.py

# 2. Base paper replication (FGSM, PGD, distillation, adversarial training)
#    Uncomment run_full_pipeline() at bottom of file first
nohup python -u replicate_base_paper.py > output.log 2>&1 &

# 3. Extended attacks (C&W, JSMA, DeepFool, AutoPGD, Boundary)
nohup python -u extended_attacks.py > extended.log 2>&1 &

# 4. Generate report figures
python generate_figures.py
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
| `milestone_report.tex` | LaTeX report |

## Reference

O. Chiejina et al., "System-level Analysis of Adversarial Attacks and Defenses on Intelligence in O-RAN based Cellular Networks," ACM WiSec, 2024.
