# Comprehensive Adversarial Attack and Defense Evaluation for O-RAN xApp ML Models

Extends [Chiejina et al. (ACM WiSec 2024)](https://dl.acm.org/doi/abs/10.1145/3643833.3656119) from 2 attacks and 1 defense to 7 adversarial attacks and 6 defense mechanisms on two O-RAN interference classification xApps (CNN for spectrograms, DNN for KPMs).

## Key Findings

- DeepFool reduces CNN accuracy to **5.0%** while FGSM only reaches 55.5% at the same perturbation budget
- AutoPGD exposes hidden DNN vulnerability (**45.6%**) that FGSM/PGD miss entirely (88.4%/85.6%)
- TRADES provides the strongest defense: **78.4% CNN** and **94.6% DNN** under PGD at epsilon=0.1
- Input transforms are counterproductive for KPM data (61.0% clean accuracy)
- MC-Dropout detection achieves only 7% to 29% TPR at 5% FPR

## Setup

```bash
python3 -m venv .venv
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
# 1. Preprocess data (generates .npy files in Dataset/)
python3 load_dataset.py

# 2. Base paper replication (FGSM, PGD, distillation, adversarial training)
nohup python3 -u replicate_base_paper.py > base.log 2>&1 &

# 3. Extended attacks (DeepFool, AutoPGD, JSMA, Boundary)
nohup python3 -u extended_attacks.py > attacks.log 2>&1 &

# 4. C&W L2 attack (fixed variant)
nohup python3 -u cw_attack_fix.py > cw.log 2>&1 &

# 5. Extended defenses (TRADES, Smoothing, Input Transforms, MC-Dropout for DNN + CNN)
nohup python3 -u extended_defenses.py > defenses.log 2>&1 &

# 6. Remaining CNN defenses (Smoothing, Input Transform, MC-Dropout evaluation)
nohup python3 -u missing_cnn_defenses.py > missing.log 2>&1 &

# 7. Generate report figures
python3 generate_figures.py
```

All scripts checkpoint results to `checkpoints/` and `extended_results/`. Re-running skips completed steps.

## Files

| File | Description |
|------|-------------|
| `load_dataset.py` | Data preprocessing: spectrogram loading, KPM JSON parsing, train/val/test splitting |
| `replicate_base_paper.py` | Base paper replication: CNN/DNN training, FGSM, PGD, distillation, adversarial training |
| `extended_attacks.py` | 5 extended attacks via ART: DeepFool, AutoPGD, C&W, JSMA, Boundary |
| `cw_attack_fix.py` | C&W L2 attack (fixed variant, replaces ineffective L-inf) |
| `extended_defenses.py` | 4 new defenses: TRADES, Randomized Smoothing, Input Transforms, MC-Dropout |
| `missing_cnn_defenses.py` | CNN evaluation for Smoothing, Input Transform, MC-Dropout |
| `test_representations.py` | Input representation comparison (grayscale vs RGB) |
| `generate_figures.py` | All report figures |

## Project Structure

```
.
├── load_dataset.py
├── replicate_base_paper.py
├── extended_attacks.py
├── cw_attack_fix.py
├── extended_defenses.py
├── missing_cnn_defenses.py
├── test_representations.py
├── generate_figures.py
├── Dataset/                  # Preprocessed .npy files
├── checkpoints/              # Trained models (.keras) and baseline results (.json)
├── extended_results/         # Attack and defense evaluation results (.json)
├── figures/                  # Generated report figures
├── newdataset/               # Raw spectrogram images (soi/, cwi/, ci/)
└── kpm-data2/                # Raw KPM JSON files
```

## Models

| Model | Architecture | Input | Parameters | Clean Accuracy |
|-------|-------------|-------|------------|----------------|
| CNN (InterClass-Spec) | 4 Conv2D + 3 MaxPool + Dense(32) | 128x128x3 RGB | 164,210 | 96.5% |
| DNN (InterClass-KPM) | Dense(64, 32, 16) | 60 features | 6,546 | 96.7% |

## Reference

O. Chiejina, B. Kim, K. Chowdhury, and V. K. Shah, "System-level Analysis of Adversarial Attacks and Defenses on Intelligence in O-RAN based Cellular Networks," in *Proc. ACM WiSec*, 2024, pp. 237-247.

Dataset: [nextgwirelesslab.org/datasets](https://www.nextgwirelesslab.org/datasets)
