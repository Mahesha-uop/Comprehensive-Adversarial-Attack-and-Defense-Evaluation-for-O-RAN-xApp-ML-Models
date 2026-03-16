"""
EDA Visualizations for Milestone Report
========================================
Generates all figures for the report:
1. Sample spectrogram comparison (SOI vs CWI)
2. Pixel intensity distributions (grayscale & RGB)
3. KPM feature distributions (clean vs jammer)
4. Class balance charts
5. Attack comparison plots (all 7 attacks)
6. Defense effectiveness comparison
"""

import os
import glob
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================================
# CONFIG
# ============================================================================
SPEC_DIR = "./newdataset"
KPM_DIR = "./kpm-data2"
CKPT_DIR = "./checkpoints"
EXT_DIR = "./extended_results"
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


# ============================================================================
# 1. SAMPLE SPECTROGRAM COMPARISON
# ============================================================================

def fig1_sample_spectrograms():
    """Show 3 SOI vs 3 CWI sample spectrograms side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for row, (folder, label) in enumerate([('soi', 'SOI (No Interference)'),
                                            ('cwi', 'CWI (Interference)')]):
        pngs = sorted(glob.glob(os.path.join(SPEC_DIR, folder, "*.png")))
        # Pick 3 evenly spaced samples
        indices = [0, len(pngs)//2, len(pngs)-1]
        for col, idx in enumerate(indices):
            img = Image.open(pngs[idx])
            axes[row, col].imshow(np.array(img))
            axes[row, col].set_title(f"{label}\n{os.path.basename(pngs[idx])}", fontsize=10)
            axes[row, col].axis('off')

    plt.suptitle("Sample Spectrograms: SOI vs CWI", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_sample_spectrograms.png"))
    plt.close()
    print("  Saved fig1_sample_spectrograms.png")


# ============================================================================
# 2. PIXEL INTENSITY DISTRIBUTIONS
# ============================================================================

def fig2_pixel_distributions():
    """Compare pixel distributions for SOI vs CWI (RGB channels + grayscale)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for row, (folder, label) in enumerate([('soi', 'SOI'), ('cwi', 'CWI')]):
        pngs = sorted(glob.glob(os.path.join(SPEC_DIR, folder, "*.png")))
        # Sample 500 images for speed
        sample_idx = np.random.choice(len(pngs), min(500, len(pngs)), replace=False)

        all_r, all_g, all_b, all_gray = [], [], [], []
        for idx in sample_idx:
            img = np.array(Image.open(pngs[idx]))
            if img.ndim == 3:
                all_r.extend(img[:,:,0].flatten()[::10])
                all_g.extend(img[:,:,1].flatten()[::10])
                all_b.extend(img[:,:,2].flatten()[::10])
            gray = np.array(Image.open(pngs[idx]).convert('L'))
            all_gray.extend(gray.flatten()[::10])

        for col, (data, ch_name, color) in enumerate([
            (all_r, 'Red', 'red'), (all_g, 'Green', 'green'),
            (all_b, 'Blue', 'blue'), (all_gray, 'Grayscale', 'gray')
        ]):
            axes[row, col].hist(data, bins=50, color=color, alpha=0.7, density=True)
            axes[row, col].set_title(f"{label} - {ch_name}")
            axes[row, col].set_xlabel("Pixel Value")
            if col == 0:
                axes[row, col].set_ylabel("Density")

    plt.suptitle("Pixel Intensity Distributions by Channel", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_pixel_distributions.png"))
    plt.close()
    print("  Saved fig2_pixel_distributions.png")


# ============================================================================
# 3. KPM FEATURE DISTRIBUTIONS
# ============================================================================

def fig3_kpm_distributions():
    """Distribution of 4 KPM features: clean vs jammer (before normalization)."""
    feature_keys = ['ul_snr', 'ul_bitrate', 'ul_bler', 'ul_mcs']
    feature_labels = ['UL SNR', 'UL Bitrate', 'UL BLER', 'UL MCS']

    clean_data = {k: [] for k in feature_keys}
    jammer_data = {k: [] for k in feature_keys}

    json_files = sorted(glob.glob(os.path.join(KPM_DIR, "*.json")))

    for jf in json_files:
        is_clean = 'clean' in os.path.basename(jf).lower()
        import json as json_mod
        with open(jf, 'r') as f:
            data = json_mod.load(f)

        for entry in data:
            if entry.get('type') != 'metrics':
                continue
            for cell in entry.get('cell_list', []):
                cc = cell.get('cell_container', {})
                for ue in cc.get('ue_list', []):
                    ue_data = ue.get('ue_container', ue)
                    target = clean_data if is_clean else jammer_data
                    for k in feature_keys:
                        try:
                            target[k].append(float(ue_data.get(k, 0)))
                        except (ValueError, TypeError):
                            pass

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, (key, label) in enumerate(zip(feature_keys, feature_labels)):
        ax = axes[i]
        c_vals = np.array(clean_data[key])
        j_vals = np.array(jammer_data[key])

        # Use reasonable bins
        all_vals = np.concatenate([c_vals, j_vals])
        bins = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)

        ax.hist(c_vals, bins=bins, alpha=0.6, label=f'Clean (n={len(c_vals)})',
                color='#2196F3', density=True)
        ax.hist(j_vals, bins=bins, alpha=0.6, label=f'Jammer (n={len(j_vals)})',
                color='#F44336', density=True)
        ax.set_title(label)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        # Add stats annotation
        ax.annotate(f"Clean: μ={c_vals.mean():.2f}, σ={c_vals.std():.2f}\n"
                    f"Jammer: μ={j_vals.mean():.2f}, σ={j_vals.std():.2f}",
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.suptitle("KPM Feature Distributions: Clean vs Jammer (Raw Values)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_kpm_distributions.png"))
    plt.close()
    print("  Saved fig3_kpm_distributions.png")


# ============================================================================
# 4. CLASS BALANCE
# ============================================================================

def fig4_class_balance():
    """Class distribution for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Spectrogram
    soi_count = len(glob.glob(os.path.join(SPEC_DIR, "soi", "*.png")))
    cwi_count = len(glob.glob(os.path.join(SPEC_DIR, "cwi", "*.png")))
    ci_count = len(glob.glob(os.path.join(SPEC_DIR, "ci", "*.png")))

    bars = axes[0].bar(['SOI\n(No Interference)', 'CWI\n(Interference)', 'CI\n(Unused)'],
                        [soi_count, cwi_count, ci_count],
                        color=['#2196F3', '#F44336', '#9E9E9E'])
    for bar, count in zip(bars, [soi_count, cwi_count, ci_count]):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                     str(count), ha='center', fontweight='bold')
    axes[0].set_title("Spectrogram Dataset")
    axes[0].set_ylabel("Number of Samples")

    # KPM
    json_files = sorted(glob.glob(os.path.join(KPM_DIR, "*.json")))
    clean_count, jammer_count = 0, 0
    for jf in json_files:
        import json as json_mod
        with open(jf, 'r') as f:
            data = json_mod.load(f)
        n = sum(1 for d in data if d.get('type') == 'metrics'
                for c in d.get('cell_list', [])
                for u in c.get('cell_container', {}).get('ue_list', []))
        if 'clean' in os.path.basename(jf).lower():
            clean_count += n
        else:
            jammer_count += n

    bars = axes[1].bar(['Clean\n(No Interference)', 'Jammer\n(Interference)'],
                        [clean_count, jammer_count],
                        color=['#2196F3', '#F44336'])
    for bar, count in zip(bars, [clean_count, jammer_count]):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                     str(count), ha='center', fontweight='bold')
    axes[1].set_title("KPM Dataset")
    axes[1].set_ylabel("Number of UE Measurements")

    plt.suptitle("Dataset Class Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_class_balance.png"))
    plt.close()
    print("  Saved fig4_class_balance.png")


# ============================================================================
# 5. ATTACK COMPARISON — CNN
# ============================================================================

def fig5_cnn_all_attacks():
    """CNN accuracy vs epsilon for all available attacks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load base results
    base_path = os.path.join(CKPT_DIR, "cnn_attack_results.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = json.load(f)
        eps = base['epsilon']
        ax.plot(eps, [base['clean'][0]*100]*len(eps), 'k-o', label='No Attack',
                linewidth=2, markersize=5)
        ax.plot(eps, [a*100 for a in base['fgsm']], 'r-s', label='FGSM', markersize=4)
        ax.plot(eps, [a*100 for a in base['pgd']], 'b-^', label='PGD (5-step)', markersize=4)

    # Load extended results
    ext_path = os.path.join(EXT_DIR, "cnn_extended_epsilon.json")
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext = json.load(f)
        eps = ext['epsilon']

        if ext.get('cw') and any(v is not None for v in ext['cw']):
            vals = [v*100 if v is not None else None for v in ext['cw']]
            valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
            if valid:
                ax.plot(*zip(*valid), 'g-D', label='C&W L-inf', markersize=4)

        if ext.get('deepfool') and any(v is not None for v in ext['deepfool']):
            vals = [v*100 if v is not None else None for v in ext['deepfool']]
            valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
            if valid:
                ax.plot(*zip(*valid), 'm-v', label='DeepFool', markersize=5, linewidth=2)

        if ext.get('autoattack') and any(v is not None for v in ext['autoattack']):
            vals = [v*100 if v is not None else None for v in ext['autoattack']]
            valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
            if valid:
                ax.plot(*zip(*valid), 'c-P', label='AutoPGD', markersize=5, linewidth=2)

    ax.set_xlabel('Perturbation Budget (ε)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('CNN (InterClass-Spec xApp): All Attacks Comparison', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, 0.105)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_cnn_all_attacks.png"))
    plt.close()
    print("  Saved fig5_cnn_all_attacks.png")


# ============================================================================
# 6. ATTACK COMPARISON — DNN
# ============================================================================

def fig6_dnn_all_attacks():
    """DNN accuracy vs epsilon for all available attacks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    base_path = os.path.join(CKPT_DIR, "dnn_attack_results.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = json.load(f)
        eps = base['epsilon']
        ax.plot(eps, [base['clean'][0]*100]*len(eps), 'k-o', label='No Attack',
                linewidth=2, markersize=5)
        ax.plot(eps, [a*100 for a in base['fgsm']], 'r-s', label='FGSM', markersize=4)
        ax.plot(eps, [a*100 for a in base['pgd']], 'b-^', label='PGD (5-step)', markersize=4)

    ext_path = os.path.join(EXT_DIR, "dnn_extended_epsilon.json")
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            ext = json.load(f)
        eps = ext['epsilon']

        if ext.get('cw') and any(v is not None for v in ext['cw']):
            vals = [v*100 if v is not None else None for v in ext['cw']]
            valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
            if valid:
                ax.plot(*zip(*valid), 'g-D', label='C&W L-inf', markersize=4)

        if ext.get('deepfool') and any(v is not None for v in ext['deepfool']):
            vals = [v*100 if v is not None else None for v in ext['deepfool']]
            valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
            if valid:
                ax.plot(*zip(*valid), 'm-v', label='DeepFool', markersize=5, linewidth=2)

        if ext.get('autoattack') and any(v is not None for v in ext['autoattack']):
            vals = [v*100 if v is not None else None for v in ext['autoattack']]
            valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
            if valid:
                ax.plot(*zip(*valid), 'c-P', label='AutoPGD', markersize=5, linewidth=2)

    ax.set_xlabel('Perturbation Budget (ε)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('DNN (InterClass-KPM xApp): All Attacks Comparison', fontsize=13)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, 0.105)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_dnn_all_attacks.png"))
    plt.close()
    print("  Saved fig6_dnn_all_attacks.png")


# ============================================================================
# 7. DEFENSE EFFECTIVENESS — CNN
# ============================================================================

def fig7_cnn_defenses():
    """CNN: Compare no defense, distillation, adversarial training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    files = {
        'No Defense': os.path.join(CKPT_DIR, "cnn_attack_results.json"),
        'Distillation': os.path.join(CKPT_DIR, "cnn_distill_results.json"),
        'Adv. Training': os.path.join(CKPT_DIR, "cnn_advtrain_results.json"),
    }

    colors = {'No Defense': '#F44336', 'Distillation': '#4CAF50', 'Adv. Training': '#FF9800'}
    markers = {'No Defense': 's', 'Distillation': 'D', 'Adv. Training': '^'}

    for atk_idx, atk_type in enumerate(['fgsm', 'pgd']):
        ax = axes[atk_idx]
        for name, path in files.items():
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                eps = data['epsilon']
                ax.plot(eps, [a*100 for a in data[atk_type]],
                        f'-{markers[name]}', color=colors[name],
                        label=name, markersize=5)

        ax.set_xlabel('Perturbation Budget (ε)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'CNN under {atk_type.upper()} Attack')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 100)

    plt.suptitle("CNN Defense Effectiveness: Distillation vs Adversarial Training",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_cnn_defenses.png"))
    plt.close()
    print("  Saved fig7_cnn_defenses.png")


# ============================================================================
# 8. DEFENSE EFFECTIVENESS — DNN
# ============================================================================

def fig8_dnn_defenses():
    """DNN: Compare no defense, distillation, adversarial training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    files = {
        'No Defense': os.path.join(CKPT_DIR, "dnn_attack_results.json"),
        'Distillation': os.path.join(CKPT_DIR, "dnn_distill_results.json"),
        'Adv. Training': os.path.join(CKPT_DIR, "dnn_advtrain_results.json"),
    }

    colors = {'No Defense': '#F44336', 'Distillation': '#4CAF50', 'Adv. Training': '#FF9800'}
    markers = {'No Defense': 's', 'Distillation': 'D', 'Adv. Training': '^'}

    for atk_idx, atk_type in enumerate(['fgsm', 'pgd']):
        ax = axes[atk_idx]
        for name, path in files.items():
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                eps = data['epsilon']
                ax.plot(eps, [a*100 for a in data[atk_type]],
                        f'-{markers[name]}', color=colors[name],
                        label=name, markersize=5)

        ax.set_xlabel('Perturbation Budget (ε)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'DNN under {atk_type.upper()} Attack')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(75, 100)

    plt.suptitle("DNN Defense Effectiveness: Distillation vs Adversarial Training",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_dnn_defenses.png"))
    plt.close()
    print("  Saved fig8_dnn_defenses.png")


# ============================================================================
# 9. PREPROCESSING PIPELINE DIAGRAM (text-based for report)
# ============================================================================

def fig9_input_representation_comparison():
    """Compare grayscale vs RGB accuracy to show representation effect."""
    # From your test_representations.py results
    configs = ['Grayscale\n128×128', 'Grayscale\n256×256', 'RGB\n128×128']
    accuracies = [96.81, 94.87, 96.11]  # From your test results
    params = [163922, 819282, 164210]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(configs, accuracies, color=['#607D8B', '#455A64', '#2196F3'])
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('CNN Accuracy by Input Representation')
    ax1.set_ylim(90, 100)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{acc:.1f}%', ha='center', fontweight='bold')

    bars2 = ax2.bar(configs, [p/1000 for p in params], color=['#607D8B', '#455A64', '#2196F3'])
    ax2.set_ylabel('Parameters (thousands)')
    ax2.set_title('Model Parameters by Input Representation')
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 f'{p:,}', ha='center', fontweight='bold', fontsize=9)

    plt.suptitle("Impact of Input Representation on Model Performance",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_representation_comparison.png"))
    plt.close()
    print("  Saved fig9_representation_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Generating EDA figures...")
    print()

    fig1_sample_spectrograms()
    fig2_pixel_distributions()
    fig3_kpm_distributions()
    fig4_class_balance()
    fig5_cnn_all_attacks()
    fig6_dnn_all_attacks()
    fig7_cnn_defenses()
    fig8_dnn_defenses()
    fig9_input_representation_comparison()

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Figures ready for milestone report.")