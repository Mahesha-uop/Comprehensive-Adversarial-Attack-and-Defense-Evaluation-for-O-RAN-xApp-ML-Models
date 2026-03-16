"""
Extended Adversarial Attacks on O-RAN xApp Models
=================================================
Attacks: C&W, JSMA, DeepFool, AutoAttack, Boundary Attack
Models: CNN (spectrograms) + DNN (KPMs)

Uses IBM ART (Adversarial Robustness Toolbox) for all attacks.
Install: pip install adversarial-robustness-toolbox

Loads baseline models from checkpoints saved by replicate_base_paper.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# ART imports
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import (
    CarliniLInfMethod,
    SaliencyMapMethod,
    DeepFool,
    BoundaryAttack,
    FastGradientMethod,
    ProjectedGradientDescent,
    AutoProjectedGradientDescent,
)

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

CKPT_DIR = "./checkpoints"
RESULTS_DIR = "./extended_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

# Limit test samples for slow attacks (C&W, Boundary, JSMA)
# Full test set for fast attacks (DeepFool, AutoAttack)
N_SAMPLES_SLOW = 200   # C&W, Boundary, JSMA — very slow per-sample
N_SAMPLES_FAST = 500   # DeepFool, AutoAttack — moderately fast
N_SAMPLES_ALL = None    # FGSM, PGD — use full test set


# ============================================================================
# DATA LOADING
# ============================================================================

def load_spectrogram_splits():
    X_train = np.load("X_spec_train.npy")
    y_train = np.load("y_spec_train.npy")
    X_val = np.load("X_spec_val.npy")
    y_val = np.load("y_spec_val.npy")
    X_test = np.load("X_spec_test.npy")
    y_test = np.load("y_spec_test.npy")
    print(f"  Spec: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_kpm_splits():
    X_train = np.load("X_kpm_train.npy")
    y_train = np.load("y_kpm_train.npy")
    X_val = np.load("X_kpm_val.npy")
    y_val = np.load("y_kpm_val.npy")
    X_test = np.load("X_kpm_test.npy")
    y_test = np.load("y_kpm_test.npy")
    print(f"  KPM: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# ART CLASSIFIER WRAPPER
# ============================================================================

def wrap_model_art(model, input_shape, nb_classes=2):
    """Wrap a Keras model as an ART classifier."""
    loss_fn = keras.losses.CategoricalCrossentropy()

    classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=nb_classes,
        input_shape=input_shape,
        loss_object=loss_fn,
        clip_values=(0.0, 1.0),
    )
    return classifier


# ============================================================================
# ATTACK FUNCTIONS
# ============================================================================

def evaluate_accuracy(model, X, y):
    """Compute accuracy manually."""
    preds = []
    for i in range(0, len(X), 256):
        batch = tf.convert_to_tensor(X[i:i+256], dtype=tf.float32)
        p = model(batch, training=False).numpy()
        preds.append(p)
    preds = np.concatenate(preds, axis=0)
    y_pred = np.argmax(preds, axis=1)
    return float(np.mean(y_pred == y))


def run_fgsm_art(classifier, X_test, y_test, epsilon):
    """FGSM via ART (targeted, class 0)."""
    attack = FastGradientMethod(
        estimator=classifier,
        eps=epsilon,
        targeted=True,
        batch_size=128,
    )
    # Target: all class 0
    y_target = to_categorical(np.zeros(len(X_test), dtype=int), 2)
    X_adv = attack.generate(x=X_test, y=y_target)
    return X_adv


def run_pgd_art(classifier, X_test, y_test, epsilon):
    """PGD via ART (targeted, class 0)."""
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=epsilon,
        eps_step=2.0 * epsilon / 5,
        max_iter=5,
        targeted=True,
        batch_size=128,
    )
    y_target = to_categorical(np.zeros(len(X_test), dtype=int), 2)
    X_adv = attack.generate(x=X_test, y=y_target)
    return X_adv


def run_cw_attack(classifier, X_test, y_test, epsilon):
    """
    Carlini & Wagner L-inf attack.
    Uses CarliniLInfMethod — optimizes perturbation, then clips to epsilon.
    """
    attack = CarliniLInfMethod(
        classifier=classifier,
        confidence=0.5,
        max_iter=30,
        learning_rate=0.01,
        batch_size=16,
        targeted=True,
        verbose=False,
    )
    y_target = to_categorical(np.zeros(len(X_test), dtype=int), 2)
    X_adv = attack.generate(x=X_test, y=y_target)

    # Clip to epsilon L-inf ball
    perturbation = np.clip(X_adv - X_test, -epsilon, epsilon)
    X_adv = np.clip(X_test + perturbation, 0.0, 1.0)
    return X_adv


def run_jsma_attack(classifier, X_test, y_test):
    """
    JSMA (Jacobian-based Saliency Map Attack).
    Note: JSMA doesn't use epsilon — it modifies individual features
    until misclassification. We run it once and measure success rate.
    """
    attack = SaliencyMapMethod(
        classifier=classifier,
        theta=0.1,       # Perturbation per step
        gamma=0.1,       # Max fraction of features modified
        batch_size=1,    # JSMA is per-sample
        verbose=False,
    )
    X_adv = attack.generate(x=X_test)
    return X_adv


def run_deepfool_attack(classifier, X_test, y_test, epsilon=None):
    """
    DeepFool attack — finds minimal perturbation.
    Epsilon used as max perturbation clip after generation.
    """
    attack = DeepFool(
        classifier=classifier,
        max_iter=10,
        epsilon=1e-6,
        nb_grads=2,
        batch_size=32,
        verbose=False,
    )
    X_adv = attack.generate(x=X_test)

    # Clip perturbation to epsilon if specified
    if epsilon is not None:
        perturbation = X_adv - X_test
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        X_adv = np.clip(X_test + perturbation, 0.0, 1.0)

    return X_adv


def run_autoattack(classifier, X_test, y_test, epsilon):
    """AutoPGD — adaptive step-size PGD, stronger than standard PGD."""
    attack = AutoProjectedGradientDescent(
        estimator=classifier,
        eps=epsilon,
        eps_step=2.0 * epsilon / 10,
        max_iter=20,
        batch_size=32,
        targeted=False,
        nb_random_init=3,
        loss_type='cross_entropy',
        verbose=False,
    )
    X_adv = attack.generate(x=X_test, y=to_categorical(y_test, 2))
    return X_adv


def run_boundary_attack(classifier, X_test, y_test):
    """
    Boundary Attack — decision-based, no gradients needed.
    Does not use epsilon — finds minimal perturbation.
    """
    attack = BoundaryAttack(
        estimator=classifier,
        targeted=True,
        max_iter=100,      # Reduced for speed
        delta=0.01,
        epsilon=0.01,
        batch_size=1,
        verbose=False,
    )
    y_target = to_categorical(np.zeros(len(X_test), dtype=int), 2)
    X_adv = attack.generate(x=X_test, y=y_target)
    return X_adv


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def save_attack_results(results, name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED] {path}")


def load_attack_results(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def evaluate_epsilon_attacks(classifier, model, X_test, y_test,
                             epsilon_values, model_name, n_samples=None):
    """
    Run FGSM, PGD, C&W, DeepFool, AutoAttack across epsilon range.
    """
    if n_samples and n_samples < len(X_test):
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sub = X_test[idx]
        y_sub = y_test[idx]
    else:
        X_sub = X_test
        y_sub = y_test

    clean_acc = evaluate_accuracy(model, X_sub, y_sub)
    print(f"\n  {model_name} Clean Accuracy: {clean_acc*100:.2f}% (n={len(X_sub)})")

    results = {
        'epsilon': epsilon_values,
        'clean': clean_acc,
        'n_samples': len(X_sub),
        'fgsm': [], 'pgd': [], 'cw': [], 'deepfool': [], 'autoattack': []
    }

    print(f"\n  {'Eps':<7} {'FGSM':<9} {'PGD':<9} {'C&W':<9} {'DeepFool':<9} {'AutoAtk':<9}")
    print("  " + "-" * 52)

    for eps in epsilon_values:
        row = {}

        # FGSM
        try:
            X_adv = run_fgsm_art(classifier, X_sub, y_sub, eps)
            acc = evaluate_accuracy(model, X_adv, y_sub)
            results['fgsm'].append(acc)
            row['fgsm'] = f"{acc*100:.1f}"
        except Exception as e:
            results['fgsm'].append(None)
            row['fgsm'] = "ERR"
            print(f"  FGSM error at eps={eps}: {e}")

        # PGD
        try:
            X_adv = run_pgd_art(classifier, X_sub, y_sub, eps)
            acc = evaluate_accuracy(model, X_adv, y_sub)
            results['pgd'].append(acc)
            row['pgd'] = f"{acc*100:.1f}"
        except Exception as e:
            results['pgd'].append(None)
            row['pgd'] = "ERR"
            print(f"  PGD error at eps={eps}: {e}")

        # C&W
        try:
            X_adv = run_cw_attack(classifier, X_sub, y_sub, eps)
            acc = evaluate_accuracy(model, X_adv, y_sub)
            results['cw'].append(acc)
            row['cw'] = f"{acc*100:.1f}"
        except Exception as e:
            results['cw'].append(None)
            row['cw'] = "ERR"
            print(f"  C&W error at eps={eps}: {e}")

        # DeepFool (clipped to epsilon)
        try:
            X_adv = run_deepfool_attack(classifier, X_sub, y_sub, eps)
            acc = evaluate_accuracy(model, X_adv, y_sub)
            results['deepfool'].append(acc)
            row['deepfool'] = f"{acc*100:.1f}"
        except Exception as e:
            results['deepfool'].append(None)
            row['deepfool'] = "ERR"
            print(f"  DeepFool error at eps={eps}: {e}")

        # AutoAttack
        try:
            X_adv = run_autoattack(classifier, X_sub, y_sub, eps)
            acc = evaluate_accuracy(model, X_adv, y_sub)
            results['autoattack'].append(acc)
            row['autoattack'] = f"{acc*100:.1f}"
        except Exception as e:
            results['autoattack'].append(None)
            row['autoattack'] = "ERR"
            print(f"  AutoAttack error at eps={eps}: {e}")

        print(f"  {eps:<7.2f} {row.get('fgsm',''):<9} {row.get('pgd',''):<9} "
              f"{row.get('cw',''):<9} {row.get('deepfool',''):<9} {row.get('autoattack',''):<9}")

    return results


def evaluate_non_epsilon_attacks(classifier, model, X_test, y_test,
                                  model_name, n_samples=None):
    """
    Run JSMA and Boundary Attack (these don't take epsilon parameter).
    """
    if n_samples and n_samples < len(X_test):
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sub = X_test[idx]
        y_sub = y_test[idx]
    else:
        X_sub = X_test
        y_sub = y_test

    clean_acc = evaluate_accuracy(model, X_sub, y_sub)
    results = {'clean': clean_acc, 'n_samples': len(X_sub)}

    # JSMA
    print(f"\n  Running JSMA on {len(X_sub)} samples...")
    try:
        X_adv = run_jsma_attack(classifier, X_sub, y_sub)
        jsma_acc = evaluate_accuracy(model, X_adv, y_sub)
        results['jsma'] = jsma_acc

        # Measure perturbation magnitude
        perturbation = X_adv - X_sub
        results['jsma_mean_l2'] = float(np.mean(np.sqrt(np.sum(perturbation**2, axis=tuple(range(1, perturbation.ndim))))))
        results['jsma_mean_linf'] = float(np.mean(np.max(np.abs(perturbation), axis=tuple(range(1, perturbation.ndim)))))

        print(f"  JSMA: {clean_acc*100:.1f}% -> {jsma_acc*100:.1f}% "
              f"(mean L-inf: {results['jsma_mean_linf']:.4f})")
    except Exception as e:
        results['jsma'] = None
        print(f"  JSMA error: {e}")

    # Boundary Attack
    print(f"\n  Running Boundary Attack on {len(X_sub)} samples...")
    try:
        X_adv = run_boundary_attack(classifier, X_sub, y_sub)
        boundary_acc = evaluate_accuracy(model, X_adv, y_sub)
        results['boundary'] = boundary_acc

        perturbation = X_adv - X_sub
        results['boundary_mean_l2'] = float(np.mean(np.sqrt(np.sum(perturbation**2, axis=tuple(range(1, perturbation.ndim))))))
        results['boundary_mean_linf'] = float(np.mean(np.max(np.abs(perturbation), axis=tuple(range(1, perturbation.ndim)))))

        print(f"  Boundary: {clean_acc*100:.1f}% -> {boundary_acc*100:.1f}% "
              f"(mean L-inf: {results['boundary_mean_linf']:.4f})")
    except Exception as e:
        results['boundary'] = None
        print(f"  Boundary error: {e}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_cnn_extended():
    """Extended attacks on CNN (spectrograms)."""
    print("\n" + "=" * 70)
    print("EXTENDED ATTACKS: CNN (InterClass-Spec xApp)")
    print("=" * 70)

    # Load data
    X_tr, X_v, X_te, y_tr, y_v, y_te = load_spectrogram_splits()

    # Load baseline CNN
    print("\n  Loading CNN baseline model...")
    cnn = keras.models.load_model(os.path.join(CKPT_DIR, "cnn_baseline.keras"))
    input_shape = X_te.shape[1:]  # (128, 128, 3)

    # Wrap for ART
    classifier = wrap_model_art(cnn, input_shape)

    # Epsilon-based attacks (FGSM, PGD, C&W, DeepFool, AutoAttack)
    result_name = "cnn_extended_epsilon"
    results = load_attack_results(result_name)
    if results is None:
        print("\n  Running epsilon-based attacks on CNN...")
        results = evaluate_epsilon_attacks(
            classifier, cnn, X_te, y_te,
            EPSILON_VALUES, "CNN", n_samples=N_SAMPLES_SLOW
        )
        save_attack_results(results, result_name)
    else:
        print(f"  [LOADED] {result_name}")

    # Non-epsilon attacks (JSMA, Boundary)
    result_name = "cnn_non_epsilon"
    ne_results = load_attack_results(result_name)
    if ne_results is None:
        print("\n  Running non-epsilon attacks on CNN...")
        ne_results = evaluate_non_epsilon_attacks(
            classifier, cnn, X_te, y_te,
            "CNN", n_samples=N_SAMPLES_SLOW
        )
        save_attack_results(ne_results, result_name)
    else:
        print(f"  [LOADED] {result_name}")

    return results, ne_results


def run_dnn_extended():
    """Extended attacks on DNN (KPMs)."""
    print("\n" + "=" * 70)
    print("EXTENDED ATTACKS: DNN (InterClass-KPM xApp)")
    print("=" * 70)

    # Load data
    X_tr, X_v, X_te, y_tr, y_v, y_te = load_kpm_splits()

    # Load baseline DNN
    print("\n  Loading DNN baseline model...")
    dnn = keras.models.load_model(os.path.join(CKPT_DIR, "dnn_baseline.keras"))
    input_shape = X_te.shape[1:]  # (60,)

    # Wrap for ART
    classifier = wrap_model_art(dnn, input_shape)

    # Epsilon-based attacks
    result_name = "dnn_extended_epsilon"
    results = load_attack_results(result_name)
    if results is None:
        print("\n  Running epsilon-based attacks on DNN...")
        results = evaluate_epsilon_attacks(
            classifier, dnn, X_te, y_te,
            EPSILON_VALUES, "DNN", n_samples=N_SAMPLES_FAST
        )
        save_attack_results(results, result_name)
    else:
        print(f"  [LOADED] {result_name}")

    # Non-epsilon attacks (JSMA, Boundary)
    result_name = "dnn_non_epsilon"
    ne_results = load_attack_results(result_name)
    if ne_results is None:
        print("\n  Running non-epsilon attacks on DNN...")
        ne_results = evaluate_non_epsilon_attacks(
            classifier, dnn, X_te, y_te,
            "DNN", n_samples=N_SAMPLES_SLOW
        )
        save_attack_results(ne_results, result_name)
    else:
        print(f"  [LOADED] {result_name}")

    return results, ne_results


if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED ADVERSARIAL ATTACK EVALUATION")
    print("7 attacks × 2 models × 10 epsilon values")
    print("=" * 70)

    # Install ART if needed
    try:
        import art
        print(f"ART version: {art.__version__}")
    except ImportError:
        print("Installing adversarial-robustness-toolbox...")
        import subprocess
        subprocess.check_call([
            "pip", "install", "adversarial-robustness-toolbox",
            "--break-system-packages"
        ])
        import art
        print(f"ART version: {art.__version__}")

    cnn_eps, cnn_ne = run_cnn_extended()
    dnn_eps, dnn_ne = run_dnn_extended()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY AT eps=0.1")
    print("=" * 70)

    print("\nCNN (Spectrograms):")
    print(f"  Clean: {cnn_eps['clean']*100:.1f}%")
    for atk in ['fgsm', 'pgd', 'cw', 'deepfool', 'autoattack']:
        val = cnn_eps[atk][-1] if cnn_eps[atk][-1] is not None else 'ERR'
        if isinstance(val, float):
            print(f"  {atk:12s}: {val*100:.1f}%")
        else:
            print(f"  {atk:12s}: {val}")
    if cnn_ne.get('jsma') is not None:
        print(f"  {'jsma':12s}: {cnn_ne['jsma']*100:.1f}%")
    if cnn_ne.get('boundary') is not None:
        print(f"  {'boundary':12s}: {cnn_ne['boundary']*100:.1f}%")

    print("\nDNN (KPMs):")
    print(f"  Clean: {dnn_eps['clean']*100:.1f}%")
    for atk in ['fgsm', 'pgd', 'cw', 'deepfool', 'autoattack']:
        val = dnn_eps[atk][-1] if dnn_eps[atk][-1] is not None else 'ERR'
        if isinstance(val, float):
            print(f"  {atk:12s}: {val*100:.1f}%")
        else:
            print(f"  {atk:12s}: {val}")
    if dnn_ne.get('jsma') is not None:
        print(f"  {'jsma':12s}: {dnn_ne['jsma']*100:.1f}%")
    if dnn_ne.get('boundary') is not None:
        print(f"  {'boundary':12s}: {dnn_ne['boundary']*100:.1f}%")