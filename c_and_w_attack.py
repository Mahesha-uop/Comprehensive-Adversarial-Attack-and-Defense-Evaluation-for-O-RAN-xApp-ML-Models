"""
Fixed C&W Attack using L2 variant (CarliniL2Method)
====================================================
The L-inf variant was ineffective. L2 is the original and stronger variant.
Runs C&W L2 once, then evaluates at each epsilon by checking if
the perturbation's L-inf falls within the budget.

Requires: adversarial-robustness-toolbox, saved baseline models in checkpoints/
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import CarliniL2Method

np.random.seed(42)
tf.random.set_seed(42)

CKPT_DIR = "./checkpoints"
RESULTS_DIR = "./extended_results"
DATA_DIR = "./Dataset"
os.makedirs(RESULTS_DIR, exist_ok=True)

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
N_SAMPLES = 200


def evaluate_accuracy(model, X, y):
    preds = []
    for i in range(0, len(X), 256):
        batch = tf.convert_to_tensor(X[i:i+256], dtype=tf.float32)
        preds.append(model(batch, training=False).numpy())
    preds = np.concatenate(preds, axis=0)
    return float(np.mean(np.argmax(preds, axis=1) == y))


def save_results(results, name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [SAVED] {path}")


def load_results(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def run_cw_l2(model_name, model_path, X_test_path, y_test_path):
    """
    Run C&W L2 attack on a model.
    Generate adversarial examples once, then evaluate at each epsilon
    by keeping only perturbations within the L-inf budget.
    """
    result_name = f"{model_name.lower()}_cw_l2_results"
    existing = load_results(result_name)
    if existing is not None:
        print(f"\n  [LOADED] {result_name}")
        return existing

    print(f"\n  --- {model_name} ---")
    model = keras.models.load_model(os.path.join(CKPT_DIR, model_path))
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # Subsample
    idx = np.random.choice(len(X_test), N_SAMPLES, replace=False)
    X_sub = X_test[idx]
    y_sub = y_test[idx]

    input_shape = X_sub.shape[1:]
    classifier = TensorFlowV2Classifier(
        model=model, nb_classes=2, input_shape=input_shape,
        loss_object=keras.losses.CategoricalCrossentropy(),
        clip_values=(0.0, 1.0),
    )

    clean_acc = evaluate_accuracy(model, X_sub, y_sub)
    print(f"  Clean accuracy: {clean_acc*100:.1f}% (n={N_SAMPLES})")

    # Run C&W L2 — targeted, class 0
    print(f"  Running C&W L2 (targeted, {N_SAMPLES} samples)...")
    print(f"  This may take 10-30 min for {model_name}...")
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=0.5,
        max_iter=50,
        learning_rate=0.01,
        batch_size=16,
        targeted=True,
        verbose=False,
    )
    y_target = to_categorical(np.zeros(N_SAMPLES, dtype=int), 2)
    X_adv = attack.generate(x=X_sub, y=y_target)

    # Measure per-sample L-inf perturbation
    perturbation = X_adv - X_sub
    per_sample_linf = np.max(np.abs(perturbation.reshape(N_SAMPLES, -1)), axis=1)
    per_sample_l2 = np.sqrt(np.sum(perturbation.reshape(N_SAMPLES, -1)**2, axis=1))

    # Check attack success (how many actually changed prediction)
    adv_preds = np.argmax(
        model(tf.convert_to_tensor(X_adv, dtype=tf.float32), training=False).numpy(), axis=1)
    attack_success = np.mean(adv_preds != y_sub)

    print(f"  Attack success rate: {attack_success*100:.1f}%")
    print(f"  Perturbation L-inf: mean={per_sample_linf.mean():.4f}, "
          f"median={np.median(per_sample_linf):.4f}, max={per_sample_linf.max():.4f}")
    print(f"  Perturbation L2: mean={per_sample_l2.mean():.4f}, "
          f"median={np.median(per_sample_l2):.4f}")

    # Evaluate at each epsilon
    results = {
        'epsilon': EPSILON_VALUES,
        'clean': clean_acc,
        'n_samples': N_SAMPLES,
        'attack_success_rate': float(attack_success),
        'mean_linf': float(per_sample_linf.mean()),
        'mean_l2': float(per_sample_l2.mean()),
        'cw_l2': [],
    }

    print(f"\n  {'Eps':<8} {'C&W L2':<10} {'In budget':<15}")
    print("  " + "-" * 33)

    for eps in EPSILON_VALUES:
        # Use adversarial if perturbation within budget, else keep original
        X_eval = np.copy(X_sub)
        in_budget = per_sample_linf <= eps
        X_eval[in_budget] = X_adv[in_budget]

        acc = evaluate_accuracy(model, X_eval, y_sub)
        results['cw_l2'].append(acc)
        print(f"  {eps:<8.2f} {acc*100:<10.1f} {np.sum(in_budget)}/{N_SAMPLES}")

    save_results(results, result_name)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("C&W L2 ATTACK EVALUATION")
    print("=" * 60)

    # CNN
    print("\n" + "=" * 60)
    print("CNN (Spectrograms)")
    print("=" * 60)
    run_cw_l2("CNN", "cnn_baseline.keras",
              os.path.join(DATA_DIR, "X_spec_test.npy"),
              os.path.join(DATA_DIR, "y_spec_test.npy"))

    # DNN
    print("\n" + "=" * 60)
    print("DNN (KPMs)")
    print("=" * 60)
    run_cw_l2("DNN", "dnn_baseline.keras",
              os.path.join(DATA_DIR, "X_kpm_test.npy"),
              os.path.join(DATA_DIR, "y_kpm_test.npy"))

    print("\nDone!")