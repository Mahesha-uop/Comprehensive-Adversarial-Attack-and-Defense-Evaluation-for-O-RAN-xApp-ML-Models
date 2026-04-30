"""
Missing CNN Defense Evaluations
================================
Runs the 3 missing CNN defense experiments:
  1. CNN Randomized Smoothing (model exists, needs evaluation)
  2. CNN Input Transforms (no retraining, uses baseline model)
  3. CNN MC-Dropout (needs training + detection evaluation)

Estimated runtime:
  - Smoothing: ~1.5 hours (30 noisy passes per sample per epsilon)
  - Input Transform: ~15 minutes (no training)
  - MC-Dropout: ~1 hour (training + 30 forward passes for detection)
  - Total: ~3 hours
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)

CKPT_DIR = "./checkpoints"
DATA_DIR = "./Dataset"
RESULTS_DIR = "./extended_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


# ============================================================================
# HELPERS
# ============================================================================

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


# ============================================================================
# ATTACKS
# ============================================================================

def fgsm_attack(model, X, y, epsilon, batch_size=128):
    adv_list = []
    for i in range(0, len(X), batch_size):
        X_b = tf.convert_to_tensor(X[i:i+batch_size], dtype=tf.float32)
        y_t = tf.convert_to_tensor(
            to_categorical(np.zeros(len(X_b), dtype=int), 2), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X_b)
            loss = keras.losses.categorical_crossentropy(y_t, model(X_b, training=False))
        grads = tape.gradient(loss, X_b)
        adv_list.append(tf.clip_by_value(X_b - epsilon * tf.sign(grads), 0., 1.).numpy())
    return np.concatenate(adv_list, axis=0)


def pgd_attack(model, X, y, epsilon, steps=5, batch_size=128):
    step_size = 2.0 * epsilon / steps
    adv_list = []
    for i in range(0, len(X), batch_size):
        X_b = tf.convert_to_tensor(X[i:i+batch_size], dtype=tf.float32)
        X_orig = tf.identity(X_b)
        X_adv = tf.identity(X_b)
        y_t = tf.convert_to_tensor(
            to_categorical(np.zeros(len(X_b), dtype=int), 2), dtype=tf.float32)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(X_adv)
                loss = keras.losses.categorical_crossentropy(y_t, model(X_adv, training=False))
            grads = tape.gradient(loss, X_adv)
            X_adv = X_adv - step_size * tf.sign(grads)
            pert = tf.clip_by_value(X_adv - X_orig, -epsilon, epsilon)
            X_adv = tf.clip_by_value(X_orig + pert, 0., 1.)
        adv_list.append(X_adv.numpy())
    return np.concatenate(adv_list, axis=0)


# ============================================================================
# 1. CNN RANDOMIZED SMOOTHING
# ============================================================================

def run_cnn_smoothing():
    name = "cnn_smoothing"
    if load_results(name) is not None:
        print(f"  [LOADED] {name}")
        return

    print("\n" + "=" * 60)
    print("CNN RANDOMIZED SMOOTHING EVALUATION")
    print("=" * 60)

    X_test = np.load(os.path.join(DATA_DIR, "X_spec_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_spec_test.npy"))

    # Load the noise-trained base model
    base_model = keras.models.load_model(os.path.join(CKPT_DIR, "cnn_smoothing_base.keras"))

    sigma = 0.05
    n_samples = 30

    # Smoothed prediction function
    def smoothed_predict(model, X, sigma, n_samples):
        preds_sum = np.zeros((len(X), 2))
        for i in range(0, len(X), BATCH_SIZE):
            batch = X[i:i+BATCH_SIZE]
            batch_sum = np.zeros((len(batch), 2))
            for _ in range(n_samples):
                noise = np.random.normal(0, sigma, batch.shape).astype(np.float32)
                noisy = np.clip(batch + noise, 0., 1.)
                batch_sum += model(tf.convert_to_tensor(noisy, dtype=tf.float32),
                                   training=False).numpy()
            preds_sum[i:i+len(batch)] = batch_sum / n_samples
        return preds_sum

    # Clean accuracy with smoothing
    print("  Evaluating clean accuracy with smoothing...")
    clean_preds = smoothed_predict(base_model, X_test, sigma, n_samples)
    clean_acc = float(np.mean(np.argmax(clean_preds, axis=1) == y_test))
    print(f"  Clean accuracy (smoothed): {clean_acc*100:.1f}%")

    results = {'epsilon': EPSILON_VALUES, 'clean': clean_acc, 'fgsm': [], 'pgd': []}

    print(f"\n  {'Eps':<8} {'FGSM':<10} {'PGD':<10}")
    print("  " + "-" * 28)

    for eps in EPSILON_VALUES:
        # Generate adversarial examples on base model (not smoothed wrapper)
        X_fgsm = fgsm_attack(base_model, X_test, y_test, eps)
        fgsm_preds = smoothed_predict(base_model, X_fgsm, sigma, n_samples)
        fgsm_acc = float(np.mean(np.argmax(fgsm_preds, axis=1) == y_test))
        results['fgsm'].append(fgsm_acc)

        X_pgd = pgd_attack(base_model, X_test, y_test, eps)
        pgd_preds = smoothed_predict(base_model, X_pgd, sigma, n_samples)
        pgd_acc = float(np.mean(np.argmax(pgd_preds, axis=1) == y_test))
        results['pgd'].append(pgd_acc)

        print(f"  {eps:<8.2f} {fgsm_acc*100:<10.1f} {pgd_acc*100:<10.1f}")

    save_results(results, name)


# ============================================================================
# 2. CNN INPUT TRANSFORMS
# ============================================================================

def apply_input_transforms(X):
    """Apply quantization + spatial smoothing + bit-depth reduction."""
    X_clean = np.copy(X)

    # 4-bit quantization
    X_clean = np.round(X_clean * 16) / 16

    # Spatial smoothing (3x3 average filter per channel)
    kernel = np.ones((3, 3, 1, 1), dtype=np.float32) / 9.0
    kernel_tf = tf.constant(kernel)
    channels = X_clean.shape[-1]
    smoothed = []
    for c in range(channels):
        ch = X_clean[:, :, :, c:c+1]
        ch_s = tf.nn.conv2d(ch, kernel_tf, strides=[1,1,1,1], padding='SAME')
        smoothed.append(ch_s.numpy())
    X_clean = np.concatenate(smoothed, axis=-1)

    # Bit-depth reduction to 4 bits
    X_clean = np.round(X_clean * 15) / 15

    return np.clip(X_clean, 0., 1.).astype(np.float32)


def run_cnn_input_transform():
    name = "cnn_input_transform"
    if load_results(name) is not None:
        print(f"  [LOADED] {name}")
        return

    print("\n" + "=" * 60)
    print("CNN INPUT TRANSFORM EVALUATION")
    print("=" * 60)

    X_test = np.load(os.path.join(DATA_DIR, "X_spec_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_spec_test.npy"))

    base_model = keras.models.load_model(os.path.join(CKPT_DIR, "cnn_baseline.keras"))

    # Clean accuracy with transforms
    X_test_transformed = apply_input_transforms(X_test)
    clean_acc = evaluate_accuracy(base_model, X_test_transformed, y_test)
    print(f"  Clean accuracy (with transforms): {clean_acc*100:.1f}%")

    results = {'epsilon': EPSILON_VALUES, 'clean': clean_acc, 'fgsm': [], 'pgd': []}

    print(f"\n  {'Eps':<8} {'FGSM':<10} {'PGD':<10}")
    print("  " + "-" * 28)

    for eps in EPSILON_VALUES:
        # Attack the BASE model, then apply transforms to adversarial inputs
        X_fgsm = fgsm_attack(base_model, X_test, y_test, eps)
        X_fgsm_t = apply_input_transforms(X_fgsm)
        fgsm_acc = evaluate_accuracy(base_model, X_fgsm_t, y_test)
        results['fgsm'].append(fgsm_acc)

        X_pgd = pgd_attack(base_model, X_test, y_test, eps)
        X_pgd_t = apply_input_transforms(X_pgd)
        pgd_acc = evaluate_accuracy(base_model, X_pgd_t, y_test)
        results['pgd'].append(pgd_acc)

        print(f"  {eps:<8.2f} {fgsm_acc*100:<10.1f} {pgd_acc*100:<10.1f}")

    save_results(results, name)


# ============================================================================
# 3. CNN MC-DROPOUT
# ============================================================================

def build_cnn_dropout(input_shape=(128, 128, 3), rate=0.2):
    return keras.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)), layers.Dropout(rate),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)), layers.Dropout(rate),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)), layers.Dropout(rate),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'), layers.Dropout(rate),
        layers.Dense(2, activation='softmax'),
    ])


def run_cnn_mcdropout():
    name = "cnn_mcdropout"
    if load_results(name) is not None:
        print(f"  [LOADED] {name}")
        return

    print("\n" + "=" * 60)
    print("CNN MC-DROPOUT EVALUATION")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_spec_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_spec_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_spec_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_spec_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_spec_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_spec_test.npy"))

    # Train or load MC-Dropout model
    mc_model_path = os.path.join(CKPT_DIR, "cnn_mcdropout.keras")
    if os.path.exists(mc_model_path):
        print(f"  [LOADED] {mc_model_path}")
        mc_model = keras.models.load_model(mc_model_path)
    else:
        print("  Training CNN with dropout...")
        mc_model = build_cnn_dropout()
        mc_model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        mc_model.fit(X_train, to_categorical(y_train, 2),
                     validation_data=(X_val, to_categorical(y_val, 2)),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        mc_model.save(mc_model_path)
        print(f"  [SAVED] {mc_model_path}")

    # MC-Dropout prediction
    n_forward = 30
    n_detect = 100  # Subsample for CNN (memory)

    def mc_dropout_entropy(model, X, n_forward):
        all_preds = []
        X_t = tf.convert_to_tensor(X, dtype=tf.float32)
        for _ in range(n_forward):
            preds = model(X_t, training=True).numpy()  # training=True keeps dropout
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        mean_preds = np.mean(all_preds, axis=0)
        entropy = -np.sum(mean_preds * np.log(np.clip(mean_preds, 1e-10, 1.)), axis=1)
        return mean_preds, entropy

    # Get clean entropy for threshold
    print("  Computing clean entropy baseline...")
    _, entropy_clean = mc_dropout_entropy(mc_model, X_test[:n_detect], n_forward)
    threshold = np.percentile(entropy_clean, 95)  # 5% FPR
    fpr = float(np.sum(entropy_clean > threshold) / len(entropy_clean))

    # Load base model for generating adversarial examples
    base_model = keras.models.load_model(os.path.join(CKPT_DIR, "cnn_baseline.keras"))

    results = {'epsilon': EPSILON_VALUES, 'threshold': float(threshold), 'detection': []}

    print(f"\n  MC-Dropout Detection (threshold={threshold:.4f})")
    print(f"  {'Eps':<8} {'TPR':<10} {'FPR':<10} {'Adv Entropy':<15}")
    print("  " + "-" * 43)

    for eps in EPSILON_VALUES:
        X_adv = fgsm_attack(base_model, X_test[:n_detect], y_test[:n_detect], eps)
        _, entropy_adv = mc_dropout_entropy(mc_model, X_adv, n_forward)
        tpr = float(np.sum(entropy_adv > threshold) / len(entropy_adv))

        det = {
            'epsilon': eps,
            'tpr': tpr,
            'fpr': fpr,
            'adv_entropy_mean': float(entropy_adv.mean()),
            'clean_entropy_mean': float(entropy_clean.mean()),
        }
        results['detection'].append(det)
        print(f"  {eps:<8.2f} {tpr*100:<10.1f} {fpr*100:<10.1f} {entropy_adv.mean():<15.4f}")

    save_results(results, name)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MISSING CNN DEFENSE EVALUATIONS")
    print("=" * 60)

    t0 = time.perf_counter()

    # 1. Input Transform (fastest, no training)
    run_cnn_input_transform()

    # 2. MC-Dropout (needs training)
    run_cnn_mcdropout()

    # 3. Randomized Smoothing (slowest)
    run_cnn_smoothing()

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.1f} minutes")
    print(f"{'='*60}")

    # Print summary
    print("\nResults files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith('.json'):
            print(f"  {f}")