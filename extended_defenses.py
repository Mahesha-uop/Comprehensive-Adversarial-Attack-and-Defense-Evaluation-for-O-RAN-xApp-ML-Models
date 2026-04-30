"""
Extended Defenses: TRADES, Randomized Smoothing, Input Transforms, MC-Dropout
==============================================================================
Evaluates 4 additional defenses on both CNN and DNN under FGSM/PGD attacks.

Requires: saved baseline models in checkpoints/, .npy data files
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)

CKPT_DIR = "./checkpoints"
RESULTS_DIR = "./extended_results"
DATA_DIR = "./Dataset"
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


def load_spec_splits():
    return (np.load(os.path.join(DATA_DIR, "X_spec_train.npy")),
            np.load(os.path.join(DATA_DIR, "X_spec_val.npy")),
            np.load(os.path.join(DATA_DIR, "X_spec_test.npy")),
            np.load(os.path.join(DATA_DIR, "y_spec_train.npy")),
            np.load(os.path.join(DATA_DIR, "y_spec_val.npy")),
            np.load(os.path.join(DATA_DIR, "y_spec_test.npy")))


def load_kpm_splits():
    return (np.load(os.path.join(DATA_DIR, "X_kpm_train.npy")),
            np.load(os.path.join(DATA_DIR, "X_kpm_val.npy")),
            np.load(os.path.join(DATA_DIR, "X_kpm_test.npy")),
            np.load(os.path.join(DATA_DIR, "y_kpm_train.npy")),
            np.load(os.path.join(DATA_DIR, "y_kpm_val.npy")),
            np.load(os.path.join(DATA_DIR, "y_kpm_test.npy")))


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_cnn(input_shape=(128, 128, 3)):
    return keras.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax'),
    ])


def build_dnn(input_shape=(60,)):
    return keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='softmax'),
    ])


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


def build_dnn_dropout(input_shape=(60,), rate=0.2):
    return keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape), layers.Dropout(rate),
        layers.Dense(32, activation='relu'), layers.Dropout(rate),
        layers.Dense(16, activation='relu'), layers.Dropout(rate),
        layers.Dense(2, activation='softmax'),
    ])


# ============================================================================
# ATTACKS FOR EVALUATION
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


def evaluate_defense(model, X_test, y_test, eps_values, defense_name):
    clean_acc = evaluate_accuracy(model, X_test, y_test)
    results = {'epsilon': eps_values, 'clean': clean_acc, 'fgsm': [], 'pgd': []}

    print(f"\n  {defense_name} — Clean: {clean_acc*100:.1f}%")
    print(f"  {'Eps':<8} {'FGSM':<10} {'PGD':<10}")
    print("  " + "-" * 28)

    for eps in eps_values:
        X_fgsm = fgsm_attack(model, X_test, y_test, eps)
        fgsm_acc = evaluate_accuracy(model, X_fgsm, y_test)
        results['fgsm'].append(fgsm_acc)

        X_pgd = pgd_attack(model, X_test, y_test, eps)
        pgd_acc = evaluate_accuracy(model, X_pgd, y_test)
        results['pgd'].append(pgd_acc)

        print(f"  {eps:<8.2f} {fgsm_acc*100:<10.1f} {pgd_acc*100:<10.1f}")

    return results


# ============================================================================
# DEFENSE 1: TRADES (Zhang et al., ICML 2019)
# ============================================================================

def train_trades(model_builder, X_train, y_train, X_val, y_val,
                 beta=6.0, epsilon=0.02, steps=5, epochs=EPOCHS):
    """
    TRADES loss = CE(x, y) + beta * KL(f(x) || f(x_adv))
    Generates adversarial examples by maximizing KL divergence during training.
    """
    print(f"\n  Training TRADES (beta={beta}, eps={epsilon})...")
    model = model_builder()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    y_train_cat = to_categorical(y_train, 2)
    step_size = 2.0 * epsilon / steps

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train_cat)).shuffle(10000).batch(BATCH_SIZE)

    for epoch in range(epochs):
        epoch_loss, n = 0, 0

        for X_b, y_b in train_ds:
            X_b = tf.cast(X_b, tf.float32)

            # Inner loop: generate adversarial by maximizing KL
            X_adv = X_b + 0.001 * tf.random.normal(X_b.shape)
            X_adv = tf.clip_by_value(X_adv, 0., 1.)

            for _ in range(steps):
                with tf.GradientTape() as tape_inner:
                    tape_inner.watch(X_adv)
                    p_clean = tf.stop_gradient(model(X_b, training=False))
                    p_adv = model(X_adv, training=False)
                    kl = tf.reduce_mean(tf.reduce_sum(
                        p_clean * tf.math.log(
                            tf.clip_by_value(p_clean, 1e-10, 1.) /
                            tf.clip_by_value(p_adv, 1e-10, 1.)
                        ), axis=1))
                grads = tape_inner.gradient(kl, X_adv)
                X_adv = X_adv + step_size * tf.sign(grads)
                pert = tf.clip_by_value(X_adv - X_b, -epsilon, epsilon)
                X_adv = tf.clip_by_value(X_b + pert, 0., 1.)
                X_adv = tf.stop_gradient(X_adv)

            # Outer loop: minimize TRADES loss
            with tf.GradientTape() as tape:
                p_clean = model(X_b, training=True)
                p_adv = model(X_adv, training=True)

                ce = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(y_b, p_clean))
                kl = tf.reduce_mean(tf.reduce_sum(
                    p_clean * tf.math.log(
                        tf.clip_by_value(p_clean, 1e-10, 1.) /
                        tf.clip_by_value(p_adv, 1e-10, 1.)
                    ), axis=1))

                loss = ce + beta * kl

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            n += 1

        if (epoch + 1) % 10 == 0:
            val_acc = evaluate_accuracy(model, X_val, y_val)
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {epoch_loss/n:.4f}, "
                  f"Val Acc: {val_acc*100:.2f}%")

    return model


# ============================================================================
# DEFENSE 2: RANDOMIZED SMOOTHING (Cohen et al., ICML 2019)
# ============================================================================

class SmoothedModel:
    """Inference wrapper: average predictions over N noisy copies."""
    def __init__(self, base_model, sigma=0.1, n_samples=50):
        self.base_model = base_model
        self.sigma = sigma
        self.n_samples = n_samples

    def __call__(self, X, training=False):
        X = tf.cast(X, tf.float32)
        preds_sum = tf.zeros((len(X), 2))
        for _ in range(self.n_samples):
            noise = tf.random.normal(X.shape, stddev=self.sigma)
            X_noisy = tf.clip_by_value(X + noise, 0., 1.)
            preds_sum += self.base_model(X_noisy, training=False)
        return preds_sum / self.n_samples


def train_with_noise(model_builder, X_train, y_train, X_val, y_val,
                     sigma=0.1, epochs=EPOCHS):
    """Train with Gaussian noise augmentation for smoothing-aware training."""
    print(f"\n  Training with noise augmentation (sigma={sigma})...")
    model = model_builder()
    y_cat = to_categorical(y_train, 2)
    y_val_cat = to_categorical(y_val, 2)

    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    for epoch in range(epochs):
        noise = np.random.normal(0, sigma, X_train.shape).astype(np.float32)
        X_noisy = np.clip(X_train + noise, 0., 1.)
        model.fit(X_noisy, y_cat, validation_data=(X_val, y_val_cat),
                  epochs=1, batch_size=BATCH_SIZE, verbose=0)

        if (epoch + 1) % 10 == 0:
            val_acc = evaluate_accuracy(model, X_val, y_val)
            print(f"    Epoch {epoch+1}/{epochs} — Val Acc: {val_acc*100:.2f}%")

    return model


# ============================================================================
# DEFENSE 3: INPUT TRANSFORMATIONS
# ============================================================================

class InputTransformModel:
    """Apply input transforms before inference to remove perturbations."""
    def __init__(self, base_model, method='all'):
        self.base_model = base_model
        self.method = method

    def __call__(self, X, training=False):
        X_np = X.numpy() if isinstance(X, tf.Tensor) else X
        X_clean = self._transform(X_np)
        return self.base_model(
            tf.convert_to_tensor(X_clean, dtype=tf.float32), training=False)

    def _transform(self, X):
        X = np.copy(X)

        if self.method in ('quantize', 'all'):
            # 4-bit quantization (simulates JPEG-like compression)
            X = np.round(X * 16) / 16

        if self.method in ('smooth', 'all'):
            if X.ndim == 4:  # Images: 3x3 average filter
                kernel = np.ones((3, 3, 1, 1), dtype=np.float32) / 9.0
                kernel_tf = tf.constant(kernel)
                channels = X.shape[-1]
                smoothed = []
                for c in range(channels):
                    ch = X[:, :, :, c:c+1]
                    ch_s = tf.nn.conv2d(ch, kernel_tf, strides=[1,1,1,1], padding='SAME')
                    smoothed.append(ch_s.numpy())
                X = np.concatenate(smoothed, axis=-1)
            else:  # KPM: simple moving average over features
                for i in range(1, X.shape[1] - 1):
                    X[:, i] = (X[:, i-1] + X[:, i] + X[:, i+1]) / 3.0

        if self.method in ('bitdepth', 'all'):
            # Reduce to 4-bit depth
            X = np.round(X * 15) / 15

        return np.clip(X, 0., 1.).astype(np.float32)


# ============================================================================
# DEFENSE 4: MC-DROPOUT DETECTION
# ============================================================================

def mc_dropout_predict(model, X, n_forward=30):
    """Run n_forward passes with dropout enabled, return mean + entropy."""
    all_preds = []
    X_t = tf.convert_to_tensor(X, dtype=tf.float32)
    for _ in range(n_forward):
        preds = model(X_t, training=True).numpy()  # training=True keeps dropout
        all_preds.append(preds)
    all_preds = np.array(all_preds)
    mean_preds = np.mean(all_preds, axis=0)
    entropy = -np.sum(mean_preds * np.log(np.clip(mean_preds, 1e-10, 1.)), axis=1)
    return mean_preds, entropy


def evaluate_mcdropout_detection(mc_model, base_model, X_test, y_test, eps_values,
                                  n_forward=30, n_samples=200):
    """Evaluate MC-Dropout adversarial detection at each epsilon."""
    X_sub = X_test[:n_samples]
    y_sub = y_test[:n_samples]

    _, entropy_clean = mc_dropout_predict(mc_model, X_sub, n_forward)
    threshold = np.percentile(entropy_clean, 95)

    results = {
        'epsilon': eps_values,
        'threshold': float(threshold),
        'clean_entropy_mean': float(entropy_clean.mean()),
        'detection': [],
    }

    print(f"\n  MC-Dropout Detection (threshold={threshold:.4f})")
    print(f"  {'Eps':<8} {'TPR':<10} {'FPR':<10} {'Adv Entropy':<15}")
    print("  " + "-" * 43)

    fpr = float(np.sum(entropy_clean > threshold) / len(entropy_clean))

    for eps in eps_values:
        X_adv = fgsm_attack(base_model, X_sub, y_sub, eps)
        _, entropy_adv = mc_dropout_predict(mc_model, X_adv, n_forward)
        tpr = float(np.sum(entropy_adv > threshold) / len(entropy_adv))

        results['detection'].append({
            'epsilon': eps, 'tpr': tpr, 'fpr': fpr,
            'adv_entropy_mean': float(entropy_adv.mean()),
        })
        print(f"  {eps:<8.2f} {tpr*100:<10.1f} {fpr*100:<10.1f} {entropy_adv.mean():<15.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_defenses_for(model_type):
    """Run all 4 defenses for CNN or DNN."""
    if model_type == "dnn":
        X_tr, X_v, X_te, y_tr, y_v, y_te = load_kpm_splits()
        builder = build_dnn
        builder_drop = build_dnn_dropout
        trades_eps = 0.05
        smooth_sigma = 0.1
        smooth_n = 50
        tag = "dnn"
    else:
        X_tr, X_v, X_te, y_tr, y_v, y_te = load_spec_splits()
        builder = build_cnn
        builder_drop = build_cnn_dropout
        trades_eps = 0.02
        smooth_sigma = 0.05
        smooth_n = 30
        tag = "cnn"

    print(f"\n{'='*60}")
    print(f"{tag.upper()} DEFENSES")
    print(f"{'='*60}")

    # --- TRADES ---
    name = f"{tag}_trades"
    if load_results(name) is None:
        model = train_trades(builder, X_tr, y_tr, X_v, y_v,
                            beta=6.0, epsilon=trades_eps)
        model.save(os.path.join(CKPT_DIR, f"{name}.keras"))
        results = evaluate_defense(model, X_te, y_te, EPSILON_VALUES, f"{tag.upper()}-TRADES")
        save_results(results, name)
    else:
        print(f"  [LOADED] {name}")

    # --- Randomized Smoothing ---
    name = f"{tag}_smoothing"
    if load_results(name) is None:
        base = train_with_noise(builder, X_tr, y_tr, X_v, y_v, sigma=smooth_sigma)
        base.save(os.path.join(CKPT_DIR, f"{name}_base.keras"))
        smooth = SmoothedModel(base, sigma=smooth_sigma, n_samples=smooth_n)
        results = evaluate_defense(smooth, X_te, y_te, EPSILON_VALUES,
                                   f"{tag.upper()}-Smoothing")
        save_results(results, name)
    else:
        print(f"  [LOADED] {name}")

    # --- Input Transforms ---
    # Note: Attack the BASE model, then apply transforms to adversarial inputs.
    # The defense is a preprocessing step, not a model change.
    name = f"{tag}_input_transform"
    if load_results(name) is None:
        base = keras.models.load_model(os.path.join(CKPT_DIR, f"{tag}_baseline.keras"))
        transform = InputTransformModel(base, method='all')

        # Evaluate clean accuracy with transforms applied
        clean_acc = evaluate_accuracy(transform, X_te, y_te)
        results = {'epsilon': EPSILON_VALUES, 'clean': clean_acc, 'fgsm': [], 'pgd': []}

        print(f"\n  {tag.upper()}-InputTransform — Clean: {clean_acc*100:.1f}%")
        print(f"  {'Eps':<8} {'FGSM':<10} {'PGD':<10}")
        print("  " + "-" * 28)

        for eps in EPSILON_VALUES:
            # Generate adversarial examples on the BASE model (not the transform wrapper)
            X_fgsm = fgsm_attack(base, X_te, y_te, eps)
            # Apply transforms to adversarial inputs, then evaluate with base model
            fgsm_acc = evaluate_accuracy(transform, X_fgsm, y_te)
            results['fgsm'].append(fgsm_acc)

            X_pgd = pgd_attack(base, X_te, y_te, eps)
            pgd_acc = evaluate_accuracy(transform, X_pgd, y_te)
            results['pgd'].append(pgd_acc)

            print(f"  {eps:<8.2f} {fgsm_acc*100:<10.1f} {pgd_acc*100:<10.1f}")

        save_results(results, name)
    else:
        print(f"  [LOADED] {name}")

    # --- MC-Dropout ---
    name = f"{tag}_mcdropout"
    if load_results(name) is None:
        print(f"\n  Training {tag.upper()} with dropout...")
        mc_model = builder_drop()
        mc_model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                        loss='categorical_crossentropy', metrics=['accuracy'])
        mc_model.fit(X_tr, to_categorical(y_tr, 2),
                    validation_data=(X_v, to_categorical(y_v, 2)),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        mc_model.save(os.path.join(CKPT_DIR, f"{name}.keras"))

        base = keras.models.load_model(os.path.join(CKPT_DIR, f"{tag}_baseline.keras"))
        n_det = 100 if model_type == "cnn" else 200
        results = evaluate_mcdropout_detection(
            mc_model, base, X_te, y_te, EPSILON_VALUES, n_samples=n_det)
        save_results(results, name)
    else:
        print(f"  [LOADED] {name}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXTENDED DEFENSE EVALUATION")
    print("4 defenses x 2 models")
    print("=" * 60)

    # DNN first (much faster)
    run_defenses_for("dnn")

    # CNN second
    run_defenses_for("cnn")

    print("\n" + "=" * 60)
    print("ALL DEFENSES COMPLETE")
    print("=" * 60)