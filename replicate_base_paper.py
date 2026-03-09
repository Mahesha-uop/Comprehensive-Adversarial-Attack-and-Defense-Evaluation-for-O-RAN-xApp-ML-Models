"""
Replication of: "System-level Analysis of Adversarial Attacks and Defenses 
on Intelligence in O-RAN based Cellular Networks" (Chiejina et al., ACM WiSec 2024)

Prerequisites: Run load_dataset.py first to generate .npy files.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

np.random.seed(42)
tf.random.set_seed(42)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

EPSILON_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
PGD_STEPS = 5
PGD_STEP_SIZE = 0.01

TEACHER_TEMPERATURE = 20.0
STUDENT_TEMPERATURE = 1.0
DISTILLATION_ALPHA = 0.1

ADV_TRAIN_EPSILON = 0.02


# ============================================================================
# 1. MODEL ARCHITECTURES
# ============================================================================

def build_cnn_model(input_shape=(128, 128, 3), num_classes=2):
    """
    CNN for InterClass-Spec xApp (Table 2).
    Using 3 channels (RGB) since spectrogram color encodes information.
    Note: param count differs slightly from paper's 163,922 (which assumed 1-ch).
    """
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ], name='InterClass_Spec_CNN')
    return model


def build_cnn_logits(input_shape=(128, 128, 3), num_classes=2):
    """CNN outputting logits (no softmax) for distillation."""
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes),
    ])
    return model


def build_dnn_model(input_shape=(60,), num_classes=2):
    """DNN for InterClass-KPM xApp (Figure 3). 6,546 parameters."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ], name='InterClass_KPM_DNN')
    return model


def build_dnn_logits(input_shape=(60,), num_classes=2):
    """DNN outputting logits for distillation."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes),
    ])
    return model


def verify_model_params():
    """Verify parameter counts."""
    cnn = build_cnn_model()
    dnn = build_dnn_model()
    print("=" * 60)
    print("MODEL PARAMETER VERIFICATION")
    print("=" * 60)
    print(f"CNN: {cnn.count_params():,} params (paper: 163,922 with 1-ch; "
          f"ours uses 3-ch RGB)")
    print(f"DNN: {dnn.count_params():,} params (paper: 6,546)")
    print(f"  DNN Match: {'YES' if dnn.count_params() == 6546 else 'NO'}")
    print()
    cnn.summary()
    print()
    dnn.summary()


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def load_spectrogram_splits():
    """Load pre-processed RGB spectrogram data from .npy files."""
    X_train = np.load("X_spec_train.npy")
    y_train = np.load("y_spec_train.npy")
    X_val = np.load("X_spec_val.npy")
    y_val = np.load("y_spec_val.npy")
    X_test = np.load("X_spec_test.npy")
    y_test = np.load("y_spec_test.npy")
    print(f"  Spectrograms: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"  Train class dist: {np.bincount(y_train)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_kpm_splits():
    """Load pre-processed KPM data from .npy files."""
    X_train = np.load("X_kpm_train.npy")
    y_train = np.load("y_kpm_train.npy")
    X_val = np.load("X_kpm_val.npy")
    y_val = np.load("y_kpm_val.npy")
    X_test = np.load("X_kpm_test.npy")
    y_test = np.load("y_kpm_test.npy")
    print(f"  KPMs: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"  Train class dist: {np.bincount(y_train)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    """Train with categorical cross-entropy."""
    y_train_cat = to_categorical(y_train, 2)
    y_val_cat = to_categorical(y_val, 2)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, label="Model"):
    """Evaluate accuracy on test set."""
    y_test_cat = to_categorical(y_test, 2)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  {label} — Acc: {acc*100:.2f}%, Loss: {loss:.4f}")
    return acc


# ============================================================================
# 4. ADVERSARIAL ATTACKS
# ============================================================================

def fgsm_attack(model, X, y, epsilon):
    """FGSM targeted attack — target class 0 (SOI)."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_target = to_categorical(np.zeros(len(y), dtype=int), 2)
    y_target_tensor = tf.convert_to_tensor(y_target, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)
        loss = keras.losses.categorical_crossentropy(y_target_tensor, predictions)

    gradients = tape.gradient(loss, X_tensor)
    X_adv = X_tensor - epsilon * tf.sign(gradients)
    X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
    return X_adv.numpy()


def pgd_attack(model, X, y, epsilon, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    """PGD targeted attack — 5-step iterative FGSM with L-inf projection."""
    X_adv = tf.identity(tf.convert_to_tensor(X, dtype=tf.float32))
    X_orig = tf.convert_to_tensor(X, dtype=tf.float32)
    y_target = to_categorical(np.zeros(len(y), dtype=int), 2)
    y_target_tensor = tf.convert_to_tensor(y_target, dtype=tf.float32)

    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            predictions = model(X_adv, training=False)
            loss = keras.losses.categorical_crossentropy(y_target_tensor, predictions)

        gradients = tape.gradient(loss, X_adv)
        X_adv = X_adv - step_size * tf.sign(gradients)
        perturbation = tf.clip_by_value(X_adv - X_orig, -epsilon, epsilon)
        X_adv = tf.clip_by_value(X_orig + perturbation, 0.0, 1.0)

    return X_adv.numpy()


def evaluate_attacks(model, X_test, y_test, epsilon_values, model_name="Model"):
    """Evaluate model under attacks across epsilon range."""
    results = {'epsilon': [], 'clean': [], 'fgsm': [], 'pgd': []}
    clean_acc = evaluate_model(model, X_test, y_test, label=f"{model_name} (clean)")

    print(f"\n  {'Eps':<8} {'Clean':<10} {'FGSM':<10} {'PGD':<10}")
    print("  " + "-" * 38)

    for eps in epsilon_values:
        X_fgsm = fgsm_attack(model, X_test, y_test, eps)
        fgsm_acc = evaluate_model(model, X_fgsm, y_test, label=f"FGSM e={eps}")

        X_pgd = pgd_attack(model, X_test, y_test, eps)
        pgd_acc = evaluate_model(model, X_pgd, y_test, label=f"PGD e={eps}")

        results['epsilon'].append(eps)
        results['clean'].append(clean_acc)
        results['fgsm'].append(fgsm_acc)
        results['pgd'].append(pgd_acc)

        print(f"  {eps:<8.2f} {clean_acc*100:<10.1f} {fgsm_acc*100:<10.1f} {pgd_acc*100:<10.1f}")

    return results


# ============================================================================
# 5. DISTILLATION DEFENSE
# ============================================================================

def train_teacher(logits_builder, X_train, y_train, X_val, y_val,



                  temperature=TEACHER_TEMPERATURE):
    """Train teacher with high temperature (Tt=20)."""
    teacher = logits_builder()
    y_train_cat = to_categorical(y_train, 2)
    y_val_cat = to_categorical(y_val, 2)

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train_cat)).shuffle(10000).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val_cat)).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        epoch_loss, n = 0, 0
        for X_b, y_b in train_ds:
            with tf.GradientTape() as tape:
                logits = teacher(X_b, training=True)
                soft_preds = tf.nn.softmax(logits / temperature)
                loss = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(y_b, soft_preds))
            grads = tape.gradient(loss, teacher.trainable_variables)
            optimizer.apply_gradients(zip(grads, teacher.trainable_variables))
            epoch_loss += loss.numpy()
            n += 1

        if (epoch + 1) % 10 == 0:
            correct, total = 0, 0
            for X_b, y_b in val_ds:
                logits = teacher(X_b, training=False)
                preds = tf.argmax(tf.nn.softmax(logits), axis=1)
                labels = tf.argmax(y_b, axis=1)
                correct += tf.reduce_sum(tf.cast(preds == labels, tf.int32)).numpy()
                total += len(y_b)
            print(f"  Teacher Epoch {epoch+1}/{EPOCHS} — "
                  f"Loss: {epoch_loss/n:.4f}, Val Acc: {correct/total*100:.2f}%")

    return teacher


def distill_student(teacher, logits_builder, X_train, y_train, X_val, y_val,
                    teacher_temp=TEACHER_TEMPERATURE, student_temp=STUDENT_TEMPERATURE,
                    alpha=DISTILLATION_ALPHA):
    """
    Distillation (Eq. 6): L = alpha * L_student + (1-alpha) * L_KL
    """
    student = logits_builder()
    y_train_cat = to_categorical(y_train, 2)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train_cat)).shuffle(10000).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        epoch_loss, n = 0, 0
        for X_b, y_b in train_ds:
            with tf.GradientTape() as tape:
                teacher_logits = teacher(X_b, training=False)
                teacher_soft = tf.nn.softmax(teacher_logits / teacher_temp)

                student_logits = student(X_b, training=True)
                student_soft = tf.nn.softmax(student_logits / student_temp)
                student_loss = keras.losses.categorical_crossentropy(y_b, student_soft)

                student_soft_temp = tf.nn.softmax(student_logits / teacher_temp)
                kl_loss = tf.reduce_sum(
                    teacher_soft * tf.math.log(
                        tf.clip_by_value(teacher_soft, 1e-10, 1.0) /
                        tf.clip_by_value(student_soft_temp, 1e-10, 1.0)
                    ), axis=1)

                total_loss = tf.reduce_mean(
                    alpha * student_loss + (1 - alpha) * kl_loss)

            grads = tape.gradient(total_loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            epoch_loss += total_loss.numpy()
            n += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Student Epoch {epoch+1}/{EPOCHS} — "
                  f"Loss: {epoch_loss/n:.4f}")

    # Wrap with softmax for inference
    class DistilledModel(keras.Model):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def call(self, x, training=False):
            return tf.nn.softmax(self.base(x, training=training))

    student_inf = DistilledModel(student)
    student_inf(X_train[:1])  # build
    return student, student_inf


# ============================================================================
# 6. ADVERSARIAL TRAINING
# ============================================================================

def adversarial_training(model_builder, X_train, y_train, X_val, y_val,
                         adv_epsilon=ADV_TRAIN_EPSILON):
    """Augment training data with adversarial examples at eps=0.02."""
    print("  Training temp model for adversarial generation...")
    temp_model = model_builder()
    train_model(temp_model, X_train, y_train, X_val, y_val, epochs=EPOCHS)

    print("  Generating FGSM adversarial examples...")
    X_adv_fgsm = fgsm_attack(temp_model, X_train, y_train, adv_epsilon)
    print("  Generating PGD adversarial examples...")
    X_adv_pgd = pgd_attack(temp_model, X_train, y_train, adv_epsilon)

    X_aug = np.concatenate([X_train, X_adv_fgsm, X_adv_pgd], axis=0)
    y_aug = np.concatenate([y_train, y_train, y_train], axis=0)
    idx = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]
    print(f"  Augmented training set: {len(X_aug)} samples")

    print("  Training robust model...")
    robust = model_builder()
    train_model(robust, X_aug, y_aug, X_val, y_val, epochs=EPOCHS)
    return robust


# ============================================================================
# 7. PLOTTING
# ============================================================================

def plot_accuracy_vs_epsilon(results_clean, results_distill, results_advtrain,
                             model_name="Model", save_path=None):
    """Replicate Figure 7 (CNN) or Figure 8 (DNN)."""
    eps = results_clean['epsilon']
    plt.figure(figsize=(8, 6))

    plt.plot(eps, [results_clean['clean'][0]*100]*len(eps),
             'k-o', label='NO-ATTACK', linewidth=2)
    plt.plot(eps, [a*100 for a in results_clean['fgsm']],
             'r-s', label='FGSM-ATTACK')
    plt.plot(eps, [a*100 for a in results_clean['pgd']],
             'b-^', label='PGD-ATTACK')
    plt.plot(eps, [a*100 for a in results_distill['fgsm']],
             'g-s', label='FGSM-DISTILLATION')
    plt.plot(eps, [a*100 for a in results_distill['pgd']],
             'c-^', label='PGD-DISTILLATION')
    plt.plot(eps, [a*100 for a in results_advtrain['fgsm']],
             'm-s', label='FGSM-ADVERSARIAL')
    plt.plot(eps, [a*100 for a in results_advtrain['pgd']],
             'y-^', label='PGD-ADVERSARIAL')

    plt.xlabel('Epsilon', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{model_name}: Accuracy vs Epsilon', fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved to {save_path}")
    plt.close()


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def run_full_pipeline():
    print("=" * 70)
    print("REPLICATION: Chiejina et al., ACM WiSec 2024")
    print("=" * 70)

    verify_model_params()

    # ==================== CNN PIPELINE ====================
    print("\n" + "=" * 70)
    print("PIPELINE 1: CNN (InterClass-Spec xApp)")
    print("=" * 70)

    print("\n[1] Loading spectrograms...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = load_spectrogram_splits()

    print("\n[2] Training baseline CNN...")
    cnn = build_cnn_model()
    train_model(cnn, X_tr, y_tr, X_v, y_v)
    acc = evaluate_model(cnn, X_te, y_te, "CNN Baseline")
    print(f"  Expected ~98%, Got {acc*100:.1f}%")

    print("\n[3] Evaluating attacks on CNN...")
    cnn_clean = evaluate_attacks(cnn, X_te, y_te, EPSILON_VALUES, "CNN")

    print("\n[4] Distillation defense...")
    print("  Training teacher (T=20)...")
    cnn_teacher = train_teacher(build_cnn_logits, X_tr, y_tr, X_v, y_v)
    print("  Distilling student...")
    _, cnn_dist = distill_student(cnn_teacher, build_cnn_logits, X_tr, y_tr, X_v, y_v)
    print("  Evaluating distilled CNN...")
    cnn_distill = evaluate_attacks(cnn_dist, X_te, y_te, EPSILON_VALUES, "CNN-Dist")

    print("\n[5] Adversarial training...")
    cnn_adv = adversarial_training(build_cnn_model, X_tr, y_tr, X_v, y_v)
    cnn_advtrain = evaluate_attacks(cnn_adv, X_te, y_te, EPSILON_VALUES, "CNN-AdvT")

    print("\n[6] Generating Figure 7...")
    plot_accuracy_vs_epsilon(cnn_clean, cnn_distill, cnn_advtrain,
                            "InterClass-Spec xApp (CNN)", "figure7_cnn.png")

    # ==================== DNN PIPELINE ====================
    print("\n" + "=" * 70)
    print("PIPELINE 2: DNN (InterClass-KPM xApp)")
    print("=" * 70)

    print("\n[1] Loading KPMs...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = load_kpm_splits()

    print("\n[2] Training baseline DNN...")
    dnn = build_dnn_model()
    train_model(dnn, X_tr, y_tr, X_v, y_v)
    acc = evaluate_model(dnn, X_te, y_te, "DNN Baseline")
    print(f"  Expected ~97.9%, Got {acc*100:.1f}%")

    print("\n[3] Evaluating attacks on DNN...")
    dnn_clean = evaluate_attacks(dnn, X_te, y_te, EPSILON_VALUES, "DNN")

    print("\n[4] Distillation defense...")
    print("  Training teacher (T=20)...")
    dnn_teacher = train_teacher(build_dnn_logits, X_tr, y_tr, X_v, y_v)
    print("  Distilling student...")
    _, dnn_dist = distill_student(dnn_teacher, build_dnn_logits, X_tr, y_tr, X_v, y_v)
    print("  Evaluating distilled DNN...")
    dnn_distill = evaluate_attacks(dnn_dist, X_te, y_te, EPSILON_VALUES, "DNN-Dist")

    print("\n[5] Adversarial training...")
    dnn_adv = adversarial_training(build_dnn_model, X_tr, y_tr, X_v, y_v)
    dnn_advtrain = evaluate_attacks(dnn_adv, X_te, y_te, EPSILON_VALUES, "DNN-AdvT")

    print("\n[6] Generating Figure 8...")
    plot_accuracy_vs_epsilon(dnn_clean, dnn_distill, dnn_advtrain,
                            "InterClass-KPM xApp (DNN)", "figure8_dnn.png")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY (at eps=0.1)")
    print("=" * 70)
    print(f"CNN: clean {cnn_clean['clean'][-1]*100:.1f}%, "
          f"FGSM {cnn_clean['fgsm'][-1]*100:.1f}%, "
          f"dist-FGSM {cnn_distill['fgsm'][-1]*100:.1f}%")
    print(f"DNN: clean {dnn_clean['clean'][-1]*100:.1f}%, "
          f"FGSM {dnn_clean['fgsm'][-1]*100:.1f}%, "
          f"dist-FGSM {dnn_distill['fgsm'][-1]*100:.1f}%")


if __name__ == "__main__":
    print("\n--- Architecture Verification ---\n")
    verify_model_params()

    # Uncomment to run full pipeline:
    run_full_pipeline()