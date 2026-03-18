
import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
tf.random.set_seed(42)

SPEC_DIR = "./newdataset"
EPOCHS = 30  # Fewer epochs for quick test
BATCH_SIZE = 32


def load_spectrograms(base_dir, img_size, grayscale=True):
    """Load spectrograms with configurable size and color mode."""
    CLASS_MAP = {'soi': 0, 'cwi': 1}
    X_list, y_list = [], []

    for folder, label in CLASS_MAP.items():
        folder_path = os.path.join(base_dir, folder)
        pngs = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        print(f"  {folder}: {len(pngs)} images")

        for img_path in pngs:
            img = Image.open(img_path)
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            img = img.resize(img_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            if grayscale:
                arr = np.expand_dims(arr, axis=-1)
            X_list.append(arr)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Shape: {X.shape}, Range: [{X.min():.3f}, {X.max():.3f}]")
    return X, y


def build_cnn(input_shape, num_classes=2):
    """Same CNN architecture, flexible input shape."""
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
    ])
    return model


def run_test(img_size, grayscale, label):
    """Train and evaluate one configuration."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  Size: {img_size}, Grayscale: {grayscale}")
    print(f"{'='*60}")

    channels = 1 if grayscale else 3
    X, y = load_spectrograms(SPEC_DIR, img_size, grayscale)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = build_cnn((img_size[0], img_size[1], channels))
    print(f"  Params: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    y_train_cat = to_categorical(y_train, 2)
    y_val_cat = to_categorical(y_val, 2)
    y_test_cat = to_categorical(y_test, 2)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n  >>> {label}: Test Acc = {acc*100:.2f}%, Loss = {loss:.4f}")
    print(f"  >>> Val Acc (best): {max(history.history['val_accuracy'])*100:.2f}%")

    return acc, history


if __name__ == "__main__":
    results = {}

    # Test 1: Grayscale 128x128 (the one that failed before)
    acc, _ = run_test((128, 128), grayscale=True,
                      label="Grayscale 128x128")
    results["gray_128"] = acc

    # Test 2: Grayscale 256x256 (original resolution)
    acc, _ = run_test((256, 256), grayscale=True,
                      label="Grayscale 256x256 (original)")
    results["gray_256"] = acc

    # Test 3: RGB 128x128 (what we've been using — baseline)
    acc, _ = run_test((128, 128), grayscale=False,
                      label="RGB 128x128 (current)")
    results["rgb_128"] = acc

    # Test 4: RGB 256x256 (original resolution, all channels)
    acc, _ = run_test((256, 256), grayscale=False,
                      label="RGB 256x256 (original)")
    results["rgb_256"] = acc

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:20s}: {acc*100:.2f}%")
    print(f"\nPaper baseline (grayscale 128x128x1): ~98%")
