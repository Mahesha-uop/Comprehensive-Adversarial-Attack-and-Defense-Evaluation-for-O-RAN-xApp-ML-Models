
import os
import json
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


SPECTROGRAM_BASE_DIR = "newdataset"
KPM_BASE_DIR = "kpm-data2"


KPM_FEATURE_KEYS = ['ul_snr', 'ul_bitrate', 'ul_bler', 'ul_mcs']
KPM_WINDOWS = 15



def load_spectrogram_dataset(base_dir, img_size=(128, 128)):
    """
    Load spectrograms as RGB (3 channels).
    RGB channels carry different information — grayscale destroys features.
    soi -> class 0, cwi -> class 1.
    """
    CLASS_MAPPING = {
        'soi': 0,
        'cwi': 1,
    }

    X_list, y_list = [], []

    for folder, label in CLASS_MAPPING.items():
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: '{folder}/' not found, skipping.")
            continue

        pngs = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        print(f"  Loading {len(pngs)} images from '{folder}/' as class {label}...")

        for img_path in pngs:
            try:
                img = Image.open(img_path).convert('RGB') 
                img = img.resize(img_size, Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0  
                X_list.append(img_array)
                y_list.append(label)
            except Exception as e:
                print(f"  ERROR: {img_path}: {e}")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n  Spectrogram dataset: X={X.shape}, y={y.shape}")
    print(f"  Class 0 (SOI): {np.sum(y==0)}, Class 1 (CWI): {np.sum(y==1)}")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")
    return X, y



def extract_kpm_from_json(filepath, feature_keys):
    """Extract UE-level KPM measurements from a single JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    measurements = []
    for entry in data:
        if entry.get('type') != 'metrics':
            continue
        for cell in entry.get('cell_list', []):
            cc = cell.get('cell_container', {})
            for ue in cc.get('ue_list', []):
                ue_data = ue.get('ue_container', ue)
                try:
                    row = [float(ue_data.get(k, 0)) for k in feature_keys]
                    measurements.append(row)
                except (ValueError, TypeError):
                    continue

    return np.array(measurements)


def create_sliding_windows(measurements, window_size=15):
    """Stack t consecutive measurements into one input vector."""
    if len(measurements) < window_size:
        return np.array([])
    windows = []
    for i in range(len(measurements) - window_size + 1):
        windows.append(measurements[i:i + window_size].flatten())
    return np.array(windows)


def load_kpm_dataset(base_dir, feature_keys, window_size=KPM_WINDOWS):
    """Load KPM dataset: 4 clean -> class 0, 6 jammer -> class 1."""
    json_files = sorted(glob.glob(os.path.join(base_dir, "*.json")))
    clean_files = [f for f in json_files if 'clean' in os.path.basename(f).lower()]
    jammer_files = [f for f in json_files if 'jammer' in os.path.basename(f).lower()]

    print(f"  Clean files: {len(clean_files)}, Jammer files: {len(jammer_files)}")
    print(f"  Feature keys: {feature_keys}")

    all_X, all_y = [], []

    print("\n  Processing clean files...")
    for jf in clean_files:
        measurements = extract_kpm_from_json(jf, feature_keys)
        windows = create_sliding_windows(measurements, window_size)
        print(f"    {os.path.basename(jf)}: {len(measurements)} -> {len(windows)} windows")
        if len(windows) > 0:
            all_X.append(windows)
            all_y.append(np.zeros(len(windows), dtype=int))

    print("\n  Processing jammer files...")
    for jf in jammer_files:
        measurements = extract_kpm_from_json(jf, feature_keys)
        windows = create_sliding_windows(measurements, window_size)
        print(f"    {os.path.basename(jf)}: {len(measurements)} -> {len(windows)} windows")
        if len(windows) > 0:
            all_X.append(windows)
            all_y.append(np.ones(len(windows), dtype=int))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0)

    # Min-max normalization
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X = (X - X_min) / X_range

    print(f"\n  KPM dataset: X={X.shape}, y={y.shape}")
    print(f"  Input dim: {X.shape[1]} (expected: {len(feature_keys) * window_size})")
    print(f"  Class 0 (clean): {np.sum(y==0)}, Class 1 (jammer): {np.sum(y==1)}")
    return X, y




def split_dataset(X, y):
    """80/10/10 stratified split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    print("\n=== SPECTROGRAMS (RGB) ===")
    if os.path.isdir(SPECTROGRAM_BASE_DIR):
        X_spec, y_spec = load_spectrogram_dataset(SPECTROGRAM_BASE_DIR)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_spec, y_spec)
        np.save("X_spec_train.npy", X_train)
        np.save("y_spec_train.npy", y_train)
        np.save("X_spec_val.npy", X_val)
        np.save("y_spec_val.npy", y_val)
        np.save("X_spec_test.npy", X_test)
        np.save("y_spec_test.npy", y_test)
        print("  Saved spectrogram .npy files")
    else:
        print(f"  Not found: {SPECTROGRAM_BASE_DIR}")

    print("\n=== KPMs ===")
    if os.path.isdir(KPM_BASE_DIR):
        X_kpm, y_kpm = load_kpm_dataset(KPM_BASE_DIR, feature_keys=KPM_FEATURE_KEYS)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_kpm, y_kpm)
        np.save("X_kpm_train.npy", X_train)
        np.save("y_kpm_train.npy", y_train)
        np.save("X_kpm_val.npy", X_val)
        np.save("y_kpm_val.npy", y_val)
        np.save("X_kpm_test.npy", X_test)
        np.save("y_kpm_test.npy", y_test)
        print("  Saved KPM .npy files")
    else:
        print(f"  Not found: {KPM_BASE_DIR}")

    print("\nDone! Now run replicate_base_paper.py")
