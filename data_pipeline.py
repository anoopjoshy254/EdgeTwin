import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE, SMOTE

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# How much extra synthetic data to generate (relative to current size).
# 1.0 = double the minority class; 2.0 = triple it, etc.
AUGMENT_FACTOR = 1.5

# Standard deviation multiplier applied to noise injection
NOISE_STD_FACTOR = 0.02

# List of typical target column names in machine failure datasets
TARGET_CANDIDATES = ['failure', 'fail', 'machine_status', 'target', 'status', 'is_failure']


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def detect_target_column(df):
    """Automatically inspect dataset columns and identify the target column."""
    columns_lower = [col.lower() for col in df.columns]
    for candidate in TARGET_CANDIDATES:
        for idx, col in enumerate(columns_lower):
            if candidate in col:
                print(f"[*] Detected target column: '{df.columns[idx]}'")
                return df.columns[idx]
    raise ValueError(f"Could not automatically detect a target column among: {TARGET_CANDIDATES}")


def gaussian_noise_augmentation(X_minority, n_samples, noise_std_factor=NOISE_STD_FACTOR):
    """
    Generate synthetic samples by adding small Gaussian noise to randomly
    selected minority-class instances.

    Args:
        X_minority: 2D array of minority-class feature vectors.
        n_samples:  Number of new samples to create.
        noise_std_factor: Scale factor multiplied by per-feature std dev.

    Returns:
        Augmented samples array of shape (n_samples, n_features).
    """
    if len(X_minority) == 0 or n_samples <= 0:
        return np.empty((0, X_minority.shape[1]))

    # Compute per-feature standard deviation (used to scale noise)
    feature_std = np.std(X_minority, axis=0) + 1e-8

    # Randomly pick base samples (with replacement)
    indices = np.random.randint(0, len(X_minority), size=n_samples)
    base_samples = X_minority[indices]

    # Add scaled Gaussian noise
    noise = np.random.normal(0, noise_std_factor * feature_std, size=base_samples.shape)
    return base_samples + noise


def feature_interpolation_augmentation(X_minority, n_samples, alpha_range=(0.3, 0.7)):
    """
    Mixup-style augmentation: blend pairs of minority-class samples at a
    random interpolation ratio α ∈ alpha_range.

    new_sample = α * sample_A + (1-α) * sample_B

    Args:
        X_minority:   2D array of minority-class feature vectors.
        n_samples:    Number of new interpolated samples to create.
        alpha_range:  (min, max) blend factor range.

    Returns:
        Interpolated samples array of shape (n_samples, n_features).
    """
    if len(X_minority) < 2 or n_samples <= 0:
        return np.empty((0, X_minority.shape[1]))

    idx_a = np.random.randint(0, len(X_minority), size=n_samples)
    idx_b = np.random.randint(0, len(X_minority), size=n_samples)

    # Ensure a != b
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % len(X_minority)

    alpha = np.random.uniform(alpha_range[0], alpha_range[1], size=(n_samples, 1))
    blended = alpha * X_minority[idx_a] + (1 - alpha) * X_minority[idx_b]
    return blended


def augment_minority_class(X, y, augment_factor=AUGMENT_FACTOR):
    """
    Applies Gaussian noise + feature interpolation augmentation to the
    minority class.  Uses augment_factor to determine the total number of
    new samples (split equally between the two methods).

    Returns:
        X_aug, y_aug — originals + synthetic minority samples appended.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y

    minority_class = classes[np.argmin(counts)]
    majority_count = counts.max()
    minority_mask = (y == minority_class)
    X_minority = X[minority_mask]

    # How many synthetic samples do we want in total?
    target_minority_count = int(majority_count * augment_factor)
    n_extra = max(0, target_minority_count - counts.min())

    if n_extra == 0:
        print("[*] Minority class already sufficiently large. Skipping custom augmentation.")
        return X, y

    half = n_extra // 2
    remainder = n_extra - half

    print(f"[*] Gaussian noise augmentation: generating {half} samples...")
    X_noise = gaussian_noise_augmentation(X_minority, half)

    print(f"[*] Feature interpolation (mixup) augmentation: generating {remainder} samples...")
    X_interp = feature_interpolation_augmentation(X_minority, remainder)

    X_synthetic = np.vstack([part for part in [X_noise, X_interp] if len(part) > 0])
    y_synthetic = np.full(len(X_synthetic), minority_class)

    X_aug = np.vstack([X, X_synthetic])
    y_aug = np.concatenate([y, y_synthetic])

    print(f"[*] After custom augmentation: total samples = {len(X_aug)} "
          f"(added {len(X_synthetic)} synthetic minority samples)")
    return X_aug, y_aug


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def process_data(csv_path, output_dir="models"):
    """
    Reads the dataset, handles missing values, encodes categories, normalises,
    applies augmentation (Gaussian noise + feature interpolation), then
    BorderlineSMOTE, and returns stratified train/test splits.
    """
    print(f"[*] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Detect target
    target_col = detect_target_column(df)

    # Separate features and labels
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # 2. Drop obvious identifier columns
    cols_to_drop = []
    for col in X.columns:
        col_lower = col.lower()
        if col_lower == 'id' or col_lower.endswith('_id') or col_lower.startswith('id_'):
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"[*] Dropping ID columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols   = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    print(f"[*] Numerical features  ({len(numerical_cols)}): {numerical_cols}")
    print(f"[*] Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # 3. Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer,     numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    print("[*] Applying preprocessing (imputation + scaling + encoding)...")
    X_processed = preprocessor.fit_transform(X)

    # Save preprocessor & feature names for inference
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))

    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"f_{i}" for i in range(X_processed.shape[1])]
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))

    # Save valid categorical values so infer.py can show real options to the user
    cat_valid_values = {}
    for col in categorical_cols:
        cat_valid_values[col] = sorted(df[col].dropna().unique().tolist())
    joblib.dump(cat_valid_values, os.path.join(output_dir, 'categorical_values.pkl'))
    print(f"[*] Saved preprocessor to {output_dir}/preprocessor.pkl")
    print(f"[*] Saved categorical value map: {cat_valid_values}")

    # 4. Class imbalance check
    classes, counts = np.unique(y, return_counts=True)
    print("\n[*] Original class distribution:")
    for cls, cnt in zip(classes, counts):
        print(f"    Class {cls}: {cnt} samples ({cnt/len(y)*100:.1f}%)")

    minority_ratio = counts.min() / len(y)

    if len(classes) > 1 and minority_ratio < 0.4:
        print("\n[*] Dataset is imbalanced — applying augmentation pipeline:")

        # Step A: Gaussian noise + feature interpolation on minority class
        X_processed, y = augment_minority_class(X_processed, y, augment_factor=AUGMENT_FACTOR)

        # Step B: BorderlineSMOTE (focuses on decision-boundary minority samples)
        print("[*] Applying BorderlineSMOTE on augmented data...")
        try:
            smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
            X_resampled, y_resampled = smote.fit_resample(X_processed, y)
            print("[*] BorderlineSMOTE applied successfully.")
        except ValueError as e:
            # Borderline SMOTE can fail if minority class is still very small
            print(f"[!] BorderlineSMOTE failed ({e}). Falling back to standard SMOTE.")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_processed, y)

        _, final_counts = np.unique(y_resampled, return_counts=True)
        print(f"[*] After full augmentation pipeline: {dict(zip(classes, final_counts))}")

    else:
        print("[*] Dataset is fairly balanced. Skipping augmentation.")
        X_resampled, y_resampled = X_processed, y

    # 5. Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2, random_state=42, stratify=y_resampled
    )

    print(f"\n[*] Final dataset sizes — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Run this via train.py or provide a CSV path directly.")
