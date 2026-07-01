import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def engineer_features(df):
    """
    Adds:
      - Amount_Log: log1p of Amount, to tame the heavy right skew
        (median £22, max £25,691)
      - Hour_Sin / Hour_Cos: cyclic encoding of hour-of-day, since fraud
        often clusters at specific hours and raw Time (seconds elapsed)
        doesn't capture that cleanly
    """
    df = df.copy()
    df['Amount_Log'] = np.log1p(df['Amount'])

    hour_of_day = (df['Time'] % 86400) / 3600.0  # 0-24
    df['Hour_Sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * hour_of_day / 24)

    return df


def preprocess(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Step 3: Feature engineering + a three-way train/val/test split.

    IMPORTANT FIX vs the original version: the scaler is now fit on the
    TRAINING split only, then reused to transform val/test. The original
    code fit StandardScaler on the full dataframe before splitting, which
    leaks test-set statistics into training.

    Returns X_train, X_val, X_test, y_train, y_val, y_test, scaler_amount
      - train: used for cross-validated hyperparameter search (steps 6-7)
      - val:   used to pick the decision threshold (step 8)
      - test:  touched exactly once, for the final evaluation (step 9)
    """
    df = engineer_features(df)

    X = df.drop(columns=['Class', 'Time', 'Amount'])
    y = df['Class']

    # Peel off the test set first
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Then peel off a validation set from what's left
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=random_state, stratify=y_train_full
    )

    # Fit scaler on TRAIN ONLY (Hour_Sin/Hour_Cos need no scaling -- already
    # bounded to [-1, 1] by construction)
    scaler_amount = StandardScaler().fit(X_train[['Amount_Log']])

    def apply_scaling(X_split):
        X_split = X_split.copy()
        X_split['Amount_Scaled'] = scaler_amount.transform(X_split[['Amount_Log']])
        return X_split.drop(columns=['Amount_Log'])

    X_train = apply_scaling(X_train)
    X_val = apply_scaling(X_val)
    X_test = apply_scaling(X_test)

    print("Features shape:", X.shape)
    print("Target shape:  ", y.shape)
    print("\nTraining set:  ", X_train.shape, f"({y_train.sum()} fraud)")
    print("Validation set:", X_val.shape, f"({y_val.sum()} fraud)")
    print("Test set:      ", X_test.shape, f"({y_test.sum()} fraud)")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_amount