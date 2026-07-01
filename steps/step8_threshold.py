import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(model, X_val, y_val, beta=2):
    """
    Step 8: Pick the probability threshold that maximises F-beta on the
    VALIDATION set (never the test set -- that would leak test information
    into a modelling decision).

    Using the default 0.5 cutoff is arbitrary; sweeping the precision-recall
    curve and picking the threshold that matches your actual cost tradeoff
    (missed fraud vs. false alarm) is usually the single biggest lever left
    once the model itself is tuned. beta=2 weights recall higher than
    precision, appropriate when missing fraud is costlier than a false alarm.
    """
    print("--- Step 8: Tuning Decision Threshold ---")

    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

    # precision/recall have one more element than thresholds; align them
    precision, recall = precision[:-1], recall[:-1]
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)

    best_idx = np.argmax(f_beta)
    best_threshold = thresholds[best_idx]

    print(f"Optimal threshold (F{beta}): {best_threshold:.4f}")
    print(f"  Validation precision at this threshold: {precision[best_idx]:.4f}")
    print(f"  Validation recall at this threshold:    {recall[best_idx]:.4f}")
    print(f"  Validation F{beta}-score at this threshold: {f_beta[best_idx]:.4f}\n")

    return best_threshold