from sklearn.metrics import classification_report, confusion_matrix, average_precision_score


def final_evaluation(model, X_test, y_test, threshold=0.5):
    """
    Step 9: The final, one-time evaluation on the untouched test set, using
    the tuned decision threshold from step 8.

    This is the only place the test set is used -- it was never touched
    during hyperparameter search (step 6-7) or threshold tuning (step 8),
    so this is an honest estimate of real-world performance.
    """
    print("--- Step 9: Final Evaluation on Held-Out Test Set ---")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("\n" + "=" * 20 + f" FINAL EVALUATION (threshold={threshold:.3f}) " + "=" * 20)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Legitimate: {cm[0][0]} | False Fraud (False Alarms): {cm[0][1]}")
    print(f"False Legitimate (Missed Fraud): {cm[1][0]} | True Fraud Caught: {cm[1][1]}")

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")
    print("=" * 70 + "\n")

    return y_pred, y_proba