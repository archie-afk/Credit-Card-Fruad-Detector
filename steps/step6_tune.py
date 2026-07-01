from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score, classification_report, confusion_matrix


def tune_xgboost(X_train, y_train, X_val, y_val, random_state=42):
    """
    Step 6: Hyperparameter tuning for XGBoost.

    IMPROVEMENTS vs the original version:
      - SMOTE now lives INSIDE the pipeline, so it's refit fresh on every
        CV fold. The original version tuned on data that had already been
        SMOTE'd before the CV split, which leaks synthetic samples derived
        from each other across folds and inflates the CV score.
      - Scoring is F2 (recall-weighted) instead of plain F1. F2 still
        penalises false alarms, but weights catching fraud more heavily,
        which better matches the actual business priority.
      - Evaluated here on the VALIDATION set (not test) -- the test set is
        reserved for the final one-time evaluation in step 9.
    """
    print("--- Step 6: Starting Hyperparameter Grid Search ---")

    pipeline = Pipeline([
        ('smote', SMOTE(sampling_strategy=0.1, random_state=random_state)),
        ('xgb', XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1)),
    ])

    param_grid = {
        'xgb__max_depth': [4, 6, 8],
        'xgb__learning_rate': [0.05, 0.1, 0.2],
        'xgb__n_estimators': [50, 100, 150],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=f2_scorer,
        n_jobs=-1,
        verbose=2,
    )

    print("Searching for the best parameters... this will take a few minutes.")
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_

    print("\nGrid Search Complete!")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Training Cross-Validation F2-Score: {grid_search.best_score_:.4f}")

    y_pred = best_pipeline.predict(X_val)

    print("\n" + "=" * 20 + " STEP 6 VALIDATION CHECK " + "=" * 20)
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("=" * 63 + "\n")

    return best_pipeline