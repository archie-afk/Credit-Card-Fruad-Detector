from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score, classification_report, confusion_matrix


def tune_smote_ratio(X_train, y_train, X_val, y_val, champion_params, random_state=42):
    """
    Step 7: Hyperparameter tuning for SMOTE sampling ratio + scale_pos_weight.

    Locks in step 6's champion XGBoost settings (max_depth, learning_rate,
    n_estimators) and sweeps SMOTE's sampling ratio alongside XGBoost's
    scale_pos_weight -- combining moderate oversampling with class weighting
    usually beats using SMOTE alone.

    IMPROVEMENTS vs the original version:
      - Scored on F2 instead of plain recall. Plain recall is trivially
        maximised by flagging everything as fraud, so the original grid
        search would drift toward high-false-alarm settings.
      - Evaluated here on the VALIDATION set, not test -- test stays
        untouched until step 9.

    champion_params: dict from step 6, e.g.
        {'max_depth': 8, 'learning_rate': 0.2, 'n_estimators': 150}
    """
    print("--- Step 7: Tuning SMOTE Sampling Ratio + Class Weight ---")

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('xgb', XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1,
            **champion_params,
        )),
    ])

    param_grid = {
        'smote__sampling_strategy': [0.05, 0.10, 0.15, 0.20, 0.30],
        'xgb__scale_pos_weight': [1, 3, 5],
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

    print("Evaluating SMOTE ratio / scale_pos_weight combinations...")
    grid_search.fit(X_train, y_train)

    print("\nGrid Search Complete!")
    print(f"Best SMOTE Strategy Found: {grid_search.best_params_['smote__sampling_strategy']}")
    print(f"Best scale_pos_weight Found: {grid_search.best_params_['xgb__scale_pos_weight']}")
    print(f"Best Training Cross-Validation F2-Score: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_val)

    print("\n" + "=" * 20 + " STEP 7 VALIDATION CHECK " + "=" * 20)
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("=" * 63 + "\n")

    return best_pipeline