import joblib

from steps.step1_load import load_data
from steps.step3_preprocessing import preprocess
from steps.step6_tune import tune_xgboost
from steps.step7_tune_SMOTE import tune_smote_ratio
from steps.step8_threshold import find_optimal_threshold
from steps.step9_evaluate import final_evaluation


# step2_eda / Data_Visulization and step4_balance / step5_train are kept in
# the project as reference/legacy files. step2 is optional (uncomment below
# to regenerate the EDA dashboard). step4 and step5 are superseded by
# step6/step7, which now apply SMOTE inside the CV pipeline instead of
# balancing the training set upfront -- that's what fixes the leakage bug
# from the original pipeline.


def main():
    # Step 1: Load
    df = load_data('data/creditcard.csv')

    # Optional: regenerate the EDA dashboard
    # from step2_eda import run_eda
    # run_eda(df)

    # Step 3: Feature engineering + train/val/test split (scaler fit on train only)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_amount = preprocess(df)

    # Step 6: Tune XGBoost hyperparameters (SMOTE inside the CV pipeline, F2-scored)
    best_xgb_pipeline = tune_xgboost(X_train, y_train, X_val, y_val)
    champion_params = {
        'max_depth': best_xgb_pipeline.named_steps['xgb'].max_depth,
        'learning_rate': best_xgb_pipeline.named_steps['xgb'].learning_rate,
        'n_estimators': best_xgb_pipeline.named_steps['xgb'].n_estimators,
    }

    # Step 7: Tune SMOTE ratio + scale_pos_weight, locking in the step 6 champion params
    best_pipeline = tune_smote_ratio(X_train, y_train, X_val, y_val, champion_params)

    # Step 8: Pick the decision threshold on the validation set
    threshold = find_optimal_threshold(best_pipeline, X_val, y_val, beta=2)

    # Step 9: Final, one-time evaluation on the untouched test set
    final_evaluation(best_pipeline, X_test, y_test, threshold=threshold)

    # Save artifacts for reuse (e.g. scoring new transactions later)
    joblib.dump(best_pipeline, 'fraud_model.joblib')
    joblib.dump(scaler_amount, 'amount_scaler.joblib')
    joblib.dump(threshold, 'decision_threshold.joblib')
    print("Saved: fraud_model.joblib, amount_scaler.joblib, decision_threshold.joblib")



if __name__ == "__main__":
    main()