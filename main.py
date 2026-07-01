from steps.step1_load import load_data
from steps.step2_eda import run_eda
from steps.step3_preprocessing import preprocess
from steps.step4_balance import balance_training_data
from steps.step5_train import train_xgboost
from steps.step6_tune import tune_xgboost
from steps.step7_tune_SMOTE import tune_smote_ratio

# ── Run the full pipeline ──────────────────────────────────────

df = load_data('data/creditcard.csv')

#run_eda(df)

X_train, X_test, y_train, y_test = preprocess(df)

X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)


fraud_model = train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)

#best_fraud_model = tune_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)

#final_production_model = tune_smote_ratio(X_train, y_train, X_test, y_test)