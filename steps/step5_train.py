import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, confusion_matrix

def train_xgboost(X_train_bal, y_train_bal, X_test, y_test):
    """
    Trains an XGBoost model on the balanced training data 
    and evaluates it on the untouched test data.
    """
    print("--- Training XGBoost Model ---")
    
    # 1. Initialize the model
    # scale_pos_weight can be left at 1 since SMOTE already balanced the data
    model = XGBClassifier(
        n_estimators=100,      # Number of trees to build
        max_depth=8,           # Maximum depth of each tree
        learning_rate=0.1,     # Step size shrinkage to prevent overfitting
        random_state=42,
        eval_metric='logloss'
    )
    
    # 2. Train the model using the BALANCED training data
    print("Fitting model to balanced training data (this may take a moment)...")
    model.fit(X_train_bal, y_train_bal)
    print("Training complete!")
    
    # 3. Predict on the UNTOUCHED real-world test data
    y_pred = model.predict(X_test)
    
    # 4. Evaluate performance
    print("\n" + "="*20 + " EVALUATION " + "="*20)
    print("Classification Report (Evaluated on Real-World Test Split):")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Legitimate: {cm[0][0]} | False Fraud (False Alarms): {cm[0][1]}")
    print(f"False Legitimate (Missed Fraud): {cm[1][0]} | True Fraud Caught: {cm[1][1]}")
    print("="*52 + "\n")
    
    return model