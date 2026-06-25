import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def tune_xgboost(X_train_bal, y_train_bal, X_test, y_test):
    """
    Runs a Grid Search across specified XGBoost hyperparameters
    to find the combination that yields the highest F1-score.
    """
    print("--- Starting Hyperparameter Grid Search ---")
    
    # 1. Initialize a base model
    base_model = XGBClassifier(random_state=42, eval_metric='logloss')
    
    # 2. Define the grid of parameters you want to test
    # (Keep it relatively small at first, as every combo multiplies training time)
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150]
    }
    
    # 3. Set up GridSearchCV
    # cv=3 means 3-fold cross-validation. scoring='f1' optimizes for precision+recall balance.
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,        # Use all available CPU cores to speed it up
        verbose=2         # Prints progress updates to the console
    )
    
    # 4. Fit the grid search on your BALANCED training data
    print("Searching for the best parameters... This will take a few minutes.")
    grid_search.fit(X_train_bal, y_train_bal)
    
    # 5. Extract the champion model
    best_model = grid_search.best_estimator_
    
    print("\nGrid Search Complete!")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Training Cross-Validation F1-Score: {grid_search.best_score_:.4f}")
    
    # 6. Evaluate the champion model on the untouched test data
    y_pred = best_model.predict(X_test)
    
    print("\n" + "="*20 + " CHAMPION MODEL EVALUATION " + "="*20)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*60 + "\n")
    
    return best_model