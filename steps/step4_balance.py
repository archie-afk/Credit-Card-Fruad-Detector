# %% [1] Function: Handle Data Imbalance with SMOTE
from imblearn.over_sampling import SMOTE
import pandas as pd

def balance_training_data(X_train, y_train):
    """
    Applies SMOTE oversampling to fix class imbalance on the training sets.
    Returns the perfectly balanced training features and targets.
    """
    print("--- Handling Data Imbalance (SMOTE) ---")
    print(f"Original training size: {X_train.shape[0]} rows")
    print(f"Original fraud cases:   {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    
    # Initialize SMOTE
    smote = SMOTE(sampling_strategy=0.10, random_state=42)
    

    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("\nOversampling complete!")
    print(f"Balanced training size: {X_train_balanced.shape[0]} rows")
    
    # Combine counts into a neat quick table for verification
    counts = y_train_balanced.value_counts()
    percentages = y_train_balanced.value_counts(normalize=True) * 100
    balance_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    print(balance_df)
    print("="*40 + "\n")
    
    return X_train_balanced, y_train_balanced