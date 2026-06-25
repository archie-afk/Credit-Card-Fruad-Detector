import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    return df