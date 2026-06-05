import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

dataset_input = input("Enter the dataset path: ")
dataset = pd.read_csv(dataset_input)
target_column = input("What you want to predict: ")
def auto_train_model(df, target):
    y = df[target]
    X = df.drop(columns=[target])

    total_rows = len(df)
    necessary_features = []
    
    for col in X.columns:
        
        #Drop column if it's 100% unique 'numbers' (like PassengerId) ----
        if X[col].dtype in ['int64'] and X[col].nunique() == total_rows:
            print(f"🗑️ Automatically dropping numerical index/ID column: {col}")
            continue
        
        #Filter high-cardinality metadata strings (IDs, Names etc)
        if X[col].dtype == 'object' or X[col].dtype == 'str' or X[col].dtype == 'category':
            if X[col].nunique() > (total_rows * 0.5):
                print(f"🗑️ Automatically dropping useless metadata column: {col}")
                continue
            
        # Filter low-variance numerical traps (Constant columns)
        if X[col].dtype in ['int64', 'float64']:
            most_frequent_pct = X[col].value_counts(normalize=True).iloc[0]
            if most_frequent_pct > 0.99:
                print(f"🗑️ Automatically dropping constant numerical column: {col}")
                continue
            
        necessary_features.append(col)
        
    #Reassigning X with .copy() to completely eliminate memory warnings
    X = X[necessary_features].copy()
    print("\n--- Final Necessary Columns(X) for Prediction ---")
    print(X.columns)
    
    #processing missing data:
    # Loop through columns one by one instead of checking the whole dataset at once
    for col in list(X.columns):
        missing_pct = X[col].isnull().mean()
        if missing_pct > 0: 
            # Condition A: Low Missing Data (<= 15%) -> Drop row samples
            if missing_pct < 0.15:
                print(f"❌ Low Missing ({missing_pct:.2%}) in '{col}': Dropping faulty rows.")
                X = X.dropna(subset=[col])
                y = y.loc[X.index] # Keep y perfectly synchronized!
            
            # Condition B: High Missing Data (> 15%) -> Split route imputation
            else:
                if X[col].dtype in ['object', 'category', 'str']:
                    print(f"📝 High Missing ({missing_pct:.2%}) in Text Column '{col}': Filling with 'unknown'")
                    X[col] = X[col].fillna("unknown")
                else:
                    print(f"🛠️ High Missing ({missing_pct:.2%}) in Numeric Column '{col}': Filling with median")
                    X[col] = X[col].fillna(X[col].median())
                    

    return X, y
    
    
X_clean, y_clean = auto_train_model(dataset, target_column)
print(f"Original Dataset Rows: {len(dataset)}")
print(f"Final Cleaned Rows: {len(X_clean)}")
print(f"Total Rows Lost: {len(dataset) - len(X_clean)}")