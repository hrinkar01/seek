import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

dataset_input = input("Enter the dataset path: ")
dataset = pd.read_csv(dataset_input)
target_column = input("What you want to predict: ")
def auto_train_model(df, target):
    if df[target].isnull().any():
        missing_count = df[target].isnull().sum()
        print(f"Target column '{target}' contains {missing_count} missing values.")
        print(f"Dropping rows where target '{target}' is blank to prevent training failure.")
        
        # Keep only the rows where the target answer actually exists
        df = df[df[target].notnull()].copy()
        
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
        
    #model type detector
    is_text = y.dtype in ['object', 'category', 'str'] 
    unique_ratio = y.nunique() / len(y)
    
    if is_text or (y.nunique() <= 20):
        problem_type = "classification"
        metric = "accuracy"
        models_pool = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Gradient Boosting Classifier": HistGradientBoostingClassifier(random_state=42)
        }
    else:
        problem_type = "regression"
        metric = "r2"
        models_pool = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor": HistGradientBoostingRegressor(random_state=42)
        }
        
    print(f"\nSeek Intelligence Layer: Detected a [{problem_type}] Problem!")
    print("=========================================================================")
    print(f"{'Machine Learning Model':<30} | {'CV Mean Evaluation Score (' + metric + ')':<18}")
    print("-------------------------------------------------------------------------")  
              
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['str','object', 'category']).columns.tolist()
    
    # Create an empty list to dynamically collect our active transformers
    active_transformers = []

    # Only build the numerical lane if we ACTUALLY have numeric columns!
    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        active_transformers.append(('num', numeric_transformer, numeric_features))
        print(f"Pipeline routed numerical features: {numeric_features}")
        

    # Only build the categorical lane if we ACTUALLY have text columns!
    elif len(categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        active_transformers.append(('cat', categorical_transformer, categorical_features))
        print(f"Pipeline routed categorical features: {categorical_features}")
        
    print("------------------------------------------------------")
    # Pass ONLY the active lanes to the Traffic Controller
    preprocessor = ColumnTransformer(transformers=active_transformers)
    
    for name, model in models_pool.items():
        seek_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        cv_scores = cross_val_score(seek_pipeline, X, y, cv=5, scoring=metric, n_jobs=-1)
        print(f"🏋️ {name:<28} | {np.mean(cv_scores):<18.2%}")
    return X, y
    
    
X_clean, y_clean = auto_train_model(dataset, target_column)
print("------------------------------------------------------")
print(f"Original Dataset Rows: {len(dataset)}")
print(f"Final Cleaned Rows: {len(X_clean)}")
print(f"Total Rows Lost: {len(dataset) - len(X_clean)}")