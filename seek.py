import pandas as pd
import numpy as np # type: ignore
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor

def auto_train_model(df, target):
    # 1. Clean Target column first
    if df[target].isnull().any():
        missing_count = df[target].isnull().sum()
        print(f"Target column '{target}' contains {missing_count} missing values.")
        print(f"Dropping rows where target '{target}' is blank to prevent training failure.")
        df = df[df[target].notnull()].copy()
        
    y = df[target]
    X = df.drop(columns=[target])
    
    # 2. First Loop: Filter out obvious structural junk (IDs, Constants)
    total_rows = len(df)
    structural_clean_cols = []
    for col in X.columns:
        # Drop unique integer/string IDs (where every single row has a unique identifier value)
        if X[col].dtype in ['int64'] and X[col].nunique() == total_rows:
            continue
        if X[col].dtype in ['object', 'str', 'category'] and X[col].nunique() > (total_rows * 0.5):
            continue
        # Drop feature constants (where 99% of rows hold the exact same value)
        if X[col].dtype in ['int64', 'float64'] and len(X[col].value_counts()) > 0:
            if X[col].value_counts(normalize=True).iloc[0] > 0.99:
                continue
        structural_clean_cols.append(col)
    
    X = X[structural_clean_cols].copy()

    # 3. Second Loop: Process remaining features using industry percentage rules
    for col in list(X.columns):
        if col not in X.columns:
            continue
            
        missing_pct = X[col].isnull().mean()
        
        if missing_pct > 0:
            # Rule 1: Extreme Missing (> 50%) -> Drop the entire column
            if missing_pct > 0.50:
                print(f"Automatically dropping column '{col}' due to extreme missing data ({missing_pct:.2%})")
                X = X.drop(columns=[col])
                
            # Rule 2: Low Missing (< 5%) -> Drop only the faulty rows safely
            elif missing_pct < 0.05:
                print(f"Low Missing ({missing_pct:.2%}) in '{col}': Dropping faulty rows.")
                X = X.dropna(subset=[col])
                y = y.loc[X.index] # Keep y perfectly synchronized!
                
            # Rule 3: Medium Missing (5% to 50%) -> Hand over to SimpleImputer lane
            else:
                print(f"Medium Missing ({missing_pct:.2%}) in '{col}': Retaining column for Imputer lane.")

    # 4. Circuit Breaker check
    if X.shape[1] == 0:
        raise ValueError(
            "🚨 Seek Automated Training Aborted: All feature columns were dropped! "
            "This happens if your dataset columns are entirely composed of unique IDs, "
            "high-cardinality metadata, constants, or contain >50% missing values."
        )
        
    # 5. Model type detector block (RUN THIS FIRST so variables exist!)
    is_text = y.dtype in ['object', 'category', 'str'] 
    
    if is_text or (y.nunique() <= 20):
        problem_type = "classification"
        metric = "accuracy"
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        gb = HistGradientBoostingClassifier(random_state=42)
        
        models_pool = {
            "Logistic Regression": lr,
            "Random Forest Classifier": rf,
            "Gradient Boosting Classifier": gb,
            "Seek Voting Ensemble": VotingClassifier(
                estimators=[('lr', lr), ('rf', rf), ('gb', gb)], 
                voting='soft'
            )
        }
    else:
        problem_type = "regression"
        metric = "r2"
        
        lin = LinearRegression()
        rf = RandomForestRegressor(random_state=42)
        gb = HistGradientBoostingRegressor(random_state=42)
        
        models_pool = {
            "Linear Regression": lin,
            "Random Forest Regressor": rf,
            "Gradient Boosting Regressor": gb,
            "Seek Voting Ensemble": VotingRegressor(
                estimators=[('lin', lin), ('rf', rf), ('gb', gb)]
            )
        }
        
    print(f"\nSeek: Detected a [{problem_type}] Problem!")
    print("\n--- Final Necessary Columns (X) for Prediction ---")
    print(list(X.columns))
    print("=========================================================================")
    print(f"{'Machine Learning Model':<30} | {'CV Mean Evaluation Score (' + metric + ')':<18}")
    print("-------------------------------------------------------------------------")  
              
    # 6. Route features down preprocessing paths
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['str','object', 'category']).columns.tolist()
    
    active_transformers = []

    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        active_transformers.append(('num', numeric_transformer, numeric_features))
        print(f"Pipeline routed numerical features: {numeric_features}")
        
    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        active_transformers.append(('cat', categorical_transformer, categorical_features))
        print(f"Pipeline routed categorical features: {categorical_features}")
        
    print("------------------------------------------------------")
    preprocessor = ColumnTransformer(transformers=active_transformers)
    
    # 7. Cross-Validation and Leaderboard Compile
    results = []

    for name, model in models_pool.items():
        seek_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        try:
            cv_scores = cross_val_score(
                seek_pipeline,X,y,cv=5,scoring=metric,n_jobs=-1
            )
            mean_score = np.mean(cv_scores)
            print(f"{name:<30} | {mean_score:<18.2%}")
            
            results.append({
                "Model": name,
                "Score": mean_score
            })
        except Exception as e:
            print(f"{name:<30} | Failed Training: {e}")

    if not results:
        raise RuntimeError("All models in the pool failed cross-validation training layouts.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Score", ascending=False)

    best_model = results_df.iloc[0]["Model"]
    best_score = results_df.iloc[0]["Score"]

    return results_df, best_model, best_score