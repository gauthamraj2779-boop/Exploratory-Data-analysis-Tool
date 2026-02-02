import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def determine_problem_type(y):
    if pd.api.types.is_numeric_dtype(y):
        # Check for float values that are not integers
        is_float = pd.api.types.is_float_dtype(y)
        if is_float:
            # Drop NA and check if values are effectively integers
            y_clean = y.dropna()
            # If any value has a non-zero decimal part, it's regression
            # Using a small tolerance for floating point comparisons
            is_integer_like = np.all(np.abs(y_clean - np.round(y_clean)) < 1e-9)
            if not is_integer_like:
                return "regression"

        nunique = y.nunique(dropna=True)
        if nunique <= 20:
            return "classification"
        return "regression"
    return "classification"

def get_preprocessor(X):
    # Identify column types using more robust selection
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns
    
    # Exclude datetime columns from categorical if any (as they might break OneHotEncoder)
    # Ideally we should handle them, but for now we'll ensure we don't treat them as simple categories unless they are object/category
    # Re-refining categorical selection:
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # Create preprocessors
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor


def train_and_evaluate(df, target, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple models (Regression or Classification) based on target type.
    """
    # Separate features and target
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]
    
    problem = determine_problem_type(y)
    
    # Split data first to avoid leakage (though preprocessor is fitted on train, splitting first is cleaner conceptually)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Exclude sequence-like/object vector columns (lists/tuples/arrays) from features
    excluded_columns = []
    for col in X.columns:
        s = X[col]
        try:
            has_sequences = s.apply(
                lambda v: isinstance(v, (list, tuple)) or (
                    hasattr(v, 'tolist') and not isinstance(v, (str, bytes))
                )
            ).any()
        except Exception:
            has_sequences = False
        if has_sequences:
            excluded_columns.append(col)
    if excluded_columns:
        X = X.drop(columns=excluded_columns)
        X_train = X_train.drop(columns=excluded_columns, errors="ignore")
        X_test = X_test.drop(columns=excluded_columns, errors="ignore")
    
    if X.shape[1] == 0:
        return {
            "problem_type": problem,
            "metrics": {},
            "models": {},
            "best_model": None,
            "best_model_score": 0.0,
            "feature_names": [],
            "feature_types": {},
            "excluded_columns": excluded_columns
        }
    
    # Get preprocessor
    preprocessor = get_preprocessor(X)
    
    results = {"problem_type": problem}
    
    if problem == "classification":
        # Initialize models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "SVC": SVC(probability=True, random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(random_state=random_state),
            "Gaussian Naive Bayes": GaussianNB(),
            "MLP Classifier": MLPClassifier(max_iter=1000, random_state=random_state)
        }
        
        metrics = {}
        trained_models = {}
        best_model_name = None
        best_score = -1
        
        for name, model in models.items():
            try:
                # Create a full pipeline for each model
                clf = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])
                
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average="macro")
                
                metrics[f"{name}_accuracy"] = float(acc)
                metrics[f"{name}_f1_macro"] = float(f1)
                trained_models[name] = clf # Store the pipeline
                
                if acc > best_score:
                    best_score = acc
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        results["metrics"] = metrics
        results["models"] = trained_models
        results["best_model"] = best_model_name
        results["best_model_score"] = best_score
        results["feature_names"] = X.columns.tolist()
        results["feature_types"] = X.dtypes.astype(str).to_dict()
        if excluded_columns:
            results["excluded_columns"] = excluded_columns
        
        # Feature importance extraction
        importances_dict = {}
        
        # Define known features for mapping
        numeric_features_list = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features_list = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Helper to get feature names from preprocessor
        def get_feature_names(model_pipeline):
            try:
                preprocessor = model_pipeline.named_steps['preprocessor']
                # Try modern sklearn get_feature_names_out
                if hasattr(preprocessor, 'get_feature_names_out'):
                    return preprocessor.get_feature_names_out()
                
                # Fallback for older sklearn or if simple lookup fails
                # Reconstruct manually based on known structure
                output_features = []
                
                # Numeric features (passed through StandardScaler)
                # We need to access the transformer list
                # format: (name, transformer, columns)
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'remainder' and trans == 'drop':
                        continue
                    if hasattr(trans, 'get_feature_names_out'):
                        names = trans.get_feature_names_out(cols)
                    elif hasattr(trans, 'get_feature_names'):
                        names = trans.get_feature_names(cols)
                    else:
                        names = cols # fallback
                    output_features.extend(names)
                return output_features
            except Exception as e:
                # print(f"Feature name extraction failed: {e}")
                return None
        
        def map_to_original_column(feature_name, num_feats, cat_feats):
            # Check for prefixes added by ColumnTransformer
            if feature_name.startswith('num__'):
                clean_name = feature_name[5:]
                if clean_name in num_feats:
                    return clean_name
            
            if feature_name.startswith('cat__'):
                clean_name = feature_name[5:]
                # Match against categorical features (longest match wins)
                matches = [col for col in cat_feats if clean_name.startswith(col)]
                if matches:
                    return max(matches, key=len)
            
            # Fallback if no prefix or standard matching fails
            if feature_name in num_feats:
                return feature_name
            
            matches = [col for col in cat_feats if feature_name.startswith(col)]
            if matches:
                return max(matches, key=len)
                
            return feature_name # Return original if no match found

        for name, pipeline in trained_models.items():
            try:
                model_step = pipeline.named_steps['classifier']
                feature_names = get_feature_names(pipeline)
                
                if feature_names is None:
                    continue

                importances = None
                if hasattr(model_step, "feature_importances_"):
                    importances = model_step.feature_importances_
                elif hasattr(model_step, "coef_"):
                    importances = np.abs(model_step.coef_)
                    if importances.ndim > 1:
                        importances = importances.mean(axis=0) # Handle multi-class coefs
                
                if importances is not None and len(importances) == len(feature_names):
                    # Create DF with raw features
                    raw_df = pd.DataFrame({
                        "Raw_Feature": feature_names,
                        "Importance": importances
                    })
                    
                    # Map to original columns
                    raw_df["Feature"] = raw_df["Raw_Feature"].apply(
                        lambda x: map_to_original_column(str(x), numeric_features_list, categorical_features_list)
                    )
                    
                    # Aggregate by original column
                    feat_imp = raw_df.groupby("Feature")["Importance"].sum().reset_index()
                    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
                    
                    importances_dict[name] = feat_imp
            except Exception as e:
                # print(f"Could not extract importance for {name}: {e}")
                continue
                
        results["feature_importances"] = importances_dict


    else:
        # Initialize models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=random_state),
            "Lasso Regression": Lasso(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "AdaBoost": AdaBoostRegressor(random_state=random_state),
            "ElasticNet": ElasticNet(random_state=random_state),
            "MLP Regressor": MLPRegressor(max_iter=1000, random_state=random_state)
        }
        
        # Train and evaluate
        metrics = {}
        trained_models = {}
        best_model_name = None
        best_score = -float('inf')
        
        for name, model in models.items():
            try:
                # Create a full pipeline for each model
                reg = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', model)])
                
                reg.fit(X_train, y_train)
                pred = reg.predict(X_test)
                
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                
                metrics[f"{name}_r2"] = float(r2)
                metrics[f"{name}_mse"] = float(mse)
                trained_models[name] = reg # Store the pipeline
                
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        results["metrics"] = metrics
        results["models"] = trained_models
        results["best_model"] = best_model_name
        results["best_model_score"] = best_score
        results["feature_names"] = X.columns.tolist()
        if excluded_columns:
            results["excluded_columns"] = excluded_columns
        
        # Feature importance extraction
        importances_dict = {}
        
        # Define known features for mapping
        numeric_features_list = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features_list = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Helper to get feature names from preprocessor (duplicated for scope safety)
        def get_feature_names_reg(model_pipeline):
            try:
                preprocessor = model_pipeline.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    return preprocessor.get_feature_names_out()
                
                output_features = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'remainder' and trans == 'drop':
                        continue
                    if hasattr(trans, 'get_feature_names_out'):
                        names = trans.get_feature_names_out(cols)
                    elif hasattr(trans, 'get_feature_names'):
                        names = trans.get_feature_names(cols)
                    else:
                        names = cols
                    output_features.extend(names)
                return output_features
            except Exception:
                return None
        
        def map_to_original_column_reg(feature_name, num_feats, cat_feats):
            # Check for prefixes added by ColumnTransformer
            if feature_name.startswith('num__'):
                clean_name = feature_name[5:]
                if clean_name in num_feats:
                    return clean_name
            
            if feature_name.startswith('cat__'):
                clean_name = feature_name[5:]
                # Match against categorical features (longest match wins)
                matches = [col for col in cat_feats if clean_name.startswith(col)]
                if matches:
                    return max(matches, key=len)
            
            # Fallback if no prefix or standard matching fails
            if feature_name in num_feats:
                return feature_name
            
            matches = [col for col in cat_feats if feature_name.startswith(col)]
            if matches:
                return max(matches, key=len)
                
            return feature_name # Return original if no match found

        for name, pipeline in trained_models.items():
            try:
                model_step = pipeline.named_steps['regressor']
                feature_names = get_feature_names_reg(pipeline)
                
                if feature_names is None:
                    continue

                importances = None
                if hasattr(model_step, "feature_importances_"):
                    importances = model_step.feature_importances_
                elif hasattr(model_step, "coef_"):
                    importances = np.abs(model_step.coef_)
                
                if importances is not None and len(importances) == len(feature_names):
                    # Create DF with raw features
                    raw_df = pd.DataFrame({
                        "Raw_Feature": feature_names,
                        "Importance": importances
                    })
                    
                    # Map to original columns
                    raw_df["Feature"] = raw_df["Raw_Feature"].apply(
                        lambda x: map_to_original_column_reg(str(x), numeric_features_list, categorical_features_list)
                    )
                    
                    # Aggregate by original column
                    feat_imp = raw_df.groupby("Feature")["Importance"].sum().reset_index()
                    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
                    
                    importances_dict[name] = feat_imp
            except Exception:
                continue
                
        results["feature_importances"] = importances_dict

    return results
