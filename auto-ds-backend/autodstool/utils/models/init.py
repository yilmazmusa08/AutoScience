import pandas as pd
import numpy as np
import re
from .binary_classification import *
from .multilabel_classification import *
from .time_series import *
from .smoothing import *
from .sarimax_optimize import *
from .regression import *
from .cluster import *
from .anomaly_detection import *
from .role import *
import warnings
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from .prep_tools import KolonTipiTahmini1, convert_column_based_on_majority, generate_warning_list, classify_columns, analyze_and_plot_distributions, fill_na, get_Date_Column, remove_outliers, replace_special_characters, fill_remaining_na_special, clean_dataframe


def preprocess(df, model=True, remove_redundant=True, Q1=5, Q3=95):
    kt = KolonTipiTahmini1()

    data_dict = {}
    delete_cols = []

    df = classify_columns(df)
    df = clean_dataframe(df)

    necessary_cols = []
    # creating a copy of dataframe in order to get the data type of the columns with kt class
    df_ = df.copy()
    df_ = df_.dropna()  # Drop all rows with NaN values

    for column in df.columns:
        try:
            df[column] = convert_column_based_on_majority(df[column])
        except Exception as e:
            print(f"An error occurred while processing column '{column}': {e}")
            # Handle the error as needed, for example, you can choose to skip the column or log the error.

    data_dict = kt.fit_transform(df_)

    if remove_redundant:
        print("=======================================================================================")
        print("Removing outliers ................")
        for col in df.columns:
            print(col, data_dict[col]["Role"])
            if col in data_dict and (data_dict[col]["Role"] == "scalar"):
                print(f"Outliers are removed for column:", col)
                df = remove_outliers(df, col, Quartile_1=Q1, Quartile_3=Q3)  # Remove outliers for numeric columns

        print("=======================================================================================")
        print("Outliers removed successfully")

    if model==True:
        
        print("=======================================================================================")
        print("Preprocessing started, model = True")
        print("=======================================================================================")
        for col in df_.columns:
            # print(f"Column Data Type Dictionary : {data_dict}")
            if col in data_dict and not (data_dict[col]["Role"] == "id" or
                                          data_dict[col]["Role"] == "object" or
                                            data_dict[col]["Role"] == "text" or data_dict[col]["Role"] == "date"):
                necessary_cols.append(col)

        # print(f"Necessary Columns with appropriate type: {necessary_cols}")
        for col in necessary_cols:
            if df[col].isnull().sum() / len(df) <= 0.5:
                df[col] = df.apply(lambda row: fill_na(row, col, df=df), axis=1)
            if df[col].isnull().sum() / len(df) > 0.5:
                df[col] = df[col].apply(lambda x: 0 if pd.isnull(x) else 1)
                print(f"{col} is converted to binary 0 for nans 1 for existing values")

        print("Deleted columns: ", delete_cols)
        print("=======================================================================================")
        df = df.drop(columns=delete_cols)
        print("Before filling special =================================================================")
        print(df.info())
        print("=======================================================================================")
        df = fill_remaining_na_special(df)
        print("After filling special =================================================================")
        print(df.info())
        print("=======================================================================================")

        df = df.dropna()  # Drop all rows with NaN values
    
    else:

        print("=======================================================================================")
        print("Preprocessing started, model = False")
        print("=======================================================================================")
        for col in df_.columns:
            # print(f"Column Data Type Dictionary : {data_dict}")
            if col in data_dict and not (data_dict[col]["Role"] == "id" or
                                          data_dict[col]["Role"] == "object" or
                                            data_dict[col]["Role"] == "text" or data_dict[col]["Role"] == "date"):
                necessary_cols.append(col)

        # print(f"Necessary Columns with appropriate type: {necessary_cols}")
        for col in necessary_cols:
            if df[col].isnull().sum() / len(df) <= 0.5:
                df[col] = df.apply(lambda row: fill_na(row, col, df=df), axis=1)
            if df[col].isnull().sum() / len(df) > 0.5:
                df[col] = df[col].apply(lambda x: 0 if pd.isnull(x) else 1)
                print(f"{col} is converted to binary 0 for nans 1 for existing values")
        print("Before filling special =================================================================")
        print(df.info())
        print("=======================================================================================")
        df = fill_remaining_na_special(df)
        print("After filling special =================================================================")
        print(df.info())
        print("=======================================================================================")
        df = df.dropna()  # Drop all rows with NaN values
        

    return df

#df = preprocess(df)
#print(df)


def get_problem_type(df, target=None):

    binary_params = {
        "cv": 5,
        "target": target,
        "models": ['Logistic Regression', 'Random Forest',
                'Decision Tree Classifier', 'Gradient Boosting Classifier',
                'Naive Bayes', 'Support Vector Machines', 'AdaBoost', 'XGBoost',
                'LightGBM', 'CatBoost'],
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    mlp_params = {
        "cv": 5,
        "target": target,
        "models": ['Logistic Regression', 'Random Forest', 
                                       'Decision Tree Classifier','Gradient Boosting Classifier', 
                                       'Naive Bayes', 'K-Nearest Neighbors','CatBoost'],
        "metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    }

    smooth_params = {
        "target": target,
        "forecast": 60,
        "method": "add",
        "seasonal_periods": 12
    }

    ts_params = {
        "target": target,
        "order": (1, 0, 0),
        "seasonal_order": (1, 0, 0, 12),
        "method": "powell",
        "forecast": 10
    }

    sarimax_params = {
        "target": target,
        "forecast": 60
    }

    reg_params = {
        "cv": 5,
        "target": target,
        "models": ['Linear Regression', 'Random Forest', 'Decision Tree Regressor', 'Gradient Boosting Regressor',
                                              'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression', 'Polynomial Regression',
                                              'Support Vector Regression', 'XGBoost Regression', 'LightGBM Regression', 'CatBoost Regression'],
        "metrics": ["neg_mean_squared_error", "neg_mean_absolute_error", "neg_mean_absolute_percentage_error", "r2"]
    }

    dbscan_params = {
        "eps": 0.5,
        "min_samples": 20
    }

    anomal_params = {
        "cv": 5,
        "target": target,
        "models": ["IsolationForest", "OneClassSVM", "EllipticEnvelope"],
        "metrics": ['accuracy', 'precision_macro', 'f1_macro'],
        "comp_ratio": 0.95
    }

    has_datetime_column = False

    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            has_datetime_column = True
        elif all(isinstance(val, str) and pd.to_datetime(val, errors='coerce') is not pd.NaT for val in df[col].dropna().values):
            has_datetime_column = True
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                has_datetime_column = True
            except:
                pass

        if df[col].dtype == "object" and df[col].nunique() >= 10:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        elif df[col].dtype == "object" and df[col].nunique() < 10:
            df = pd.get_dummies(df, columns=[col])

                
    problem_type = None
    if target is not None:
        if df[target].nunique() == 2:
            min_count = df[target].value_counts().min()
            if min_count < 0.01 * len(df):
                problem_type = "anomaly_detection"
                # print("Anomaly Detection Confirmed")
            else:
                problem_type = "binary classification"
                # print("Binary Classification Confirmed")
        elif 2 < df[target].nunique() < 20:
            problem_type = "multi-class classification"
            # print("Multiclass Classification Confirmed")
        elif has_datetime_column or df.sort_values(by=[col], ascending=True).index.is_monotonic_increasing:
            # if df[target].dtype in [np.int, np.float, np.int32, np.float32, np.int64, np.float64]:
            problem_type = "time series"
            # print("Time Series Confirmed")

        else:
            problem_type = "scoring"
    else:
        problem_type = "clustering"
        # print("Clustering Confirmed")



    if problem_type == "binary classification":
        print("Problem Type : Binary Classification")
        params = []
        params.append(binary_params)
        return problem_type, params

    elif problem_type == "time series":
        print("Problem Type : Time Series")
        params = []
        params.append(smooth_params)
        params.append(ts_params)
        params.append(sarimax_params)
        return problem_type, params


    elif problem_type == "multi-class classification":
        print("Problem Type : Multi-Class Classification")
        params = []
        params.append(mlp_params)
        return problem_type, params


    elif problem_type == "scoring":
        print("Problem Type : Regression")
        params = []
        params.append(reg_params)
        return problem_type, params


    elif problem_type == "anomaly_detection":
        print("Problem Type : Anomaly Detection")
        params = []
        params.append(anomal_params)
        return problem_type, params


    elif problem_type == "clustering":
        print("Problem Type : Clustering")
        params = []
        params.append(dbscan_params)
        return problem_type, params

    else:
        problem_type = ""
        params = ""
        return problem_type, params

def create_model(df, problem_type=None, params=[], max_rows=5000):

    if len(df) > max_rows:
        df=df.sample(n=max_rows)

    if problem_type == "binary classification":
        print("params : ", params)
        result = binary_classification(df, **params[0])

    elif problem_type == "time series":
        print("params : ", params)
        df = get_Date_Column(df)
        result = {}
        result["Smoothing Methods"] = smoothing(df, **params[0])
        result["Statistical Methods"] = time_series(df, **params[1])
        result["Optimized SARIMA"] = sarimax_optimize(df, **params[2])

    elif problem_type == "multi-class classification":
        print("params : ", params)
        result = multiclass_classification(df, **params[0])

    elif problem_type == "scoring":
        print("params : ", params)
        result = regression(df, **params[0])

    elif problem_type == "anomaly_detection":
        print("Problem Type : Anomaly Detection")
        print("params : ", params)
        result = anomaly_detection(df, **params[0])

    elif problem_type == "clustering":
        print("params : ", params)
        result = dbscan(df, **params[0])
        print(visualize_clusters(df, result))

    else:
        result = None

    return result

    

def analysis(df: pd.DataFrame, target=None, threshold_target=0.2, max_rows=5000):
    """
    This function is designed to analyze a dataset. The function takes the following parameters:
    
    Parameters:
    -----------
    1. df: The dataset to be analyzed.
    
    2. threshold_col: The threshold value for high correlation between columns (default: 0.8).
    
    3. threshold_target: The threshold value for high correlation between columns and the target variable (default: 0.4).
    
    4. target: A boolean indicating the target variable (default: None).
    
    5. warning: The warning parameter can be used to display any warning messages. 
       This warning message can help guide the user in subsequent operations. If warning is set to True,
       it will remove the columns with warnings.
    """


    high_corr_target = []
    kt = KolonTipiTahmini1()
    data_dict = kt.fit_transform(df)
    # Create a copy of the DataFrame 'df' after dropping rows with any NaN values
    df_copy = df.dropna(axis=0).copy()
    

    # Loop through each column in the DataFrame
    for col in df_copy.columns:
        # Check if the column has a numeric data type and is a float
        if pd.api.types.is_numeric_dtype(df_copy[col]) and df_copy[col].dtype == float:
            # Check if all values in the column are integers (have no fractional parts)
            if df_copy[col].apply(lambda x: x.is_integer() or x == int(x)).all():
                # Convert the column to the integer data type
                df_copy[col] = df_copy[col].astype(int)
        else:
            # If the column is not numeric or not a float, continue to the next column
            continue

    # Find and store column names that have duplicate data in 'df_copy'
    duplicated_columns = [col2 for i, col1 in enumerate(df_copy.columns) for col2 in df_copy.columns[i+1:] if df_copy[col1].equals(df_copy[col2])]

    # Drop the columns with duplicate data from the original DataFrame 'df'
    df.drop(duplicated_columns, axis=1, inplace=True)

    warning_list = generate_warning_list(df)
    categorical_columns = [column for column, data in data_dict.items() if data.get("Role") == "categoric" or data.get("Role") == "flag"]
   
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    

    if target is not None:
        if not np.issubdtype(df[target].dtype, np.number):
            df[target] = le.fit_transform(df[target])
            corr = True
        else:
            corr = True
    else:
        corr = False

    if corr:
        numeric_columns = df.select_dtypes(include=['number']).columns
        corr_matrix=df[numeric_columns].corr().abs()
        high_corr_target = []
        for col in corr_matrix.columns:
            if target is not None and col != target and corr_matrix[target][col] >= threshold_target:
                high_corr_target.append(col)

# In this code, the suitability of different probability distribution functions for a dataset is tested.
# ==========================================================================================
    plots, result_dict = analyze_and_plot_distributions(df)


# Determining the Problem Type and Metrics
# ======================================
    if target:
        problem_type = get_problem_type(df = df, target = target)

# When the target is active, this code block checks for NaN values, sparse values, and unique values. It also checks for high correlation among numerical columns. If the warning variable is True, it removes the warning columns.
# =======================================================================================================================================================================================================

        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        korelasyonlar = {}
        if isinstance(target, int):
            corr_matrix = df[numeric_columns].corr()[[target]]
            print('Correlation Matrix', corr_matrix)
            corr_matrix = corr_matrix.drop(target)
            kolonlar = list(corr_matrix.index)
            for i in range(len(kolonlar)):
                korelasyonlar[kolonlar[i]] = corr_matrix.iloc[i, 0]
        else:
            pass

# When the target is active, feature selection (feature importance) is performed using the target variable in the dataset.
# =============================================================================================================

        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                df[col].fillna(df[col].mode()[0], inplace=True)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            elif df[col].dtype != 'object':
                df[col].fillna(df[col].mean(), inplace=True)

        X = df.select_dtypes(include=['float', 'int'])
        if target in X.columns:
            X.drop(target, axis=1, inplace=True)

        if target is not None:
            if len(X) > max_rows:
                X = X.sample(n=max_rows, random_state=42)
                y = df.loc[X.index, target]
            else:
                y = df[target]
        feature_names = X.columns
        print('X',feature_names)
        print('y',y)

        if len(feature_names) >= 2:
            if isinstance(y.values[0], (int, float)):  # Kontrol eder: hedef değişken sürekli mi?
                rf = RandomForestRegressor(n_jobs=-1, n_estimators=500, oob_score=True, max_depth=5)
            else:
                rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, oob_score=True, max_depth=5)

            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
            feat_selector.fit(X.values, y.values)
            
            importance = feat_selector.ranking_

            feature_importance = {}
            total_importance = 0
            for i in range(len(feature_names)):
                feature_importance[feature_names[i]] = importance[i]
                total_importance += importance[i]

            for feature in feature_importance:
                feature_importance[feature] = round((feature_importance[feature] / total_importance) * 100, 2)

            if target and len(feature_importance) >= 2:
                result = {
                    "Column Roles": data_dict,
                    "Warnings": warning_list,
                    "Distributions": result_dict,
                    "High Correlation with Target": high_corr_target,
                    "Feature Importance": {k: f"{v}%" for k, v in feature_importance.items()},
                    "Problem Type": problem_type
                }
            else:
                result = {
                    "Column Roles": data_dict,
                    "Warnings": warning_list,
                    "Distributions": result_dict,
                    "High Correlation with Target": high_corr_target,
                    "Problem Type": problem_type
        
                }
        else:
            result = {
                "Column Roles": data_dict,
                "Warnings": warning_list,
                "Distributions": result_dict,
                "High Correlation with Target": high_corr_target,
                "Problem Type": problem_type
                
            }

        return result


# If the target is not active or warning is None, a warning calculation is performed.
# ==================================================================================================================
    if target is None:
        clustering = 'Clustering'
        clustering = {clustering}
        problem_type = {}

        result = {
            "Column Roles": data_dict,
            "Warnings": warning_list,
            "Distributions": result_dict,
            "Problem Type": clustering}
        return result
    
################################################################################################
def calculate_pca(df, comp_ratio=0.95, target=None):
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() >= 10:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        elif df[col].dtype == "object" and df[col].nunique() < 10:
            df = pd.get_dummies(df, columns=[col])
            
    if target is not None:
        df = df.drop(columns=[target])
    df.fillna(df.mean(), inplace=True)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    pca = PCA()
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    if comp_ratio <= 1:
        n_components = np.argmax(cumsum_var_ratio >= comp_ratio) + 1
    else:
        n_components = int(comp_ratio)
    n_components = min(n_components, 6)  # Limit the number of components to 6 if it exceeds that number
    pca = PCA(n_components=n_components)
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    result_dict = {
        'Cumulative Explained Variance Ratio': np.round(cumsum_var_ratio, 2).tolist()}
    result_dict['Principal Component'] = list(range(1, len(explained_var_ratio) + 1))
    return result_dict



def set_to_list(data):
    if isinstance(data, set):
        return list(data)
    if isinstance(data, dict):
        return {set_to_list(key): set_to_list(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [set_to_list(item) for item in data]
    if isinstance(data, np.int64):
        return int(data)
    return data
