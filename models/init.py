import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from binary_classification import *
from multilabel_classification import *
from time_series import *
from smoothing import *
from sarimax_optimize import *
from regression import *
from cluster import *
from anomaly_detection import *
from role import *
from prep_tools import fill_na, get_Date_Column


# df = pd.read_csv('time2.csv')
# df = pd.read_csv('train.csv')
# df = pd.read_csv('Salary_Data.csv')
# df = pd.read_csv('Ratings.csv')
def preprocess(df):
    kt = KolonTipiTahmini1()

    data_dict = {}
    delete_cols = []

    if len(df) > 5000:
        df = df.sample(n=5000)

    for col in df.columns:
        try:
            data_dict[col] = kt.fit_transform(df[[col]])[col]
        except KeyError:
            continue

        if col in data_dict and (data_dict[col]["Role"] == "identifier" or data_dict[col]["Role"] == "text" or data_dict[col]["Role"] == "date"):
            delete_cols.append(col)

    print("DELETED COLUMNS: ", delete_cols)
    df = df.drop(columns=delete_cols)
    return df

#df = preprocess(df)
#print(df)


def get_problem_type(df, target=None):

    binary_params = {
        "cv": 5,
        "target": target,
        "models": ["Logistic Regression", "Random Forest", "Decision Tree Classifier", "Gradient Boosting Classifier"],
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }

    mlp_params = {
        "cv": 5,
        "target": target,
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
        "models": ["Linear Regression", "Random Forest", "Decision Tree Regressor", "Gradient Boosting Regressor"],
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


    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() < 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        numeric_columns = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
        for col in numeric_columns:
            if df[col].isnull().sum() / len(df) > 0.5:
                df.drop(col, axis=1, inplace=True)
            elif 0.05 <= df[col].isnull().sum() / len(df) <= 0.5:
                print(df.info())
                df[col] = df.apply(lambda row: fill_na(row, col, df=df), axis=1)
                print(df.info())
            else:
                median_value = df[col].median()
                print(df.info())
                df[col].fillna(median_value, inplace=True)
                print(df.info())
    problem_type = None
    if target is not None:
        if len(numeric_columns) > 0:
            if df[target].nunique() == 2:
                min_count = df[target].value_counts().min()
                if min_count < 0.01 * len(df):
                    problem_type = "anomaly_detection"
                    print("Anomaly Detection Confirmed")
                else:
                    problem_type = "binary classification"
                    print("Binary Classification Confirmed")
            elif 2 < df[target].nunique() < 20:
                problem_type = "multi-class classification"
                print("Multiclass Classification Confirmed")
            elif len(df.columns) <= 6:
                has_datetime_column = False
                for col in df:
                    if df[col].dtype == "object":
                        try:
                            df[col] = pd.to_datetime(df[col])
                            has_datetime_column = True
                        except:
                            pass
                if has_datetime_column or df.sort_values(by=[col], ascending=True).index.is_monotonic_increasing:
                    problem_type = "time series"
                    print("Time Series Confirmed")

                elif (
                    len(df.columns) < 5
                    and len([col for col in df.columns if isinstance(df[col].iloc[0], int)]) == 2
                    or any(re.search(r"(id|ID|Id|iD>|ıd)", col) for col in df.columns)
                ):
                    problem_type = "recommendation"
                    print("Recommendation Confirmed")
                else:
                    problem_type = "scoring"
                    print("Regression Confirmed")
        else:
            problem_type = None
    else:
        problem_type = "clustering"
        print("Clustering Confirmed")


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
        return None

def create_model(df, problem_type=None, params=[]):

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

# print(create_model(df, target='Salary')) # Salary_Data.csv
# print(create_model(df, date='DATE', target='Value')) # time2.csv
#print(create_model(df))  Online_Retail.xlsx