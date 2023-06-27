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
from role import *
from prep_tools import fill_na

# df = pd.read_csv('time2.csv')
# df = pd.read_csv('train.csv')
# df = pd.read_csv('Salary_Data.csv')
# df = pd.read_csv('Ratings.csv')


def preprocess(df): 
    kt = KolonTipiTahmini1()  # KolonTipiTahmini1 sınıfı veya modülünün doğru şekilde yüklendiğinden emin olun

    data_dict = {}
    delete_cols = []  # Silinecek sütunları depolamak için boş bir liste oluşturun

    if len(df) > 5000:
        df=df.sample(n=5000)

    for col in df.columns:
        data_dict[col] = kt.fit_transform(df[[col]])[col]

        if data_dict[col]["Role"] == "identifier" or data_dict[col]["Role"] == "text" or data_dict[col]["Role"] == "date":
            delete_cols.append(col)  # Silinecek sütunları listeye ekleyin

    print("DELETED COLUMNS : ", delete_cols)
    df = df.drop(columns=delete_cols)  # Silinecek sütunları DataFrame'den kaldırın
    return df




def preprocess(df):
    kt = KolonTipiTahmini1()  # KolonTipiTahmini1 sınıfı veya modülünün doğru şekilde yüklendiğinden emin olun

    data_dict = {}
    delete_cols = []  # Silinecek sütunları depolamak için boş bir liste oluşturun

    if len(df) > 5000:
        df = df.sample(n=5000)

    for col in df.columns:
        data_dict[col] = kt.fit_transform(df[[col]])[col]

        if data_dict[col]["Role"] == "identifier" or data_dict[col]["Role"] == "text" or data_dict[col]["Role"] == "date":
            delete_cols.append(col)  # Silinecek sütunları listeye ekleyin

    print("DELETED COLUMNS: ", delete_cols)
    df = df.drop(columns=delete_cols)  # Silinecek sütunları DataFrame'den kaldırın
    return df


#df = preprocess(df)
#print(df)

def create_model(df, date=None, target=None):

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
        "date": date,
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
        "date": date,
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


    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() < 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        numeric_columns = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
        for col in numeric_columns:
            if df[col].isnull().sum() / len(df) > 0.5:
                df.drop(col, axis=1, inplace=True)
            elif 0.05 <= df[col].isnull().sum() / len(df) <= 0.5:
                df[col] = df.apply(lambda row: fill_na(row, col, df=df), axis=1)
            else:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)

    if target is not None:
        if len(numeric_columns) > 0:
            if df[target].nunique() == 2:
                min_count = df[target].value_counts().min()
                if min_count < 0.01 * len(df):
                    problem_type = "anomaly detection"
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
        print("params : ", binary_params)
        result = binary_classification(df, **binary_params)

    elif problem_type == "time series":
        print("Problem Type : Time Series")
        print("params : ", ts_params)
        result = smoothing(df, **smooth_params)
        result.update(time_series(df, **ts_params))
        result.update(sarimax_optimize(df, **sarimax_params))

    elif problem_type == "multi-class classification":
        print("Problem Type : Multi-Class Classification")
        print("params : ", mlp_params)
        result = multiclass_classification(df, **mlp_params)

    elif problem_type == "scoring":
        print("Problem Type : Regression")
        print("params : ", reg_params)
        result = regression(df, **reg_params)

    elif problem_type == "anomaly_detection":
        print("Problem Type : Anomaly Detection")
        # print("params : ", anomal_params)
        # result = anomaly_detection(df, **anomal_params)

    elif problem_type == "clustering":
        print("Problem Type : Clustering")
        print("params : ", dbscan_params)
        result = dbscan(df, **dbscan_params)
        print(visualize_clusters(df, result))

    else:
        result = None

    return result

# print(create_model(df, target='Salary')) # Salary_Data.csv
# print(create_model(df, date='DATE', target='Value')) # time2.csv
#print(create_model(df))  Online_Retail.xlsx