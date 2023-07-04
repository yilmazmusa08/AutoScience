import pandas as pd
import numpy as np
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
import warnings
from fitter import Fitter
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.stats
from models.prep_tools import generate_warning_list, analyze_and_plot_distributions

#CLASS MODEL
class KolonTipiTahmini1:
    def __init__(self, threshold=20):
        self.threshold=threshold
        
    def fit_transform(self, data, columns=None):
        if not isinstance(data, pd.DataFrame):
            data=pd.DataFrame(data)
        if len(data) > 5000:
            data=data.sample(n=5000)
             
        distm=data.apply(lambda x: np.mean(pd.Series(x).value_counts(normalize=True).values))# Calculates the mean of normalized frequency distribution values for unique values in each column.
        dists=data.apply(lambda x: np.std(pd.Series(x).value_counts(normalize=True).values))# Calculates the standard deviation of normalized frequency distribution values for unique values in each column.
        wc=data.apply(lambda x: round(len(str(x).split()) / len(str(x)), 5))# Calculates the average number of words in each column.
        wcs=data.apply(lambda x: np.std([len(str(kelime).split()) for kelime in x]))# Calculates the standard deviation of the word counts in each column.
        Len=data.apply(lambda x: len(str(x)) / len(x))#  Calculates the average number of characters in each column.
        lens=data.apply(lambda x: np.std([len(str(s).replace(" ", "")) for s in x]))# Calculates the standard deviation of the character counts in each column.
        uniq=data.apply(lambda x: pd.Series(x).nunique())# Calculates the number of unique values in each column.
        most=data.apply(lambda x: pd.Series(x).value_counts().idxmax() if len(x)>0 and not x.isnull().all() else np.nan)# Calculates the most frequently occurring value in each column.
        mostR=data.apply(lambda x: pd.Series(x).value_counts(normalize=True).values[0] if len(x)>0 and not x.isnull().all() else np.nan)# Calculates the ratio of the most frequently occurring value in each column.
        uniqR=data.apply(lambda x: pd.Series(x).nunique() / len(x))# Calculates the ratio of unique values in each column.
        allunique=data.apply(lambda x: int(x.nunique() == len(x)))# Determines if all values in each column are unique (1 if True, 0 if False).
        kolon_role=[] # An empty list that can be used to store the role of each column.
        
        if columns:
            data=data[columns]

        for col in data.columns:
            if len(data[col].unique()) == 1:
                data.drop(col, axis=1, inplace=True)
            elif (len(data[col].unique()) == 2):
                kolon_role.append('flag')
            elif len(data[col].unique()) == len(data):
                kolon_role.append('unique')
            elif (all(isinstance(val, int) for val in data[col]) and len(data[col].unique()) == len(data) and set(data[col]) == set(range(1, len(data)+1))):
                kolon_role.append('id')
            elif (all(isinstance(val, int) for val in data[col]) and (re.search(r'(id|ID|Id|iD>|ıd)', col))):
                kolon_role.append('id')
            elif all(isinstance(val, int) for val in data[col]) and len(data[col].unique()) == len(data):
                digits=len(str(data[col].iloc[0]))
                if (data[col].apply(lambda x: len(str(x)) == digits).sum() / len(data[col])) > 0.9:
                    kolon_role.append('id')           
            elif all([isinstance(val, str) and pd.to_datetime(val, errors='coerce') is not pd.NaT for val in data[col].values]) or (data[col].dtype == 'datetime64[ns]'):
                    kolon_role.append('date')
            elif isinstance(data[col].iloc[0], str) and data[col].str.split().str.len().mean() > 4:
                kolon_role.append('text')
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                if data[col].apply(lambda x: (isinstance(x, (int,float)) and len(str(x).split('.')) > 1 and str(x).split('.')[1] != '0' and str(x).split('.')[1] != '0000')).sum() > 0:
                    kolon_role.append('scalar')
                elif data[col].apply(lambda x: isinstance(x, (int))).all():
                    kolon_role.append('numeric')
                else:
                    kolon_role.append('numeric')
            elif isinstance(data[col].iloc[0], str) and len(data[col].unique()) >= 0.9*len(data) and data[col].str.split().str.len().mean()<5:
                kolon_role.append('identifier')     
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                kolon_role.append('numeric')
            elif len(data[col].unique()) < self.threshold:
                kolon_role.append('categoric')
            else:
                kolon_role.append("identifier")

        if 'date' in kolon_role:
            date_cols=[col for col, role in zip(data.columns, kolon_role) if role == 'date']
            for date_col in date_cols:
                col_idx=data.columns.get_loc(date_col)
                kolon_role[col_idx]='date'
        # If the column values are datetime, it will display the format in which they are stored.
        def get_Date_Format(data) -> str:
            if not isinstance(data, pd.DataFrame):
                dataframe=pd.DataFrame(data)

            date_formats=[   
                "%m/%Y",
                "%m-%Y",    
                "%d/%m/%y",  
                "%m/%d/%y",  
                "%d.%m.%Y",  
                "%d/%m/%Y", 
                "%m/%d/%Y",  
                "%Y-%m-%d",  
                "%Y/%m/%d",  
                "%m-%d-%Y",  
                "%d-%m-%Y", 
                "%d.%m.%y",
                "%m.%d.%y",
                "%Y/%m",
                "%Y-%m",
                "%m/%d",
                "%d.%m",
                "%d/%m",
                "%m.%d",
                "%Y",
                "%d-%m"
                ]
            dict1={}
            for column in data.columns:
                values=data[column]
                for f in date_formats:
                    try:
                        date=pd.to_datetime(values, format=f)
                        if date.dt.strftime(f).eq(values).all():
                            dict1=f
                            break
                    except ValueError:
                        pass
                else: 
                    dict1=None
            return dict1
        date_formats = data.apply(lambda col: get_Date_Format(col.to_frame()))

        sonuc=[]
        for i in data.columns:
            kolon_role_val=kolon_role[data.columns.get_loc(i)]
            sonuc.append({
                'col_name': i,
                'Role': kolon_role_val})

        result={}
        for d in sonuc:
            col_name=d.pop('col_name')
            result[col_name]=d
        return result
    

def analysis(df: pd.DataFrame, target=None, threshold_target=0.2):
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
    categorical_columns = [column for column, data in data_dict.items() if data.get("Role") == "categoric" or data.get("Role") == "flag"]

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    if target is not None:
        if df[target].dtype == 'object':
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target])
            corr = True
        else:
            corr = True
    else:
        corr = False

    if corr:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
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
        problem_type = None  # The problem_type variable is initially set to None

        # If the number of columns in the dataset is greater than 5 and the data type of the target variable is int64 or float64, and the number of unique values in the target variable is greater than 20, assign 'scoring' to the problem_type variable
        if len(df.columns) > 5 and df[target].dtype in ['int64', 'float64'] and df[target].nunique() > 20:
            problem_type = 'Scoring'
        elif df[target].nunique() == 2:  # If the number of unique values in the target variable is 2
            min_count = df[target].value_counts().min()
            # If the number of classes of the target variable is less than 1% of the sample size:
            if min_count < 0.01 * len(df):
                # Assign 'anomaly detection' to the problem_type variable.
                problem_type = 'Anomaly Detection'
            else:
                # Otherwise, assign 'binary classification' to the problem_type variable.
                problem_type = 'Binary Classification'
        # If the number of unique values in the target variable is between 2 and 20:
        elif 2 < df[target].nunique() < 20:
            # Assign 'multi-class classification' to the problem_type variable.
            problem_type = 'Multi-Class Classification'
        # If the number of columns in the dataset is less than 5:
        elif len(df.columns) < 5:
            for col in df:
                # If the data type of each column in the dataset is 'object':
                if df[col].dtype == 'object':
                    try:
                        # If the columns can be converted to 'datetime64[ns]' data type:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            if any(df[col].dtype == 'datetime64[ns]' for col in df.columns):
                # Assign 'time series' to the problem_type variable.
                problem_type = 'Time Series'
                metrics = ['mae', 'mse', 'rmse']
            elif len(df.columns) < 5 and len([col for col in df.columns if isinstance(df[col], "int")]) == 2 or any(re.search(r'(id|ID|Id|iD>|ıd)', col) for col in df.columns):
                # Assign 'recommendation' to the problem_type variable.
                problem_type = 'Recommendation'


        # If the problem_type variable is assigned, it is stored as a dictionary in the format {'problem_type': problem_type, 'metrics': metrics}.
        if problem_type is not None:
            problem_type = {'Problem Type': problem_type}

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

        warning_list = generate_warning_list(df)


# When the target is active, feature selection (feature importance) is performed using the target variable in the dataset.
# =============================================================================================================
        # Silme işlemi
        null_counts = df.isnull().sum()
        empty_cols = null_counts[null_counts >= len(df) * 0.6].index
        df.drop(empty_cols, axis=1, inplace=True)
        print(df.columns)

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
            if len(X) > 5000:
                X = X.sample(n=5000, random_state=42)
                y = df.loc[X.index, target]
            else:
                y = df[target]
        feature_names = X.columns

        if len(feature_names) >= 2:
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
        warning_list = generate_warning_list(df)

    
    if target is None:
        result = {
            "Column Roles": data_dict,
            "Warnings": warning_list,
            "Distributions": result_dict,
            "Problem Type": clustering}
        return result
    
################################################################################################
def calculate_pca(df, comp_ratio=0.95, target=None):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() < 20:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
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

