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
                kolon_role.append('indentifier')     
            elif len(data[col].unique()) > self.threshold and (all(isinstance(val, int) for val in data[col]) or all(isinstance(val, float) for val in data[col])):
                kolon_role.append('numeric')
            elif len(data[col].unique()) < self.threshold:
                kolon_role.append('categoric')
            else:
                kolon_role.append("indentifier")

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
            distm_val=round(distm[i], 5) if distm.get(i) is not None else None
            dists_val=round(dists[i], 5) if dists.get(i) is not None else None
            wc_val=round(wc[i], 5) if wc.get(i) is not None else None
            wcs_val=round(wcs[i], 5) if wcs.get(i) is not None else None
            len_val=round(Len[i], 5) if Len.get(i) is not None else None
            lens_val=round(lens[i], 5) if lens.get(i) is not None else None
            uniq_val=round(uniq[i], 5) if uniq.get(i) is not None else None
            most_val=most[i] if most.get(i) is not None else None
            mostR_val=round(mostR[i], 5) if mostR.get(i) is not None else None
            uniqR_val=round(uniqR[i], 5) if uniqR.get(i) is not None else None
            allunique_val=round(allunique[i], 5) if allunique.get(i) is not None else None
            kolon_role_val=kolon_role[data.columns.get_loc(i)]
            sonuc.append({
                'col_name': i,
                'distm': distm_val,
                'date_formats': date_formats[i],
                'dists': dists_val,
                "wc":wc_val,
                "wcs":wcs_val,
                "len":len_val,
                "lens":lens_val,
                "uniq":uniq_val,
                "uniqR %":uniqR_val,
                "most":most_val,
                "mostR %":mostR_val,
                "allunique":allunique_val,
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
# ===========================================================================================
    for col in df:
        try:  # Start an error handling block for each column
            # Define a lambda function to check if all rows are integers to determine if the column is of integer type, and apply it to the entire column using the apply() method. The results are checked with the all() method and assigned to the is_int variable.
            is_int = df[col].apply(lambda x: x.is_integer()).all()
            if is_int:
                # If all rows are integers, the data type of the column is changed to 'int'.
                df[col] = df[col].astype("int")
                print(df.info())
        except:
            pass  # If an error occurs, continue to the next column.

    result_dict = {}
    distributions = ['norm', 'uniform', 'binom', 'poisson', 'gamma', 'beta', 'lognorm', 'weibull_min', 'weibull_max', 'expon', 'pareto', 'cauchy', 'chi', 'f', 't', 'laplace',
                     'bernoulli', 'exponential', 'geometric', 'hypergeometric', 'normal_mix', 'rayleigh', 'student_t', 'weibull']  # Kullanılacak dağılımların listesi oluşturulur.
    for col in df.columns:
        if df[col].nunique() <= 20:
            df[col] = LabelEncoder().fit_transform(df[col])

        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            if len(df) >= 5000:
                df_sampled = df.sample(n=5000, random_state=42)
            else:
                df_sampled = df
            f = Fitter(df_sampled[col], distributions=distributions)
            f.fit()
            if len(f.df_errors) > 0:
                try:
                    best_dist = list(f.get_best().keys())[0]
                except KeyError:
                    continue

                if best_dist == 'weibull_min' or best_dist == 'weibull_max':
                    best_dist = 'weibull'
                elif best_dist == 'lognormvariate':
                    best_dist = 'lognorm'
                elif best_dist == 'norm':
                    best_dist = 'gauss'

                result_dict[col] = best_dist

# Determining the Problem Type and Metrics
# ======================================
    if target:
        problem_type = None  # The problem_type variable is initially set to None
        metrics = None  # The metrics variable is initially set to None

        # If the number of columns in the dataset is greater than 5 and the data type of the target variable is int64 or float64, and the number of unique values in the target variable is greater than 20, assign 'scoring' to the problem_type variable
        if len(df.columns) > 5 and df[target].dtype in ['int64', 'float64'] and df[target].nunique() > 20:
            problem_type = 'scoring'
            metrics = ['r2_score', 'mean_absolute_error', 'mean_squared_error']
        elif df[target].nunique() == 2:  # If the number of unique values in the target variable is 2
            min_count = df[target].value_counts().min()
            # If the number of classes of the target variable is less than 1% of the sample size:
            if min_count < 0.01 * len(df):
                # Assign 'anomaly detection' to the problem_type variable.
                problem_type = 'anomaly detection'
                metrics = ['precision_score', 'recall_score', 'f1_score']
            else:
                # Otherwise, assign 'binary classification' to the problem_type variable.
                problem_type = 'binary classification'
                metrics = ['f1_score', 'precision_score', 'recall_score']
        # If the number of unique values in the target variable is between 2 and 20:
        elif 2 < df[target].nunique() < 20:
            # Assign 'multi-class classification' to the problem_type variable.
            problem_type = 'multi-class classification'
            metrics = ['accuracy_score', 'precision_score', 'recall_score']
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
                problem_type = 'time series'
                metrics = ['mae', 'mse', 'rmse']
            elif len(df.columns) < 5 and len([col for col in df.columns if isinstance(df[col], "int")]) == 2 or any(re.search(r'(id|ID|Id|iD>|ıd)', col) for col in df.columns):
                # Assign 'recommendation' to the problem_type variable.
                problem_type = 'recommendation'
                metrics = ['recall_score', 'precision_score', 'map_score']


        # If the problem_type variable is assigned, it is stored as a dictionary in the format {'problem_type': problem_type, 'metrics': metrics}.
        if problem_type is not None:
            problem_type = {'problem_type': problem_type, 'metrics': metrics}

# When the target is active, this code block checks for NaN values, sparse values, and unique values. It also checks for high correlation among numerical columns. If the warning variable is True, it removes the warning columns.
# =======================================================================================================================================================================================================

        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        korelasyonlar = {}
        if isinstance(target, int):
            corr_matrix = df[numeric_columns].corr()[[target]]
            print('corr_matrix', corr_matrix)
            corr_matrix = corr_matrix.drop(target)
            kolonlar = list(corr_matrix.index)
            for i in range(len(kolonlar)):
                korelasyonlar[kolonlar[i]] = corr_matrix.iloc[i, 0]
        else:
            pass

        warning_list = []
        for col in df.columns:
            if df[col].dtype == "int64" or df[col].dtype == "float64":
                null_values = df[col].isna().sum()
                zero_values = (df[col] == 0).sum()
                total_values = len(df[col])
                null_ratio = null_values / total_values
                zero_ratio = zero_values / total_values
                unique_ratio = len(df[col].unique()) / total_values

                ratio_str = ""

                if null_ratio > 0:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)

                if zero_ratio > 0.6 and null_ratio == 0:
                    if len(ratio_str) > 0:
                        ratio_str += ", "
                    ratio_str += "sparse:{:.2f}%".format(zero_ratio * 100)

                if unique_ratio >= 0.95 and null_ratio == 0 and zero_ratio == 0:
                    if len(ratio_str) > 0:
                        ratio_str += ", "
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)

                if ratio_str:
                    warning_list.append([col, ratio_str])

        if not warning_list:
            warning_list.append(["no warning"])        

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
                    "Role": data_dict,
                    "warning_list": warning_list,
                    "distributions": result_dict,
                    "high_corr_target": high_corr_target,
                    "feature_importance": {k: f"{v}%" for k, v in feature_importance.items()},
                    "problem_type": problem_type
                }
            else:
                result = {
                    "Role": data_dict,
                    "warning_list": warning_list,
                    "distributions": result_dict,
                    "high_corr_target": high_corr_target,
                    "problem_type": problem_type
        
                }
        else:
            result = {
                "Role": data_dict,
                "warning_list": warning_list,
                "distributions": result_dict,
                "high_corr_target": high_corr_target,
                "problem_type": problem_type
                
            }

        return result


# If the target is not active or warning is None, a warning calculation is performed.
# ==================================================================================================================
    if target is None:
        clustering = 'clustering'
        clustering = {clustering}
        warning_list = []
        problem_type = {}

        for col in df.columns:
            if df[col].dtype == "int64" or df[col].dtype == "float64":
                null_values = df[col].isna().sum()
                zero_values = (df[col] == 0).sum()
                total_values = len(df[col])
                null_ratio = null_values / total_values
                zero_ratio = zero_values / total_values
                unique_ratio = len(df[col].unique()) / total_values
                ratio_str = ""

                if null_ratio > 0:
                    ratio_str += "NaN:{:.2f}%".format(null_ratio * 100)

                if zero_ratio > 0.6 and null_ratio == 0:
                    if len(ratio_str) > 0:
                        ratio_str += ", "
                    ratio_str += "sparse:{:.2f}%".format(zero_ratio * 100)

                if unique_ratio >= 0.95 and null_ratio == 0 and zero_ratio == 0:
                    if len(ratio_str) > 0:
                        ratio_str += ", "
                    ratio_str += "unique:{:.2f}%".format(unique_ratio * 100)

                if ratio_str:
                    warning_list.append([col, ratio_str])

        if not warning_list:
            warning_list.append(["no warning"])
    
    if target is None:
        result = {
            "Role": data_dict,
            "warning_list": warning_list,
            "distributions": result_dict,
            "problem_type": clustering}
        return result
    
################################################################################################
# Principal Component
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
    n_components = len(cumsum_var_ratio)  # The n_components is directly set as the length of the cumsum_var_ratio.
    pca = PCA(n_components=n_components)
    pca.fit(df[numeric_cols])
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(explained_var_ratio)
    result_dict = {
        'Cumulative Explained Variance Ratio': cumsum_var_ratio.tolist()}
    result_dict['Principal Component'] = list(
        range(1, len(explained_var_ratio)+1))
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

