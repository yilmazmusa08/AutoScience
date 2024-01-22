#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib
import pandas as pd
import joblib
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier,export_graphviz, export_text
import warnings 
warnings.filterwarnings("ignore")
# import statsmodels.stats.api as sms
# from statsmodels.stats.proportion import proportions_ztest
# from skompiler import skompile
# import graphviz
import scipy.stats
from sklearn.preprocessing import LabelEncoder
from fitter import Fitter


def col_type_pred(df):
    kt = KolonTipiTahmini1()

    data_dict = {}
    scalar_cols = []
    categoric_cols = []
    id_cols = []
    text_cols = []
    numeric_cols = []
    date_cols = []

    df_ = df.copy()
    df_ = df_.dropna()  # Drop all rows with NaN values

    for col in df_.columns:
        try:
            data_dict[col] = kt.fit_transform(df_[[col]])[col]
            # print(f"Column Data Type Dictionary : {data_dict}")
            if col in data_dict and (data_dict[col]["Role"] == "date"):
                date_cols.append(col)
            elif col in data_dict and (data_dict[col]["Role"] == "text"):
                text_cols.append(col)
            elif col in data_dict and (data_dict[col]["Role"] == "scalar"):
                scalar_cols.append(col)
            elif col in data_dict and (data_dict[col]["Role"] == "id" or data_dict[col]["Role"] == "identifier" or data_dict[col]["Role"] == "object"):
                id_cols.append(col)
            elif col in data_dict and (data_dict[col]["Role"] == "categoric"):
                categoric_cols.append(col)
                
        except KeyError:
            continue


    return scalar_cols, numeric_cols, categoric_cols, date_cols, text_cols, id_cols

def classify_columns(df):
    scalar_cols, numeric_cols, categoric_cols, date_cols, text_cols, id_cols = col_type_pred(df)
    if text_cols:
        for col in text_cols:
            length = df[col].astype(str).str.len()
            df.loc[length < 3, col] = "Very Short"
            df.loc[(3 <= length) & (length < 6) & (df[col] != "Very Short"), col] = "Short"
            df.loc[(6 <= length) & (length < 10) & (df[col].isin(["Very Short", "Short"])) == False, col] = "Medium"
            df.loc[(10 <= length) & (length < 15) & (df[col].isin(["Very Short", "Short", "Medium"])) == False, col] = "Long"
            df.loc[length >= 15, col] = "Very Long or Text"
    
    if categoric_cols:
        try:
            dummy_df = pd.get_dummies(df[categoric_cols], drop_first=True)
        except Exception as e:
            print("Error creating dummy variables:", e)
            dummy_df = None

        if dummy_df is not None:
            df = pd.concat([df, dummy_df], axis=1)

    return df


def clean_dataframe(df, forbidden_symbols=["'", '"', ":", "=", "/", "[", "]", "{", "}", "(", ")", " ", ".", ",", ">", "<"]):
    print(f"Column name lists: {df.columns}")

    # Clean column names
    cleaned_columns = []
    for col in df.columns:
        col_cleaned = col
        for symbol in forbidden_symbols:
            col_cleaned = col_cleaned.replace(symbol, "")
        col_cleaned = col_cleaned.strip()  # Remove leading and trailing spaces
        cleaned_columns.append(col_cleaned)

    # Rename columns
    df.columns = cleaned_columns

    # Clean values in each column of type 'object'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.replace('|'.join(forbidden_symbols), "")


    print(f"Cleaned column name lists: {df.columns}")
    return df

# Function to convert column based on majority type
def convert_column_based_on_majority(column):
    try:
        majority_value = column.mode().iloc[0]
        
        if isinstance(majority_value, str):
            majority_type = str
        else:
            majority_type = majority_value.dtype
        
        print(f"Column: {column}, Majority type: {majority_type}")
        return column.astype(majority_type)
    except Exception as e:
        # Handle the exception, you can print an error message or take other actions
        print(f"Error converting column: {e}")
        return column  # Returning the original column in case of an error




def remove_outliers(df, column_name, Quartile_1=1, Quartile_3=99, remove=True):
    # Create a copy of the original DataFrame
    print("=======================================================================================")
    print(f"Before Outlier Removal")
    print(df.info())
    print("=======================================================================================")
    df_copy = df.copy()
    
    # Select the column for outlier removal
    column_data = df_copy[column_name]

    # Calculate the Quartiles using numpy
    q1 = np.percentile(column_data, Quartile_1)
    q3 = np.percentile(column_data, Quartile_3)
    
    # Calculate the IQR (Interquartile Range)
    IQR = q3 - q1
    
    # Define lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (column_data < lower_bound) | (column_data > upper_bound)
    
    if remove:
        # Remove outliers from the DataFrame
        df_copy = df_copy[~outliers]
    print("=======================================================================================")
    print(f"After Outlier Removal")
    print(df_copy.info())
    print("=======================================================================================")
    
    return df_copy


def fill_na(row, col, lower_coeff=0.8, upper_coeff=1.20, df=None, max_attempts=2):
    forbidden_symbols = ["'", '"', "/", "[", "]", "{", "}", "(", ")"]

    for _ in range(max_attempts):
        # If the value is not NaN, return the value directly
        if pd.notnull(row[col]) or str(row[col]).lower() not in ["nan", "na"]:
            # print(f"Returning the value {row[col]}")
            return row[col]

        query = []

        # Object and numerical columns separation
        obj_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(exclude=['object']).columns

        # Categorizing strings by extended length categories
        print(f"Checking object columns for column {col}")
        for o in obj_cols:
            if not pd.isna(row[o]):
                value = str(row[o])
                for i in forbidden_symbols:
                    value = value.replace(i, "")
                query.append(f'({o} == "{value}")')

        # Creating query for numerical columns
        print(f"Checking numeric columns for column {col}")
        for n in num_cols:
            if not pd.isna(row[n]):
                query.append(f"({n} >= {row[n] * lower_coeff})")
                query.append(f"({n} <= {row[n] * upper_coeff})")

        # Joining the created query
        query = " and ".join(query)
        print(f"query is: {query}")


        try:
            # Creating subset using the query
            sub = df.query(query)
            print(f"subquery is: {sub}")
            print("Type of the values:", df[col].dtype)

            if len(sub) == 0:
                raise ValueError
            else:
                if df[col].dtype == 'object':
                    # For object columns, fill with the mode
                    print(f"Returning the value: {sub[col].mode().iloc[0]}")
                    return sub[col].mode().iloc[0]
                else:
                    # For numerical columns, fill with the mean
                    print(f"Returning the value: {sub[col].mean()}")
                    return sub[col].mean()

        except ValueError:
            lower_coeff *= 0.8  # Decrease lower limit by 20%
            upper_coeff *= 1.2  # Increase upper limit by 20%

    # If all attempts fail, return None
    return None



def fill_remaining_na_special(df, nan_replacement=-9999):
    """
    Fill remaining NaN and specific values in the DataFrame using a specialized method.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing NaN values and specific values.
        nan_replacement: The specific value to be treated as NaN during similarity calculation.
    
    Returns:
        pd.DataFrame: The DataFrame with NaN and specific values filled using the specified method.
    """
    # Create a copy of the DataFrame to work with
    filled_df = df.copy()
    
    # Replace NaN values with a specific placeholder value
    filled_df.fillna(nan_replacement, inplace=True)
    
    # Fill numeric columns using a specialized method (cosine similarity in this example)
    numeric_columns = filled_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        for index, row in filled_df.iterrows():
            if row[col] == nan_replacement:
                # Create a mask to exclude specific values (e.g., -9999)
                mask = filled_df[col] != nan_replacement
                
                if mask.any():
                    # Convert row[col] to a NumPy array and then reshape it
                    value = np.array([row[col]])
                    similarities = cosine_similarity(value.reshape(1, -1),
                                                     filled_df.loc[mask][col].values.reshape(-1, 1))
                    
                    # Find the most similar non-specific row
                    most_similar_index = np.argmax(similarities)
                    most_similar_row = filled_df.loc[mask].iloc[most_similar_index]
                    
                    # Fill NaN value in the current numeric column with the most similar row's value
                    filled_df.at[index, col] = most_similar_row[col]
    
    # Fill object columns with mode value
    object_columns = filled_df.select_dtypes(include=[object])
    
    for column in object_columns:
        mode_result = filled_df[column].mode()
        if not mode_result.empty:
            filled_df[column].fillna(mode_result.iloc[0], inplace=True)

    
    return filled_df

def get_Date_Column(DataFrame) -> pd.DataFrame:
    if not isinstance(DataFrame, pd.DataFrame):
        DataFrame = pd.DataFrame(DataFrame)

    date_formats = [
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

    for column in DataFrame.columns:
        values = DataFrame[column]
        for f in date_formats:
            try:
                date = pd.to_datetime(values, format=f)
                if date.dt.strftime(f).eq(values).all():
                    try:
                        DataFrame[column] = pd.to_datetime(DataFrame[column], errors='coerce')
                        if all([isinstance(val, str) and pd.to_datetime(val, errors='coerce') is not pd.NaT for val in DataFrame[column].values]) or (DataFrame[column].dtype == 'datetime64[ns]'):
                            DataFrame = DataFrame.set_index(column)
                            return DataFrame
                    except:
                        pass
            except ValueError:
                pass

    return DataFrame

def generate_warning_list(df):
    warning_list = []

    for col in df.columns:
        if df[col].dtype == "int64" or df[col].dtype == "float64":
            null_values = df[col].isna().sum()
            zero_values = (df[col] == 0).sum()
            total_values = len(df[col])
            null_ratio = int(null_values) / total_values
            zero_ratio = int(zero_values) / total_values
            unique_ratio = len(df[col].unique()) / total_values

            if null_ratio > 0.30:
                ratio_str = "NaN Rate : {:.2f}%".format(null_ratio * 100)
                column = "Column : " + str(col)
                warning_list.append([column, ratio_str])

            if zero_ratio > 0.50:
                ratio_str = "Sparsity Rate : {:.2f}%".format(zero_ratio * 100)
                column = "Column : " + str(col)
                warning_list.append([col, ratio_str])

            if unique_ratio > 0.80:
                ratio_str = "Unique Rate : {:.2f}%".format(unique_ratio * 100)
                column = "Column : " + str(col)
                warning_list.append([col, ratio_str])

    if len(warning_list) == 0:
        warning_list.append(["No Warning Exist"])

    return warning_list


def analyze_and_plot_distributions(df):
    result_dict = {}
    distributions = ['norm', 'uniform', 'binom', 'poisson', 'gamma', 'beta', 'lognorm', 'weibull_min', 'weibull_max', 'expon', 'pareto', 'cauchy', 'chi', 'f', 'laplace',
                    'bernoulli', 'exponential', 'geometric', 'hypergeometric', 'normal_mix', 'rayleigh', 'student_t', 'dweibull']

    for col in df.columns:
        try:
            is_int = df[col].apply(lambda x: x.is_integer()).all()
            if is_int:
                df[col] = df[col].astype("int")
                print(df.info())
        except:
            pass

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
                    best_dist = 'dweibull'
                elif best_dist == 'lognormvariate':
                    best_dist = 'lognorm'
                elif best_dist == 'norm':
                    best_dist = 'gauss'

                result_dict[col] = best_dist

    plot_results = {}

    for col, dist in result_dict.items():
        try:
            plt.figure()
            df_sampled[col].plot(kind='hist', bins=50, density=True, alpha=0.5)
            x = np.linspace(df_sampled[col].min(), df_sampled[col].max(), 100)
            x = np.tile(x, len(df_sampled[col]) // 100 + 1)[:len(df_sampled[col])]

            if dist == 'norm':
                loc, scale = scipy.stats.norm.fit(df_sampled[col])
                y = scipy.stats.norm(loc=loc, scale=scale).pdf(x)
            elif dist == 'beta':
                a, b, loc, scale = scipy.stats.beta.fit(df_sampled[col])
                y = scipy.stats.beta(a=a, b=b, loc=loc, scale=scale).pdf(x)
            else:
                if dist == 't':
                    y = scipy.stats.t.pdf(x, df_sampled[col])
                else:
                    y = getattr(scipy.stats, dist).pdf(x)

            plt.plot(x, y, label=dist)
            plt.xlabel(col)
            plt.legend()

            plot_results[col] = (x, y)
        except:
            continue


    return plot_results, result_dict



def replace_special_characters(col_name):
    forbidden_symbols = ['/', '|', '\\', '.', '*']
    for symbol in forbidden_symbols:
        col_name = col_name.replace(symbol, '_')
    return col_name

def separate_words(input_string):
    words = re.findall('[A-Z]+[a-z]*', input_string)
    filtered_words = [word for word in words if len(word) >= 2]
    return filtered_words

class ColumnTypePredictor:
    def __init__(self, model_path='col_pred_model.joblib'):
        # Get the absolute path to the script's directory
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the absolute path to the model file
        self.model_path = os.path.join(script_directory, model_path)
        self.loaded_model = None

    def load_model(self):
        # Load the trained model
        self.loaded_model = joblib.load(self.model_path)

    def predict_column_type(self, df, col):
        # Data processing logic
        data = pd.DataFrame(columns=['column', 'distm', 'dists', 'wc', 'wcs', 'len', 'lens', 'type', 'uniqR'])

        if len(df) > 5000:
            df = df.sample(5000)
        for c in df:
            m = df[c].mode()
            if m is not None and len(m) > 0:
                m = m[0]
                length = np.mean([len(str(s)) for s in df[c].values])
                lengths = np.std([len(str(s)) for s in df[c].values])
                wc = np.mean([len(str(s).split(' ')) for s in df[c].values])
                wcs = np.std([len(str(s).split(' ')) for s in df[c].values])

                dist = list(df[c].value_counts().to_dict().values())
                dist = [d / len(df) for d in dist]
                distm = np.mean(dist)
                dists = np.std(dist)

                data.loc[len(data) + 1] = [c, distm, dists, wc, wcs, length, lengths,
                                            df[c].dtype, df[c].nunique() / len(df)]

        data['type'] = data['type'].astype(str)  # Convert the column to string type

        # Use a lambda function and apply to replace values
        data['type'] = data['type'].apply(lambda x: 1 if 'float' in x else (0 if 'int' in x else None))

        # Drop rows with None values (rows with other types)
        data = data.dropna()

        # Optional: Convert the column back to int if needed
        data['type'] = data['type'].astype(float)

        type_col = data['type']
        data = data.drop(columns='type')
        data['type'] = type_col

        # Filter data for the specified column
        data = data[data['column'] == col]
        data = data.drop(columns=['column']).reset_index(drop=True)

        # Make predictions using the loaded model
        print(f"Predictions for the data {data}")
        predictions = self.loaded_model.predict(data)

        # Return the result
        return "scalar" if predictions[0] == 1 else "identifier"
    

# CLASS MODEL
class KolonTipiTahmini1:
    def __init__(self, threshold=20):
        self.threshold = threshold

    def fit_transform(self, data, columns=None, max_rows=5000):
        # Make a copy of the DataFrame 'data' after dropping rows with any NaN values
        if not isinstance(data, pd.DataFrame):
            data=pd.DataFrame(data)
        data_copy = data.dropna(axis=0).copy()

        if len(data_copy) > max_rows:
            data_copy = data_copy.sample(n=max_rows)

        kolon_role = []  # An empty list that can be used to store the role of each column.

        if columns:
            data_copy = data_copy[columns]

        for col in data_copy.columns:
            if len(data_copy[col].unique()) == 1:
                data_copy.drop(col, axis=1, inplace=True)
            elif len(data_copy[col].unique()) == 2:
                kolon_role.append('flag')
            elif isinstance(data_copy[col].iloc[0], str) and data_copy[col].str.split().str.len().mean() > 4:
                kolon_role.append('text')
            elif all([isinstance(val, str) and pd.to_datetime(val, errors='coerce') is not pd.NaT for val in
                     data_copy[col].values]) or (data_copy[col].dtype == 'datetime64[ns]'):
                kolon_role.append('date')
            elif len(data_copy[col].unique()) < self.threshold:
                kolon_role.append('categoric')
            elif all(isinstance(val, str) for val in data_copy[col].values):
                kolon_role.append('object')
            else:
                result = separate_words(col)

                if 'id' in map(str.lower, result):
                    print(f"Role of the column {col} is: identifier as column name includes 'id'")
                    kolon_role.append('id')
                else:
                    predictor = ColumnTypePredictor()
                    predictor.load_model()
                    # print(f"This is our data : {data_copy}")
                    role = predictor.predict_column_type(data_copy, col)
                    print(f"Role of the column {col} is: {role}")
                    
                    if role == 'scalar':
                        kolon_role.append('scalar')
                    else:
                        kolon_role.append('identifier')


        if 'date' in kolon_role:
            date_cols = [col for col, role in zip(data_copy.columns, kolon_role) if role == 'date']
            for date_col in date_cols:
                col_idx = data_copy.columns.get_loc(date_col)
                kolon_role[col_idx] = 'date'

        sonuc = []
        for i in data_copy.columns:
            if data_copy.columns.get_loc(i) < len(kolon_role):
                kolon_role_val = kolon_role[data_copy.columns.get_loc(i)]
            else:
                kolon_role_val = "unknown"
            sonuc.append({
                'col_name': i,
                'Role': kolon_role_val})

        result = {}
        for d in sonuc:
            col_name = d.pop('col_name')
            result[col_name] = d
        return result
    

