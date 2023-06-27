#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier,export_graphviz, export_text
import warnings 
warnings.filterwarnings("ignore")
# import statsmodels.stats.api as sms
from scipy.stats import shapiro, levene, ttest_ind
# from statsmodels.stats.proportion import proportions_ztest
# from skompiler import skompile
# import graphviz

def describe_dataframe(df):
    """
    Verilen veri çerçevesindeki tüm sütunların NaN, benzersiz değerler, eşsiz değerler, açıklama vb. gibi 
    istatistiksel özelliklerini döndürür.
    
    Parametreler:
        df (pandas.DataFrame): Özetlenecek veri çerçevesi
        
    Döndürülen Değerler:
        pandas.DataFrame: Tüm sütunların istatistiksel özellikleri
    
    """
    df = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32', 'float16', 'int16'])
    na_count = df.isna().sum()
    
    unique_values = df.nunique()
    
    nunique_values = df.apply(pd.Series.nunique)
    
    desc_df = df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    
    result_df = pd.concat([na_count, unique_values, nunique_values, desc_df], axis=1)
    
    # Create a DataFrame with Calculated Results
    result_df.columns = ['NaN_values', 'Unique_values', 'Nunique_values', 'Count', 'Mean', 'Std', 'Min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'Max']
    
    return result_df

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Head #####################")
    return dataframe.head(head)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def col_types(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    num_but_car = [col for col in num_cols if dataframe[col].nunique() == len(dataframe) and
                  dataframe[col].dtypes == "int"]
    num_cols = [col for col in num_cols if col not in num_but_car]
    
    # date_cols
    date_cols = [col for col in cat_cols if dataframe[col].apply(lambda x: pd.to_datetime(x, errors='coerce')).notnull().all()]

    cat_cols = [col for col in cat_cols if col not in date_cols]

    return cat_cols, num_cols, cat_but_car + num_but_car, date_cols

def fill_na(row, col, lower_coeff=0.75, upper_coeff=1.25, df=None):
    if pd.notna(row[col]):
        return row[col]
    
    query = []
    
    obj = list(df.select_dtypes(include=['object']).columns)
    num = list(df.select_dtypes(exclude=['object']).columns)

    for o in obj:
        if pd.notna(row[o]):
            value = str(row[o]).replace("'", "").replace('"', "")  # Escape single and double quotes
            query.append(f"({o} == '{value}')")
    
    for n in num:
        if pd.notna(row[n]):
            query.append(f"({n} >= {row[n] * lower_coeff})")
            query.append(f"({n} <= {row[n] * upper_coeff})")
    
    query = " and ".join(query)
    sub = df.query(query)
    
    if len(sub) == 0:
        return None
    
    return sub[col].mean()

def fillnan(row, col, lower_coeff = 0.75, upper_coeff = 1.25):
    # to use an example : df['bmi'] = df.apply(lambda row: fillna(row, 'bmi'), axis = 1)
    query = []
    for o in obj:
        if not pd.isna(row[o]):
            query.append("(" + str(o) + " == '" + str(row[o]) + "')")
    for n in num:
        if not pd.isna(row[n]):
            query.append("(" + n + " >= " + str(row[n] * lower_coeff) + ")")
            query.append("(" + n + " <= " + str(row[n] * upper_coeff) + ")")

    query = " and ".join(query)
    sub = df.query(query)
    if len(sub) == 0:
        return None
    return sub[col].mean()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
    

def test_normality(df, col1_name, col2_name, alpha=0.05):
    '''
    Test whether two columns in a pandas DataFrame are normally distributed using the Shapiro-Wilk test.
    Returns True if both columns are likely normally distributed (p-values > alpha), False otherwise.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the data to test.
    col1_name : str
        The name of the first column to test.
    col2_name : str
        The name of the second column to test.
    alpha : float, optional
        The significance level to use. Default is 0.05.
        
    Returns:
    --------
    bool
        True if both columns are likely normally distributed, False otherwise.
    '''
    col1 = df[col1_name]
    col2 = df[col2_name]
    
    stat1, p1 = shapiro(col1)
    stat2, p2 = shapiro(col2)
    
    if p1 > alpha and p2 > alpha:
        return True
    else:
        return False

def test_variance_homogeneity(df, col1_name, col2_name, alpha=0.05):
    '''
    Test for variance homogeneity between two columns in a pandas DataFrame using the Levene's test.
    Returns True if the variances are homogeneous (p-value > alpha), False otherwise.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the data to test.
    col1_name : str
        The name of the first column to test.
    col2_name : str
        The name of the second column to test.
    alpha : float, optional
        The significance level to use. Default is 0.05.
        
    Returns:
    --------
    bool
        True if the variances are homogeneous, False otherwise.
    '''
    col1 = df[col1_name]
    col2 = df[col2_name]
    
    stat, p = levene(col1, col2)
    
    if p > alpha:
        return True
    else:
        return False

def compare_means(df, col1_name, col2_name, alpha=0.05):
    '''
    Compare the means of two columns in a pandas DataFrame using a two-sample t-test.
    Returns True if there is a significant difference between the means (p-value < alpha), False otherwise.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the data to test.
    col1_name : str
        The name of the first column to test.
    col2_name : str
        The name of the second column to test.
    alpha : float, optional
        The significance level to use. Default is 0.05.
        
    Returns:
    --------
    bool
        True if there is a significant difference between the means, False otherwise.
    '''
    col1 = df[col1_name]
    col2 = df[col2_name]
    
    stat, p = ttest_ind(col1, col2)
    
    if p < alpha:
        return True
    else:
        return False

