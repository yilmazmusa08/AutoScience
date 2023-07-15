#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
import scipy.stats
from sklearn.preprocessing import LabelEncoder
from fitter import Fitter

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
            print(col,'null ratio ', null_ratio)
            print(col,'zero ratio ', zero_ratio)
            print(col,'unique ratio ', unique_ratio)



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

