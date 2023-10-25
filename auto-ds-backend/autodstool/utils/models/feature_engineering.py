import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def perform_operation(df, col_1, col_2, method='add'):
    """
    Perform addition, subtraction, multiplication, or division of two columns from a DataFrame and return a new DataFrame with the result.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_1 (str): The name of the first column.
        col_2 (str): The name of the second column.
        operation (str): The operation to perform ('add', 'subtract', 'multiply', or 'divide').

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing the result
                      of the specified operation on the specified columns as integers or floats.
    """
    # Check if the specified columns exist in the DataFrame
    if col_1 not in df.columns or col_2 not in df.columns:
        raise ValueError("Specified columns do not exist in the DataFrame.")
    
    # Check if the values in both columns are numeric
    if not pd.api.types.is_numeric_dtype(df[col_1]) or not pd.api.types.is_numeric_dtype(df[col_2]):
        raise ValueError("Both columns must contain numeric values.")

    if 'divide' in method and (df[col_2] == 0).any():
        raise ValueError("Division by zero encountered. Aborting operation.")

    # Perform the specified operation and store it in a new column
    if method == 'add':
        new_col_name = f"{col_1}_{col_2}_sum"
        df[new_col_name] = df[col_1] + df[col_2]
    elif method == 'subtract':
        new_col_name = f"{col_1}_{col_2}_difference"
        df[new_col_name] = df[col_1] - df[col_2]
    elif method == 'multiply':
        new_col_name = f"{col_1}_{col_2}_multiplied"
        df[new_col_name] = df[col_1] * df[col_2]
    elif method == 'divide':
        new_col_name = f"{col_1}_{col_2}_division"
        df[new_col_name] = df[col_1] / df[col_2]
    else:
        raise ValueError("Invalid method. Use 'add', 'subtract', 'multiply', or 'divide'.")

    return df


def take_first_n(df, col_name, n):
    """
    Create a new column in a DataFrame based on the first "n" characters from the values of the input column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the input column.
        n (int): The number of characters to take from the beginning of each value.

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing the first "n" characters from the input column values.
    """
    # Check if the specified column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Convert the values in the column to strings
    df[col_name] = df[col_name].apply(str)
    
    # Check if any value has more than "n" characters
    if (df[col_name].str.len() > n).any():
        raise ValueError(f"Some values in column '{col_name}' have more than {n} characters.")

    # Create a new column with the first "n" characters from the input column values
    new_col_name = f"{col_name}_{n}_chars"
    df[new_col_name] = df[col_name].str[:n]

    return df


def take_last_n(df, col_name, n):
    """
    Create a new column in a DataFrame based on the last "n" characters from the values of the input column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the input column.
        n (int): The number of characters to take from the end of each value.

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing the last "n" characters from the input column values.
    """
    # Check if the specified column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Convert the values in the column to strings
    df[col_name] = df[col_name].apply(str)
    
    # Check if any value has more than "n" characters
    if (df[col_name].str.len() > n).any():
        raise ValueError(f"Some values in column '{col_name}' have more than {n} characters.")
    
    # Create a new column with the last "n" characters from the input column values
    new_col_name = f"{col_name}_last_{n}_chars"
    df[new_col_name] = df[col_name].str[-n:]

    return df


def create_transformed_column(df, col_name, transform_type='log', n=None):
    """
    Create a new column in a DataFrame by applying various transformations (logarithm, power, root) to the values in the input column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the input column.
        transform_type (str): The type of transformation to apply ('log', 'power', 'root').
        n (float): The power or root to apply (only used when transform_type is 'power' or 'root').

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing the transformed values.
    """
    # Check if the specified column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Check if the values in the column are numeric
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        raise ValueError(f"Column '{col_name}' contains non-numeric values.")
    
    # Check if any value has more than "n" characters
    if (df[col_name].str.len() > n).any():
        raise ValueError(f"Some values in column '{col_name}' have more than {n} characters.")
    
    # Create a new column with the transformed values
    new_col_name = f"{col_name}_{transform_type}"
    
    if transform_type == 'log':
        try:
            df[new_col_name] = np.log(df[col_name])
        except ValueError as e:
            error_message = str(e)
            if "divide by zero" in error_message:
                raise ValueError(f"Logarithm of 0 encountered in column '{col_name}'.") from e
            elif "invalid value encountered in log" in error_message:
                raise ValueError(f"Logarithm of inf or -inf encountered in column '{col_name}'.") from e
    elif transform_type == 'power':
        if n is None:
            raise ValueError("Parameter 'n' is required for 'power' transformation.")
        df[new_col_name] = np.power(df[col_name], n)
    elif transform_type == 'root':
        if n is None:
            raise ValueError("Parameter 'n' is required for 'root' transformation.")
        df[new_col_name] = np.power(df[col_name], 1/n)
    else:
        raise ValueError("Invalid transform_type. Use 'log', 'power', or 'root'.")

    return df


def replace_values(df, col_name, val_search, val_replace):
    """
    Replace specific values in a DataFrame column with a specified replacement value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to search and replace values in.
        val_search: The value to search for in the specified column.
        val_replace: The value to replace the found values with.

    Returns:
        pd.DataFrame: A new DataFrame with the specified replacements made in the specified column.
    """
    # Check if the specified column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Replace the specified values in the column
    df[col_name] = df[col_name].replace(val_search, val_replace)

    return df


def create_flag_column(df, col_name, val_search):


    """
    Create a new column in a DataFrame with 1 if the specified value is found in the specified column, and 0 otherwise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to search for the value.
        val_search: The value to search for in the specified column.

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing 1 if the value is found, and 0 otherwise.
    """
    # Check if the specified column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Create a new column with 1 if the value is found, and 0 otherwise
    new_col_name = f"{col_name}_flag"
    df[new_col_name] = df[col_name].apply(lambda x: 1 if x == val_search else 0)

    return df


def scale_column_in_dataframe(df, col_name, scaler_type):
    # Create a copy of the original DataFrame
    df_copy = df.copy()
    
    # Select the column to scale
    column_to_scale = df_copy[col_name].values.reshape(-1, 1)
    
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler_type. Please use 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'.")
    
    # Fit and transform the selected column
    scaled_values = scaler.fit_transform(column_to_scale)
    
    # Replace the original column in the DataFrame with the scaled values
    df_copy[col_name] = scaled_values
    
    return df_copy


def remove_outliers(df, col_name, Q1=0.1, Q3=0.99, remove=True):
    # Create a copy of the original DataFrame
    df_copy = df.copy()
    
    # Select the column for outlier removal
    column_data = df_copy[col_name]
    
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (column_data < lower_bound) | (column_data > upper_bound)
    
    if remove:
        # Remove outliers from the DataFrame
        df_copy = df_copy[~outliers]
    
    return df_copy, outliers

