import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def perform_operation(df, column_1, column_2, method='add'):
    """
    Perform addition, subtraction, multiplication, or division of two columns from a DataFrame and return a new DataFrame with the result.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_1 (str): The name of the first column.
        column_2 (str): The name of the second column.
        operation (str): The operation to perform ('add', 'subtract', 'multiply', or 'divide').

    Returns:
        pd.DataFrame: A new DataFrame with an additional column containing the result
                      of the specified operation on the specified columns as integers or floats.
    """
    # Check if the specified columns exist in the DataFrame
    if column_1 not in df.columns or column_2 not in df.columns:
        raise ValueError("Specified columns do not exist in the DataFrame.")
    
    # Check if the values in both columns are numeric
    if not pd.api.types.is_numeric_dtype(df[column_1]) or not pd.api.types.is_numeric_dtype(df[column_2]):
        raise ValueError("Both columns must contain numeric values.")

    if 'divide' in method and (df[column_2] == 0).any():
        raise ValueError("Division by zero encountered. Aborting operation.")

    # Perform the specified operation and store it in a new column
    if method == 'add':
        new_column_name = f"{column_1}_{column_2}_sum"
        df[new_column_name] = df[column_1] + df[column_2]
    elif method == 'subtract':
        new_column_name = f"{column_1}_{column_2}_difference"
        df[new_column_name] = df[column_1] - df[column_2]
    elif method == 'multiply':
        new_column_name = f"{column_1}_{column_2}_multiplied"
        df[new_column_name] = df[column_1] * df[column_2]
    elif method == 'divide':
        new_column_name = f"{column_1}_{column_2}_division"
        df[new_column_name] = df[column_1] / df[column_2]
    else:
        raise ValueError("Invalid method. Use 'add', 'subtract', 'multiply', or 'divide'.")

    return df


def take_first_n(df, column_name, n):
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
    if column_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Convert the values in the column to strings
    df[column_name] = df[column_name].apply(str)
    
    # Check if any value has more than "n" characters
    if (df[column_name].str.len() < n).any():
        raise ValueError(f"Some values in column '{column_name}' have more than {n} characters.")

    # Create a new column with the first "n" characters from the input column values
    new_column_name = f"{column_name}_{n}_chars"
    df[new_column_name] = df[column_name].str[:n]

    return df


def take_last_n(df, column_name, n):
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
    if column_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Convert the values in the column to strings
    df[column_name] = df[column_name].apply(str)
    
    # Check if any value has more than "n" characters
    if (df[column_name].str.len() < n).any():
        raise ValueError(f"Some values in column '{column_name}' have more than {n} characters.")
    
    # Create a new column with the last "n" characters from the input column values
    new_column_name = f"{column_name}_last_{n}_chars"
    df[new_column_name] = df[column_name].str[-n:]

    return df


def create_transformed_column(df, column_name, transform_type='log', n=None):
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
    if column_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Check if the values in the column are numeric
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' contains non-numeric values.")
    
    # Create a new column with the transformed values
    new_column_name = f"{column_name}_{transform_type}"
    
    if transform_type == 'log':
        try:
            df[new_column_name] = np.log(df[column_name])
        except ValueError as e:
            error_message = str(e)
            if "divide by zero" in error_message:
                raise ValueError(f"Logarithm of 0 encountered in column '{column_name}'.") from e
            elif "invalid value encountered in log" in error_message:
                raise ValueError(f"Logarithm of inf or -inf encountered in column '{column_name}'.") from e
    elif transform_type == 'power':
        if n is None:
            raise ValueError("Parameter 'n' is required for 'power' transformation.")
        df[new_column_name] = np.power(df[column_name], n)
    elif transform_type == 'root':
        if n is None:
            raise ValueError("Parameter 'n' is required for 'root' transformation.")
        df[new_column_name] = np.power(df[column_name], 1/n)
    else:
        raise ValueError("Invalid transform_type. Use 'log', 'power', or 'root'.")

    return df


def replace_values(df, column_name, value_to_search, value_to_replace):
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
    if column_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Replace the specified values in the column
    df[column_name] = df[column_name].replace(value_to_search, value_to_replace)

    return df


def create_flag_column(df, column_name, value_to_search):


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
    if column_name not in df.columns:
        raise ValueError("Specified column does not exist in the DataFrame.")
    
    # Create a new column with 1 if the value is found, and 0 otherwise
    new_column_name = f"{column_name}_flag"
    df[new_column_name] = df[column_name].apply(lambda x: 1 if x == value_to_search else 0)

    return df


def scale_column_in_dataframe(df, column_name, scaler_type=StandardScaler):
    # Create a copy of the original DataFrame
    df_copy = df.copy()
    
    # Select the column to scale
    column_to_scale = df_copy[column_name].values.reshape(-1, 1)
    
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
    df_copy[column_name] = scaled_values
    
    return df_copy


def remove_outliers(df, column_name, Quartile_1=1, Quartile_3=99, remove=True):
    # Create a copy of the original DataFrame
    df_copy = df.copy()
    
    # Select the column for outlier removal
    column_data = df_copy[column_name]
    
    # Convert Quartiles to a percentage value
    Quartile_1 = Quartile_1 / 100
    Quartile_3 = Quartile_3 / 100

    # Calculate the IQR (Interquartile Range)
    IQR = Quartile_3 - Quartile_1
    
    # Define lower and upper bounds for outliers
    lower_bound = Quartile_1 - 1.5 * IQR
    upper_bound = Quartile_3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (column_data < lower_bound) | (column_data > upper_bound)
    
    if remove:
        # Remove outliers from the DataFrame
        df_copy = df_copy[~outliers]
    
    return df_copy

