import io
import pandas as pd

def determine_csv_encoding(file_content):
    csv_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32']
    for encoding in csv_encodings:
        try:
            return pd.read_csv(io.BytesIO(file_content), encoding=encoding)
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
    return None

def determine_excel_df(file_content):
    try:
        return pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
    except (pd.errors.ParserError, UnicodeDecodeError):
        return None