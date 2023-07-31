
import pandas as pd

data = pd.read_csv("time2.csv")

def get_Date_Column(DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
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

data = get_Date_Column(data)
print(data)