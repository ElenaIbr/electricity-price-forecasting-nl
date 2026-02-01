import pandas as pd

def check_date_range(df: pd.DataFrame) -> tuple:
    return df.index.min(), df.index.max()


def check_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def check_duplicates(df: pd.DataFrame) -> int:
    return df.index.duplicated().sum()


def basic_report(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("Date range:", check_date_range(df))
    print("Missing values:")
    print(check_missing(df))
    print("Duplicate timestamps:", check_duplicates(df))
