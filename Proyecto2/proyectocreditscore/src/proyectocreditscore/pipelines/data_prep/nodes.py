"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""
import pandas as pd


def load_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data"""
    query_string = (
        "age > 0 and "
        "age <= 120 and "
        "num_bank_accounts >= 0 and "
        "num_of_loan >= 0 and "
        "num_of_delayed_payment >= 0 and "
        "monthly_balance >= 0"
    )
    df = df.query(query_string)
    df.loc[:, "occupation"] = df["occupation"].astype(str)
    df.loc[:, "payment_of_min_amount"] = df["payment_of_min_amount"]\
        .replace("nm", "no")
    df = df[df["payment_behaviour"] != "9#%8"]
    df.dropna(inplace=True)
    return df
