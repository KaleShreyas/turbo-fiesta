import os
import pickle
import click
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

class CONFIG:
    NAMES_DTYPES = {
        "Source" : str,
        "Production" : np.float32
    }


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_csv(
        filename,
        parse_dates=["Date and Hour", "Date"],
        infer_datetime_format=True,
        dtype=CONFIG.NAMES_DTYPES
        )
    
    df.dropna(inplace=True)

    df['StartHour'] = df['StartHour'].replace('24:00:00', '00:00:00')
    df['EndHour'] = df['EndHour'].replace('24:00:00', '00:00:00')
    df['StartHour'] = pd.to_datetime(df['StartHour'])
    df['EndHour'] = pd.to_datetime(df['EndHour'])
    df['TimeDifference'] = df['EndHour'] - df['StartHour']

    df['Total_time'] = df['TimeDifference'].dt.components['hours']
    df['Total_time']=df['Total_time']*60

    df = df.drop(['TimeDifference', 'StartHour', 'EndHour', 'Date and Hour', 'Date'], axis=1)
    lst = ['Source','dayName','monthName']
    for i in lst:
        df[i] = le.fit_transform(df[i])

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    dicts = df.to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw data is saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "intermittent-renewables-production-france.csv"):
    # Load input files
    df = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}")
    )

    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])

    # Extract the target
    target = 'Production'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Drop the target
    df_train.drop([target], axis=1, inplace=True)
    df_val.drop([target], axis=1, inplace=True)
    df_test.drop([target], axis=1, inplace=True)

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

    print("Data Pickled")


if __name__ == '__main__':
    le = LabelEncoder()
    run_data_prep()
