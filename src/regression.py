"""
LightGBM regression for predicting committor probabilities from structural features.

Usage (from a notebook):

    from src.regression import regression
    score, model, X_train, X_test, y_train, y_test = regression(
        df_features,
        committor_csv="data/cluster_40_prob.csv",
        down=0.1,    # use stratified 10% downsample; 1 = no downsampling
    )
"""

import re

import lightgbm
import pandas as pd
from sklearn.model_selection import train_test_split

from .downsample import dsample


def regression(df_x, committor_csv, committor_col="Committor_prob", down=1):
    """
    Train a LightGBM regressor to predict committor probability.

    Parameters
    ----------
    df_x : pd.DataFrame
        Feature matrix (one row per trajectory frame, same order as committor CSV).
    committor_csv : str
        Path to the cluster probability CSV (must contain `committor_col`).
    committor_col : str
        Column name for the committor probability in `committor_csv`.
    down : float
        Downsampling fraction passed to dsample() (1 = no downsampling).

    Returns
    -------
    score : float
        R² on the test set.
    model : LGBMRegressor
        Trained model.
    X_train, X_test, y_train, y_test : DataFrames/Series
        Train/test splits (useful for SHAP analysis).
    """
    dfy = pd.read_csv(committor_csv)[committor_col]
    df = pd.concat([df_x.reset_index(drop=True), dfy.reset_index(drop=True)], axis=1)

    # Drop boundary frames labeled Null or NaN
    df = df[df[committor_col] != "Null"]
    df = df.dropna(subset=[committor_col])
    df = df.astype(float)

    # Sanitise column names (PyEMMA puts colons in feature labels)
    df.rename(columns=lambda x: re.sub(":", "_", x), inplace=True)

    # Min-max normalise all columns
    for col in df.columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)

    if down != 1:
        df = dsample(df, down)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = lightgbm.LGBMRegressor(seed=1)
    print("Training LightGBM …")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"R² (test): {score:.4f}")
    return score, model, X_train, X_test, y_train, y_test
