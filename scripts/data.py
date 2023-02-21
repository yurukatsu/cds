import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

def create_train_test_data(
    base_date: datetime.datetime,
    endog_csv_file: str | Path,
    exog_csv_file: str | Path,
    selected_factors_csv_file: str | Path = None,
    prediction_lag: int = 2,
    use_factor_lag: int = 1,
    rolling_span: int = 250,
    normalize_exog: bool = True,
    varbose: bool = True
) -> Tuple:
    """訓練データとテストデータを作成する

    Args:
        base_date (datetime.datetime): 基準日（予測したい日）
        endog_csv_file (str | Path): 目的変数csv
        exog_csv_file (str | Path): 説明変数csv
        selected_factors_csv_file (str | Path): 選択ファクターcsv
        prediction_lag (int, optional): 予測ラグ. Defaults to 1.
        rolling_span (int, optional): 回帰期間. Defaults to 250.

    Returns:
        Tuple: y訓練，yテスト, X訓練, Xテスト
    """
    # データの読み込み
    df_X = pd.read_csv(endog_csv_file, index_col=0) # 説明変数
    df_y = pd.read_csv(exog_csv_file, index_col=0) # 非説明変数
    if selected_factors_csv_file is not None:
        df_factor_selected = pd.read_csv(selected_factors_csv_file, index_col=0) # 選択された変数
    
    # インデックスを日付型にする
    df_X.index = pd.to_datetime(df_X.index)
    df_y.index = pd.to_datetime(df_y.index)
    if selected_factors_csv_file is not None:
        df_factor_selected.index = pd.to_datetime(df_factor_selected.index)
    
    # 選択されたファクター
    if selected_factors_csv_file is not None:
        _factor_selected = df_factor_selected.loc[
            df_factor_selected.index < base_date].iloc[-use_factor_lag]
        factor_selected = _factor_selected.to_list()
    else:
        factor_selected = df_X.columns.to_list()
    # 被説明変数（学習期間（rolling_span日）+予測期間（正解）（1日））
    y = df_y.loc[df_y.index <= base_date].iloc[-rolling_span - 1:, :]
    # 説明変数（学習用（rolling_span日）+予測用（1日））
    if prediction_lag > 1:
        X = df_X.loc[df_X.index < base_date].loc[:, factor_selected].iloc[-rolling_span - prediction_lag:-prediction_lag+1, :]
    else:
        X = df_X.loc[df_X.index < base_date].loc[:, factor_selected].iloc[-rolling_span - prediction_lag:, :]
    # 標準化する
    if normalize_exog:
        # X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = X/np.abs(X).max(axis=0)
        
    # print(X.mean(axis=0), X.shape)
        
    if varbose:
        print(
                "base_date: {}".format(base_date),
                "date_use_factor: {}".format(_factor_selected.name),
                "date_y_train: {}".format(y.index[-2]),
                "date_X_train: {}".format(X.index[-2]),
                "date_y_test: {}".format(y.index[-1]),
                "date_X_test: {}".format(X.index[-1])
        )

    return y.iloc[:-1 ], y.iloc[-1].values, X.iloc[:-1, :], X.iloc[-1, :]