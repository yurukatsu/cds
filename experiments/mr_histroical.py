import datetime
import json
import pickle
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

sys.path.append("../src")
sys.path.append("../scripts")

from data import create_train_test_data
from model import RegimeSwitchingModel

# Setting
# パス，変数
# ディレクトリ
# DATA_INPUT_DIR = Path(r"O:\プロジェクト\P00770\01_Simulation\Linear_Model") # インプットデータディレクトリ
DATA_INPUT_DIR = Path("../data/input")
DATA_OUTPUT_DIR = Path("../data/output") # 出力データディレクトリ
# 分析条件
prediction_lag = 1
corr_rolling_span = 250
rolling_span = corr_rolling_span
# ファイル
endog_file = DATA_INPUT_DIR / "X.csv"
exog_file = DATA_INPUT_DIR / "Y.csv"
selected_factors_file = DATA_INPUT_DIR / f"Selected_Factors_Col{corr_rolling_span}_lag{prediction_lag}.csv"
# 日付リスト作成
_df = pd.read_csv(selected_factors_file, index_col=0)
_df.index = pd.to_datetime(_df.index)
date_list = _df.index.to_list()[:-1]
del _df
# 説明変数を正規化するか
normalize_exog: bool = True
normalize_exog_type = "divmax" # @params ["divmax", "z"]

# 非説明変数を正規化するか
normalize_endog = True
normalize_endog_type = "z"

settings = {
    "prediction_lag": prediction_lag,
    "corr_rolling_span": corr_rolling_span,
    "rolling_span": rolling_span,
    "normalize_endog": normalize_endog,
    "normalize_endog_type": normalize_endog_type,
    "normalize_exog": normalize_exog,
    "normalize_exog_type": normalize_exog_type
}

def simulate(
    base_date: datetime.datetime,
    results,
    save_path,
    max_iter: int = 10
):
    y_train, y_test, X_train, X_test = create_train_test_data(
        base_date,
        endog_file,
        exog_file,
        selected_factors_csv_file = selected_factors_file,
        prediction_lag = prediction_lag,
        rolling_span = rolling_span,
        normalize_exog = normalize_exog,
        normalize_exog_type = normalize_exog_type,
        varbose=False
    )
    y_train.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    
    # normalize y
    if normalize_endog:
        mu = y_train.mean()[0]
        std = y_train.std()[0]
    else:
        mu = 0
        std = 1
    y_train = (y_train - mu) / std
    
    # folder
    save_path = save_path / "tmp" / base_date.strftime("%Y%m%d")
    if not save_path.exists():
        save_path.mkdir(parents=True)
    res = []
    
    for _ in tqdm(range(1, max_iter + 1)):
        # Regime Switching
        rs_model = RegimeSwitchingModel(
            y_train,
            k_regimes=2,
            exog=X_train,
            trend='c',
            switching_trend=True,
            switching_exog=True,
            switching_variance=True
        )
        y_train,
        # パラメラ推定（モデル作成）
        rs_model.fit(
            maxiter=10000,
            em_iter=10000,
            cov_type="robust",
            method="powell",
            search_reps=100,
            search_iter=1000,
            search_scale=1.0,
            disp=True
        )
        p00 = rs_model.res.params["p[0->0]"]
        check = (p00 > 0)
        if check:
            no_error = True
            # 予測
            y_pred_avg = rs_model.forecast(X_test.to_numpy(), method="avg") * std + mu
            y_pred_max = rs_model.forecast(X_test.to_numpy(), method="max") * std + mu
            target = y_test[0]
        else:
            no_error = False
            y_pred_avg = None
            y_pred_max = None
            target = y_test[0]
            
        summary = {
            "date": base_date,
            "target": target,
            "pred_max": y_pred_max,
            "pred_avg": y_pred_avg,
            "no_error": no_error,
        }
        res.append(summary)
        with (save_path / f"model_{_:02}.pkl").open("wb") as f:
            pickle.dump(rs_model, f)
            
        results.append(summary)
        
    df_summary = pd.DataFrame.from_dict(res)
    df_summary.to_csv(save_path / f"results.csv", index=False)
        
def main():
    cpu_count = os.cpu_count()
    results = []
    # folder
    save_dir = Path("../data/output/markov_regression/01")
    while save_dir.exists():
        num = int(save_dir.stem) + 1
        save_dir = save_dir.parent / f"{num:02}"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        for base_date in tqdm(date_list[::-1]):
            executor.submit(
                simulate,
                base_date,
                results,
                save_dir
            )
    df_result = pd.DataFrame.from_dict(results)
    df_result.sort_values("date", inplace=True)
        
    df_result.to_csv(save_dir / "results.csv", index=False)
    
    with (save_dir / "setting.json").open("w") as f:
        json.dump(settings, f)

if __name__ == "__main__":
    main()