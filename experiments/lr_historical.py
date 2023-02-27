import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

sys.path.append("../src")
sys.path.append("../scripts")

from data import create_train_test_data

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
normalize_exog: bool = False
normalize_exog_type = "z2" # @params ["divmax", "z", "z2"]

# 非説明変数を正規化するか
normalize_endog = False
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
    results
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
    
    # normalize y
    if normalize_endog:
        mu = y_train.mean()[0]
        std = y_train.std()[0]
    else:
        mu = 0
        std = 1
    y_train = (y_train - mu) / std
    
    # fit
    model_rl = sm.OLS(y_train.to_numpy(), X_train.to_numpy())
    res_rl = model_rl.fit()

    # predict
    y_pred = res_rl.predict(X_test.to_numpy()) * std + mu
    
    summary_rl = {
        "date": base_date,
        "target": y_test[0],
        "pred": y_pred[0]
    }
    
    results.append(summary_rl)

def main():
    cpu_count = os.cpu_count()
    results = []
    with ThreadPoolExecutor(
        max_workers=cpu_count
    ) as executer:
        for base_date in tqdm(date_list):
            executer.submit(
                simulate,
                base_date,
                results
            )
    df_result = pd.DataFrame.from_dict(results)
    df_result.sort_values("date", inplace=True)
    
    save_dir = Path("../data/output/linear_regression/01")
    while save_dir.exists():
        num = int(save_dir.stem) + 1
        save_dir = save_dir.parent / f"{num:02}"
    save_dir.mkdir(exist_ok=True, parents=True)
        
    df_result.to_csv(save_dir / "results.csv", index=False)
    
    with (save_dir / "setting.json").open("w") as f:
        json.dump(settings, f)
    
if __name__ == "__main__":
    main()