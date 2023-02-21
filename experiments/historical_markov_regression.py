import datetime
import sys
import os
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import pandas as pd
from tqdm import tqdm

sys.path.append("../src")
sys.path.append("../scripts")

from model import RegimeSwitchingModel
from data import create_train_test_data

# パス，変数
# ディレクトリ
DATA_INPUT_DIR = Path("../data/input") # インプットデータディレクトリ
DATA_OUTPUT_DIR = Path("../data/output") # 出力データディレクトリ
# 分析条件
prediction_lag = 1
rolling_span = 250
# ファイル
endog_file = DATA_INPUT_DIR / "X.csv"
exog_file = DATA_INPUT_DIR / "Y.csv"
selected_factors_file = DATA_INPUT_DIR / f"Selected_Factors_Col{rolling_span}_lag{prediction_lag}.csv"
# 日付リスト作成
_df = pd.read_csv(selected_factors_file, index_col=0)
_df.index = pd.to_datetime(_df.index)
date_list = _df.index.to_list()[:-1]
del _df

def _make_model(base_date: datetime.datetime, save_path:Path):
    y_train, y_test, X_train, X_test = create_train_test_data(
        base_date,
        endog_file,
        exog_file,
        selected_factors_csv_file = selected_factors_file,
        prediction_lag = prediction_lag,
        rolling_span = rolling_span
    )
    # マルコフ回帰
    rs = RegimeSwitchingModel(
        y_train,
        k_regimes=2,
        exog=X_train,
        trend='c',
        switching_trend=True,
        switching_exog=True,
        switching_variance=True
    )
    # パラメラ推定（モデル作成）
    rs.fit(
        maxiter=10000,
        em_iter=10000,
        cov_type="approx",
        # method="bfgs",
        method="powell",
        search_reps=50,
        search_iter=50,
        disp=True
    )
    result = {
        "base_date": base_date, 
        "model": rs,
        "y_test": y_test,
        "X_test": X_test
    }
    output_file = save_path / "result_markov_regression_{}.pkl".format(base_date.strftime("%Y%m%d"))
    
    with output_file.open(mode="wb") as f:
        pickle.dump(result, f)
    
def main():
    cpu_count = os.cpu_count()
    output_file = DATA_OUTPUT_DIR / "markov_regression_lag1"
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        for base_date in tqdm(date_list):
            executor.submit(
                _make_model,
                base_date,
                output_file
            )
        
if __name__ == "__main__":
    main()