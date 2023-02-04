import numpy as np
import statsmodels.api as sm

class RegimeSwitchingModel(sm.tsa.MarkovRegression):
    def __init__(
        self,
        endog,
        k_regimes: int,
        **statsmodels_kwargs
    ) -> None:
        """Regime Switching Model（マルコフ回帰モデル）

        Args:
            endog (array): 目的変数
            k_regimes (int): レジーム数
        """
        super().__init__(endog, k_regimes, **statsmodels_kwargs)
        self.res = None

    def fit(self, *args, **kwargs):
        self.res = super().fit(*args, **kwargs)
    
    def forecast(self, exog, method="avgs"):
        # 直近の状態確率（フィルター化確率）
        latest_marginal_prob = np.array(self.res.filtered_marginal_probabilities.iloc[-1, :])
        # 遷移行列
        transition_matrix = self.res.regime_transition.squeeze()
        # 確率
        state_prob = transition_matrix  @ latest_marginal_prob
        # 被説明変数の名前（定数項含む）
        exog_names0 = [p + "[0]" for p in self.exog_names]
        exog_names1 = [p + "[1]" for p in self.exog_names]
        # 回帰係数
        coef0 = np.array(self.res.params.loc[exog_names0])
        coef1 = np.array(self.res.params.loc[exog_names1])
        # 説明変数に定数項をつける
        x = np.insert(exog, 0, 1)
        #予測値（レジームごと）
        pred0 = np.dot(x, coef0)
        pred1 = np.dot(x, coef1)
        # 予測値（状態確率を用いた重み月平均）
        if method == "avg":
            pred = np.array([pred0, pred1]) @ state_prob
        # 状態確率が高いほうを採用
        if method == "max":
            pred = pred0 if state_prob[0] > state_prob[1] else pred1
        return pred