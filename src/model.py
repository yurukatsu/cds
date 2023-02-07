from typing import Tuple

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


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
    
class MarkovRegressionWithPenalty:
    def __init__(
        self,
        y: np.array,
        X: np.array,
        k_regimes: int
    ):
        # 目的変数
        self.y = y
        # 観測数
        self.nobs = y.shape[0]
        if y.shape != (self.nobs, 1):
            raise ValueError("目的変数の型は（サンプル数, 1）")
        # 説明変数
        self.X = X
        self.k_exogs = X.shape[1]
        if X.shape != (self.nobs, self.k_exogs):
            raise ValueError("説明変数の型は（サンプル数，変数数）")
        # レジーム数
        self.k_regimes = k_regimes
        if k_regimes != 2:
            raise NotImplementedError
        
    # f(y_t| \Omega_{t-1}, s_t = s)
    # 時刻tで状態sにいるときに，y_tが観測される確率
    # パラメタとして，各状態における，定数項，回帰係数，分散を与える
    @classmethod
    def calc_conditional_observation_probabilities(
        cls,
        y: np.array,
        X: np.array,
        trend: np.array,
        beta: np.array,
        variance: np.array
    ) -> np.array:
        """時刻tで状態sにいるときに, y_tが観測される確率を計算

        Args:
            y (np.array): 目的変数（観測回数, 1）
            X (np.array): 説明変数（観測回数, 変数の数）
            trend (np.array): 定数項（トレンド項）（レジーム数, ）
            beta (np.array): 回帰係数（レジーム数，変数の数）
            variance (np.array): 分散（レジーム数，）

        Returns:
            np.array: 時刻tで状態sにいるときに, y_tが観測される確率（観測回数, レジーム数）
        """
        # データの型を確認する
        # 観測数が同じであることをチェック
        if y.shape[0] != X.shape[0]:
            raise ValueError("目的変数と説明変数の観測数が異なります．")
        # レジーム数が同じであること確認
        if not (trend.shape[0] == beta.shape[0] == variance.shape[0]):
            raise ValueError("レジーム数がパラメータ間で異なります．")
        # 説明変数の数があっていることを確認
        if X.shape[1] != beta.shape[1]:
            raise ValueError("説明変数の数が異なります．")
        
        # 時刻tで状態sにいるときに, y_tが観測される確率
        prob = np.exp(- (y - X @ beta.T - trend)**2 / 2 / variance)
        prob /= np.sqrt(2 * np.pi * variance)
        
        return prob

    # パラメタp1, p2から，遷移行列を計算する
    # p11 = 1 / (1 + exp(-p1))
    # p22 = 1 / (1 + exp(-p2))
    @classmethod
    def make_2d_transition_matrix_from_parameters(
        cls,
        p1: float,
        p2: float
    ) -> np.array:
        """

        Args:
            p1 (float): p11を与えるパラメタ
            p2 (float): p22を与えるパラメタ

        Returns:
            np.array: 遷移行列
        """
        p11 = 1 / (1 + np.exp(-p1))
        p22 = 1 / (1 + np.exp(-p2))
        # 遷移行列
        transition_matrix = np.array(
            [
                [p11, 1 - p22],
                [1 - p11, p22]
            ]
        )
        return transition_matrix

    # 遷移行列から定常確率を求める
    # A = (I - P, 1^T)^Tとして，
    # (A^T A)^{-1} A.Tの（レジーム数+1） 列目のベクトル
    @classmethod
    def cal_ergotic_stationary_prob(
        cls,
        transition_matrix: np.array
        ) -> np.array:
        """エルゴード的定常確率を求める

        Args:
            transition_matrix (array): 遷移行列（レジーム数，レジーム数）
        """
        # レジーム数
        k_regimes = transition_matrix.shape[0]
        # Aを計算
        A = np.ones((k_regimes+1, k_regimes))
        A[:k_regimes, :k_regimes] = np.eye(k_regimes) - transition_matrix
        # 定常確率を計算
        stationary_prob = (np.linalg.inv(A.T @ A) @ A.T)[:, -1]
        return stationary_prob

    # パラメタを1行のarrayにする
    @classmethod
    def gather_params(
        cls,
        trend: np.array,
        beta: np.array,
        variance: np.array,
        p1: float,
        p2: float
    ) -> np.array:
        """

        Args:
            trend (np.array): 定数項（トレンド項）（レジーム数, ）
            beta (np.array): 回帰係数（レジーム数，変数の数）
            variance (np.array): 分散（レジーム数，）
            p1 (float): p11を与えるパラメタ
            p2 (float): p22を与えるパラメタ

        Returns:
            np.array: パラメタをまとめた配列
        """
        # レジーム数
        k_regimes = trend.shape[0]
        # パラメタをレジームごとの配列にまとめる
        params = np.concatenate(
            [
                trend.reshape((k_regimes, 1)),
                beta,
                variance.reshape((k_regimes, 1)),
                np.array([[p1], [p2]])
            ],
            axis=1
        )
        return params.flatten()

    # 1列の配列にまとめられたパラメタを各々にばらす
    @classmethod
    def disassemble_params(
        cls,
        params: np.array,
        k_regimes: int,
        k_exogs: int
    ) -> Tuple:
        """1列の配列にまとめられたパラメタを各々にばらす

        Args:
            params (np.array): 1列の配列にまとめられたパラメタ（k_regimes + k_regimes * k_exogs + k_regimes + 1 + 1）
            k_regimes (int): レジームす
            k_exogs (int): 説明変数の数

        Returns:
            Tuple: それぞれのパラメタ（trend, beta, variance, p1, p2）
        """
        # レジームごとの配列にする
        params_ = params.reshape((k_regimes, -1))
        # トレンド
        trend = params_[:, 0]
        # 回帰係数
        beta = params_[:, 1:1+k_exogs]
        # 分散
        variance = params_[:, 1+k_exogs]
        # p11, p22をあたえるパラメタ
        p1 = params_[0, -1]
        p2 = params_[1, -1]
        
        return trend, beta, variance, p1, p2

    # 条件付き観測確率を周辺化したものとフィルター化確率を全観測期間で計算
    @classmethod
    def calc_marginal_observation_probs_and_filtered_probs(
        cls,
        conditional_observation_probabilities: np.array,
        transition_matrix: np.array,
        initial_filtered_prob: np.array
    ) -> Tuple:
        """条件付き観測確率を周辺化したものとフィルター化確率を全観測期間で計算

        Args:
            conditional_observation_probabilities (np.array): 条件付き観測確率
            transition_matrix (np.array): 遷移行列
            initial_filtered_prob (np.array): フィルター化確率の初期値

        Returns:
            Tuple: （条件付き観測確率を周辺化したもの，フィルター化確率）
        """
        # フィルター化確率を初期化
        filtered_prob = initial_filtered_prob
        # 結果の保存用にリスト作成
        list_marginal_observation_prob = []
        list_filtered_prob = []
        
        # 条件付き観測確率の周辺化とフィルター化確率を求める
        for conditional_observation_prob in conditional_observation_probabilities:
            # p(s_{t+1} = j | \Omega_t)
            next_state_prob = transition_matrix @ filtered_prob
            # 条件付き観測確率の周辺化（P(y | Omega_{t-1})）
            marginal_observation_prob = np.dot(
                conditional_observation_prob,
                next_state_prob
            )
            # フィルター化確率を更新
            filtered_prob = (next_state_prob * conditional_observation_prob) / marginal_observation_prob
            # 結果を蓄積
            list_marginal_observation_prob.append(marginal_observation_prob)
            list_filtered_prob.append(filtered_prob)
        return np.array(list_marginal_observation_prob), np.array(list_filtered_prob)
    
    # （重み付き）対数尤度
    @classmethod
    def weighted_likelihood(
        cls,
        marginal_observation_probabilities: np.array,
        weights: np.array,
        epsilon : float = 0
    ) -> float:
        """（重み付き）対数尤度を計算

        Args:
            marginal_observation_probabilities (np.array): 条件付き観測確率の周辺
            weights (np.array): 重み
            epsilon (float, optional): ゼロ回避用の微小量. Defaults to 1e-15.

        Returns:
            float: （重み付き）対数尤度
        """
        logL = np.dot(
            np.log(marginal_observation_probabilities + epsilon),
            weights
        ) / weights.sum() * len(weights)
        return logL
    
    @classmethod
    def exp_decay_weight(cls, n_sample: int, t_half: int = 125) -> np.array:
        return np.exp(- np.log(2) * np.linspace(0, n_sample-1, n_sample) / t_half)[::-1]
    
    def _exp_decay_weight(self, t_half: int = 125) -> np.array:
        return self.exp_decay_weight(self.nobs, t_half=t_half)
    
    @classmethod
    def equal_weight(cls, n_sample: int) -> np.array:
        return np.ones(n_sample) / n_sample
    
    def _equal_weight(self) -> np.array:
        """

        Returns:
            np.array: 等ウェイト 
        """
        return self.equal_weight(self.nobs)
        
    # コスト関数
    @classmethod
    def cost(
        cls,
        marginal_observation_probabilities: np.array,
        weights: np.array,
        p11: float,
        p22: float,
        gamma: float = 5,
        p11_base: float = 0.5,
        p22_base: float = 0.5,
        epsilon : float = 0
    ) -> float:
        logL = cls.weighted_likelihood(
            marginal_observation_probabilities,
            weights,
            epsilon=epsilon
        )
        penalty = gamma * ((p11 - p11_base)**2 + (p22 - p22_base)**2)
        return - logL + penalty

    def fit(
        self,
        start_params: np.array,
        cost_weights: np.array,
        method="powell",
        maxiter=100,
        **cost_kwargs
    ):
        # def constraint_1(params):
        #     _, _, log_variance, _, _ = self.disassemble_params(
        #         params, self.k_regimes, self.k_exogs
        #     )
        #     return log_variance[0] - log_variance[1]
        
        # constraints = (
        #     {"type": "ineq", "func": constraint_1}
        # )
        
        def _cost(
            params: np.array,
        ) -> float:
            trend, beta, log_variance, p1, p2 = self.disassemble_params(
                    params, self.k_regimes, self.k_exogs)
            
            # swap
            if log_variance[1] > log_variance[0]:
                log_variance = log_variance[::-1]
            
            # 条件付きの観測確率
            conditional_observation_probabilities = self.calc_conditional_observation_probabilities(self.y, self.X, trend, beta, np.exp(log_variance))
            # 遷移確率
            transition_matrix = self.make_2d_transition_matrix_from_parameters(p1, p2)
            # 初期フィルター化確率
            initial_filtered_prob = self.cal_ergotic_stationary_prob(transition_matrix)
            # 条件付きの観測確率の周辺
            marginal_observation_probabilities, _ = \
                self.calc_marginal_observation_probs_and_filtered_probs(
                    conditional_observation_probabilities,
                    transition_matrix,
                    initial_filtered_prob
                )
            # コスト関数
            cost_ = self.cost(
                marginal_observation_probabilities,
                cost_weights,
                transition_matrix[0, 0],
                transition_matrix[1, 1],
                **cost_kwargs
            )
            return cost_
        
        res = minimize(
            _cost,
            x0=start_params,
            method=method,
            options={
                "disp": True,
                "maxiter": maxiter
            }
        )
        
        trend, beta, log_variance, p1, p2 = self.disassemble_params(res.x, self.k_regimes, self.k_exogs)
        # swap
        if log_variance[1] > log_variance[0]:
            log_variance = log_variance[::-1]
        # 条件付きの観測確率
        conditional_observation_probabilities = self.calc_conditional_observation_probabilities(self.y, self.X, trend, beta, np.exp(log_variance))
        # 遷移確率
        transition_matrix = self.make_2d_transition_matrix_from_parameters(p1, p2)
        # 初期フィルター化確率
        initial_filtered_prob = self.cal_ergotic_stationary_prob(transition_matrix)
        # 条件付きの観測確率の周辺
        marginal_observation_probabilities, filtered_probabilities = \
            self.calc_marginal_observation_probs_and_filtered_probs(
                conditional_observation_probabilities,
                transition_matrix,
                initial_filtered_prob
            )
        summary = {
            "trend": trend,
            "beta": beta,
            "variance": np.sqrt(np.exp(log_variance)),
            "p1": p1,
            "p2": p2,
            "transition_matrix": transition_matrix,
            "marginal_observation_probabilities": marginal_observation_probabilities,
            "filtered_probabilities": filtered_probabilities
        }
        
        return summary