import numpy as np


class IMMFilter:
    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None):
        """
        改进版 IMM Filter
        参数:
            transition_probabilities: 转移概率矩阵
            initial_state: 初始状态 (6,)
            initial_cov: 初始协方差 (6,6)
            r_cov: 观测噪声协方差矩阵 (3,3) [关键修正]
        """
        self.M = 3  # 模型数量
        self.trans_prob = transition_probabilities
        # 初始模型概率均匀分布
        self.model_probs = np.array([1 / 3, 1 / 3, 1 / 3])

        self.dim = 6
        self.x = np.zeros((self.M, self.dim))
        self.P = np.zeros((self.M, self.dim, self.dim))

        # 初始状态和协方差
        for i in range(self.M):
            self.x[i] = initial_state.copy()
            self.P[i] = initial_cov.copy()

        # 模型参数 (加速度标准差)
        self.sigmas = [2.0, 25.0, 100.0]

        # 测量矩阵 (只观测位置 x, y, z)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 2] = 1
        self.H[2, 4] = 1

        # 观测噪声 R
        if r_cov is not None:
            self.R = r_cov
        else:
            # 默认给一个，但建议外部传入
            print("Warning: Using default R matrix (std=4).")
            self.R = np.eye(3) * 16

    def get_F_matrix(self, dt):
        """生成 CV 模型的转移矩阵 (依赖 dt)"""
        F = np.eye(6)
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        return F

    def get_kinematic_Q(self, dt, sigma):
        """
        生成运动学过程噪声矩阵 (依赖 dt)
        """
        q_var = sigma ** 2
        q2 = (dt ** 3) / 3.0
        q3 = (dt ** 2) / 2.0
        q4 = dt

        # 2x2 块
        block = np.array([
            [q2, q3],
            [q3, q4]
        ]) * q_var

        # 组装成 6x6
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = block
        Q[2:4, 2:4] = block
        Q[4:6, 4:6] = block

        return Q

    def interact(self):
        """步骤 1: 交互/混合"""
        EPS = 1e-20
        c_bar = np.dot(self.trans_prob.T, self.model_probs)

        # 计算混合概率 mu_ij
        mixing_probs = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(self.M):
                mixing_probs[i, j] = (self.trans_prob[i, j] * self.model_probs[i]) / (c_bar[j] + EPS)

        # 计算混合后的初始状态和协方差
        x_mixed = np.zeros((self.M, self.dim))
        P_mixed = np.zeros((self.M, self.dim, self.dim))

        for j in range(self.M):
            # 状态混合
            for i in range(self.M):
                x_mixed[j] += mixing_probs[i, j] * self.x[i]

            # 协方差混合
            for i in range(self.M):
                diff = (self.x[i] - x_mixed[j]).reshape(-1, 1)
                P_mixed[j] += mixing_probs[i, j] * (self.P[i] + diff @ diff.T)

        return x_mixed, P_mixed, c_bar

    def update(self, z, dt):
        """步骤 2 & 3: 预测与更新"""

        # [修正] 在 update 内部根据传入的 dt 实时生成 F 和 Q
        # 这样即使 dt 变化，或者是 0.033 这种奇葩数值，都能保证物理模型正确
        models = []
        for sigma in self.sigmas:
            models.append({
                'F': self.get_F_matrix(dt),
                'Q': self.get_kinematic_Q(dt, sigma)
            })

        EPS = 1e-20
        x_mixed, P_mixed, c_bar = self.interact()

        likelihoods = np.zeros(self.M)

        for i in range(self.M):
            # --- 预测 ---
            F = models[i]['F']
            Q = models[i]['Q']

            x_pred = F @ x_mixed[i]
            P_pred = F @ P_mixed[i] @ F.T + Q

            # --- 更新 ---
            y = z - self.H @ x_pred  # 残差
            S = self.H @ P_pred @ self.H.T + self.R  # 新息协方差

            try:
                S_inv = np.linalg.inv(S)
                K = P_pred @ self.H.T @ S_inv
            except np.linalg.LinAlgError:
                K = np.zeros((6, 3))
                S_inv = np.eye(3)

            self.x[i] = x_pred + K @ y
            # Joseph form update for stability (optional but recommended)
            self.P[i] = (np.eye(6) - K @ self.H) @ P_pred @ (np.eye(6) - K @ self.H).T + K @ self.R @ K.T
            # self.P[i] = (np.eye(6) - K @ self.H) @ P_pred#todo

            # --- 计算似然 ---
            S_det = np.linalg.det(S)
            denom = np.sqrt((2 * np.pi) ** 3 * S_det)
            mahalanobis = -0.5 * y.T @ S_inv @ y
            likelihoods[i] = np.exp(mahalanobis) / (denom + EPS)

        # --- 步骤 4: 更新模型概率 ---
        new_probs = likelihoods * c_bar
        sum_probs = np.sum(new_probs)

        if sum_probs < EPS:
            self.model_probs = np.ones(self.M) / self.M
        else:
            self.model_probs = new_probs / sum_probs

        # --- 步骤 5: 融合状态 ---
        x_out = np.zeros(self.dim)
        for i in range(self.M):
            x_out += self.model_probs[i] * self.x[i]

        return x_out, self.model_probs