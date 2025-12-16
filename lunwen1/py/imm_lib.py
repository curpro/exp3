import numpy as np


class IMMFilter:
    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None):
        """
        IMM Filter (F-16 轨迹跟踪优化版)
        基于 Enhanced 逻辑重构，包含精确 CT 模型与 CV 加速度衰减机制。
        """
        self.dim = 9
        self.M = 3  # 模型数量：CV, CA, CT

        # 转移概率矩阵 (复制一份防止外部修改影响)
        self.trans_prob = transition_probabilities.copy()

        # 初始模型概率
        self.model_probs = np.array([0.4, 0.4, 0.2])

        # 初始化状态和协方差
        self.x = np.zeros((self.M, self.dim))
        self.P = np.zeros((self.M, self.dim, self.dim))

        for i in range(self.M):
            self.x[i] = initial_state.copy()
            self.P[i] = initial_cov.copy()

        # --- Q 参数设置 ---
        # CV: 低噪声，依靠速度预测
        # CA: 中等噪声，允许加速度变化
        # CT: 高噪声，适应转弯不确定性
        self.q_params = [1.0, 50.0, 100.0]

        # 核心超参数
        self.cv_tau = 0.5  # CV 模型加速度衰减时间常数 (秒)，越小回归匀速越快
        self.omega_ct = 0.22  # CT 模型默认转弯角速度 (rad/s)

        # 测量矩阵 H (观测 x, y, z)
        self.H = np.zeros((3, self.dim))
        self.H[0, 0] = 1  # x
        self.H[1, 3] = 1  # y
        self.H[2, 6] = 1  # z

        # 观测噪声 R
        if r_cov is not None:
            self.R = r_cov
        else:
            self.R = np.eye(3) * 100.0

        # 内部状态标记
        self._has_prediction = False
        self.c_bar = np.zeros(self.M)
        self.x_pred = np.zeros_like(self.x)
        self.P_pred = np.zeros_like(self.P)

    # ================= 物理模型定义 (Enhanced Logic) =================

    def get_F_CV(self, dt):
        """
        优化版 CV：使用 (x,v,a) 统一状态，但让加速度快速衰减到 0。
        这解决了传统 CV 模型直接忽略加速度导致模型切换时的状态突变问题。
        """
        F = np.eye(self.dim)
        # 一阶 Markov 衰减：a_{k+1} = rho * a_k
        tau = getattr(self, 'cv_tau', 1.0)
        rho = float(np.exp(-dt / tau)) if tau > 0 else 0.0

        # 对每个轴的 3x3 block: [pos, vel, acc]
        block = np.array([
            [1.0, dt, 0.5 * dt ** 2],
            [0.0, 1.0, dt],
            [0.0, 0.0, rho],  # 加速度衰减
        ])
        for i in [0, 3, 6]:
            F[i:i + 3, i:i + 3] = block
        return F

    def get_F_CA(self, dt):
        """标准 CA 模型：牛顿运动学"""
        F = np.eye(self.dim)
        block = np.array([[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]])
        for i in [0, 3, 6]:
            F[i:i + 3, i:i + 3] = block
        return F

    def get_F_CT(self, dt, omega=0.22):
        """
        优化版 CT：精确离散化 (Exact Discretization)。
        在 x-y 平面上求解 v' = Ωv + a 的解析解，避免线性近似误差。
        """
        F = np.eye(self.dim)
        w = float(omega)
        t = float(dt)

        # --- x-y 平面 6x6 block ---
        eps = 1e-8
        if abs(w) < eps:
            # 如果转弯率极小，退化为 CA
            ca3 = np.array([[1.0, t, 0.5 * t ** 2],
                            [0.0, 1.0, t],
                            [0.0, 0.0, 1.0]])
            # 映射到 9x9 稍显复杂，这里简化逻辑，直接复用 CA 逻辑即可
            # 但为了严谨，我们仅在 w!=0 时计算旋转
            return self.get_F_CA(dt)
        else:
            sw = np.sin(w * t)
            cw = np.cos(w * t)
            R = np.array([[cw, -sw],
                          [sw, cw]])

            # 速度对位置的积分 I1 = ∫ R(s) ds
            I1 = (1.0 / w) * np.array([[sw, -(1.0 - cw)],
                                       [1.0 - cw, sw]])

            # 加速度对位置的二次积分 I2 = ∫ s R(s) ds
            A = (t * sw) / w + (cw - 1.0) / (w ** 2)
            B = (-t * cw) / w + (sw) / (w ** 2)
            I2 = np.array([[A, -B],
                           [B, A]])

            Z = np.zeros((2, 2))
            I = np.eye(2)

            # 构建 6x6 矩阵，顺序为 [pos_vec, vel_vec, acc_vec]
            # 这里的 block6 是基于向量 [x, y, vx, vy, ax, ay] 的排列
            block6 = np.block([
                [I, I1, I2],
                [Z, R, t * R],
                [Z, Z, R],
            ])

        # 将 block6 映射回 [x, vx, ax, y, vy, ay] 的 9x9 格式
        # 原始向量: [x, y, vx, vy, ax, ay] -> 索引 [0, 3, 1, 4, 2, 5]
        map_idx = [0, 3, 1, 4, 2, 5]
        F[np.ix_(map_idx, map_idx)] = block6

        # Z 轴保持 CA 模型
        ca3_z = np.array([[1.0, t, 0.5 * t ** 2],
                          [0.0, 1.0, t],
                          [0.0, 0.0, 1.0]])
        F[6:9, 6:9] = ca3_z
        return F

    def get_Q(self, dt, q_std, model_type='CA'):
        Q = np.zeros((self.dim, self.dim))
        var = q_std ** 2

        if model_type == 'CV':
            # CV 模型：噪声主要驱动速度 (Discrete White Noise Acceleration)
            # 加速度项给极小噪声防止奇异
            q_block = np.array([
                [dt ** 3 / 3, dt ** 2 / 2, 0],
                [dt ** 2 / 2, dt, 0],
                [0, 0, 1e-6]
            ]) * var
        else:
            # CA/CT 模型：噪声驱动加加速度 (Discrete White Noise Jerk)
            q_block = np.array([
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt]
            ]) * var

        for i in [0, 3, 6]:
            Q[i:i + 3, i:i + 3] = q_block
        return Q

    # ================= 滤波核心逻辑 =================

    def interact(self):
        """步骤 1: 交互 (Interaction) - 混合状态"""
        self.c_bar = np.dot(self.trans_prob.T, self.model_probs)
        EPS = 1e-20

        # 计算混合概率 mu_{i|j}
        mixing_probs = (self.trans_prob * self.model_probs[:, None]) / (self.c_bar + EPS)

        x_mixed = np.zeros_like(self.x)
        P_mixed = np.zeros_like(self.P)

        for j in range(self.M):
            # 状态加权平均
            for i in range(self.M):
                x_mixed[j] += mixing_probs[i, j] * self.x[i]
            # 协方差加权 (包含 Spread term)
            for i in range(self.M):
                diff = (self.x[i] - x_mixed[j]).reshape(-1, 1)
                P_mixed[j] += mixing_probs[i, j] * (self.P[i] + diff @ diff.T)

        return x_mixed, P_mixed

    def predict(self, dt):
        """步骤 2: 模型预测"""
        x_mixed, P_mixed = self.interact()

        # 定义当次迭代的模型参数
        self.model_defs = [
            {'F': self.get_F_CV(dt), 'Q': self.get_Q(dt, self.q_params[0], model_type='CV')},
            {'F': self.get_F_CA(dt), 'Q': self.get_Q(dt, self.q_params[1])},
            {'F': self.get_F_CT(dt, omega=self.omega_ct), 'Q': self.get_Q(dt, self.q_params[2], model_type='CT')}
        ]

        for i in range(self.M):
            F = self.model_defs[i]['F']
            Q = self.model_defs[i]['Q']

            self.x_pred[i] = F @ x_mixed[i]
            self.P_pred[i] = F @ P_mixed[i] @ F.T + Q

        self._has_prediction = True

    def update(self, z, dt):
        """步骤 3 & 4: 测量更新与概率更新"""

        # 兼容性设计：如果外部没调用 predict，这里自动调用
        if not self._has_prediction:
            self.predict(dt)

        log_likelihoods = np.zeros(self.M)

        for i in range(self.M):
            # 测量残差
            y_res = z - self.H @ self.x_pred[i]
            # 创新协方差 S
            S = self.H @ self.P_pred[i] @ self.H.T + self.R

            # 数值稳定处理：加微小 Jitter 防止 S 非正定
            S += np.eye(3) * 1e-6

            try:
                # 使用 solve 代替 inv 提高精度
                # K = P H^T S^-1
                PHt = self.P_pred[i] @ self.H.T
                K = np.linalg.solve(S, PHt.T).T
            except np.linalg.LinAlgError:
                # 极罕见情况：退化处理
                K = np.zeros((self.dim, 3))

            # 卡尔曼状态更新
            self.x[i] = self.x_pred[i] + K @ y_res

            # 卡尔曼协方差更新 (Joseph form 保证对称正定)
            I_KH = np.eye(self.dim) - K @ self.H
            self.P[i] = I_KH @ self.P_pred[i] @ I_KH.T + K @ self.R @ K.T

            # 计算对数似然
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0: logdet = 50.0  # 惩罚非正定矩阵

            try:
                mahalanobis = float(y_res.T @ np.linalg.solve(S, y_res))
            except:
                mahalanobis = 100.0

            log_likelihoods[i] = -0.5 * (3 * np.log(2 * np.pi) + logdet + mahalanobis)

        # --- 概率更新 (核心修正) ---
        log_c_bar = np.log(self.c_bar + 1e-100)
        log_unnorm_probs = log_likelihoods + log_c_bar

        # Log-Sum-Exp 技巧防止溢出
        max_log = np.max(log_unnorm_probs)
        if np.isinf(max_log) or np.isnan(max_log): max_log = -1e9

        sum_exp = np.sum(np.exp(log_unnorm_probs - max_log))
        # total_log_likelihood = max_log + np.log(sum_exp + 1e-100) # (可选：如果需要返回总似然)

        unnorm_probs = np.exp(log_unnorm_probs - max_log)
        new_probs = unnorm_probs / (sum_exp + 1e-100)

        # 【核心修正】：防止模型概率坍塌 (Mode Probability Floor)
        # 强制每个模型至少保留 0.1% 的概率
        MIN_PROB = 0.001
        new_probs = np.maximum(new_probs, MIN_PROB)
        self.model_probs = new_probs / np.sum(new_probs)  # 重新归一化

        # 融合输出
        x_out = np.zeros(self.dim)
        for i in range(self.M):
            x_out += self.model_probs[i] * self.x[i]

        # 重置标志位
        self._has_prediction = False

        return x_out, self.model_probs