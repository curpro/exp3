import numpy as np


class IMMFilterEnhanced:
    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None):
        """
        增强版 IMM Filter (适配第五章在线优化 - Final Strict Version)
        """
        self.dim = 9
        self.M = 3  # 模型数量

        # 转移概率矩阵 (复制一份防止引用修改)
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
        self.q_params = [1.0, 50.0, 100.0]

        # 可调超参数（不改接口，外部可直接改 self.cv_tau / self.omega_ct）
        self.cv_tau = 1.0   # CV 模型中加速度衰减时间常数（秒），越小越接近“常速”
        self.omega_ct = 0.22  # CT 模型默认转弯角速度（rad/s）
        self._has_prediction = False

        # 测量矩阵 H
        self.H = np.zeros((3, self.dim))
        self.H[0, 0] = 1
        self.H[1, 3] = 1
        self.H[2, 6] = 1

        # 观测噪声 R
        if r_cov is not None:
            self.R = r_cov
        else:
            self.R = np.eye(3) * 225.0

        # 混合概率归一化常数
        self.c_bar = np.zeros(self.M)

        # 预分配临时变量
        self.x_pred = np.zeros_like(self.x)
        self.P_pred = np.zeros_like(self.P)

    # ================= 物理模型定义 =================
    def get_F_CV(self, dt):
        """近似 CV：使用 (x,v,a) 统一状态，但让加速度快速衰减到 0。
        这比“把 a 维度悬空”更自洽，也更利于 IMM 混合。
        """
        F = np.eye(self.dim)
        # 一阶 Markov 衰减：a_{k+1} = rho * a_k, rho in (0,1)
        tau = getattr(self, 'cv_tau', 1.0)
        if tau is None or tau <= 0:
            rho = 0.0
        else:
            rho = float(np.exp(-dt / tau))

        # 对每个轴的 3x3 block: [pos, vel, acc]
        block = np.array([
            [1.0, dt, 0.5 * dt ** 2],
            [0.0, 1.0, dt],
            [0.0, 0.0, rho],
        ])
        for i in [0, 3, 6]:
            F[i:i + 3, i:i + 3] = block
        return F

    def get_F_CA(self, dt):
        F = np.eye(self.dim)
        block = np.array([[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]])
        for i in [0, 3, 6]: F[i:i + 3, i:i + 3] = block
        return F

    def get_F_CT(self, dt, omega=0.22):
        """协调转弯 CT（增强版）：在 x-y 平面上对 (pos, vel, acc) 做一致的转弯离散化。
        状态顺序仍为 [x,vx,ax, y,vy,ay, z,vz,az]。

        连续时间近似：
            p' = v
            v' = Ω v + a
            a' = Ω a
        其中 Ω = ω * [[0,-1],[1,0]]。
        该模型的离散化有闭式形式，避免“a 维度悬空”。
        """
        F = np.eye(self.dim)
        w = float(omega)
        t = float(dt)

        # --- x-y 平面 6x6 block（按 [x,y,vx,vy,ax,ay] 排列） ---
        eps = 1e-8
        if abs(w) < eps:
            # ω -> 0 时退化为 CA（无旋转）
            ca3 = np.array([[1.0, t, 0.5 * t ** 2],
                            [0.0, 1.0, t],
                            [0.0, 0.0, 1.0]])
            block6 = np.zeros((6, 6))
            # x axis (x,vx,ax) corresponds to rows/cols [0,2,4] in [x,y,vx,vy,ax,ay]
            # but our ordering is [x,y,vx,vy,ax,ay], so x triple indices are [0,2,4]
            idx_x = [0, 2, 4]
            idx_y = [1, 3, 5]
            for r,c,blk in [(idx_x, idx_x, ca3), (idx_y, idx_y, ca3)]:
                block6[np.ix_(r, c)] = blk
        else:
            sw = np.sin(w * t)
            cw = np.cos(w * t)
            R = np.array([[cw, -sw],
                          [sw,  cw]])

            # I1 = ∫0^t R(s) ds
            I1 = (1.0 / w) * np.array([[sw, -(1.0 - cw)],
                                       [1.0 - cw,  sw]])

            # I2 = ∫0^t s R(s) ds
            A = (t * sw) / w + (cw - 1.0) / (w ** 2)     # ∫ s cos(ws) ds
            B = (-t * cw) / w + (sw) / (w ** 2)          # ∫ s sin(ws) ds
            I2 = np.array([[A, -B],
                           [B,  A]])

            Z = np.zeros((2, 2))
            I = np.eye(2)
            block6 = np.block([
                [I,  I1,  I2],
                [Z,  R,   t * R],
                [Z,  Z,   R],
            ])

        # 把 block6 写回 9x9：映射 [x,y,vx,vy,ax,ay] -> [0,3,1,4,2,5]
        map_idx = [0, 3, 1, 4, 2, 5]
        F[np.ix_(map_idx, map_idx)] = block6

        # --- z 轴：保持 CA ---
        ca3_z = np.array([[1.0, t, 0.5 * t ** 2],
                          [0.0, 1.0, t],
                          [0.0, 0.0, 1.0]])
        F[6:9, 6:9] = ca3_z
        return F

    def get_Q(self, dt, q_std, model_type='CA'):
        Q = np.zeros((self.dim, self.dim))
        var = q_std ** 2

        if model_type == 'CV':
            # CV 模型：噪声主要进入速度层 (Discrete White Noise Acceleration)
            # 对应状态顺序: x, vx, ax
            # Q_block 2x2 for pos/vel, leaving acc with tiny noise
            q_block = np.array([
                [dt ** 3 / 3, dt ** 2 / 2, 0],
                [dt ** 2 / 2, dt, 0],
                [0, 0, 1e-6]  # 极小的噪声防止奇异矩阵
            ]) * var
        else:
            # CA/CT 模型：噪声进入加速度层 (Discrete White Noise Jerk)
            q_block = np.array([
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt]
            ]) * var

        for i in [0, 3, 6]:
            Q[i:i + 3, i:i + 3] = q_block
        return Q

    def interact(self):
        self.c_bar = np.dot(self.trans_prob.T, self.model_probs)
        EPS = 1e-20

        mixing_probs = (self.trans_prob * self.model_probs[:, None]) / (self.c_bar + EPS)

        x_mixed = np.zeros_like(self.x)
        P_mixed = np.zeros_like(self.P)

        for j in range(self.M):
            for i in range(self.M):
                x_mixed[j] += mixing_probs[i, j] * self.x[i]
            for i in range(self.M):
                diff = (self.x[i] - x_mixed[j]).reshape(-1, 1)
                P_mixed[j] += mixing_probs[i, j] * (self.P[i] + diff @ diff.T)

        return x_mixed, P_mixed

    def predict(self, dt):
        x_mixed, P_mixed = self.interact()

        self.model_defs = [
            {'F': self.get_F_CV(dt), 'Q': self.get_Q(dt, self.q_params[0],model_type='CV')},
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
        if not getattr(self, '_has_prediction', False):
            self.predict(dt)

        log_likelihoods = np.zeros(self.M)

        for i in range(self.M):
            y_res = z - self.H @ self.x_pred[i]
            S = self.H @ self.P_pred[i] @ self.H.T + self.R + np.eye(3) * 1e-5  # Jitter

            try:
                PHt = self.P_pred[i] @ self.H.T
                K = np.linalg.solve(S, PHt.T).T
            except np.linalg.LinAlgError:
                K = np.zeros((self.dim, 3))

            self.x[i] = self.x_pred[i] + K @ y_res
            I_KH = np.eye(self.dim) - K @ self.H
            self.P[i] = I_KH @ self.P_pred[i] @ I_KH.T + K @ self.R @ K.T

            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0: logdet = 50.0
            try:
                mahalanobis = float(y_res.T @ np.linalg.solve(S, y_res))
            except:
                mahalanobis = 100.0
            log_likelihoods[i] = -0.5 * (3 * np.log(2 * np.pi) + logdet + mahalanobis)

        log_c_bar = np.log(self.c_bar + 1e-100)
        log_unnorm_probs = log_likelihoods + log_c_bar
        max_log = np.max(log_unnorm_probs)
        if np.isinf(max_log) or np.isnan(max_log): max_log = -1e9

        sum_exp = np.sum(np.exp(log_unnorm_probs - max_log))
        total_log_likelihood = max_log + np.log(sum_exp + 1e-100)

        unnorm_probs = np.exp(log_unnorm_probs - max_log)
        new_probs = unnorm_probs / (sum_exp + 1e-100)

        # 【核心修正】：防止模型概率坍塌 (Mode Probability Floor)
        # 给所有模型保留至少 0.1% 的概率，防止在切换机动时反应迟钝
        MIN_PROB = 0.001
        new_probs = np.maximum(new_probs, MIN_PROB)
        self.model_probs = new_probs / np.sum(new_probs)  # 重新归一化

        x_out = np.zeros(self.dim)
        for i in range(self.M):
            x_out += self.model_probs[i] * self.x[i]

        self._has_prediction = False
        return x_out, total_log_likelihood

    def get_state_snapshot(self):
        return {
            'x': self.x.copy(), 'P': self.P.copy(),
            'model_probs': self.model_probs.copy(),
            'trans_prob': self.trans_prob.copy(), 'c_bar': self.c_bar.copy()
        }

    def set_state_snapshot(self, snapshot):
        self.x[:] = snapshot['x']
        self.P[:] = snapshot['P']
        self.model_probs[:] = snapshot['model_probs']
        self.trans_prob[:] = snapshot['trans_prob']
        self.c_bar[:] = snapshot['c_bar']

    def set_transition_matrix(self, matrix):
        self.trans_prob[:] = matrix