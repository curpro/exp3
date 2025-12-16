import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize

device = torch.device("cpu")
dtype = torch.double


class OnlineBoOptimizer:
    def __init__(self, base_imm, dt):
        self.base_imm = base_imm
        self.dt = dt
        # 约束：p1+p2 <= 0.999
        self.constraints = [
            (torch.tensor([0, 1], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999),
            (torch.tensor([2, 3], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999),
            (torch.tensor([4, 5], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
        ]

    @staticmethod
    def construct_matrix_static(p_vec):
        """
        静态方法：供 Optimizer 和 Main 共同调用，确保逻辑唯一。
        """
        p11, p12, p21, p22, p31, p32 = p_vec

        def norm_row(p_a, p_b):
            if p_a < 0: p_a = 1e-6
            if p_b < 0: p_b = 1e-6
            s = p_a + p_b
            if s > 0.99999:
                scale = 0.99999 / s
                p_a *= scale
                p_b *= scale
                p_c = 1.0 - (p_a + p_b)
            else:
                p_c = 1.0 - s
            return p_a, p_b, p_c

        r1 = norm_row(p11, p12)
        r2 = norm_row(p21, p22)
        r3 = norm_row(p31, p32)
        return np.array([r1, r2, r3])

    def objective_function(self, params, obs_window, start_snapshot):
        self.base_imm.set_state_snapshot(start_snapshot)
        try:
            p = params.detach().cpu().numpy()
        except:
            p = params.cpu().numpy()

        new_matrix = self.construct_matrix_static(p)
        self.base_imm.set_transition_matrix(new_matrix)

        total_log_likelihood = 0.0
        steps = obs_window.shape[1]
        valid_steps = 0

        try:
            for k in range(steps):
                self.base_imm.predict(self.dt)
                _, ll = self.base_imm.update(obs_window[:, k], self.dt)
                if np.isnan(ll) or np.isinf(ll): return -1e9
                if ll < -1e5: ll = -1e5
                total_log_likelihood += ll
                valid_steps += 1
        except Exception:
            return -1e9

        if valid_steps == 0: return -1e9
        return total_log_likelihood / valid_steps

    def run_optimization(self, obs_window, start_snapshot, current_params, default_params, n_iter=5):
        # --- 1. 定义局部搜索范围 (Search Bounds) ---
        # 这是为了限制优化器只在当前参数附近寻找，防止突变
        delta = 0.15
        center = torch.tensor(current_params, dtype=dtype, device=device)
        anchor = torch.tensor(default_params, dtype=dtype, device=device)

        lb = torch.clamp(center - delta, 0.02, 0.99)
        ub = torch.clamp(center + delta, 0.02, 0.99)
        search_bounds = torch.stack([lb, ub])

        # --- 2. 定义全局归一化范围 (Normalization Bounds) ---
        # 这是为了告诉 GP 模型：参数的物理极限是 0 到 1
        # 这样无论 anchor 在哪里，归一化后都在 [0,1] 之间，消灭 Warning
        global_lb = torch.zeros(6, dtype=dtype, device=device)
        global_ub = torch.ones(6, dtype=dtype, device=device)
        norm_bounds = torch.stack([global_lb, global_ub])

        # 初始化样本：包含当前点和锚点(默认点)
        train_x = [center, anchor]

        # 随机采样补充 (在局部搜索范围内采样)
        max_sample_attempts = 100
        valid_count = 0
        for _ in range(max_sample_attempts):
            if valid_count >= 100: break
            rand_pt = torch.rand(6, dtype=dtype, device=device) * (ub - lb) + lb

            # 简单的行和约束检查
            c1 = rand_pt[0] + rand_pt[1]
            c2 = rand_pt[2] + rand_pt[3]
            c3 = rand_pt[4] + rand_pt[5]
            if (c1 < 0.999) and (c2 < 0.999) and (c3 < 0.999):
                train_x.append(rand_pt)
                valid_count += 1

        train_x = torch.stack(train_x)

        # 计算目标函数值
        train_y = []
        for x in train_x:
            val = self.objective_function(x, obs_window, start_snapshot)
            train_y.append(val)

        train_y_tensor = torch.tensor(train_y, dtype=dtype, device=device).unsqueeze(-1)
        train_y_tensor = torch.nan_to_num(train_y_tensor, nan=-1e9)

        # 如果所有样本结果都一样(方差为0)，GP无法拟合，直接返回
        if train_y_tensor.std() < 1e-6:
            return current_params

        # --- 3. BO 迭代 ---
        for i in range(n_iter):
            try:
                # 【关键修正】：使用 norm_bounds (0~1) 进行归一化，而不是局部的 search_bounds
                model = SingleTaskGP(train_x, train_y_tensor,
                                     input_transform=Normalize(d=6, bounds=norm_bounds),
                                     outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                UCB = UpperConfidenceBound(model, beta=2.0)

                # 【关键保持】：搜索依然限制在局部 search_bounds 内
                candidate, _ = optimize_acqf(
                    UCB, bounds=search_bounds, q=1, num_restarts=3, raw_samples=100,
                    inequality_constraints=self.constraints
                )

                new_val = self.objective_function(candidate[0], obs_window, start_snapshot)
                train_x = torch.cat([train_x, candidate])
                train_y_tensor = torch.cat([train_y_tensor, torch.tensor([[new_val]], dtype=dtype, device=device)])
            except Exception as e:
                # print(f"Optimization step failed: {e}") # 调试时可开启
                continue

        self.base_imm.set_state_snapshot(start_snapshot)
        best_idx = train_y_tensor.argmax()

        if train_y_tensor[best_idx] < -1e5:
            # print("Warning: Optimization failed or diverged. Resetting to Default Params.")
            return default_params

        return train_x[best_idx].cpu().numpy()