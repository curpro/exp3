import os
import time
import warnings
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# BoTorch 依赖
from botorch import fit_gpytorch_model, fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
# [关键修复] 引入归一化和标准化工具，解决 Gradient NaN 问题
from botorch.models.transforms import Standardize, Normalize

# 导入本地的 IMM 算法模块
# 假设 imm_lib.py 与此文件在同一目录
from helper_data import generate_truth_data
from imm_lib import IMMFilter

# 设置环境和警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# 绘图设置
matplotlib_font = 'SimHei'  # Windows请用 SimHei
plt.rcParams['font.sans-serif'] = [matplotlib_font]
plt.rcParams['axes.unicode_minus'] = False

# 颜色定义
GREEN = '\033[32m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
CYAN = '\033[36m'
RED = '\033[37m'  # Random 用的颜色
RESET = '\033[0m'

# ==========================================
# 1. 全局配置与数据准备
# ==========================================
# 设备配置
device = torch.device("cpu")
dtype = torch.double

# 优化参数
N_INIT = 20       # 初始随机样本点数量
N_BATCH = 50    # 迭代次数
BATCH_SIZE = 1    # 每次推荐点数
MC_SAMPLES = 500  # 蒙特卡洛采样数

# 参数范围 [a, b, c, d]
lower_bounds = torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=dtype, device=device)
upper_bounds = torch.tensor([0.99, 0.99, 0.99, 0.99], dtype=dtype, device=device)
bounds = torch.stack([lower_bounds, upper_bounds])

# ------------------------------------------
# 生成固定的基准数据
# ------------------------------------------
print("正在生成基准轨迹数据...")
np.random.seed(2018)
truth_state_full, time_vec = generate_truth_data()

# 自动计算 dt (对应您说的 0.1s)
dt = time_vec[1] - time_vec[0]
num_steps = len(time_vec)
print(f"检测到数据采样间隔 dt = {dt:.4f} s")

true_pos = truth_state_full[[0, 2, 4], :]
true_vel = truth_state_full[[1, 3, 5], :]

# 【修改1】定义观测噪声标准差
MEAS_NOISE_STD = 4.0

# 生成带噪声的观测
meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
meas_pos = true_pos + meas_noise

# 初始化用的观测
meas_noise_b = np.random.randn(6, num_steps) * MEAS_NOISE_STD
meas_pos_b = truth_state_full + meas_noise_b
init_state = meas_pos_b[:, 0]

# 初始协方差
init_cov = np.diag([1, 1e4, 1, 1e4, 1, 1e4])

# 【修改2】生成 R 矩阵，适配新版 IMMFilter
r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

GLOBAL_DATA = {
    'meas_pos': meas_pos,
    'true_pos': true_pos,
    'true_vel': true_vel,
    'dt': dt,
    'init_state': init_state,
    'init_cov': init_cov,
    'r_cov': r_cov  # 将 R 存入全局数据
}


# ==========================================
# 2. 核心评估函数
# ==========================================
def run_imm_and_get_score(params):
    """ 输入 params=[a,b,c,d], 输出 -RMSE """
    a, b, c, d = params

    # 计算剩余概率
    p13 = 1 - a - b
    p23 = 1 - c - d
    p31 = 1 - a - c
    p32 = 1 - b - d
    p33 = a + b + c + d - 1

    # 物理约束检查
    if p13 < 0 or p23 < 0 or p31 < 0 or p32 < 0 or p33 < 0:
        return -200.0

    trans_matrix = np.array([
        [a, b, p13],
        [c, d, p23],
        [p31, p32, p33]
    ])

    try:
        # 【修改3】初始化 IMMFilter 时传入 r_cov
        imm = IMMFilter(
            transition_probabilities=trans_matrix,
            initial_state=GLOBAL_DATA['init_state'],
            initial_cov=GLOBAL_DATA['init_cov'],
            r_cov=GLOBAL_DATA['r_cov']  # 传入噪声矩阵
        )
    except Exception as e:
        return -200.0

    est_pos = np.zeros((6, len(time_vec)))
    est_pos[:, 0] = GLOBAL_DATA['init_state']
    meas = GLOBAL_DATA['meas_pos']
    dt_val = GLOBAL_DATA['dt']

    for i in range(1, len(time_vec)):
        z = meas[:, i]
        # 调用 update，传入 z 和 dt (新版 IMMFilter 需要 dt 来计算 F 和 Q)
        est, _ = imm.update(z, dt_val)
        est_pos[:, i] = est

    err_pos = est_pos[[0, 2, 4], :] - GLOBAL_DATA['true_pos']
    dist_err = np.sqrt(np.sum(err_pos ** 2, axis=0))
    rmse_pos = np.sqrt(np.mean(dist_err ** 2)) * 1.7  # todo 系数

    return -rmse_pos


def evaluate_y_batch(X_tensor):
    results = []
    for i in range(X_tensor.shape[0]):
        params = X_tensor[i].cpu().numpy()
        score = run_imm_and_get_score(params)
        results.append([score])
    return torch.tensor(results, device=device, dtype=dtype)


# ==========================================
# 3. 约束条件定义
# ==========================================
constraint_row1 = (torch.tensor([0, 1], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
constraint_row2 = (torch.tensor([2, 3], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
constraint_row3 = (torch.tensor([0, 2], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
constraint_row4 = (torch.tensor([1, 3], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
constraint_row5 = (torch.tensor([0, 1, 2, 3], device=device), torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=dtype, device=device), -1.999)
constraint_sum = (
torch.tensor([0, 1, 2, 3], device=device), torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device), 1.001)
constraints_list = [constraint_row1, constraint_row2, constraint_row3,constraint_row4,constraint_row5,constraint_sum]


# ==========================================
# 4. 辅助功能函数
# ==========================================
def initialize_model(train_x, train_y, state_dict=None):
    # [关键修复] 加入 input_transform 和 outcome_transform
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=train_y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def generate_valid_random_params(n=1):
    """生成满足概率约束的随机参数 (用于 Random Search 对照组)"""
    valid_points = []
    while len(valid_points) < n:
        proposal = torch.rand(4, dtype=dtype, device=device) * (upper_bounds - lower_bounds) + lower_bounds
        a, b, c, d = proposal.cpu().numpy()
        p13 = 1 - a - b
        p23 = 1 - c - d
        p31 = 1 - a - c
        p32 = 1 - b - d
        p33 = a + b + c + d - 1
        if (p13 > 0.001) and (p23 > 0.001) and \
                (p31 > 0.001) and (p32 > 0.001) and \
                (p33 > 0.001):
            valid_points.append(proposal)
    return torch.stack(valid_points)


# ==========================================
# 5. 主流程
# ==========================================
def main():
    print(f"{GREEN}=== 开始 Python IMM 贝叶斯优化 (含随机基准) ==={RESET}")
    print(f"初始样本: {N_INIT}, 迭代: {N_BATCH}")

    # 1. 初始化数据 (Warm-up)
    print(f"正在生成 {N_INIT} 个初始随机样本...")
    train_x = generate_valid_random_params(N_INIT)
    train_y = evaluate_y_batch(train_x)

    # 复制给四种策略 (EI, qEI, UCB, Random)
    train_x_ei, train_y_ei = train_x.clone(), train_y.clone()
    train_x_qei, train_y_qei = train_x.clone(), train_y.clone()
    train_x_ucb, train_y_ucb = train_x.clone(), train_y.clone()
    train_x_rnd, train_y_rnd = train_x.clone(), train_y.clone()  # [新增] Random策略数据

    # 记录最佳值轨迹
    best_y_ei = [train_y_ei.max().item()]
    best_y_qei = [train_y_qei.max().item()]
    best_y_ucb = [train_y_ucb.max().item()]
    best_y_rnd = [train_y_rnd.max().item()] # [新增]

    print(f"初始最佳 RMSE: {-train_y.max().item():.4f} m")

    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # 2. 迭代优化循环
    for i in range(N_BATCH):
        t0 = time.time()

        # --- A. 更新模型 (Random 不需要模型) ---
        mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)
        mll_qei, model_qei = initialize_model(train_x_qei, train_y_qei)
        mll_ucb, model_ucb = initialize_model(train_x_ucb, train_y_ucb)

        fit_gpytorch_mll(mll_ei)
        fit_gpytorch_mll(mll_qei)
        fit_gpytorch_mll(mll_ucb)

        # --- B. 定义采集函数 ---
        EI = ExpectedImprovement(model=model_ei, best_f=train_y_ei.max())
        qEI = qExpectedImprovement(model=model_qei, best_f=train_y_qei.max(), sampler=qmc_sampler)
        UCB = UpperConfidenceBound(model=model_ucb, beta=3.0)

        # --- C. 获取推荐点 ---
        def get_next_point(acq_f):
            candidates, _ = optimize_acqf(
                acq_function=acq_f,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=10,
                raw_samples=128,
                inequality_constraints=constraints_list
            )
            new_x = candidates.detach()
            new_y = evaluate_y_batch(new_x)
            return new_x, new_y

        # AI 策略推荐
        new_x_ei, new_y_ei = get_next_point(EI)
        new_x_qei, new_y_qei = get_next_point(qEI)
        new_x_ucb, new_y_ucb = get_next_point(UCB)

        # [新增] Random 策略推荐 (纯随机生成)
        new_x_rnd = generate_valid_random_params(BATCH_SIZE)
        new_y_rnd = evaluate_y_batch(new_x_rnd)

        # --- D. 更新数据集 ---
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_y_ei = torch.cat([train_y_ei, new_y_ei])

        train_x_qei = torch.cat([train_x_qei, new_x_qei])
        train_y_qei = torch.cat([train_y_qei, new_y_qei])

        train_x_ucb = torch.cat([train_x_ucb, new_x_ucb])
        train_y_ucb = torch.cat([train_y_ucb, new_y_ucb])

        train_x_rnd = torch.cat([train_x_rnd, new_x_rnd]) # [新增]
        train_y_rnd = torch.cat([train_y_rnd, new_y_rnd])

        # --- E. 记录当前最佳值 ---
        current_best_ei = train_y_ei.max().item()
        current_best_qei = train_y_qei.max().item()
        current_best_ucb = train_y_ucb.max().item()
        current_best_rnd = train_y_rnd.max().item() # [新增]

        best_y_ei.append(current_best_ei)
        best_y_qei.append(current_best_qei)
        best_y_ucb.append(current_best_ucb)
        best_y_rnd.append(current_best_rnd)

        t1 = time.time()
        print(f"\nBatch {i + 1}/{N_BATCH} | Time: {t1 - t0:.2f}s")
        print(f"{BLUE}[EI]  RMSE: {-new_y_ei.item():.4f} | Best: {-current_best_ei:.4f}{RESET}")
        print(f"{BLUE}[qEI] RMSE: {-new_y_qei.item():.4f} | Best: {-current_best_qei:.4f}{RESET}")
        print(f"{CYAN}[UCB] RMSE: {-new_y_ucb.item():.4f} | Best: {-current_best_ucb:.4f}{RESET}")
        print(f"{RED}[Rand] RMSE: {-new_y_rnd.item():.4f} | Best: {-current_best_rnd:.4f}{RESET}")

    # ==========================================
    # 6. 结果展示
    # ==========================================
    # 转换回正的 RMSE
    trace_ei = [-x for x in best_y_ei]
    trace_qei = [-x for x in best_y_qei]
    trace_ucb = [-x for x in best_y_ucb]
    trace_rnd = [-x for x in best_y_rnd] # [新增]

    # 找到最佳参数 (以 qEI 为例，通常效果最好)
    idx_best = train_y_qei.argmax()
    best_params = train_x_qei[idx_best].cpu().numpy()
    best_rmse = -train_y_qei[idx_best].item()

    print(f"\n{GREEN}=== 优化完成 ==={RESET}")
    print(f"qEI 最佳参数 [a, b, c, d]: {best_params.round(6)}")
    print(f"qEI 最佳 RMSE: {best_rmse:.6f} m")
    print(f"Random 最佳 RMSE: {min(trace_rnd):.6f} m")

    # 绘图
    iters = np.arange(len(trace_ei))
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(iters, trace_ei, linewidth=2, label="EI")
    plt.plot(iters, trace_qei, linewidth=2, label="qEI")
    plt.plot(iters, trace_ucb, linewidth=2, label="UCB")
    plt.plot(iters, trace_rnd, linewidth=2, color='red', label="Random")

    plt.title('贝叶斯优化 vs 随机搜索')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳 RMSE (m)')
    plt.legend()

    save_path = f'BO_IMM_Comparison_{datetime.now().strftime("%H%M%S")}.png'
    plt.savefig(save_path, dpi=300)
    print(f"结果图已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()