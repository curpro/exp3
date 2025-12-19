import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt  # [新增] 用于画图
from collections import deque
from scipy.signal import savgol_filter
import lunwen1.chapter5.network.paper_plotting as pp
import time

# 请确保路径与您项目结构一致
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced

# ================= 配置 =================
# [修改] 指向您的 F16 测试文件路径
TEST_DATA_PATH = r'D:\AFS\lunwen\dataSet\test_data\f16_super_maneuver_a.csv'

MODEL_PATH = '../net/n/3_MLP/imm_param_net.pth'
SCALER_PATH = '../net/n/3_MLP/scaler_params.json'
WINDOW_SIZE = 90
DT = 1 / 30  # 假设采样率为 30Hz，如果CSV里有时间戳，最好通过时间戳计算
OPTIMIZE_INTERVAL = 20
SAVGOL_WINDOW = 25 # [新增] 与训练一致
SAVGOL_POLY = 2


# === [核心修改] 模型定义必须与训练代码完全一致 ===
class ParamPredictorMLP(nn.Module):
    def __init__(self, seq_len=90, input_dim=9):
        super(ParamPredictorMLP, self).__init__()
        # 输入维度 = 时间步长 * 特征数 (例如 90 * 9 = 810)
        self.input_flat_dim = seq_len * input_dim
        # 定义 MLP 网络结构：输入 -> 隐层 -> 输出
        # 这里设计了一个 3 层网络 (810 -> 512 -> 128 -> 9)
        self.net = nn.Sequential(
            nn.Linear(self.input_flat_dim, 128),
            nn.BatchNorm1d(128),  # 加速收敛，防止过拟合
            nn.ReLU(),
            nn.Dropout(0.4),  # 防止过拟合
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 9)  # 输出 9 个值，对应 3x3 矩阵
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, 90, 9)
        b, s, f = x.shape
        # [关键] 将时间序列展平：(batch, 810)
        x = x.reshape(b, -1)
        logits = self.net(x)
        # 后续处理保持不变，与原代码兼容
        logits = logits.view(-1, 3, 3)
        temperature = 2.0
        return torch.log_softmax(logits / temperature, dim=2)


# ================= 特征提取函数 (与 Step1 完全一致) =================
def calculate_derivatives(pos_data, dt):
    """
    [修改] 使用 scipy.signal.savgol_filter 计算速度和加速度。
    逻辑完全复用 step1_generate_data.py 中的代码。
    """
    # 如果数据太短无法滤波，回退到原来的逻辑（防止报错）
    if len(pos_data) < SAVGOL_WINDOW:
        vel = np.zeros_like(pos_data)
        vel[1:] = (pos_data[1:] - pos_data[:-1]) / dt
        vel[0] = vel[1]

        acc = np.zeros_like(pos_data)
        acc[1:] = (vel[1:] - vel[:-1]) / dt
        acc[0] = acc[1]
        return vel, acc

    # 使用 scipy 的 savgol_filter
    # deriv=1 算一阶导(速度), deriv=2 算二阶导(加速度)
    vel = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY,
                        deriv=1, delta=dt, axis=0)
    acc = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY,
                        deriv=2, delta=dt, axis=0)

    return vel, acc

def load_test_data(filepath):
    """
        读取测试数据，同时提取位置真值和速度真值（如果有的话）。
        """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到测试文件: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # 1. 必须要有位置
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError(f"CSV 文件缺少必要的位置列 x, y, z")

    pos_gt = df[['x', 'y', 'z']].values

    # 2. 尝试读取速度真值 (vx, vy, vz)
    # 假设 CSV 列名是 vx, vy, vz，如果是 v_x 等请自行修改
    vel_cols = ['vx', 'vy', 'vz']
    if all(col in df.columns for col in vel_cols):
        print("  > 成功读取 CSV 中的真值速度 (用于评估)")
        vel_gt = df[vel_cols].values
    else:
        print("  > CSV 中未找到速度列，将使用差分计算替代 (评估精度受限)")
        vel_gt = None

    return pos_gt, vel_gt


def main_inference():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f">>> 随机种子已固定为: {SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载测试数据 (F16)
    try:
        gt_pos_data, gt_vel_data = load_test_data(TEST_DATA_PATH)
        sim_steps = len(gt_pos_data)
        print(f">>> 已加载测试集: {os.path.basename(TEST_DATA_PATH)}, 共 {sim_steps} 帧")
    except Exception as e:
        print(f"读取数据出错: {e}")
        return

    meas_noise_std = 15  # 对齐 BoIMM
    noise_matrix = np.random.randn(3, sim_steps) * meas_noise_std
    _ = np.random.randn(9)

    # 1. 加载模型与参数
    try:
        model = ParamPredictorMLP(seq_len=WINDOW_SIZE, input_dim=9).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        with open(SCALER_PATH, 'r') as f:
            scaler = json.load(f)
        mean = np.array(scaler['mean'], dtype=np.float32)
        std = np.array(scaler['std'], dtype=np.float32)
        print(">>> 模型与参数加载完毕。")
    except Exception as e:
        print(f"加载资源失败: {e}")
        return



    # 3. 初始化 IMM
    initial_trans_prob = np.array([[0.81388511, 0.18511489, 0.001], [0.989, 0.01, 0.001], [0.01, 0.01, 0.98]])
    # initial_trans_prob = np.array([[0.989, 0.01, 0.001], [0.08635608, 0.91264392, 0.001], [0.26990839, 0.01, 0.72009161]])

    initial_state = np.zeros(9)
    # [修改] 使用 CSV 第一帧作为初始位置
    initial_state[0] = gt_pos_data[0, 0]  # x
    initial_state[3] = gt_pos_data[0, 1]  # y
    initial_state[6] = gt_pos_data[0, 2]  # z
    # 可以简单估算初始速度，或者设为0
    if sim_steps > 1:
        initial_state[1] = gt_vel_data[0, 0]  # vx
        initial_state[4] = gt_vel_data[0, 1]  # vy
        initial_state[7] = gt_vel_data[0, 2]  # vz

    init_cov_diag = np.zeros(9)
    init_cov_diag[[0, 3, 6]] = 100.0  # Pos
    init_cov_diag[[1, 4, 7]] = 25.0  # Vel
    init_cov_diag[[2, 5, 8]] = 10.0  # Acc
    initial_cov = np.diag(init_cov_diag)


    r_cov = np.eye(3) * (meas_noise_std ** 2)

    # === [修改] 同时实例化两个滤波器 ===
    # 1. 自适应 IMM (NN 控制)
    imm_adaptive = IMMFilterEnhanced(initial_trans_prob, initial_state, initial_cov, r_cov=r_cov)

    # 2. 固定 IMM (对照组，永远不更新矩阵)
    imm_fixed = IMMFilterEnhanced(initial_trans_prob, initial_state, initial_cov, r_cov=r_cov)

    pos_buffer = deque(maxlen=WINDOW_SIZE)
    last_pred_params = None
    alpha_smooth = 0.9

    # 用于绘图记录
    history_true_pos = []

    history_est_pos_adaptive = []
    history_est_vel_adaptive = []

    history_est_pos_fixed = []
    history_est_vel_fixed = []

    history_probs = []
    history_tpm_params = []

    print(">>> 开始在线仿真...")

    nn_inference_times = []

    # 4. 仿真循环
    for k in range(sim_steps):
        # --- (A) 获取当前帧数据 ---
        true_pos = gt_pos_data[k]

        # 模拟观测值：真值 + 噪声
        # 如果您的CSV本身就是观测数据（含噪），则不需要加 noise
        # 这里假设 CSV 是真值，所以手动加噪声模拟雷达观测
        noise = noise_matrix[:, k]
        z_k = true_pos + noise

        if len(pos_buffer) == WINDOW_SIZE and k % OPTIMIZE_INTERVAL == 0:
            pos_seq = np.array(pos_buffer)

            ref_point = pos_seq[-1]
            rel_pos_seq = pos_seq - ref_point

            vel_seq, acc_seq = calculate_derivatives(pos_seq, DT)

            raw_features = np.hstack([rel_pos_seq, vel_seq, acc_seq])
            norm_features = (raw_features - mean) / std

            input_tensor = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                log_probs = model(input_tensor)  # 输出 (1, 3, 3)
                pred_params = torch.exp(log_probs).cpu().numpy()[0]

            t_cost = (time.perf_counter() - t0) * 1000
            nn_inference_times.append(t_cost)
            if last_pred_params is not None:
                pred_params = alpha_smooth * pred_params + (1 - alpha_smooth) * last_pred_params
            last_pred_params = pred_params

            new_matrix = np.clip(pred_params, 1e-6, 1.0)
            row_sums = np.sum(new_matrix, axis=1, keepdims=True)
            new_matrix = new_matrix / row_sums
            imm_adaptive.set_transition_matrix(new_matrix)

        est_state_adapt, _ = imm_adaptive.update(z_k, DT)
        est_state_fixed, _ = imm_fixed.update(z_k, DT)
        est_pos = est_state_adapt[[0, 3, 6]]

        pos_buffer.append(z_k)

        history_true_pos.append(true_pos)

        history_est_pos_adaptive.append(est_state_adapt[[0, 3, 6]])
        history_est_vel_adaptive.append(est_state_adapt[[1, 4, 7]])

        history_est_pos_fixed.append(est_state_fixed[[0, 3, 6]])
        history_est_vel_fixed.append(est_state_fixed[[1, 4, 7]])

        history_probs.append(imm_adaptive.model_probs.copy())

        curr_P = imm_adaptive.trans_prob
        # 顺序对应: P11(CV->CV), P12(CV->CA), P21(CA->CV), P22(CA->CA), P31(CT->CV), P32(CT->CA)
        history_tpm_params.append([
            curr_P[0, 0], curr_P[0, 1],
            curr_P[1, 0], curr_P[1, 1],
            curr_P[2, 0], curr_P[2, 1]
        ])

        if k % 50 == 0:
            err = np.linalg.norm(est_pos - true_pos)
            print(f"Frame {k:03d}: Pos Error {err:.2f}m | Model Probs: {np.round(imm_adaptive.model_probs, 2)}")

    # 5. 结果可视化 [新增]
    hist_true_pos = np.array(history_true_pos)
    hist_est_pos_adapt = np.array(history_est_pos_adaptive)
    hist_est_vel_adapt = np.array(history_est_vel_adaptive)
    hist_est_pos_fix = np.array(history_est_pos_fixed)
    hist_est_vel_fix = np.array(history_est_vel_fixed)

    # 计算 RMSE (跳过前 WINDOW_SIZE 帧，避开初始化震荡)
    eval_start_idx = WINDOW_SIZE

    # 位置误差序列
    err_pos_fix_seq = np.linalg.norm(hist_est_pos_fix - hist_true_pos, axis=1)
    err_pos_adapt_seq = np.linalg.norm(hist_est_pos_adapt - hist_true_pos, axis=1)

    # 速度误差序列 (对比计算出的真值速度)
    if gt_vel_data is not None:
        # 使用 CSV 真值计算误差
        err_vel_fix_seq = np.linalg.norm(hist_est_vel_fix - gt_vel_data, axis=1)
        err_vel_adapt_seq = np.linalg.norm(hist_est_vel_adapt - gt_vel_data, axis=1)
    else:
        # 回退到使用计算出的导数 (原逻辑)
        # 注意：这里要重新对 gt_pos_data 算一遍导数作为参考
        print("警告：使用计算导数作为真值参考")
        gt_vel_calc, _ = calculate_derivatives(gt_pos_data, DT)
        err_vel_fix_seq = np.linalg.norm(hist_est_vel_fix - gt_vel_calc, axis=1)
        err_vel_adapt_seq = np.linalg.norm(hist_est_vel_adapt - gt_vel_calc, axis=1)

    # 均值 RMSE
    rmse_pos_fix = np.sqrt(np.mean(err_pos_fix_seq[eval_start_idx:] ** 2))
    rmse_pos_adapt = np.sqrt(np.mean(err_pos_adapt_seq[eval_start_idx:] ** 2))
    rmse_vel_fix = np.sqrt(np.mean(err_vel_fix_seq[eval_start_idx:] ** 2))
    rmse_vel_adapt = np.sqrt(np.mean(err_vel_adapt_seq[eval_start_idx:] ** 2))

    # [新增] Variance (方差)
    var_pos_fix = np.var(err_pos_fix_seq[eval_start_idx:])
    var_pos_adapt = np.var(err_pos_adapt_seq[eval_start_idx:])
    var_vel_fix = np.var(err_vel_fix_seq[eval_start_idx:])
    var_vel_adapt = np.var(err_vel_adapt_seq[eval_start_idx:])

    max_pos_fix = np.max(err_pos_fix_seq[eval_start_idx:])
    max_pos_adapt = np.max(err_pos_adapt_seq[eval_start_idx:])
    max_vel_fix = np.max(err_vel_fix_seq[eval_start_idx:])
    max_vel_adapt = np.max(err_vel_adapt_seq[eval_start_idx:])

    # 提升率计算
    pos_improv = (rmse_pos_fix - rmse_pos_adapt) / rmse_pos_fix * 100
    vel_improv = (rmse_vel_fix - rmse_vel_adapt) / rmse_vel_fix * 100

    # === 打印结果表格 (包含 Max Error) ===
    print("\n" + "=" * 90)
    print(f"     仿真结果性能对比 (File: {os.path.basename(TEST_DATA_PATH)})")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Fixed IMM (RMSE/Max)':<25} | {'NN-IMM (RMSE/Max)':<25} | {'Improv(RMSE)':<11}")
    print("-" * 90)
    print(
        f"{'Position':<10} | {rmse_pos_fix:<6.4f} / {max_pos_fix:<6.4f} {'(m)':<3} | {rmse_pos_adapt:<6.4f} / {max_pos_adapt:<6.4f} {'(m)':<3} | {pos_improv:>8.2f}%")
    print(
        f"{'Velocity':<10} | {rmse_vel_fix:<6.4f} / {max_vel_fix:<6.4f} {'(m/s)':<3}| {rmse_vel_adapt:<6.4f} / {max_vel_adapt:<6.4f} {'(m/s)':<3}| {vel_improv:>8.2f}%")
    print("=" * 90 + "\n")

    plot_slice = slice(WINDOW_SIZE, None)
    plot_x = np.arange(WINDOW_SIZE, sim_steps)

    # plt.figure(figsize=(12, 14))

    # # 子图1: 3D 轨迹对比
    # fig1 = plt.figure(figsize=(10, 8))
    # ax1 = fig1.add_subplot(111, projection='3d')
    #
    # ax1.plot(hist_true_pos[:, 0], hist_true_pos[:, 1], hist_true_pos[:, 2],
    #          'k-', linewidth=1.5, label='True', alpha=0.6)
    #
    # # 2. 画固定 IMM (Fixed) - 蓝色虚线
    # ax1.plot(hist_est_pos_fix[:, 0], hist_est_pos_fix[:, 1], hist_est_pos_fix[:, 2],
    #          'b--', linewidth=1, label='Fixed IMM', alpha=0.5)
    #
    # # 3. 画 NN-IMM (Ours) - 红色实线
    # ax1.plot(hist_est_pos_adapt[:, 0], hist_est_pos_adapt[:, 1], hist_est_pos_adapt[:, 2],
    #          'r-', linewidth=1.5, label='NN-IMM (Ours)')
    #
    # # 4. 标记起点和终点
    # ax1.scatter(hist_true_pos[0, 0], hist_true_pos[0, 1], hist_true_pos[0, 2],
    #             c='g', marker='o', s=50, label='Start')
    # ax1.scatter(hist_true_pos[-1, 0], hist_true_pos[-1, 1], hist_true_pos[-1, 2],
    #             c='m', marker='x', s=80, label='End')
    #
    # ax1.set_title(f"3D Trajectory Tracking Comparison\n({os.path.basename(TEST_DATA_PATH)})")
    # ax1.set_xlabel("X (m)")
    # ax1.set_ylabel("Y (m)")
    # ax1.set_zlabel("Z (m)")
    # ax1.legend(loc='best')
    # ax1.grid(True)

    # 子图2: 模型概率变化 (纯净版，无转弯率背景)
    plt.figure(figsize=(12, 5))
    hist_probs = np.array(history_probs)
    plt.plot(hist_probs[:, 0], 'g-', label='CV Prob', alpha=0.6, linewidth=1.5)
    plt.plot(hist_probs[:, 1], 'b-', label='CA Prob', alpha=0.6, linewidth=1.5)
    plt.plot(hist_probs[:, 2], 'r-', label='CT Prob', linewidth=2.0)  # 红色加粗 CT

    plt.ylabel("Probability")
    plt.xlabel("Frame")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("NN-IMM Model Probabilities Evolution")
    plt.tight_layout()


    # 子图3: 误差对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(plot_x, err_pos_fix_seq[plot_slice], 'b--', linewidth=1, label='Fixed IMM Error', alpha=0.6)
    plt.plot(plot_x, err_pos_adapt_seq[plot_slice], 'r-', linewidth=1.5, label='NN-IMM Error')

    # 标记 NN 更新时刻 #todo 注释掉了
    # for t in range(WINDOW_SIZE, sim_steps, OPTIMIZE_INTERVAL):
    #     plt.axvline(x=t, color='m', alpha=0.1)

    plt.title(f"Position Estimation Error Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Error (m)")
    plt.xlim([WINDOW_SIZE, sim_steps])  # 锁定 X 轴范围
    plt.legend()
    plt.grid(True)

    # [新增] 子图4: 速度误差对比
    plt.subplot(2, 1, 2)
    plt.plot(plot_x, err_vel_fix_seq[plot_slice], 'b--', linewidth=1, label='Fixed IMM Vel Error', alpha=0.6)
    plt.plot(plot_x, err_vel_adapt_seq[plot_slice], 'r-', linewidth=1.5, label='NN-IMM Vel Error')
    # 标记 NN 更新时刻
    # for t in range(WINDOW_SIZE, sim_steps, OPTIMIZE_INTERVAL):
    #     plt.axvline(x=t, color='m', alpha=0.1)
    plt.title(f"Velocity Estimation Error (Improv: {vel_improv:.2f}%)")
    plt.xlabel("Frame")
    plt.ylabel("Vel Error (m/s)")
    plt.xlim([WINDOW_SIZE, sim_steps])  # 锁定 X 轴范围
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # [新增 3] 绘制转移矩阵参数变化图 (6个子图)
    hist_params = np.array(history_tpm_params)
    t_axis = np.arange(sim_steps) * DT  # 如果你想用时间轴，或者直接用 frame 也可以

    param_labels = [
        r'$P_{11}$ (CV$\to$CV)', r'$P_{12}$ (CV$\to$CA)',
        r'$P_{21}$ (CA$\to$CV)', r'$P_{22}$ (CA$\to$CA)',
        r'$P_{31}$ (CT$\to$CV)', r'$P_{32}$ (CT$\to$CA)'
    ]

    plt.figure(figsize=(12, 10))
    plt.suptitle(f'NN-Adjusted Transition Probability Parameters (File: {os.path.basename(TEST_DATA_PATH)})')

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t_axis, hist_params[:, i], color='purple', linewidth=1.5, label='NN Output')

        plt.title(param_labels[i])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(-0.05, 1.05)  # 限制在 0-1 之间，防止视图跑偏

        if i % 2 == 0:
            plt.ylabel('Probability')
        if i >= 4:
            plt.xlabel('Time (s)')

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    # === [新增] 图表：位置误差的累积分布函数 (CDF) ===
    # 将误差从小到大排序 (只取 WINDOW_SIZE 之后的稳定阶段)
    sorted_err_fix = np.sort(err_pos_fix_seq[WINDOW_SIZE:])
    sorted_err_adapt = np.sort(err_pos_adapt_seq[WINDOW_SIZE:])

    # 计算百分比 (0 到 1)
    y_vals = np.arange(len(sorted_err_fix)) / float(len(sorted_err_fix))

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_err_fix, y_vals, 'b--', linewidth=2, label='Fixed IMM')
    plt.plot(sorted_err_adapt, y_vals, 'r-', linewidth=2, label='NN-IMM (Ours)')

    plt.title(f'CDF of Position Estimation Error\n(File: {os.path.basename(TEST_DATA_PATH)})')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Cumulative Probability (Confidence)')
    plt.grid(True)
    plt.legend(loc='lower right')

    pp.set_paper_style()

    # 2. 论文标准图：误差热力图 (可选，展示轨迹哪里误差大)
    print("绘制误差热力图...")
    pp.plot_error_heatmap(hist_true_pos, hist_est_pos_adapt, title="NN-IMM Error Distribution")

    # 3. 论文标准图：统计箱线图 (Box Plot)
    print("绘制误差统计箱线图(Pos & Vel)...")
    # 注意：去掉初始化的前 WINDOW_SIZE 帧，使统计更准确
    valid_err_pos_fix = err_pos_fix_seq[WINDOW_SIZE:]
    valid_err_pos_adapt = err_pos_adapt_seq[WINDOW_SIZE:]
    valid_err_vel_fix = err_vel_fix_seq[WINDOW_SIZE:]
    valid_err_vel_adapt = err_vel_adapt_seq[WINDOW_SIZE:]

    pp.plot_statistical_boxplot(
        valid_err_pos_fix, valid_err_pos_adapt,
        valid_err_vel_fix, valid_err_vel_adapt
    )

    print("绘制计算开销对比图...")
    # 1. 加载 BO 的实测数据
    if os.path.exists('../../bayes_imm/bo_costs_measured.npy'):
        real_bo_times = np.load('../../bayes_imm/bo_costs_measured.npy')
        print(f"成功加载 {len(real_bo_times)} 条 BO 实测数据，均值: {np.mean(real_bo_times):.2f}ms")

        # 如果实测数据比仿真帧数少，我们就循环使用它 (或只截取 NN 的前 N 帧)
        needed_length = len(nn_inference_times)
        simulated_bo_times = np.resize(real_bo_times, needed_length)

    else:
        print("警告：未找到 bo_costs_measured.npy，使用备用模拟数据")
        # 备用方案：万一你没跑上面的步骤，还是得有个托底
        simulated_bo_times = np.random.normal(loc=400.0, scale=20.0, size=len(nn_inference_times))

    pp.set_paper_style()
    pp.plot_computational_cost(nn_inference_times, simulated_bo_times, bo_interval=OPTIMIZE_INTERVAL)

    print("绘制局部放大细节图...")
    # 1. 自动寻找 Fixed IMM 误差最大的时刻 (也就是最需要放大的时刻)
    # 注意：只在稳定运行阶段 (WINDOW_SIZE 之后) 寻找，避开初始化的震荡
    valid_err_seq = err_pos_fix_seq[WINDOW_SIZE:]
    max_err_idx = np.argmax(valid_err_seq) + WINDOW_SIZE

    # 2. 设定放大范围 (例如：前后各看 50 帧)
    zoom_radius = 30
    # 防止索引越界
    start_f = max(0, max_err_idx - 3*zoom_radius)
    end_f = min(sim_steps, max_err_idx)

    print(f"  -> 自动定位到误差最大区域: Frame {start_f} - {end_f} (Center: {max_err_idx})")

    # 3. 调用 paper_plotting.py 中的函数画图
    pp.plot_3d_zoom_with_context(
        hist_true_pos,
        hist_est_pos_fix,
        hist_est_pos_adapt,
        start_frame=start_f,
        end_frame=end_f
    )

    plt.show()


if __name__ == '__main__':
    main_inference()