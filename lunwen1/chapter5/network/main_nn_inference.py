import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt  # [新增] 用于画图
from collections import deque
from scipy.signal import savgol_filter

# 请确保路径与您项目结构一致
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced
from lunwen1.chapter5.bayes_imm.online_optimizer import OnlineBoOptimizer

# ================= 配置 =================
# [修改] 指向您的 F16 测试文件路径
TEST_DATA_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver_a.csv'

MODEL_PATH = 'imm_param_net.pth'
SCALER_PATH = 'scaler_params.json'
WINDOW_SIZE = 90
DT = 1 / 30  # 假设采样率为 30Hz，如果CSV里有时间戳，最好通过时间戳计算
OPTIMIZE_INTERVAL = 20
SAVGOL_WINDOW = 25  # [新增] 与训练一致
SAVGOL_POLY = 2

# === 模型定义 (保持不变) ===
class ParamPredictorLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=6):
        super(ParamPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


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

def safe_construct_matrix(pred_params):
    """安全构建转移矩阵"""
    try:
        pred_params = np.nan_to_num(pred_params, nan=0.01, posinf=0.99, neginf=0.01)
        pred_params = np.clip(pred_params, 1e-4, 0.9999)
        new_matrix = OnlineBoOptimizer.construct_matrix_static(pred_params)
        new_matrix = np.array(new_matrix)
        new_matrix = np.maximum(new_matrix, 0.0)
        row_sums = new_matrix.sum(axis=1, keepdims=True) + 1e-8
        row_sums[row_sums == 0] = 1.0
        new_matrix = new_matrix / row_sums
        if not np.all(np.isfinite(new_matrix)):
            raise ValueError("Constructed matrix contains NaN or Inf.")
        return new_matrix
    except Exception as e:
        print(f"Matrix construction warning: {e}")
        return np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])


def load_test_data(filepath):
    """[新增] 读取 CSV 测试数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到测试文件: {filepath}")

    df = pd.read_csv(filepath)
    # 去除列名可能存在的空格
    df.columns = df.columns.str.strip()

    # 检查必要的列
    required_cols = ['x', 'y', 'z']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 文件缺少列: {col}")

    # 提取 x, y, z 真值
    return df[['x', 'y', 'z']].values


def main_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型与参数
    try:
        model = ParamPredictorLSTM().to(device)
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

    # 2. 加载测试数据 (F16)
    try:
        gt_pos_data = load_test_data(TEST_DATA_PATH)
        sim_steps = len(gt_pos_data)
        print(f">>> 已加载测试集: {os.path.basename(TEST_DATA_PATH)}, 共 {sim_steps} 帧")
    except Exception as e:
        print(f"读取数据出错: {e}")
        return

    # 3. 初始化 IMM
    initial_trans_prob = np.array([[0.81388511, 0.18511489, 0.001], [0.989, 0.01, 0.001], [0.01, 0.01, 0.98]])

    initial_state = np.zeros(9)
    # [修改] 使用 CSV 第一帧作为初始位置
    initial_state[0] = gt_pos_data[0, 0]  # x
    initial_state[3] = gt_pos_data[0, 1]  # y
    initial_state[6] = gt_pos_data[0, 2]  # z
    # 可以简单估算初始速度，或者设为0
    if sim_steps > 1:
        initial_state[1] = (gt_pos_data[1, 0] - gt_pos_data[0, 0]) / DT
        initial_state[4] = (gt_pos_data[1, 1] - gt_pos_data[0, 1]) / DT
        initial_state[7] = (gt_pos_data[1, 2] - gt_pos_data[0, 2]) / DT

    init_cov_diag = np.zeros(9)
    init_cov_diag[[0, 3, 6]] = 100.0  # Pos
    init_cov_diag[[1, 4, 7]] = 25.0  # Vel
    init_cov_diag[[2, 5, 8]] = 10.0  # Acc
    initial_cov = np.diag(init_cov_diag)

    meas_noise_std = 15  # 对齐 BoIMM
    r_cov = np.eye(3) * (meas_noise_std ** 2)

    imm_filter = IMMFilterEnhanced(initial_trans_prob, initial_state, initial_cov, r_cov=r_cov)

    pos_buffer = deque(maxlen=WINDOW_SIZE)
    last_pred_params = None
    alpha_smooth = 0.7

    # 用于绘图记录
    history_true = []
    history_est = []
    history_probs = []

    print(">>> 开始在线仿真...")

    # 4. 仿真循环
    for k in range(sim_steps):
        # --- (A) 获取当前帧数据 ---
        true_pos = gt_pos_data[k]

        # 模拟观测值：真值 + 噪声
        # 如果您的CSV本身就是观测数据（含噪），则不需要加 noise
        # 这里假设 CSV 是真值，所以手动加噪声模拟雷达观测
        noise = np.random.normal(0, meas_noise_std, 3)
        z_k = true_pos + noise

        if len(pos_buffer) == WINDOW_SIZE and k % OPTIMIZE_INTERVAL == 0:
            pos_seq = np.array(pos_buffer)

            ref_point = pos_seq[-1]
            rel_pos_seq = pos_seq - ref_point

            vel_seq, acc_seq = calculate_derivatives(pos_seq, DT)

            raw_features = np.hstack([rel_pos_seq, vel_seq, acc_seq])
            norm_features = (raw_features - mean) / std

            input_tensor = torch.tensor(norm_features, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_params = model(input_tensor).cpu().numpy()[0]

            if last_pred_params is not None:
                pred_params = alpha_smooth * pred_params + (1 - alpha_smooth) * last_pred_params
            last_pred_params = pred_params

            new_matrix = safe_construct_matrix(pred_params)
            imm_filter.set_transition_matrix(new_matrix)

            # 可选：打印更新信息
            # print(f"[Frame {k}] TPM Updated. CV Prob: {imm_filter.model_probs[0]:.4f}")

        est_state, _ = imm_filter.update(z_k, DT)
        est_pos = est_state[[0, 3, 6]]

        pos_buffer.append(z_k)

        # --- (D) 记录历史 ---
        history_true.append(true_pos)
        history_est.append(est_pos)
        history_probs.append(imm_filter.model_probs.copy())

        if k % 50 == 0:
            err = np.linalg.norm(est_pos - true_pos)
            print(f"Frame {k:03d}: Pos Error {err:.2f}m | Model Probs: {np.round(imm_filter.model_probs, 2)}")

    # 5. 结果可视化 [新增]
    hist_true = np.array(history_true)
    hist_est = np.array(history_est)
    hist_probs = np.array(history_probs)

    plt.figure(figsize=(12, 12))

    # 子图1: X-Y 平面轨迹
    plt.subplot(3, 1, 1)
    plt.plot(hist_true[:, 0], hist_true[:, 1], 'k-', linewidth=1.5, label='True Trajectory')
    plt.plot(hist_est[:, 0], hist_est[:, 1], 'r--', linewidth=1.5, label='IMM Estimate')
    plt.title(f"Trajectory Tracking (File: {os.path.basename(TEST_DATA_PATH)})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)


    # 子图2: 模型概率变化
    plt.subplot(3, 1, 2)
    plt.plot(hist_probs[:, 0], 'g-', label='CV (Const Vel)')
    plt.plot(hist_probs[:, 1], 'b-', label='CA (Const Acc)')
    plt.plot(hist_probs[:, 2], 'r-', label='CT (Turn)')
    plt.title("IMM Model Probabilities Evolution")
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)

    # 子图3: 神经网络介入时机 (可视化)
    # 我们可以画出 Position Error 随时间变化，标记 NN 更新的时间点
    plt.subplot(3, 1, 3)
    errors = np.linalg.norm(hist_est - hist_true, axis=1)
    plt.plot(errors, 'k-', linewidth=1)
    # 画出 NN 更新的时刻 (每 20 帧)
    for t in range(WINDOW_SIZE, len(errors), OPTIMIZE_INTERVAL):
        plt.axvline(x=t, color='m', alpha=0.1)  # 紫色竖线代表 NN 介入更新了参数
    plt.title("Estimation Error (Vertical Lines = NN Update Steps)")
    plt.xlabel("Frame");
    plt.ylabel("Error (m)");
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main_inference()