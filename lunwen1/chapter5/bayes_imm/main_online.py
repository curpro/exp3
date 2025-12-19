import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imm_lib_enhanced import IMMFilterEnhanced
from online_optimizer import OnlineBoOptimizer
from collections import deque
import time

# =================配置=================
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver_a.csv'
DT = 1 / 30
MEAS_NOISE_STD = 15.0
WINDOW_SIZE = 90
OPTIMIZE_INTERVAL = 20


# ======================================

def load_data(filepath):
    df = pd.read_csv(filepath)
    pos = df[['x', 'y', 'z']].values.T
    try:
        truth = df[['x', 'vx', 'ax', 'y', 'vy', 'ay', 'z', 'vz', 'az']].values.T
    except KeyError:
        truth = np.zeros((9, pos.shape[1]))
        truth[[0, 3, 6], :] = pos
    return pos, truth


def get_initial_matrix_from_chapter4():
    p = np.array([
        [0.81388511, 0.18511489, 0.001],
        [0.989, 0.01, 0.001],
        [0.01, 0.01, 0.98]
    ])
    params = np.array([p[0, 0], p[0, 1], p[1, 0], p[1, 1], p[2, 0], p[2, 1]])
    return p, params


def main():
    print("=== 第五章：在线自适应 BO-IMM 仿真开始 (Final Strict) ===")

    try:
        true_pos_raw, true_state_full = load_data(CSV_FILE_PATH)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    num_steps = true_pos_raw.shape[1]
    np.random.seed(42)
    meas_pos = true_pos_raw + np.random.randn(*true_pos_raw.shape) * MEAS_NOISE_STD
    _ = np.random.randn(9)

    R_true = np.eye(3) * (MEAS_NOISE_STD ** 2)

    # 初始化
    init_state = np.zeros(9)
    init_state[[0, 3, 6]] = true_pos_raw[:, 0]
    init_cov = np.eye(9) * 100

    init_matrix, current_params = get_initial_matrix_from_chapter4()
    default_params = current_params.copy()

    imm_online = IMMFilterEnhanced(init_matrix, init_state, init_cov, r_cov=R_true)
    imm_fixed = IMMFilterEnhanced(init_matrix, init_state, init_cov, r_cov=R_true)
    optimizer = OnlineBoOptimizer(imm_online, DT)

    est_online = np.zeros((9, num_steps))
    est_fixed = np.zeros((9, num_steps))
    param_history = []

    buffer_z = deque(maxlen=WINDOW_SIZE)
    buffer_state = deque(maxlen=WINDOW_SIZE)

    print(f"Start Processing {num_steps} frames...")

    for k in range(num_steps):
        z = meas_pos[:, k]

        # 1. 记录 Snapshot
        current_snapshot = imm_online.get_state_snapshot()

        # 2. Filter Update
        imm_online.predict(DT)
        est_x_on, _ = imm_online.update(z, DT)
        est_online[:, k] = est_x_on

        imm_fixed.predict(DT)
        est_x_fix, _ = imm_fixed.update(z, DT)
        est_fixed[:, k] = est_x_fix

        # 3. Buffer Update
        buffer_state.append(current_snapshot)
        buffer_z.append(z)

        bo_real_times = []
        # === 优化触发点 ===
        if len(buffer_z) == WINDOW_SIZE and k % OPTIMIZE_INTERVAL == 0:
            current_window_idx = k // OPTIMIZE_INTERVAL
            print(f"\n[进度] 第 {k}/{num_steps} 帧 | 正在优化第 {current_window_idx} 个窗口 (BO running)...")

            obs_window = np.array(buffer_z).T
            start_snapshot = buffer_state[0]

            # ================= [修复开始] =================
            latest_snapshot = imm_online.get_state_snapshot()

            t0 = time.perf_counter()
            new_params = optimizer.run_optimization(
                obs_window, start_snapshot, current_params, default_params, n_iter=10
            )
            t_bo = (time.perf_counter() - t0) * 1000
            bo_real_times.append(t_bo)
            print(f"cost: {t_bo:.2f} ms")
            if len(bo_real_times) % 10 == 0:
                np.save('bo_costs_measured.npy', np.array(bo_real_times))
                print(f"  >>> 已保存 {len(bo_real_times)} 条实测时间数据到 bo_costs_measured.npy")

            imm_online.set_state_snapshot(latest_snapshot)
            # ================= [修复结束] =================

            current_params = new_params
            new_matrix = OnlineBoOptimizer.construct_matrix_static(new_params)
            imm_online.set_transition_matrix(new_matrix)

            # === [新增] 打印当前矩阵核心元素 ===
            print(f"    >>> 优化完成。更新后的转移矩阵关键元素:")
            print(f"        P_CV->CV (P11): {new_matrix[0, 0]:.4f} | P_CV->CA (P12): {new_matrix[0, 1]:.4f}")
            print(f"        P_CA->CV (P21): {new_matrix[1, 0]:.4f} | P_CA->CA (P22): {new_matrix[1, 1]:.4f}")
            print(f"        P_CT->CV (P31): {new_matrix[2, 0]:.4f} | P_CT->CA (P32): {new_matrix[2, 1]:.4f}")
            print("-" * 60)

        # 简单的进度心跳 (非优化帧也偶尔显示一下，防止焦虑)
        elif k % 100 == 0:
            print(f"[Run] 处理中... 第 {k} 帧")

        p = imm_online.trans_prob
        param_history.append([p[0, 0], p[0, 1], p[1, 0], p[1, 1], p[2, 0], p[2, 1]])

    # ================= 绘图代码保持不变 =================
    # ... (原有绘图代码) ...

    # 为了完整性，这里简写了绘图部分，实际运行时请保留你原有的绘图代码
    print("\n仿真结束，开始绘图...")

    err_online_pos = np.sqrt(np.sum((est_online[[0, 3, 6]] - true_pos_raw) ** 2, axis=0))
    err_fixed_pos = np.sqrt(np.sum((est_fixed[[0, 3, 6]] - true_pos_raw) ** 2, axis=0))
    start_plot = 80
    t = np.arange(num_steps) * DT

    true_vel = true_state_full[[1, 4, 7], :]
    est_online_vel = est_online[[1, 4, 7], :]
    est_fixed_vel = est_fixed[[1, 4, 7], :]
    err_online_vel = np.sqrt(np.sum((est_online_vel - true_vel) ** 2, axis=0))
    err_fixed_vel = np.sqrt(np.sum((est_fixed_vel - true_vel) ** 2, axis=0))

    # ================= [新增] 打印详细对比数据 =================
    # 使用与绘图相同的起始帧，跳过初始化的不稳定阶段
    eval_start_idx = 80

    # 1. 计算平均 RMSE
    rmse_pos_fix_val = np.sqrt(np.mean(err_fixed_pos[eval_start_idx:] ** 2))
    rmse_pos_onl_val = np.sqrt(np.mean(err_online_pos[eval_start_idx:] ** 2))
    rmse_vel_fix_val = np.sqrt(np.mean(err_fixed_vel[eval_start_idx:] ** 2))
    rmse_vel_onl_val = np.sqrt(np.mean(err_online_vel[eval_start_idx:] ** 2))

    # 2. [新增] 计算误差方差 (Variance)
    # 计算公式：对误差序列求方差 var = mean(abs(x - x.mean())**2)
    var_pos_fix_val = np.var(err_fixed_pos[eval_start_idx:])
    var_pos_onl_val = np.var(err_online_pos[eval_start_idx:])
    var_vel_fix_val = np.var(err_fixed_vel[eval_start_idx:])
    var_vel_onl_val = np.var(err_online_vel[eval_start_idx:])

    # 2. 计算提升率 (Improvement Percentage)
    # 公式：(旧 - 新) / 旧 * 100%
    pos_rmse_improv = (rmse_pos_fix_val - rmse_pos_onl_val) / rmse_pos_fix_val * 100
    vel_rmse_improv = (rmse_vel_fix_val - rmse_vel_onl_val) / rmse_vel_fix_val * 100

    pos_var_improv = (var_pos_fix_val - var_pos_onl_val) / var_pos_fix_val * 100
    vel_var_improv = (var_vel_fix_val - var_vel_onl_val) / var_vel_fix_val * 100

    # 4. 打印制表 (包含 RMSE 和 Variance)
    print("\n" + "=" * 65)
    print("               仿真结果性能对比 (RMSE & Variance)")
    print("=" * 65)
    print(f"{'Metric':<16} | {'Fixed IMM':<12} | {'BO-IMM (Ours)':<15} | {'Improvement':<11}")
    print("-" * 65)
    # RMSE 行
    print(f"{'Pos RMSE (m)':<16} | {rmse_pos_fix_val:<12.4f} | {rmse_pos_onl_val:<15.4f} | {pos_rmse_improv:>9.2f}%")
    print(f"{'Vel RMSE (m/s)':<16} | {rmse_vel_fix_val:<12.4f} | {rmse_vel_onl_val:<15.4f} | {vel_rmse_improv:>9.2f}%")
    print("-" * 65)
    # Variance 行
    print(f"{'Pos Var (m^2)':<16} | {var_pos_fix_val:<12.4f} | {var_pos_onl_val:<15.4f} | {pos_var_improv:>9.2f}%")
    print(f"{'Vel Var (m/s)^2':<16} | {var_vel_fix_val:<12.4f} | {var_vel_onl_val:<15.4f} | {vel_var_improv:>9.2f}%")
    print("=" * 65 + "\n")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t[start_plot:], err_fixed_pos[start_plot:], 'b--', label='Fixed IMM', alpha=0.6)
    plt.plot(t[start_plot:], err_online_pos[start_plot:], 'r-', label='Adaptive IMM (Ours)', linewidth=1.5)
    plt.ylabel('Position RMSE (m)')
    plt.legend()
    plt.grid(True)
    rmse_fix_plot = np.sqrt(np.mean(err_fixed_pos[start_plot:] ** 2))
    rmse_onl_plot = np.sqrt(np.mean(err_online_pos[start_plot:] ** 2))
    plt.title(
        f'Position RMSE: Fixed={rmse_fix_plot:.2f}m, Online={rmse_onl_plot:.2f}m')

    plt.subplot(2, 1, 2)
    plt.plot(t[start_plot:], err_fixed_vel[start_plot:], 'b--', label='Fixed IMM', alpha=0.6)
    plt.plot(t[start_plot:], err_online_vel[start_plot:], 'r-', label='Adaptive IMM (Ours)', linewidth=1.5)
    plt.ylabel('Velocity RMSE (m/s)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    rmse_fix_plot_v = np.sqrt(np.mean(err_fixed_vel[start_plot:] ** 2))
    rmse_onl_plot_v = np.sqrt(np.mean(err_online_vel[start_plot:] ** 2))
    plt.title(
        f'Velocity RMSE: Fixed={rmse_fix_plot_v:.2f}m/s, Online={rmse_onl_plot_v:.2f}m/s')
    plt.tight_layout()
    plt.show()

    ph = np.array(param_history)
    param_labels = [
        r'$P_{11}$ (CV$\to$CV)', r'$P_{12}$ (CV$\to$CA)',
        r'$P_{21}$ (CA$\to$CV)', r'$P_{22}$ (CA$\to$CA)',
        r'$P_{31}$ (CT$\to$CV)', r'$P_{32}$ (CT$\to$CA)'
    ]

    plt.figure(figsize=(12, 10))
    plt.suptitle('Adaptive Transition Probability Parameters (BO-IMM)')

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, ph[:, i], label=param_labels[i], linewidth=1)
        plt.title(param_labels[i])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1.0)
        if i % 2 == 0:
            plt.ylabel('Probability')
        if i >= 4:
            plt.xlabel('Time (s)')
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.show()

if __name__ == "__main__":
    main()