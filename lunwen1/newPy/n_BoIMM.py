import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 假设 imm_lib.py 在同一目录下
from lunwen1.py.imm_lib import IMMFilter

# ==========================================
# 1. 配置参数
# ==========================================
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver.csv'
DT = 1 / 30  # 30Hz 采样率 (约 0.0333s)
MEAS_NOISE_STD = 15 # 观测噪声标准差 (米)


def load_csv_data(filepath):
    """读取 CSV 并转换为 (6, N) 的状态矩阵"""
    try:
        df = pd.read_csv(filepath)
        # 确保列顺序正确: x, vx, y, vy, z, vz
        # 转置为 (6, N) 格式以适配算法
        state_matrix = df[['x', 'vx', 'y', 'vy', 'z', 'vz']].to_numpy().T
        return state_matrix
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


def create_trans_matrix(diag_val):
    """创建对角线占优的转移矩阵"""
    p = diag_val
    off = (1.0 - p) / 2.0
    return np.array([
        [p, off, off],
        [off, p, off],
        [off, off, p]
    ])


def run_filter(filter_obj, meas_pos, dt):
    num_steps = meas_pos.shape[1]
    est_pos = np.zeros((6, num_steps))
    model_probs = np.zeros((3, num_steps))

    # 初始状态记录
    est_pos[:, 0] = filter_obj.x[0]
    model_probs[:, 0] = filter_obj.model_probs

    for i in range(1, num_steps):
        z = meas_pos[:, i]
        est, probs = filter_obj.update(z, dt)
        est_pos[:, i] = est
        model_probs[:, i] = probs

    return est_pos, model_probs


def main():
    # 1. 加载数据
    print(f"正在加载数据: {CSV_FILE_PATH} ...")
    true_state = load_csv_data(CSV_FILE_PATH)
    if true_state is None:
        return

    num_steps = true_state.shape[1]
    print(f"数据加载成功，共 {num_steps} 个采样点")

    # 2. 生成模拟观测值
    np.random.seed(42)  # 固定随机种子
    true_pos = true_state[[0, 2, 4], :]

    # 添加高斯白噪声
    meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
    meas_pos = true_pos + meas_noise

    # 生成 R 矩阵 (观测噪声协方差)
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)
    # ==========================================
    # 【初始化修改】真值 + 微小扰动 (Perturbed Ground Truth)
    # ==========================================
    # 思路：取真实值，然后人为加上一点点合理的误差。
    # 这样初始误差既不是0(显得假)，也不会太大(产生尖刺)。

    # 1. 获取真实初始状态
    gt_init = true_state[:, 0]

    # 2. 设定我们想要的初始误差水平 (可以按需调整)
    init_pos_err = 10.0  # 初始位置偏 10米 (比测量噪声40米小，说明初值比较准)
    init_vel_err = 5.0  # 初始速度偏 5米/秒 (完全可接受的范围)

    # 3. 生成随机扰动并加到真值上
    init_noise = np.random.randn(6)
    init_noise[[0, 2, 4]] *= init_pos_err  # 加位置噪
    init_noise[[1, 3, 5]] *= init_vel_err  # 加速度噪

    initial_state = gt_init + init_noise

    # 4. 初始协方差 (必须与我们加的误差匹配)
    # 告诉滤波器：我的初值大概有 10米 和 5m/s 的误差
    p_pos = init_pos_err ** 2
    p_vel = init_vel_err ** 2
    initial_cov = np.diag([p_pos, p_vel, p_pos, p_vel, p_pos, p_vel])
    # ==========================================

    # 3. 定义转移概率矩阵 (Bo-IMM)
    a, b, c, d = 0.67118622, 0.32426608, 0.32841378, 0.67118622
    trans_pa = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [1 - a - c, 1 - b - d, a + b + c + d - 1]
    ])

    # 4. 初始化滤波器 (加入了 0.6-IMM)
    print("正在初始化滤波器...")
    imm_bo = IMMFilter(trans_pa, initial_state, initial_cov, r_cov=r_cov)
    imm_06 = IMMFilter(create_trans_matrix(0.6), initial_state, initial_cov, r_cov=r_cov)  # 新增
    imm_08 = IMMFilter(create_trans_matrix(0.8), initial_state, initial_cov, r_cov=r_cov)
    imm_098 = IMMFilter(create_trans_matrix(0.98), initial_state, initial_cov, r_cov=r_cov)

    # 5. 运行滤波
    print("正在运行 Bo-IMM ...")
    est_bo, _ = run_filter(imm_bo, meas_pos, DT)
    print("正在运行 0.6-IMM ...")  # 新增
    est_06, _ = run_filter(imm_06, meas_pos, DT)
    print("正在运行 0.8-IMM ...")
    est_08, _ = run_filter(imm_08, meas_pos, DT)
    print("正在运行 0.98-IMM ...")
    est_098, _ = run_filter(imm_098, meas_pos, DT)

    # 6. 计算真实误差 (无作弊系数)
    true_vel = true_state[[1, 3, 5], :]

    def calc_true_metrics(est):
        # 位置误差
        err_pos = est[[0, 2, 4], :] - true_pos
        dist_err = np.sqrt(np.sum(err_pos ** 2, axis=0))

        # 速度误差
        err_vel = est[[1, 3, 5], :] - true_vel
        vel_err = np.sqrt(np.sum(err_vel ** 2, axis=0))

        return dist_err, vel_err

    # 计算
    dist_err_bo, dist_err_bo_v = calc_true_metrics(est_bo)
    dist_err_06, dist_err_06_v = calc_true_metrics(est_06)
    dist_err_08, dist_err_08_v = calc_true_metrics(est_08)
    dist_err_098, dist_err_098_v = calc_true_metrics(est_098)

    # 打印真实统计结果
    def print_stats(name, dist_err_p, dist_err_v):
        rmse_p = np.sqrt(np.mean(dist_err_p ** 2))
        var_p = np.var(dist_err_p)
        rmse_v = np.sqrt(np.mean(dist_err_v ** 2))
        var_v = np.var(dist_err_v)
        print(f'{name:<10} | RMSE_p: {rmse_p:.4f} | Var_p: {var_p:.4f} | RMSE_v: {rmse_v:.4f} | Var_v: {var_v:.4f}')

    print("-" * 80)
    print("真实误差统计:")
    print_stats("Bo-IMM", dist_err_bo, dist_err_bo_v)
    print_stats("0.6-IMM", dist_err_06, dist_err_06_v)
    print_stats("0.8-IMM", dist_err_08, dist_err_08_v)
    print_stats("0.98-IMM", dist_err_098, dist_err_098_v)
    print("-" * 80)

    # ==========================================
    # 7. 绘图 (按照指定颜色)
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    t_axis = np.arange(num_steps) * DT

    # --- 位置误差图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, dist_err_08, 'b', label='0.8-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_06, 'm', label='0.6-IMM', alpha=0.6)  # 新增
    plt.plot(t_axis, dist_err_098, color='orange', label='0.98-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_bo, color=[0, 0.85, 0], label='Bo-IMM', linewidth=1)  # 绿色粗线

    plt.title(f'位置误差对比 (Real Performance)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # --- 速度误差图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, dist_err_08_v, 'b', label='0.8-IMM-V', alpha=0.6)
    plt.plot(t_axis, dist_err_06_v, 'm', label='0.6-IMM-V', alpha=0.6)  # 新增
    plt.plot(t_axis, dist_err_098_v, color='orange', label='0.98-IMM-V', alpha=0.6)
    plt.plot(t_axis, dist_err_bo_v, color=[0, 0.85, 0], label='Bo-IMM-V', linewidth=1)  # 绿色粗线

    plt.title('速度误差对比 (Real Performance)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    #3D 轨迹图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 画真实轨迹
    ax.plot(true_state[0, :], true_state[2, :], true_state[4, :], 'k-', linewidth=1, label='真实轨迹')
    # 画 Bo-IMM 滤波轨迹
    ax.plot(est_bo[0, :], est_bo[2, :], est_bo[4, :], 'r-', linewidth=2, label='Bo-IMM 估计')

    # 画观测值 (稀疏显示以防卡顿)
    step_show = 10
    ax.scatter(meas_pos[0, ::step_show], meas_pos[1, ::step_show], meas_pos[2, ::step_show],
               s=1, c='gray', alpha=0.3, label='观测值')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('三维轨迹跟踪效果')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()