import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import glob
import numpy as np
import pandas as pd
import warnings
from scipy.signal import savgol_filter
from lunwen1.chapter5.bayes_imm.online_optimizer import OnlineBoOptimizer
from lunwen1.chapter5.bayes_imm.imm_lib_enhanced import IMMFilterEnhanced  # [新增] 需要引入 IMM

# ================= 配置 =================
DATA_FOLDER = r'D:\AFS\lunwen\dataSet\processed_data_4'
OUTPUT_DATA_FILE = '../network/npz/training_data_part5.npz'

EXCLUDED_FILES = [
]

REPEAT_PER_FILE = 3  # 每个文件重复次数
WINDOW_SIZE = 90  # 观测窗口
OPTIMIZE_INTERVAL = 20
DT = 1 / 30
NOISE_STD = 15.0

SAVGOL_WINDOW = 25
SAVGOL_POLY = 2

def setup_seed(seed):
    np.random.seed(seed)
    print(f">>> 随机种子已固定为: {seed}")

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        required_cols = ['x', 'y', 'z']  # 只需要位置真值即可
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  [跳过] 缺列: {missing} in {os.path.basename(filepath)}")
            return None
        return df[required_cols].values
    except Exception as e:
        print(f"  [错误] {os.path.basename(filepath)}: {e}")
        return None


def calculate_derivatives(pos_data, dt):
    """
    [修改2] 使用 Savitzky-Golay 滤波器计算速度和加速度。
    原代码使用简单差分 (pos[k]-pos[k-1])/dt 会导致噪声放大30倍(速度)和900倍(加速度)。
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

    # deriv=1 算一阶导(速度), deriv=2 算二阶导(加速度)
    # delta=dt 自动处理 /dt 的缩放
    vel = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY,
                        deriv=1, delta=dt, axis=0)
    acc = savgol_filter(pos_data, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY,
                        deriv=2, delta=dt, axis=0)

    return vel, acc


def process_single_trajectory(raw_data, file_id):
    """
    [修复版] 增加了 float64 精度转换和 try-except 异常捕获，防止 Cholesky 报错中断。
    """
    X_list = []
    Y_list = []
    G_list = []  # [新增] 存储组ID

    pos_gt = raw_data[:, :3]  # 真值
    n_steps = len(pos_gt)

    if n_steps <= WINDOW_SIZE + 20:
        return [], [], []

    # === 循环重复生成 ===
    for rep in range(REPEAT_PER_FILE):
        # 1. 生成带噪观测
        noise = np.random.normal(0, NOISE_STD, pos_gt.shape)
        pos_measured = pos_gt + noise

        # 2. 初始化 IMM
        initial_trans = np.array([[0.81388511, 0.18511489, 0.001], [0.989, 0.01, 0.001], [0.01, 0.01, 0.98]])
        init_state = np.zeros(9)
        init_state[0] = pos_measured[0, 0]
        init_state[3] = pos_measured[0, 1]
        init_state[6] = pos_measured[0, 2]
        init_state[1] = 265.0  # 粗略速度

        # 协方差初始化
        init_cov_diag = np.zeros(9)
        init_cov_diag[[0, 3, 6]] = 100.0
        init_cov_diag[[1, 4, 7]] = 25.0
        init_cov_diag[[2, 5, 8]] = 10.0
        init_cov = np.diag(init_cov_diag)

        r_cov = np.eye(3) * (NOISE_STD ** 2)

        # 实例化 IMM 和 优化器
        imm = IMMFilterEnhanced(initial_trans, init_state, init_cov, r_cov=r_cov)
        optimizer = OnlineBoOptimizer(imm, DT)

        default_params = [0.98, 0.01, 0.01, 0.98, 0.01, 0.01]
        current_params = default_params

        # 3. 运行滤波循环
        for k in range(n_steps):
            z_k = pos_measured[k]

            # (A) 记录当前时刻的数据用于生成特征
            if k >= WINDOW_SIZE and k % OPTIMIZE_INTERVAL == 0:
                if k + WINDOW_SIZE <= n_steps:
                    # 1. 准备特征数据
                    hist_pos = pos_measured[k - WINDOW_SIZE: k]
                    hist_vel, hist_acc = calculate_derivatives(hist_pos, DT)
                    rel_pos = hist_pos - hist_pos[-1]
                    features = np.hstack([rel_pos, hist_vel, hist_acc])

                    # 2. 准备优化所需数据
                    snapshot = imm.get_state_snapshot()

                    # [关键修复 1] 强制转换为 float64，提高矩阵分解的稳定性
                    future_window = pos_gt[k:k + WINDOW_SIZE].T.astype(np.float64)
                    # future_window = pos_measured[k:k + WINDOW_SIZE].T.astype(np.float64)

                    best_p = None
                    # [关键修复 2] 增加 try-except 捕获 Cholesky/Numerical 错误
                    try:
                        # 暂时抑制那烦人的警告，或者让它报错以便捕获
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")  # 忽略警告继续运行

                            best_p = optimizer.run_optimization(
                                future_window,
                                snapshot,
                                current_params,
                                default_params,
                                n_iter=5
                            )
                    except Exception as e:
                        print(f"  [警告] Frame {k} 优化失败 (跳过): {e}")
                        best_p = None

                    if best_p is not None:
                        # 转 numpy 方便检查
                        best_p_arr = np.array(best_p, dtype=np.float32)

                        # 检查1: 是否全为有限数 (非 NaN, 非 Inf)
                        is_finite = np.all(np.isfinite(best_p_arr))

                        # 检查2: (可选) 是否在 [0, 1] 范围内
                        # 虽然 Sigmoid 会处理，但如果优化器跑出 1e10 这种数也是不正常的
                        is_in_range = np.all((best_p_arr >= 0.0) & (best_p_arr <= 1.0))

                        if is_finite and is_in_range:
                            # 只有数据完全干净，才加入数据集
                            X_list.append(features.astype(np.float32))
                            Y_list.append(best_p_arr)
                            G_list.append(file_id)  # [新增] 记录当前样本属于哪个文件

                            # 更新下一轮优化的起点
                            current_params = best_p

                            # 更新 IMM 矩阵，继续跟踪
                            new_mtx = optimizer.construct_matrix_static(best_p)
                            imm.set_transition_matrix(new_mtx)
                        else:
                            # 即使算出了结果，如果结果是 NaN 或离谱值，视为优化失败
                            # print(f"  [过滤] Frame {k} 产生无效参数 (NaN/Inf): {best_p}")
                            # 不更新 current_params，保持上一次的参数，防止雪崩
                            pass

            # (B) IMM 步进
            imm.update(z_k, DT)

    return X_list, Y_list, G_list


def main():
    setup_seed(42)

    print("=== Step 1: 生成增强训练数据 (修正闭环版) ===")
    search_path = os.path.join(DATA_FOLDER, "*.csv")
    csv_files = glob.glob(search_path)

    # ... (后续主循环代码与您之前的一致，保持不变即可) ...
    # 只要确保 process_single_trajectory 被替换即可

    if not csv_files:
        print(f"在 {DATA_FOLDER} 未找到 CSV 文件")
        return

    all_X = []
    all_Y = []
    all_G = []  # [新增]
    skipped_count = 0

    for f_idx, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        if filename in EXCLUDED_FILES:
            print(f"[{f_idx + 1}/{len(csv_files)}] 🚫 跳过测试集: {filename}")
            skipped_count += 1
            continue

        print(f"[{f_idx + 1}/{len(csv_files)}] 处理训练集: {filename}")
        raw_data = load_data(filepath)
        if raw_data is None: continue

        x_batch, y_batch, g_batch = process_single_trajectory(raw_data, f_idx)
        if len(x_batch) > 0:
            all_X.extend(x_batch)
            all_Y.extend(y_batch)
            all_G.extend(g_batch)  # [新增]
            print(f"  > 生成样本 {len(x_batch)} 个")

    # 保存部分
    if not all_X:
        print("未生成数据。")
        return

    all_X = np.array(all_X, dtype=np.float32)
    all_Y = np.array(all_Y, dtype=np.float32)
    all_G = np.array(all_G, dtype=np.int32)  # [新增]

    print("-" * 30)
    print(f"生成完毕。X: {all_X.shape}, Y: {all_Y.shape}, G: {all_G.shape}")
    np.savez(OUTPUT_DATA_FILE, X=all_X, Y=all_Y, G=all_G)
    print(f"数据已保存至 {OUTPUT_DATA_FILE}")


if __name__ == '__main__':
    main()