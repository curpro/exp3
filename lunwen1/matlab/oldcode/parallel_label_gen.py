import numpy as np
import pandas as pd
import glob
from multiprocessing import Pool, cpu_count
from bayes_opt import BayesianOptimization
import warnings

# --- 导入你的 BO 评估函数 ---
# 确保你的 chapter5_adaptive_bo_imm.py 在 python 路径下
# 如果不在，可能需要 sys.path.append
from lunwen1.matlab.adaptive_bo_imm import get_trans_matrix_6d, IMMFilter

warnings.filterwarnings("ignore")

# --- 配置 ---
WINDOW_SIZE = 20
BO_ITERATIONS = 5 # 针对大数据集生成，稍微降低迭代次数以平衡速度
BO_INIT_POINTS = 2
MEAS_NOISE_STD = 15.0
DT = 1 / 30

# 优化边界 (与你论文一致)
BOUNDS = {
    'p00': (0.5, 0.98), 'p01': (0.01, 0.4),
    'p10': (0.01, 0.4), 'p11': (0.6, 0.99),
    'p20': (0.01, 0.4), 'p21': (0.01, 0.4)
}


def process_single_window(args):
    """
    处理单个窗口的 Worker 函数
    这部分代码会在每个 CPU 核心上独立运行
    """
    window_data, start_snapshot, r_cov = args

    # 这里的 start_snapshot 是窗口起始时刻的 IMM 状态
    # 为了简化离线生成，我们可以假设窗口起始时刻状态就是真实值（或者加一点点噪声）
    # 这样可以解耦前后依赖，实现完全并行

    # 这是一个 wrapper，因为 BO 库需要函数输入参数
    def objective(p00, p01, p10, p11, p20, p21):
        # 注意：你需要稍微修改一下 evaluate_window_6d
        # 让它接受 window_data 而不是全局 measurements
        # 这里假设你已经适配好了，或者我们直接在这里重写核心逻辑

        trans_mat = get_trans_matrix_6d(p00, p01, p10, p11, p20, p21)
        if trans_mat is None: return -1e6

        # 构建临时 IMM 进行评估
        # 使用窗口第一个点的真实值初始化 (模拟理想状态)
        x_start = window_data[:, 0]  # 9维状态
        P_start = np.diag([100] * 9)

        imm = IMMFilter(trans_mat, x_start, P_start, r_cov=r_cov)

        total_err = 0.0
        # 跑完这个窗口
        for k in range(window_data.shape[1]):
            z_meas = window_data[[0, 3, 6], k] + np.random.randn(3) * MEAS_NOISE_STD
            est_x, _ = imm.update(z_meas, DT)
            # 计算位置误差
            err = np.sum((est_x[[0, 3, 6]] - window_data[[0, 3, 6], k]) ** 2)
            total_err += err

        return -np.sqrt(total_err / window_data.shape[1])

    # 运行 BO
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=BOUNDS,
        verbose=0,
        random_state=np.random.randint(10000)
    )

    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_ITERATIONS)

    best_params = optimizer.max['params']

    # 返回: [特征(如窗口末端的平均加速度, 新息等), 标签(6个参数)]
    # 这里我们简单提取窗口最后时刻的真实加速度作为特征 (输入给神经网络用)
    last_acc = window_data[[2, 5, 8], -1]  # ax, ay, az
    last_vel = window_data[[1, 4, 7], -1]  # vx, vy, vz

    # 也可以计算新息序列的统计特征，这里先存原始物理量
    features = np.concatenate([last_vel, last_acc])

    labels = [best_params['p00'], best_params['p01'],
              best_params['p10'], best_params['p11'],
              best_params['p20'], best_params['p21']]

    return np.concatenate([features, labels])


def process_trajectory_file(file_path):
    print(f"正在处理文件: {file_path}")
    df = pd.read_csv(file_path)
    data = df[['x', 'vx', 'ax', 'y', 'vy', 'ay', 'z', 'vz', 'az']].to_numpy().T
    num_steps = data.shape[1]
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

    tasks = []
    # 滑动窗口切片
    # 步长可以设为 5 或 10，不必每一步都切，减少计算量且数据相关性不那么高
    step_stride = 20

    for k in range(0, num_steps - WINDOW_SIZE, step_stride):
        window_segment = data[:, k: k + WINDOW_SIZE]

        # 初始快照 (其实在上面的 worker 里我们用了简化初始化，这里传 None 即可)
        start_snapshot = None

        tasks.append((window_segment, start_snapshot, r_cov))

    return tasks


def main():
    raw_files = glob.glob(r"/lunwen1/oldcode/train_dataset/raw_csv/*.csv")
    all_tasks = []

    # 1. 准备所有任务
    for f in raw_files:
        tasks = process_trajectory_file(f)
        all_tasks.extend(tasks)

    print(f"总共有 {len(all_tasks)} 个窗口需要优化。")
    print(f"启动并行池，核心数: {cpu_count()}")

    # 2. 并行执行
    # 这一步是加速的关键
    with Pool(processes=cpu_count()) as pool:
        # 使用 tqdm 显示进度条 (可选)
        # results = list(tqdm.tqdm(pool.imap(process_single_window, all_tasks), total=len(all_tasks)))
        results = pool.map(process_single_window, all_tasks)

    # 3. 保存结果
    results_mat = np.array(results)

    cols = ['vx', 'vy', 'vz', 'ax', 'ay', 'az', 'p00', 'p01', 'p10', 'p11', 'p20', 'p21']
    df_result = pd.DataFrame(results_mat, columns=cols)

    output_file = "train_dataset/labeled_dataset_final.csv"
    df_result.to_csv(output_file, index=False)
    print(f"数据集生成完毕！已保存至 {output_file}")
    print(f"数据形状: {df_result.shape}")


if __name__ == "__main__":
    main()