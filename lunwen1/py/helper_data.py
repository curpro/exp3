import numpy as np


def generate_truth_data():
    """
    生成三维轨迹的真实状态 (对应 helperGenerateTruthData4.m)
    返回:
        Xgt: 真实状态矩阵 (9 x num_steps) [x, vx, ax, y, vy, ay, z, vz, az]
        tt: 时间序列
    """
    np.random.seed(2019)

    # 参数定义
    vx = 10.0  # m/s
    omegaA = np.deg2rad(5)
    omegaB = np.deg2rad(6)
    omegaC = np.deg2rad(6)
    # omegaD = np.deg2rad(10) # 未使用
    # omegaE = np.deg2rad(-10) # 未使用
    acc = 5.0
    accD = -3.0  # m/s^2
    dt = 0.1
    num_steps = 4000
    tt = np.arange(0, num_steps * dt, dt)

    Xgt = np.zeros((9, len(tt)))  # 状态包括位置、速度、加速度 (x, vx, ax, y, vy, ay, z, vz, az)
    # 注意：为了方便计算，这里暂存9维，最后输出会按需调整
    # 映射：Python索引 0:x, 1:vx, 2:ax, 3:y, 4:vy, 5:ay, 6:z, 7:vz, 8:az

    # 初始状态
    Xgt[1, 0] = vx
    Xgt[4, 0] = 0
    Xgt[7, 0] = 0

    # 定义各段结束索引 (Python range是左闭右开，所以这里直接用作切片终点)
    seg1 = 400  # 匀速 1
    seg2 = 800  # 加速 1
    seg3 = 1200  # 旋转 1
    seg4 = 1600  # 匀速 2
    seg5 = 2000  # 减速 1
    seg6 = 2400  # 旋转 2
    seg7 = 2800  # 匀速 3
    seg8 = 3200  # 旋转 3
    seg9 = 3600  # 加速 2
    seg10 = 4000  # 旋转 4 (螺旋)
    # 后面的 seg11-13 在原代码逻辑中被 seg10 覆盖或未完全使用，暂略

    attitude = np.zeros(len(tt))

    # --- 1. 匀速 1 ---
    for m in range(1, seg1):
        Xgt[:, m] = Xgt[:, m - 1]
        Xgt[0, m] = Xgt[0, m - 1] + Xgt[1, m - 1] * dt  # x
        Xgt[3, m] = Xgt[3, m - 1] + Xgt[4, m - 1] * dt  # y
        Xgt[6, m] = Xgt[6, m - 1] + Xgt[7, m - 1] * dt  # z
        attitude[m] = 1

    # --- 2. 加速 1 ---
    acc_vector = np.array([acc, acc, acc])
    for m in range(seg1, seg2):
        Xgt[:, m] = Xgt[:, m - 1]
        # 更新速度
        Xgt[1, m] = Xgt[1, m - 1] + acc_vector[0] * dt
        Xgt[4, m] = Xgt[4, m - 1] + acc_vector[1] * dt
        Xgt[7, m] = Xgt[7, m - 1] + acc_vector[2] * dt
        # 更新位置
        Xgt[0, m] = Xgt[0, m - 1] + Xgt[1, m] * dt
        Xgt[3, m] = Xgt[3, m - 1] + Xgt[4, m] * dt
        Xgt[6, m] = Xgt[6, m - 1] + Xgt[7, m] * dt
        attitude[m] = 2

    # --- 3. 旋转 1 ---
    for m in range(seg2, seg3):
        X0 = Xgt[:, m - 1].copy()
        omega = omegaB
        # 计算当前航向角 phi
        phi = np.arctan2(X0[4], np.sqrt(X0[1] ** 2 + X0[7] ** 2))
        v_total = np.sqrt(X0[1] ** 2 + X0[4] ** 2 + X0[7] ** 2)

        # 旋转逻辑复现
        time_step = m - seg3  # 注意：MATLAB原代码这里用了 (m-seg3-1)，在seg2阶段其实是负数，为了保持一致：
        # MATLAB: m从801到1200. seg3=1200. (m-seg3-1) 是 -399 到 -1.
        # Python: m从800到1199.
        t_param = (m - seg3)

        vx_new = v_total * np.cos(phi) * np.cos(omega * dt * t_param)
        vy_new = v_total * np.sin(phi)
        vz_new = v_total * np.cos(phi) * np.sin(omega * dt * t_param)

        Xgt[0, m] = Xgt[0, m - 1] + vx_new * dt
        Xgt[3, m] = Xgt[3, m - 1] + vy_new * dt
        Xgt[6, m] = Xgt[6, m - 1] + vz_new * dt
        Xgt[1, m] = vx_new
        Xgt[4, m] = vy_new
        Xgt[7, m] = vz_new
        attitude[m] = 3

    # --- 4. 匀速 2 ---
    for m in range(seg3, seg4):
        Xgt[:, m] = Xgt[:, m - 1]
        Xgt[0, m] = Xgt[0, m - 1] + Xgt[1, m - 1] * dt
        Xgt[3, m] = Xgt[3, m - 1] + Xgt[4, m - 1] * dt
        Xgt[6, m] = Xgt[6, m - 1] + Xgt[7, m - 1] * dt
        attitude[m] = 1

    # --- 5. 减速 1 ---
    acc_vector_d = np.array([accD, accD, accD])
    for m in range(seg4, seg5):
        Xgt[:, m] = Xgt[:, m - 1]
        Xgt[1, m] = Xgt[1, m - 1] + acc_vector_d[0] * dt
        Xgt[4, m] = Xgt[4, m - 1] + acc_vector_d[1] * dt
        Xgt[7, m] = Xgt[7, m - 1] + acc_vector_d[2] * dt
        Xgt[0, m] = Xgt[0, m - 1] + Xgt[1, m] * dt
        Xgt[3, m] = Xgt[3, m - 1] + Xgt[4, m] * dt
        Xgt[6, m] = Xgt[6, m - 1] + Xgt[7, m] * dt
        attitude[m] = 2

    # --- 6. 旋转 2 ---
    for m in range(seg5, seg6):
        X0 = Xgt[:, m - 1].copy()
        omega = omegaC
        phi = np.arctan2(X0[4], np.sqrt(X0[1] ** 2 + X0[7] ** 2))
        v_total = np.sqrt(X0[1] ** 2 + X0[4] ** 2 + X0[7] ** 2)

        # MATLAB: (m-seg3-1) -> 这里原代码似乎还在引用 seg3，我们照抄逻辑
        # 但注意 seg5 远大于 seg3，这会导致参数很大
        t_param = (m - seg3)

        vx_new = v_total * np.cos(phi) * np.cos(omega * dt * t_param)
        vy_new = v_total * np.sin(phi)
        vz_new = v_total * np.cos(phi) * np.sin(omega * dt * t_param)

        Xgt[0, m] = Xgt[0, m - 1] + vx_new * dt
        Xgt[3, m] = Xgt[3, m - 1] + vy_new * dt
        Xgt[6, m] = Xgt[6, m - 1] + vz_new * dt
        Xgt[1, m] = vx_new
        Xgt[4, m] = vy_new
        Xgt[7, m] = vz_new
        attitude[m] = 3

    # --- 7. 匀速 3 ---
    for m in range(seg6, seg7):
        Xgt[:, m] = Xgt[:, m - 1]
        Xgt[0, m] = Xgt[0, m - 1] + Xgt[1, m - 1] * dt
        Xgt[3, m] = Xgt[3, m - 1] + Xgt[4, m - 1] * dt
        Xgt[6, m] = Xgt[6, m - 1] + Xgt[7, m - 1] * dt
        attitude[m] = 1

    # --- 8. 旋转 3 ---
    for m in range(seg7, seg8):
        X0 = Xgt[:, m - 1].copy()
        omega = omegaC
        phi = np.arctan2(X0[4], np.sqrt(X0[1] ** 2 + X0[7] ** 2))
        v_total = np.sqrt(X0[1] ** 2 + X0[4] ** 2 + X0[7] ** 2)
        t_param = (m - seg3)

        vx_new = v_total * np.cos(phi) * np.cos(omega * dt * t_param)
        vy_new = v_total * np.sin(phi)
        vz_new = v_total * np.cos(phi) * np.sin(omega * dt * t_param)

        Xgt[0, m] = Xgt[0, m - 1] + vx_new * dt
        Xgt[3, m] = Xgt[3, m - 1] + vy_new * dt
        Xgt[6, m] = Xgt[6, m - 1] + vz_new * dt
        Xgt[1, m] = vx_new
        Xgt[4, m] = vy_new
        Xgt[7, m] = vz_new
        attitude[m] = 3

    # --- 9. 加速 2 ---
    acc_vector = np.array([acc, acc, acc])
    for m in range(seg8, seg9):
        Xgt[:, m] = Xgt[:, m - 1]
        Xgt[1, m] = Xgt[1, m - 1] + acc_vector[0] * dt
        Xgt[4, m] = Xgt[4, m - 1] + acc_vector[1] * dt
        Xgt[7, m] = Xgt[7, m - 1] + acc_vector[2] * dt
        Xgt[0, m] = Xgt[0, m - 1] + Xgt[1, m] * dt
        Xgt[3, m] = Xgt[3, m - 1] + Xgt[4, m] * dt
        Xgt[6, m] = Xgt[6, m - 1] + Xgt[7, m] * dt
        attitude[m] = 2

    # --- 10. 旋转 4 (螺旋) ---
    for m in range(seg9, seg10):
        X0 = Xgt[:, m - 1].copy()
        omega = omegaA
        phi = np.arctan2(X0[4], np.sqrt(X0[1] ** 2 + X0[7] ** 2))
        v_total = np.sqrt(X0[1] ** 2 + X0[4] ** 2 + X0[7] ** 2)

        # MATLAB: (m-2)
        t_param = (m - 1)  # Python index adjust

        vx_new = v_total * np.cos(phi) * np.cos(omega * dt * t_param)
        vy_new = v_total * np.sin(phi)
        vz_new = v_total * np.cos(phi) * np.sin(omega * dt * t_param)

        Xgt[0, m] = Xgt[0, m - 1] + vx_new * dt
        Xgt[3, m] = Xgt[3, m - 1] + vy_new * dt
        Xgt[6, m] = Xgt[6, m - 1] + vz_new * dt
        Xgt[1, m] = vx_new
        Xgt[4, m] = vy_new
        Xgt[7, m] = vz_new
        attitude[m] = 3

    # 重组输出格式，仅保留状态中的 [x, vx, y, vy, z, vz] (6xN)
    # 对应 MATLAB: Xgt = Xgt([1, 2, 4, 5, 7, 8], :);
    Xgt_out = Xgt[[0, 1, 3, 4, 6, 7], :]

    # 添加测量噪声到真值 (helper 中的逻辑)
    # 注意：helper中是对 Truth 添加了噪声并覆盖了 Xgt，这有点奇怪，但我们照做
    measNoiseTruePos = np.random.randn(*Xgt_out.shape) * 2
    Xgt_out = Xgt_out + measNoiseTruePos

    return Xgt_out, tt