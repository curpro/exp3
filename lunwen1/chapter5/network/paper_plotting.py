import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # 建议安装 pip install seaborn，画图更美观
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


def set_paper_style():
    """设置学术论文通用的绘图风格"""
    plt.style.use('default')  # 或者 'seaborn-whitegrid'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


# ================= 训练阶段图表 =================

def plot_residuals_distribution(y_true, y_pred, labels=None):
    """
    绘制预测残差直方图 (检查模型是否有系统性偏差)
    y_true, y_pred: shape (N, 3, 3) 或者是展平后的
    """
    residuals = (y_true - y_pred).flatten()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='navy', bins=50, alpha=0.6)
    plt.axvline(0, color='r', linestyle='--', linewidth=1.5)

    plt.title("Prediction Residuals Distribution (Target - Prediction)")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ================= 推理阶段图表 =================

def plot_trajectory_projections(true_pos, est_pos_fix, est_pos_adapt, title_suffix=""):
    """
    [解决看不清问题] 绘制 3D 轨迹 + 三个 2D 投影面 (XY, XZ, YZ)
    """
    fig = plt.figure(figsize=(16, 12))

    # 子图1: 3D 视图
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 'k-', lw=1.5, label='Ground Truth', alpha=0.7)
    ax3d.plot(est_pos_fix[:, 0], est_pos_fix[:, 1], est_pos_fix[:, 2], 'b--', lw=1, label='Fixed IMM', alpha=0.5)
    ax3d.plot(est_pos_adapt[:, 0], est_pos_adapt[:, 1], est_pos_adapt[:, 2], 'r-', lw=1.5, label='NN-IMM')
    ax3d.set_title(f"3D View {title_suffix}")
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.legend()

    # 子图2: XY 平面 (俯视图)
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(true_pos[:, 0], true_pos[:, 1], 'k-', lw=1.5, label='Truth')
    ax_xy.plot(est_pos_fix[:, 0], est_pos_fix[:, 1], 'b--', lw=1, label='Fixed', alpha=0.6)
    ax_xy.plot(est_pos_adapt[:, 0], est_pos_adapt[:, 1], 'r-', lw=1.5, label='NN-IMM')
    ax_xy.set_title("Top View (X-Y Plane)")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, linestyle=':')
    ax_xy.axis('equal')  # 保持比例一致

    # 子图3: XZ 平面 (侧视图)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(true_pos[:, 0], true_pos[:, 2], 'k-', lw=1.5)
    ax_xz.plot(est_pos_fix[:, 0], est_pos_fix[:, 2], 'b--', lw=1, alpha=0.6)
    ax_xz.plot(est_pos_adapt[:, 0], est_pos_adapt[:, 2], 'r-', lw=1.5)
    ax_xz.set_title("Side View (X-Z Plane)")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.grid(True, linestyle=':')

    # 子图4: YZ 平面 (后视图)
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(true_pos[:, 1], true_pos[:, 2], 'k-', lw=1.5)
    ax_yz.plot(est_pos_fix[:, 1], est_pos_fix[:, 2], 'b--', lw=1, alpha=0.6)
    ax_yz.plot(est_pos_adapt[:, 1], est_pos_adapt[:, 2], 'r-', lw=1.5)
    ax_yz.set_title("Back View (Y-Z Plane)")
    ax_yz.set_xlabel("Y (m)")
    ax_yz.set_ylabel("Z (m)")
    ax_yz.grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()


def plot_error_heatmap(true_pos, est_pos, title="Trajectory Error Heatmap"):
    """
    [高级可视化] 绘制轨迹，颜色深浅代表误差大小
    """
    error = np.linalg.norm(true_pos - est_pos, axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 散点图，c=error 控制颜色
    p = ax.scatter(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
                   c=error, cmap='jet', s=5, alpha=0.8)

    fig.colorbar(p, ax=ax, label='Position Error (m)')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_statistical_boxplot(pos_err_fix, pos_err_adapt, vel_err_fix, vel_err_adapt):
    """
    [升级版] 同时绘制位置和速度的误差箱线图 (双子图)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- 左图：位置误差 ---
    data_pos = [pos_err_fix, pos_err_adapt]
    labels = ['Fixed IMM', 'NN-IMM (Ours)']

    bplot1 = ax1.boxplot(data_pos, patch_artist=True, labels=labels, showfliers=False, widths=0.5)

    # 颜色填充
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Position Error Distribution')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 计算位置改善率 (中位数)
    med_pos_fix = np.median(pos_err_fix)
    med_pos_adapt = np.median(pos_err_adapt)
    improv_pos = (med_pos_fix - med_pos_adapt) / med_pos_fix * 100
    ax1.text(1.5, med_pos_fix, f"Median Improv:\n{improv_pos:.1f}%",
             ha='center', va='bottom', color='darkred', fontweight='bold')

    # --- 右图：速度误差 ---
    data_vel = [vel_err_fix, vel_err_adapt]

    bplot2 = ax2.boxplot(data_vel, patch_artist=True, labels=labels, showfliers=False, widths=0.5)

    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('Velocity Error (m/s)')
    ax2.set_title('Velocity Error Distribution')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 计算速度改善率
    med_vel_fix = np.median(vel_err_fix)
    med_vel_adapt = np.median(vel_err_adapt)
    improv_vel = (med_vel_fix - med_vel_adapt) / med_vel_fix * 100
    ax2.text(1.5, med_vel_fix, f"Median Improv:\n{improv_vel:.1f}%",
             ha='center', va='bottom', color='darkred', fontweight='bold')

    plt.suptitle('Statistical Error Distribution Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_computational_cost(nn_times, bo_times, bo_interval=20):
    """
    绘制计算开销对比图
    nn_times: list/array, 每次 NN 推理的耗时 (ms)
    bo_times: list/array, 每次 BO 优化的耗时 (ms)
    bo_interval: int, 优化的间隔帧数
    """
    avg_nn = np.mean(nn_times)
    avg_bo = np.mean(bo_times)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # === 左图：平均耗时对比 (Bar Chart) ===
    labels = ['NN-IMM (Inference)', 'BO-IMM (Optimization)']
    times = [avg_nn, avg_bo]
    colors = ['#d62728', '#1f77b4']  # NN用红，BO用蓝

    bars = ax1.bar(labels, times, color=colors, alpha=0.8, width=0.5)

    # 在柱状图上方标注具体数值
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{t:.2f} ms',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Avg. Time Cost per Update (ms)')
    ax1.set_title('Computational Cost Comparison (Log Scale)')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # [关键] 使用对数坐标，因为 BO 通常比 NN 慢几个数量级
    ax1.set_yscale('log')

    # === 右图：时间轴延迟分布 (Timeline) ===
    # 模拟构建一个时间轴来展示“脉冲”
    # 假设一共运行了 len(nn_times) * bo_interval 帧
    n_updates = min(len(nn_times), len(bo_times))
    frames = np.arange(n_updates) * bo_interval

    ax2.plot(frames, nn_times[:n_updates], 'r-o', label='NN-IMM (Ours)', markersize=4, linewidth=1.5)
    ax2.plot(frames, bo_times[:n_updates], 'b-s', label='BO-IMM', markersize=4, linewidth=1, alpha=0.6)

    ax2.set_xlabel('Simulation Frame')
    ax2.set_ylabel('Processing Time (ms)')
    ax2.set_title('Real-time Processing Latency')
    ax2.legend()
    ax2.grid(True, linestyle=':')
    ax2.set_yscale('log')  # 同样使用对数坐标

    plt.suptitle("Real-time Feasibility Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_3d_zoom_with_context(true_pos, est_fix, est_adapt, start_frame, end_frame):
    """
    [Final Correction]
    1. Global trajectory is solid black line.
    2. Arrow points EXACTLY at the red segment with 'Expand' text.
    """
    fig = plt.figure(figsize=(16, 8))

    # ==========================================
    # 1. 左图：全局视角 (Global View)
    # ==========================================
    ax_global = fig.add_subplot(1, 2, 1, projection='3d')

    # [修改 1] 全局轨迹：黑色实线 (Solid Black), 清晰可见
    ax_global.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
                   color='black', linestyle='-', linewidth=1.0, alpha=0.6, label='Global Trajectory')

    # [修改 2] 高亮区域：红色粗线
    segment_true = true_pos[start_frame:end_frame]
    ax_global.plot(segment_true[:, 0], segment_true[:, 1], segment_true[:, 2],
                   color='red', linewidth=4, label='Zoom Region')

    # 起点终点
    ax_global.scatter(true_pos[0, 0], true_pos[0, 1], true_pos[0, 2], c='g', s=40, label='Start')
    ax_global.scatter(true_pos[-1, 0], true_pos[-1, 1], true_pos[-1, 2], c='k', s=40, label='End')

    # 设置标题和轴
    ax_global.set_title("Global Context", fontsize=14, fontweight='bold')
    ax_global.set_xlabel('X')
    ax_global.set_ylabel('Y')
    ax_global.set_zlabel('Z')

    # 调整视角 (关键：视角决定了 3D 点在 2D 屏幕上的位置)
    ax_global.view_init(elev=30, azim=-45)
    ax_global.legend(loc='upper right')

    # ==========================================
    # 2. 右图：3D 局部放大 (3D Zoom-in)
    # ==========================================
    ax_zoom = fig.add_subplot(1, 2, 2, projection='3d')

    seg_fix = est_fix[start_frame:end_frame]
    seg_adapt = est_adapt[start_frame:end_frame]

    # 右图也用实线
    ax_zoom.plot(segment_true[:, 0], segment_true[:, 1], segment_true[:, 2],
                 'k-', linewidth=5, alpha=0.3, label='Ground Truth')

    ax_zoom.plot(seg_fix[:, 0], seg_fix[:, 1], seg_fix[:, 2],
                 color='blue', linestyle='--', linewidth=2,
                 marker='x', markersize=6, markevery=5, label='Fixed IMM')

    ax_zoom.plot(seg_adapt[:, 0], seg_adapt[:, 1], seg_adapt[:, 2],
                 color='red', linestyle='-', linewidth=3,
                 marker='o', markersize=4, markevery=5, label='NN-IMM (Ours)')

    ax_zoom.set_title(f"Detail Zoom-in (Frame {start_frame}-{end_frame})", fontsize=14, fontweight='bold')
    ax_zoom.set_xlabel('X')
    ax_zoom.set_ylabel('Y')
    ax_zoom.set_zlabel('Z')
    ax_zoom.legend(loc='best')

    # ==========================================
    # [修改 3] 精确的箭头标注 (The Exact Arrow)
    # ==========================================
    # 我们先强制渲染一次，以便计算 3D 点在屏幕上的投影坐标
    fig.canvas.draw()

    # 1. 获取红线中心的 3D 坐标
    mid_idx = len(segment_true) // 2
    x, y, z = segment_true[mid_idx]

    # 2. 将 3D 坐标投影到 2D 屏幕坐标
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax_global.get_proj())

    # 3. 在 2D 屏幕坐标上画箭头
    # xy=(x2, y2) 是箭头尖端的位置 (即红线中心)
    # xytext=(-30, 30) 是文字偏移量 (箭头尾巴的位置)
    ax_global.annotate(
        "ZoomIn",
        xy=(x2, y2), xycoords='data',
        xytext=(-40, 40), textcoords='offset points',  # 文字在左上方
        fontsize=12, fontweight='bold', color='red',
        arrowprops=dict(arrowstyle="->", color='red', linewidth=2, mutation_scale=15)
    )

    plt.tight_layout()
    plt.show()