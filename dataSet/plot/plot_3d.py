import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_3d_trajectory_black(file_path):
    # 1. 检查文件
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    print(f"正在读取: {file_path}")
    df = pd.read_csv(file_path)

    # 2. 提取数据
    x = df['x']
    y = df['y']
    z = df['z']

    # 3. 创建画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 4. 绘制轨迹 (纯黑色实线)
    # color='black' 设置颜色
    # linewidth=2 设置线条粗细
    ax.plot(x, y, z, color='black', linewidth=2, label='Flight Path')

    # 5. 标记起点和终点 (保留标记以便区分方向)
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], c='green', s=100, marker='^', label='Start')
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], c='red', s=100, marker='x', label='End')

    # 添加文字标签
    ax.text(x.iloc[0], y.iloc[0], z.iloc[0], "  Start", color='green', fontweight='bold')
    ax.text(x.iloc[-1], y.iloc[-1], z.iloc[-1], "  End", color='red', fontweight='bold')

    # 6. 设置坐标轴标签
    ax.set_title(f'3D Flight Trajectory\n(File: {os.path.basename(file_path)})', fontsize=14)
    ax.set_xlabel('East (X) [m]', fontsize=12)
    ax.set_ylabel('North (Y) [m]', fontsize=12)
    ax.set_zlabel('Altitude (Z) [m]', fontsize=12)

    # 7. 强制坐标轴比例一致 (防止变形)
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 请根据你的实际路径修改这里
    csv_path = r'D:\AFS\lunwen\dataSet\processed_data\f16_complex_maneuver.csv'

    plot_3d_trajectory_black(csv_path)