import pandas as pd
import plotly.graph_objects as go
import os


def plot_interactive_HD(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    print("读取数据中...")
    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 核心优化：创建更清晰的 3D 轨迹
    # ---------------------------------------------------------
    fig = go.Figure()

    # 1. 绘制主轨迹线
    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='lines',
        name='飞行轨迹',
        line=dict(
            color=df['z'],  # 根据高度上色
            colorscale='Turbo',  # 【优化】使用高对比度色谱 (Turbo 比 Jet 更清晰)
            width=6,  # 【优化】线条加粗，看得更清楚
            showscale=True,  # 显示颜色条
            colorbar=dict(title='高度 (m)', x=0.85)
        ),
        # 鼠标悬停显示具体信息
        hovertemplate='<b>X</b>: %{x:.1f}m<br><b>Y</b>: %{y:.1f}m<br><b>Z</b>: %{z:.1f}m<extra></extra>'
    ))

    # 2. 绘制“地面投影” (帮助理解空间位置)
    # 这一步会在 Z轴的底部画出灰色的影子，极大地增强立体感
    z_min = df['z'].min()
    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=[z_min - 100] * len(df),  # 投影在最低点下方一点
        mode='lines',
        name='地面投影',
        line=dict(color='gray', width=3),
        opacity=0.4,  # 半透明
        hoverinfo='skip'
    ))

    # 3. 标记起点 (加大图标)
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
        mode='markers+text',
        name='起点',
        marker=dict(size=12, color='#00FF00', symbol='diamond'),  # 亮绿色菱形
        text=["START"],
        textposition="top center",
        textfont=dict(color='#00FF00', size=14, family="Arial Black")
    ))

    # 4. 标记终点 (加大图标)
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
        mode='markers+text',
        name='终点',
        marker=dict(size=12, color='#FF0000', symbol='x'),  # 红色叉号
        text=["END"],
        textposition="top center",
        textfont=dict(color='#FF0000', size=14, family="Arial Black")
    ))

    # ---------------------------------------------------------
    # 布局优化：深色背景 + 比例锁定
    # ---------------------------------------------------------
    fig.update_layout(
        title={
            'text': f"3D 飞行轨迹可视化: {os.path.basename(file_path)}",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color='white')
        },
        template='plotly_dark',  # 【优化】使用深色背景，线条更显眼
        width=1200,
        height=900,
        margin=dict(r=0, l=0, b=0, t=50),
        scene=dict(
            xaxis=dict(title='东向距离 (X)', gridcolor='gray', showbackground=True, backgroundcolor='black'),
            yaxis=dict(title='北向距离 (Y)', gridcolor='gray', showbackground=True, backgroundcolor='black'),
            zaxis=dict(title='飞行高度 (Z)', gridcolor='gray', showbackground=True, backgroundcolor='black'),
            # 【关键】强制 xyz 比例 1:1:1，防止轨迹被压扁
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)  # 默认视角调整得舒服一点
            )
        )
    )

    # 显示并保存
    output_html = file_path.replace('.csv', '_3d_HD.html')
    fig.write_html(output_html)
    print(f"✅ 高清图表已生成: {output_html}")

    # 尝试自动打开浏览器 (如果运行环境支持)
    try:
        fig.show()
    except:
        print("请在文件夹中打开生成的 HTML 文件查看。")


if __name__ == "__main__":
    # 请确保路径正确
    csv_path = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver.csv'
    plot_interactive_HD(csv_path)