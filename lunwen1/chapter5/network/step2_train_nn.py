import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit  # [修改] 引入 sklearn 进行划分
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt  # [新增] 记得引入

# === 配置 ===
MODEL_FILE = 'imm_param_net.pth'
SCALER_FILE = 'scaler_params.json'

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 15
GRAD_CLIP = 1.0


class ParamPredictorLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=6):
        super(ParamPredictorLSTM, self).__init__()
        # batch_first=True -> input shape (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # 输出归一化到 (0, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def plot_training_results(history, model, val_loader, device):
    """
    [新增] 专门的画图函数
    1. Loss 曲线
    2. 真值 vs 预测 散点图
    """
    print(">>> 正在生成训练结果图表...")

    # 1. 绘制 Loss 曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (SmoothL1)')
    plt.legend()
    plt.grid(True)

    # 2. 绘制 预测 vs 真值 散点图
    # 为了画图，我们需要跑一遍验证集，把所有预测值拿出来
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 展平数据：不区分具体参数，看整体拟合度
    flat_preds = all_preds.flatten()
    flat_targets = all_targets.flatten()

    plt.subplot(1, 2, 2)
    # alpha=0.05 让密集的点显示得更清楚
    plt.scatter(flat_targets, flat_preds, alpha=0.05, s=1, c='blue')

    # 画一条 y=x 的参考线
    lims = [0, 1]
    plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label='Ideal Fit')

    plt.title('Ground Truth vs Prediction (Val Set)')
    plt.xlabel('Ground Truth (Optimizer Output)')
    plt.ylabel('Prediction (NN Output)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print(">>> 图表已生成。")


def load_and_merge_data(file_pattern):
    """
    加载所有分块数据并合并，同时处理 Group ID 偏移
    """
    file_list = glob.glob(file_pattern)
    if not file_list:
        raise FileNotFoundError("未找到任何 training_data_part*.npz 文件")

    all_X, all_Y, all_G = [], [], []
    group_offset = 0

    print(f"发现 {len(file_list)} 个数据文件，开始合并...")

    for fname in sorted(file_list):
        data = np.load(fname)
        X = data['X']
        Y = data['Y']
        # 兼容性处理：如果没重新跑 Step 1，可能没有 G，这里会报错提醒你
        if 'G' not in data:
            raise ValueError(f"文件 {fname} 中缺少 Group ID ('G')。请重新运行 Step 1 生成带标签的数据。")
        G = data['G']

        # [关键] 偏移 Group ID，确保不同 Part 文件的 ID 不冲突
        # 例如 Part1 有 10 个文件(ID 0-9)，Part2 的 ID 0 应该变成 10
        G_offset = G + group_offset

        all_X.append(X)
        all_Y.append(Y)
        all_G.append(G_offset)

        # 更新偏移量 (当前最大ID + 1)
        group_offset += (np.max(G) + 1)
        print(f"  -> 已加载 {fname}: {len(X)} 样本, Group ID 范围 [{np.min(G_offset)}, {np.max(G_offset)}]")

    return np.concatenate(all_X), np.concatenate(all_Y), np.concatenate(all_G)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载并合并数据
    # 这里会自动寻找 part0.npz, part1.npz, part2.npz ...
    try:
        X_raw, Y_raw, G_raw = load_and_merge_data('training_data_part*.npz')
        print(f"数据合并完毕: X={X_raw.shape}, Y={Y_raw.shape}, Groups={len(np.unique(G_raw))}")
    except Exception as e:
        print(e)
        return

    # 2. 按 Group (轨迹) 进行划分
    # n_splits=1 表示只分一次，test_size=0.1 表示 10% 的轨迹作为验证集
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    # 获取划分索引
    train_idx, val_idx = next(gss.split(X_raw, Y_raw, groups=G_raw))

    X_train_raw = X_raw[train_idx]
    Y_train = Y_raw[train_idx]

    X_val_raw = X_raw[val_idx]
    Y_val = Y_raw[val_idx]

    print(f"划分完成 (按轨迹):")
    print(f"  训练集: {len(X_train_raw)} 样本 (来自 {len(np.unique(G_raw[train_idx]))} 条轨迹)")
    print(f"  验证集: {len(X_val_raw)} 样本 (来自 {len(np.unique(G_raw[val_idx]))} 条轨迹)")
    print("-" * 40)

    # 3. 归一化 (逻辑不变，仅使用训练集统计量)
    mean_X = np.mean(X_train_raw, axis=(0, 1))
    std_X = np.std(X_train_raw, axis=(0, 1)) + 1e-8

    X_train_norm = (X_train_raw - mean_X) / std_X
    X_val_norm = (X_val_raw - mean_X) / std_X

    # 保存参数
    scaler_params = {'mean': mean_X.tolist(), 'std': std_X.tolist()}
    with open(SCALER_FILE, 'w') as f:
        json.dump(scaler_params, f)

    # 4. 创建 Dataset
    train_dataset = TensorDataset(torch.from_numpy(X_train_norm).float(), torch.from_numpy(Y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val_norm).float(), torch.from_numpy(Y_val).float())

    # ================= [核心修改结束] =================

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 模型初始化
    model = ParamPredictorLSTM().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # [新增] 初始化 history 字典用于记录
    history = {'train_loss': [], 'val_loss': []}

    # 7. 训练循环
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # [新增] 记录历史数据
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        # --- 早停检查 ---
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"早停触发! 验证集 Loss 在 {PATIENCE} 个 epoch 内未下降。停止训练。")
                break

    print(f"训练结束。最佳模型已保存，Val Loss: {best_val_loss:.6f}")

    # ================= [新增调用] =================
    # 加载最佳模型参数用于画图 (防止画的是最后一次早停前的过拟合模型)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))

    # 调用画图函数
    plot_training_results(history, model, val_loader, device)

if __name__ == '__main__':
    main()