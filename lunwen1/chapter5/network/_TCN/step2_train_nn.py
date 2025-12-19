import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit  # [修改] 引入 sklearn 进行划分
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt

# === 配置 ===
MODEL_FILE = 'imm_param_net.pth'
SCALER_FILE = 'scaler_params.json'

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 30
GRAD_CLIP = 1.0


# 修改 ParamPredictorCNN 类
class Chomp1d(nn.Module):
    """
    TCN 的核心组件：裁剪掉卷积后多出来的 padding，
    确保输出的时间步 t 只依赖于 t 及之前的输入。
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN 的基本残差块：包含两个膨胀卷积层、权重归一化、ReLU、Dropout
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 第一次卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二次卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1x1 卷积用于调整残差连接的维度
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) # 你想要的每层膨胀系数
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # 计算 Padding 以保持因果性
            padding = (kernel_size - 1) * dilation_size

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ParamPredictorTCN(nn.Module):
    def __init__(self, input_dim=9, output_dim=9):
        super(ParamPredictorTCN, self).__init__()

        # 配置 TCN 结构
        # 5层网络，通道数逐渐增加
        # 膨胀系数: 1, 2, 4, 8, 16 -> 感受野 > 90帧
        num_channels = [32, 32, 64, 64, 128]
        kernel_size = 3
        dropout = 0.2

        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)

        # TCN 的输出取最后一个时间步，维度等于最后一层的通道数
        self.linear_dim = num_channels[-1]
        self.use_attn = True
        self.attn_fc = nn.Linear(self.linear_dim, 1)

        self.mlp_head = nn.Sequential(
            nn.Linear(self.linear_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x input shape: (Batch, Seq_Len=90, Dim=9)
        # TCN expects: (Batch, Dim, Seq_Len)
        x = x.transpose(1, 2)

        # Run TCN
        y = self.tcn(x)  # Output: (Batch, 128, 90)

        # 取最后一个时间步 (TCN是因果的，最后一个点包含了全部历史信息)
        if self.use_attn:
            yt = y.transpose(1, 2)  # (B,90,128)
            scores = self.attn_fc(yt).squeeze(-1)  # (B,90)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B,90,1)
            context = torch.sum(yt * weights, dim=1)  # (B,128)
        else:
            context = y[:, :, -1]

        logits = self.mlp_head(context)

        logits = logits.view(-1, 3, 3)
        return F.log_softmax(logits, dim=2)

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # 保证 CUDA 卷积算法一致
    torch.backends.cudnn.benchmark = False


# === [核心修改 2] 标签转换辅助函数 ===
def construct_target_matrix_tensor(params_batch):
    """
    将 (Batch, 6) 的参数标签转换为 (Batch, 3, 3) 的完整概率矩阵。
    假设 6 个参数分别是 [p11, p12, p21, p22, p31, p32]
    """
    # params_batch: (Batch, 6)
    p11, p12 = params_batch[:, 0], params_batch[:, 1]
    p21, p22 = params_batch[:, 2], params_batch[:, 3]
    p31, p32 = params_batch[:, 4], params_batch[:, 5]

    # 计算每行第三个元素，确保非负
    p13 = torch.clamp(1.0 - p11 - p12, min=1e-6)
    p23 = torch.clamp(1.0 - p21 - p22, min=1e-6)
    p33 = torch.clamp(1.0 - p31 - p32, min=1e-6)

    # 堆叠成 3x3 矩阵
    # dim=1 堆叠成行向量，再 stack 成矩阵
    row1 = torch.stack([p11, p12, p13], dim=1)
    row2 = torch.stack([p21, p22, p23], dim=1)
    row3 = torch.stack([p31, p32, p33], dim=1)

    # Result: (Batch, 3, 3)
    target_matrix = torch.stack([row1, row2, row3], dim=1)

    # 再次归一化确保和为1 (防止浮点误差)
    target_matrix = target_matrix / target_matrix.sum(dim=2, keepdim=True)

    return target_matrix


def plot_training_results(history, model, val_loader, device):
    """
    更新后的绘图：绘制 KL Divergence Loss 和 矩阵元素的拟合情况
    """
    print(">>> 正在生成详细训练结果图表...")
    model.eval()
    all_preds_probs = []
    all_targets_probs = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 模型输出 log_prob，转回概率
            log_preds = model(batch_x)
            preds = torch.exp(log_preds)

            # 目标转为矩阵
            targets = construct_target_matrix_tensor(batch_y)

            all_preds_probs.append(preds.cpu().numpy())
            all_targets_probs.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds_probs, axis=0)  # (N, 3, 3)
    all_targets = np.concatenate(all_targets_probs, axis=0)  # (N, 3, 3)

    # 1. Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss (KL)')
    plt.plot(history['val_loss'], label='Val Loss (KL)', linestyle='--')
    plt.title('Training & Validation Loss (KL Divergence)')
    plt.xlabel('Epoch')
    plt.ylabel('KL Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. 矩阵元素散点图 (9个子图)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    labels = [f"P{i + 1}{j + 1}" for i in range(3) for j in range(3)]

    for i in range(9):
        row, col = i // 3, i % 3
        ax = axes[i]

        y_true = all_targets[:, row, col]
        y_pred = all_preds[:, row, col]

        ax.scatter(y_true, y_pred, alpha=0.05, s=2, c='blue')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # 对角线

        ax.set_title(f'{labels[i]} (True vs Pred)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Transition Matrix Element-wise Prediction', fontsize=16)
    plt.tight_layout()
    plt.show()


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
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载并合并数据
    # 这里会自动寻找 part0.npz, part1.npz, part2.npz ...
    try:
        X_raw, Y_raw, G_raw = load_and_merge_data(os.path.join('../npz', 'training_data_part*.npz'))
        # X_raw, Y_raw, G_raw = load_and_merge_data(os.path.join('npz_pos', 'training_data_part*.npz'))
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
    model = ParamPredictorTCN(input_dim=9, output_dim=9).to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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
            log_probs = model(batch_x)
            with torch.no_grad():
                target_probs = construct_target_matrix_tensor(batch_y)
            loss = criterion(log_probs, target_probs)
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
                log_probs = model(batch_x)
                target_probs = construct_target_matrix_tensor(batch_y)
                loss = criterion(log_probs, target_probs)
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
        if avg_val_loss < best_val_loss - 1e-5:
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