import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from sklearn.manifold import TSNE

# ================= 配置 =================
MODEL_FILE = 'net/n/3_MLP/imm_param_net.pth'
SCALER_FILE = 'net/n/3_MLP/scaler_params.json'
DATA_PATTERN = os.path.join('npz', 'training_data_part*.npz')
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    'Rel Pos X', 'Rel Pos Y', 'Rel Pos Z',
    'Vel X', 'Vel Y', 'Vel Z',
    'Acc X', 'Acc Y', 'Acc Z'
]


# === 模型定义 (保持不变) ===
class ParamPredictorMLP(nn.Module):
    def __init__(self, seq_len=90, input_dim=9):
        super(ParamPredictorMLP, self).__init__()
        self.input_flat_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        b, s, f = x.shape
        x = x.reshape(b, -1)
        logits = self.net(x)
        logits = logits.view(-1, 3, 3)
        temperature = 2.0
        return torch.log_softmax(logits / temperature, dim=2)

    # [新增] 提取倒数第二层特征用于 t-SNE
    def get_embedding(self, x):
        b, s, f = x.shape
        x = x.reshape(b, -1)
        # 手动前向传播直到倒数第二层 (32维)
        out = x
        for i, layer in enumerate(self.net):
            out = layer(out)
            # 这里的索引取决于你的 Sequential 定义
            # Linear(flat, 128) -> BN -> ReLU -> Drop -> Linear(128, 32) -> BN -> ReLU -> Drop -> Linear(32, 9)
            # 我们想取 Linear(128, 32) 之后或者 ReLU 之后的值
            # 0:Lin, 1:BN, 2:ReLU, 3:Drop, 4:Lin(to 32), 5:BN, 6:ReLU
            if i == 6:
                return out
        return out


# === 数据加载 (保持不变) ===
def load_data():
    files = glob.glob(DATA_PATTERN)
    if not files:
        # 为了演示，如果没有文件生成一些假数据
        print("警告: 未找到数据文件，生成随机数据用于测试代码功能...")
        X = np.random.randn(1000, 90, 9).astype(np.float32)
        # 生成假标签: 随机概率
        Y = np.random.rand(1000, 6).astype(np.float32)
        return torch.FloatTensor(X), torch.FloatTensor(Y)

    all_X, all_Y = [], []
    print(f"Loading data from {len(files)} files...")
    for f in files:
        data = np.load(f)
        all_X.append(data['X'])
        all_Y.append(data['Y'])

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE, 'r') as f:
            scaler = json.load(f)
        mean = np.array(scaler['mean'], dtype=np.float32)
        std = np.array(scaler['std'], dtype=np.float32)
        X = (X - mean) / std

    return torch.FloatTensor(X), torch.FloatTensor(Y)


# === [核心新增 1] 梯度显著性分析 (Saliency Map) ===
def analyze_saliency_map(model, X, Y):
    print("\n>>> 1. 正在计算时空显著性图 (Saliency Map)...")
    model.eval()

    # 随机采样一部分数据
    idx = np.random.choice(len(X), min(200, len(X)), replace=False)
    inputs = X[idx].to(DEVICE)
    inputs.requires_grad = True  # 关键：开启输入梯度

    # 前向传播
    log_probs = model(inputs)  # (B, 3, 3)

    # 我们不仅想知道loss对输入的梯度，更想知道“预测结果”对输入的梯度
    # 这里我们最大化预测概率最大的那个类别的输出
    # 简单的做法：对 output 求 sum 也可以得到梯度幅度
    target_score = log_probs.sum()

    # 反向传播
    model.zero_grad()
    target_score.backward()

    # 获取梯度：(B, 90, 9)
    grads = inputs.grad.data.cpu().numpy()

    # 取绝对值并平均 (Magnitude of gradients)
    # (90, 9) -> 时间步 x 特征
    saliency = np.mean(np.abs(grads), axis=0)

    # 绘图
    plt.figure(figsize=(12, 6))
    # 转置一下：行是特征，列是时间
    sns.heatmap(saliency.T, cmap='viridis', xticklabels=10, yticklabels=FEATURE_NAMES)
    plt.xlabel('Time Step (0-90)')
    plt.ylabel('Features')
    plt.title('Saliency Map: Which time steps matter most?\n(Brighter = Higher Gradient Impact)')
    plt.tight_layout()
    plt.show()

    return saliency


# === [核心新增 2] t-SNE 隐空间可视化 ===
def analyze_tsne_distribution(model, X, Y):
    print("\n>>> 2. 正在计算 t-SNE 分布 (Latent Space Visualization)...")

    # 采样 1000 个点
    n_samples = min(1000, len(X))
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[idx].to(DEVICE)
    Y_sub = Y[idx].numpy()  # (N, 6)

    model.eval()
    with torch.no_grad():
        # 获取倒数第二层的特征向量 (High-level features)
        embeddings = model.get_embedding(X_sub).cpu().numpy()  # (N, 32)

    # 定义标签类别：我们要看模型是否区分了不同的主导模型
    # 假设 IMM 是 3 模型 (CV, CT, CA)，我们看对角线谁最大
    # Y_sub 是参数 P11, P12... 我们需要还原对角线
    p11 = Y_sub[:, 0]
    p22 = Y_sub[:, 3]
    # P33 近似为 1 - P31 - P32 (这里简化处理，假设主要由前两个决定，或者直接比较前两个)
    # 为了简化可视化，我们根据 P11 和 P22 的大小着色，或者假设有某种主导模式

    # 构造一个简单的类别标签：根据 P11 (保持CV) 和 P22 (保持CT) 的强度
    # 0: High P11 (CV Dominated), 1: High P22 (CT Dominated), 2: Mixed/Other
    labels = []
    for i in range(n_samples):
        if p11[i] > 0.8:
            labels.append(0)  # CV
        elif p22[i] > 0.8:
            labels.append(1)  # CT (假设)
        else:
            labels.append(2)  # Transition/Mixed

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Dominant Mode (0:Model1, 1:Model2, 2:Mix)')
    plt.title('t-SNE Visualization of Feature Space (Layer before output)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


# === [优化] 统计相关性矩阵 ===
def analyze_statistical_correlation(X, Y):
    print("\n>>> 3. 正在绘制统计特征相关性热力图...")

    # 计算序列的统计特征
    X_np = X.numpy()  # (N, 90, 9)
    X_mean = np.mean(X_np, axis=1)  # (N, 9) 平均值
    X_std = np.std(X_np, axis=1)  # (N, 9) 标准差

    # 拼接 Mean 和 Std
    # 特征名扩展
    feat_names_ext = [f'{n}_mean' for n in FEATURE_NAMES] + [f'{n}_std' for n in FEATURE_NAMES]
    data_feat = np.hstack([X_mean, X_std])  # (N, 18)

    Y_np = Y.numpy()
    param_names = ['P11', 'P12', 'P21', 'P22', 'P31', 'P32']

    # 计算相关性 (feat vs params)
    full_data = np.hstack([data_feat, Y_np])
    corr_matrix = np.corrcoef(full_data, rowvar=False)

    # 截取 (Feature rows, Param cols)
    # Feat indices: 0~17, Param indices: 18~23
    corr_sub = corr_matrix[:18, 18:]

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=param_names, yticklabels=feat_names_ext)
    plt.title("Correlation: Sequence Statistics (Mean/Std) vs IMM Parameters")
    plt.tight_layout()
    plt.show()


def main():
    # 1. 准备模型
    model = ParamPredictorMLP().to(DEVICE)
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
            print(f"模型 {MODEL_FILE} 加载成功。")
        except:
            print("模型加载失败，将使用随机初始化模型运行（仅演示功能）")
    else:
        print(f"提示：找不到模型文件 {MODEL_FILE}，使用随机初始化模型。")

    # 2. 准备数据
    X, Y = load_data()
    print(f"数据形状: X={X.shape}, Y={Y.shape}")

    # 3. 运行高级分析
    # (A) 梯度显著性图 - 可以在论文中分析模型的时间敏感性
    analyze_saliency_map(model, X, Y)

    # (B) t-SNE 分布 - 可以在论文中分析模型是否学习到了模式的差异
    analyze_tsne_distribution(model, X, Y)

    # (C) 统计相关性 - 比单纯看最后一帧更有物理意义
    analyze_statistical_correlation(X, Y)


if __name__ == '__main__':
    main()