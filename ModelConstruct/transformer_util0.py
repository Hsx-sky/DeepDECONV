import math
import numpy as np
import torch
from transformer_ModelParaSet import maxLen, embeddingSize, headNum
torch.cuda.set_device(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 注意力计算函数
def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 50, 8]

    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    score /= 8 ** 0.5

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask.to(device), -float('inf'))
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, maxLen, embeddingSize)

    return score


# 多头注意力计算层
class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(embeddingSize, embeddingSize).to(device)
        self.fc_K = torch.nn.Linear(embeddingSize, embeddingSize).to(device)
        self.fc_V = torch.nn.Linear(embeddingSize, embeddingSize).to(device)

        self.out_fc = torch.nn.Linear(embeddingSize, embeddingSize).to(device)

        self.norm = torch.nn.LayerNorm(normalized_shape=embeddingSize, elementwise_affine=True).to(device)

        self.dropout = torch.nn.Dropout(p=0.1).to(device)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V = [b, 50, 32]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, maxLen, headNum, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, maxLen, headNum, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, maxLen, headNum, 8).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score


# 位置编码层
class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(maxLen, embeddingSize)
        for i in range(maxLen):
            for j in range(embeddingSize):
                pe[i, j] = get_pe(i, j, embeddingSize)
        pe = pe.unsqueeze(0)
        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层 1+65535+3
        # self.embed = torch.nn.Embedding(1004, 32).to(device)
        self.embed = torch.nn.Linear(4, embeddingSize).to(device)
        # self.embed = torch.nn.Linear(1, 32).to(device)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]
        # x = x.unsqueeze(-1)  # 调整 x 的形状为 (batch_size, num_features, 1)
        if x.ndimension() == 1:
            x = x.unsqueeze(0)

        if x.size(-1) == 2:
            layer_norm = torch.nn.LayerNorm(normalized_shape=[x.size(1), 2]).to(device)
            log = torch.log(x + 1)
            x = torch.cat((layer_norm(x.to(device)),
                             layer_norm(log.to(device))), dim=2)
        else:
            layer_norm = torch.nn.LayerNorm(normalized_shape=x.size(1)).to(device)
            diff = torch.diff(x.to(device), n=1, dim=1, prepend=torch.zeros(x.size(0), 1).to(device), append=None)
            log = torch.log(x+1)
            x2 = x ** 2
            x = torch.stack((layer_norm(x.to(device)), layer_norm(diff.to(device)),
                             layer_norm(log.to(device)), layer_norm(x2.to(device))), dim=2)

        embed = self.embed(x.to(device))  # 现在 x 的形状是 (batch_size, num_features, 32)

        # 词编码和位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed = embed + self.pe.to(device)
        return embed

class PositionEmbeddingY(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(maxLen, embeddingSize)
        for i in range(maxLen):
            for j in range(embeddingSize):
                pe[i, j] = get_pe(i, j, embeddingSize)
        pe = pe.unsqueeze(0)
        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层 1+65535+3
        # self.embed = torch.nn.Embedding(1004, 32).to(device)
        self.embed = torch.nn.Linear(1, embeddingSize).to(device)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]
        x = x.unsqueeze(-1)  # 调整 x 的形状为 (batch_size, num_features, 1)
        embed = self.embed(x.to(device))  # 现在 x 的形状是 (batch_size, num_features, 32)

        # 词编码和位置编码相加
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32]
        embed = embed + self.pe.to(device)
        return embed

# 全连接输出层
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=embeddingSize, out_features=embeddingSize*2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embeddingSize*2, out_features=embeddingSize),
            torch.nn.Dropout(p=0.1),
        ).to(device)

        self.norm = torch.nn.LayerNorm(normalized_shape=embeddingSize,
                                       elementwise_affine=True).to(device)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out
