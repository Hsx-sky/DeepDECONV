import torch
from DataLoad.embedding2tensor import dict_pol, dict_polc
from ModelConstruct.transformer_mask0 import mask_pad, mask_tril
from ModelConstruct.transformer_util0 import MultiHead, PositionEmbedding, FullyConnectedOutput, PositionEmbeddingY
from transformer_ModelParaSet import maxLen, embeddingSize, headNum, outputCharacter2, LayerNumber

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 编码器层
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()
        if LayerNumber == 6:
            self.layer_4 = EncoderLayer()
            self.layer_5 = EncoderLayer()
            self.layer_6 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        if LayerNumber == 6:
            x = self.layer_4(x, mask)
            x = self.layer_5(x, mask)
            x = self.layer_6(x, mask)
        return x


# 解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mh1 = MultiHead()
        self.mh2 = MultiHead()

        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.mh2(y, x, x, mask_pad_x)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)

        return y


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()
        if LayerNumber == 6:
            self.layer_4 = DecoderLayer()
            self.layer_5 = DecoderLayer()
            self.layer_6 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        if LayerNumber == 6:
            y = self.layer_4(x, y, mask_pad_x, mask_tril_y)
            y = self.layer_5(x, y, mask_pad_x, mask_tril_y)
            y = self.layer_6(x, y, mask_pad_x, mask_tril_y)
        return y


# 主模型
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y1 = PositionEmbeddingY()
        self.embed_y2 = PositionEmbeddingY()
        self.encoder = Encoder()
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        # self.fc_out1 = torch.nn.Linear(32, addCharacter + outputCharacter1)
        # self.fc_out2 = torch.nn.Linear(32, addCharacter + outputCharacter2)
        self.fc1 = torch.nn.Linear(embeddingSize, embeddingSize//2)
        self.dropout1 = torch.nn.Dropout(0.1)  # 在全连接层后添加Dropout
        self.fc2 = torch.nn.Linear(embeddingSize, embeddingSize//2)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.fc_out1 = torch.nn.Linear(embeddingSize//2, 1)
        self.fc_out2 = torch.nn.Linear(embeddingSize//2, 1)
        self.RELU = torch.nn.ReLU()
    def forward(self, x, y1, y2):
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y1 = mask_tril(y1)
        mask_tril_y2 = mask_tril(y2)

        # x, y1, y2 = self.RELU(x), self.RELU(y1), self.RELU(y2)
        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        x, y1, y2 = self.embed_x(x), self.embed_y1(y1), self.embed_y2(y2)

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x.to(device))

        # 解码层计算
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y1 = self.decoder1(x, y1, mask_pad_x, mask_tril_y1)
        y2 = self.decoder2(x, y2, mask_pad_x, mask_tril_y2)
        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]

        y1 = torch.relu(self.fc1(y1))
        y1 = self.dropout1(y1)  # 应用Dropout
        y2 = torch.relu(self.fc2(y2))
        y2 = self.dropout2(y2)

        y1 = self.fc_out1(y1)
        y2 = self.fc_out2(y2)

        return y1, y2
