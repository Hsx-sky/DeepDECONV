import torch
from ModelConstruct.OrdinalRegressionLoss import OrdinalRegressionLoss
from DataLoad.embedding2tensor import dict_pol, dict_polc
from ModelConstruct.transformer_mask0 import mask_pad, mask_tril
from ModelConstruct.transformer2head_model0 import Transformer
from transformer_ModelParaSet import maxLen, outputCharacter2
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
model = Transformer().to(device)
# 预测函数
def predict(x, model_in):
    ord_loss = OrdinalRegressionLoss(outputCharacter2, train_cutpoints=False)
    model_in.eval()
    model_in.to(device)
    mask_pad_x = mask_pad(x)

    # 初始化输出
    target1 = [0]+[0] * (maxLen-1)
    target1 = torch.FloatTensor(target1).unsqueeze(0)
    target2 = [0]+[0] * (maxLen-1)
    target2 = torch.FloatTensor(target2).unsqueeze(0)
    # x编码,添加位置信息
    x = model_in.embed_x(x)
    # 编码层计算,维度不变
    x = model_in.encoder(x, mask_pad_x)
    # 遍历生成
    for i in range(maxLen-1):
        y1 = target1
        y2 = target2
        mask_tril_y1 = mask_tril(y1)
        mask_tril_y2 = mask_tril(y2)
        # y编码,添加位置信息
        y1 = model_in.embed_y1(y1)
        y2 = model_in.embed_y2(y2)
        # 解码层计算,维度不变
        y1 = model_in.decoder1(x, y1, mask_pad_x, mask_tril_y1)
        y2 = model_in.decoder2(x, y2, mask_pad_x, mask_tril_y2)
        # 输出
        y1 = torch.relu(model_in.fc1(y1))
        y1 = model_in.dropout1(y1)  # 应用Dropout
        y2 = torch.relu(model_in.fc2(y2))
        y2 = model_in.dropout2(y2)
        out1 = model_in.fc_out1(y1)
        out2 = model_in.fc_out2(y2)
        # 取出当前词的输出
        out1 = out1[:, i, :]
        out2 = out2[:, i, :]
        out2 = out2.squeeze(dim=1)
        # 取出分类结果
        loss2, likelihoods, out2 = ord_loss(out2.to(device), label=None)
        # 以当前词预测下一个词,填到结果中
        target1[:, i + 1] = out1
        target2[:, i + 1] = out2
    return target1, target2

def r2_score(predictions, targets):
    # 计算总平方和（总变异）
    total_variance = torch.sum((targets - torch.mean(targets)) ** 2)
    # 计算残差平方和（剩余变异）
    residual_variance = torch.sum((targets - predictions) ** 2)
    # 计算 R²
    return 1 - (residual_variance / total_variance)


def train(loader, nEpoch, saveName, IncrementalLearning, learningRate, stepGamma):
    loss_func = torch.nn.MSELoss()
    ord_loss = OrdinalRegressionLoss(outputCharacter2, train_cutpoints=False)
    optim = torch.optim.Adam(model.parameters(), lr=learningRate)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=stepGamma)

    if not IncrementalLearning == '':
        model.load_state_dict(torch.load('../Model/' + IncrementalLearning + '.pth'))
    best_r2 = 0
    for epoch in range(nEpoch):
        for i, (x, y1, y2) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{nEpoch}")):
            x.to(device)
            y1.to(device)
            y2.to(device)
            pred1, pred2 = model(x, y1[:, :-1], y2[:, :-1])
            pred1 = pred1.reshape(-1)
            pred2 = pred2.reshape(-1)
            y1 = y1[:, 1:].reshape(-1)
            y2 = y2[:, 1:].reshape(-1)
            # 忽略pad
            select = y1 != -float('inf')
            pred1 = pred1[select]
            y1 = y1[select]
            select = y2 != -float('inf')
            pred2 = pred2[select]
            y2 = y2[select]

            loss1 = loss_func(pred1, y1.to(device))
            loss2, likelihoods, pred2 = ord_loss(pred2.to(device), y2.to(device))
            loss = 0.5 * loss1 + 0.5 * loss2

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 1000 == 0:
                r2_1 = r2_score(pred1, y1.to(device))
                r2_2 = r2_score(pred2, y2.to(device))
                lr = optim.param_groups[0]['lr']

                if r2_2 > best_r2:
                    best_r2 = r2_2
                    torch.save(model.state_dict(), f'../Model/{saveName}_best.pth')
                    tqdm.write(f"Saved best model with R2: {best_r2:.2f}")

                tqdm.write(f"LR: {lr:.4f}, "
                           f"Loss1: {loss1.item():.2f}, Loss2: {loss2.item():.2f}, "
                           f"R1: {r2_1:.2f}, R2: {r2_2:.2f}")

        sched.step()
        torch.save(model.state_dict(), '../Model/'+saveName+'.pth')
