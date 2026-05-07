# 定义字典
import numpy as np
import torch
from DataLoad.Simulation_pol2rna import get_data
from transformer_ModelParaSet import inputCharacter, outputCharacter1, outputCharacter2, trainDataSize, batchSize

dict_pol = ['<SOS>', '<EOS>', '<PAD>']
dict_pol.extend([str(i) for i in np.array(range(outputCharacter2)).tolist()])
dict_pol = {word: i for i, word in enumerate(dict_pol)}
dict_polr = [k for k, v in dict_pol.items()]

dict_polc = ['<SOS>', '<EOS>', '<PAD>']
dict_polc.extend([str(i+1) for i in np.array(range(outputCharacter1)).tolist()])
dict_polc = {word: i for i, word in enumerate(dict_polc)}
dict_polcr = [k for k, v in dict_polc.items()]

dict_rna = ['<SOS>', '<EOS>', '<PAD>']
dict_rna.extend([str(i) for i in np.array(range(inputCharacter)).tolist()])
dict_rna = {word: i for i, word in enumerate(dict_rna)}
dict_ranr = [k for k, v in dict_rna.items()]



def data2tensor(timeStepLenRange, maxStepLen):
    # maxStepLen=max(timeStepLenRange)+3
    x, y, polC = get_data(timeStepLenRange)
    # Choose x
    x = [x[0]] + x
    x2 = [polC[0]] + polC

    x = [0] + x
    x2 = [0] + x2
    y = [0] + y
    # Add pad to a fixed length
    x = x + [0] * maxStepLen
    x2 = x2 + [0] * maxStepLen
    y = y + [0] * (maxStepLen-1)
    x = x[:maxStepLen]
    x2 = x2[:maxStepLen]
    y = y[:(maxStepLen-1)]


    # transform type to tensor
    output_y1 = torch.FloatTensor(x2)
    output_y2 = torch.FloatTensor(x)
    input_x = torch.FloatTensor(y)

    return input_x, output_y1, output_y2

def get_loader(timeStepLenRange,maxStepLen):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            super(Dataset, self).__init__()
            self.timelen = timeStepLenRange
            self.maxlen = maxStepLen
        def __len__(self):
            return trainDataSize
        def __getitem__(self, i):
            return data2tensor(self.timelen, self.maxlen)

    # 数据加载器dataset=
    loader = torch.utils.data.DataLoader(Dataset(),
                                         batch_size=batchSize,
                                         drop_last=True,
                                         shuffle=True,
                                         collate_fn=None)
    return loader