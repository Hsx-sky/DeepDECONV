# 定义字典
import numpy as np
import torch
from DataLoad.Simulation_pol2rna import get_data
from transformer_ModelParaSet import inputCharacter, outputCharacter

dict_pol = '<SOS>,<EOS>,<PAD>,0,1'
dict_pol = {word: i for i, word in enumerate(dict_pol.split(','))}
dict_polr = [k for k, v in dict_pol.items()]

dict_polc = ['<SOS>', '<EOS>', '<PAD>']
dict_polc.extend([str(i+1) for i in np.array(range(outputCharacter)).tolist()])
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

    # Add tag at start and end
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']

    # Add pad to a fixed length
    x = x + ['<PAD>'] * maxStepLen
    y = y + ['<PAD>'] * (maxStepLen-1)
    x = x[:maxStepLen]
    y = y[:(maxStepLen-1)]

    # embedding
    x = [dict_pol[i] for i in x]
    y = [dict_rna[i] for i in y]

    # transform type to tensor
    output_y = torch.LongTensor(x)
    input_x = torch.LongTensor(y)

    return input_x, output_y

def dataAddPad(x, maxStepLen):
    # Add tag at start and end
    x = ['<SOS>'] + [str(i) for i in list(np.round(x).astype(int))] + ['<EOS>']

    # Add pad to a fixed length
    x = x + ['<PAD>'] * maxStepLen
    x = x[:maxStepLen]
    if x[-1] != '<PAD>':
        x[-2] = '<EOS>'
        x[-1] = '<PAD>'

    # embedding
    x = [dict_rna[i] for i in x]

    # transform type to tensor
    input_x = torch.LongTensor(x)

    return input_x

def dataAddPadReg(x, maxStepLen):
    # Add tag at start and end
    x = [0] + x

    # Add pad to a fixed length
    x = x.tolist() + [0] * maxStepLen
    x = x[:maxStepLen]
    if x[-1] != 0:
        x[-1] = 0


    # transform type to tensor
    input_x = torch.FloatTensor(x)

    return input_x

def get_loader(timeStepLenRange,maxStepLen):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            super(Dataset, self).__init__()
            self.timelen = timeStepLenRange
            self.maxlen = maxStepLen
        def __len__(self):
            return 30000
        def __getitem__(self, i):
            return data2tensor(self.timelen, self.maxlen)

    # 数据加载器dataset=
    loader = torch.utils.data.DataLoader(Dataset(),
                                         batch_size=8,
                                         drop_last=True,
                                         shuffle=True,
                                         collate_fn=None)
    return loader