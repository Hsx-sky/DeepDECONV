import numpy as np
from DataLoad.embedding2tensorTwoHead import get_loader as get_loader_input1
from DataLoad.embedding2tensorTwoHead_input2 import get_loader as get_loader_input2
from ModelConstruct.transformer2head_regClassify_train_predict0 import train
from transformer_ModelParaSet import intensity_random, InputNum
import torch
torch.cuda.set_device(0)
# torch.cuda.set_per_process_memory_fraction(0.5,device=0)
# ini parameter
timeStepLenRange = [64, 128]
maxStepLen = max(timeStepLenRange)+3
N_EPOCHS = 30

InheritName = ('Transformer2head_Classify_Layer6_Head12_Reg_Len128'
               'Pc[1 5][0 20][0 0][1 10]OscOff_Pi[0 5]_Int[1 1]'
               '_NoiseSigma[0 0.5]Self_BS64_NormLink_TrainLossOFF_LenAll_v1')

saveName = ('Transformer2head_Classify_Layer6_Head12_Reg_Len128'
            'Pc[1 5][0 20][0 0][1 10]OscOff_Pi[0 5]_Int[1 1]'
            '_NoiseSigma[0 0.5]Self_BS64_NormLink_TrainLossOFF_LenAll_v2')

learningRate = 0.5e-3
stepGamma = 0.5
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'm.,.......

# Data & Train
if InputNum == 1:
    loader = get_loader_input1(timeStepLenRange, maxStepLen)
elif InputNum == 2:
    loader = get_loader_input2(timeStepLenRange, maxStepLen)

train(loader, N_EPOCHS, saveName, InheritName, learningRate, stepGamma)
