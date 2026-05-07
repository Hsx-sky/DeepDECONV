import os
import re
import numpy as np
from ModelConstruct.transformer2head_regClassify_train_predict0 import predict
import matplotlib.pyplot as plt
from transformer_ModelParaSet import maxLen, inputCharacter
from DataLoad.embedding2tensor import dataAddPadReg
from ModelConstruct.transformer2head_model0 import Transformer
import torch
from scipy.io import savemat
# ini parameter
timeStepLenRange = [128, 768]
MaxLenOnlySignal = max(timeStepLenRange)
Overlap = 100
MaxLenPolyIIContribution = 30
maxStepLen = max(timeStepLenRange)+3
ModelSaveName = ('Transformer2head_Classify_Layer6_Head12_Reg_Len768'
                 'Pc[5 15][10 25][0 0][1 1]OscOff_Pi[0 5]_Int[1 1]'
                 '_NoiseSigma[0 0.5]Self_BS32_Norm_TrainLossOFF_LenAll_v3OffIA')
'''
('Transformer2head_Classify_Layer6_Head12_Reg_Len768'
'Pc[5 15][10 25][0 0][1 1]OscOff_Pi[0 5]_Int[1 1]'
'_NoiseSigma[0 0.5]Self_BS32_Norm_TrainLossOFF_LenAll_v2')
'''
filterSwitch = False
# loads (data & model)
# OutputPath = 'E:/Shihe/RnaPredict/nosGFP/Test/'
# loadFile = './Data/RNA_signals.npy'
# nameRe = 'Histogram/.*_new'
# OutputPath = 'E:/Shihe/RnaPredict/FilterOFF/Public/QuantitativeImagingNC/'+ModelSaveName+'_snaE/'
OutputPath = os.path.join(
    r"\\?\E:\Shihe\RnaPredict\FilterOFF\Public\QuantitativeImagingNC",
    ModelSaveName+'_snaEIlp4-INRPr',
    ""
)
loadFile = '../Data/RNA_signals_public_QI.npy'
nameRe = 'NatComm Source Data/.*-'
ShortNameRe = r'NatComm Source Data/.*\\'
ShortNameReLast = r'\\.*-'
if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)
nameChoose = r'snaE-Ilp4-INRPr'
# r'snaE-sna\+INRPr' r'snaE-snaPr'

RNA_signals_dict = np.load(loadFile, allow_pickle=True).item()
model = Transformer()
model.load_state_dict(torch.load('../Model/'+ModelSaveName+'.pth'))


polII_ini_predict_dict = {}
polII_contri_predict_dict = {}
polII_RnaSignal_predict_dict = {}
polII_RnaSignalRec_predict_dict = {}
# Data & Train
plt.figure(figsize=(15, 6))
for key, value in enumerate(RNA_signals_dict):
    signalSeries = RNA_signals_dict[value]
    # emboryName = re.findall(nameRe, value)[0][10:-4]
    emboryName = re.findall(nameRe, value)[0][20:-2]
    emboryShortName = (re.findall(ShortNameRe, value)[0][20:-2] +
                       re.findall(ShortNameReLast, value)[0][2:-2])
    emboryName = emboryName.replace('\\', '-')
    if not re.findall(nameChoose, emboryName):
        continue
    signalsDetect = np.size(signalSeries, 1)
    signalSeriesI = signalSeries[:, 1]
    signalSeriesILen = len(signalSeriesI)
    yp0G = np.zeros([MaxLenPolyIIContribution, signalsDetect])
    yp1G = np.zeros([signalSeriesILen, signalsDetect])
    x0 = np.zeros([signalSeriesILen, signalsDetect])
    xRec = np.zeros([signalSeriesILen, signalsDetect])
    # ini predict save
    for i in range(signalsDetect):
        signalSeriesI = signalSeries[:, i]
        signalSeriesI = (np.array(signalSeriesI) / max(signalSeriesI) * (inputCharacter - 1)).astype(int)
        signalSeriesFilterI = signalSeriesI
        signalSeriesFilterI[signalSeriesFilterI < 0] = 0
        signalSeriesFilterI[signalSeriesFilterI > 1e3] = 1e3
        signalSeriesILen = len(signalSeriesI)
        # cat to suit len
        splitNumber = signalSeriesILen//MaxLenOnlySignal+1
        splitLen = np.ceil(signalSeriesILen/splitNumber)

        start_i = 0
        end_i = 0
        index = 0
        while not end_i == signalSeriesILen:
            index += 1
            end_i = int(min([start_i+MaxLenOnlySignal, signalSeriesILen]))
            start_i_next = int(end_i-Overlap)
            coverAdd = int(Overlap / 2)
            if index == 1:
                coverAdd = 0
            coverStart = start_i + coverAdd
            signalSeriesIc = signalSeriesFilterI[start_i: end_i]
            input_x = dataAddPadReg(signalSeriesIc, maxLen)

            # predict
            x00 = np.array(input_x[input_x != -float('inf')].tolist())
            x00 = [int(x) for x in x00][1:-1]
            with torch.no_grad():
                yp = predict(input_x, model)

            yp0 = yp[0].squeeze(0).tolist()
            yp0 = yp0 + [0 for _ in range(MaxLenPolyIIContribution)]
            yp0 = yp0[1:MaxLenPolyIIContribution + 1]
            yp1 = yp[1].squeeze(0).tolist()
            yp1 = yp1 + [0 for _ in range(MaxLenOnlySignal)]
            yp1 = yp1[1:MaxLenOnlySignal + 2]

            yp0 = np.array(yp0)
            yp0[yp0 < 0] = 0
            # save predict
            yp0G[:, i] = np.array(yp0)
            yp1GAdd = np.array(yp1[1 + coverAdd:len(x00)+1])
            yp1G[coverStart: end_i, i] = yp1GAdd[:end_i-coverStart]
            # reconstruct rna signal
            x00_rec = np.convolve(yp1[1:], np.array(yp0[1:]), mode='full')
            # x00_rec = f_cov(yp1[1:], np.array(yp0[1:]))
            x00_rec = x00_rec[0:len(x00)].tolist()
            if not np.max(x00_rec) < 1:
                x00_rec_rescale = list(np.round(np.array(x00_rec) * np.mean(x00) / np.mean(x00_rec)).astype(int))
            else:
                x00_rec_rescale = list(np.round(np.array(x00_rec)).astype(int))

            if np.mean(np.array(input_x)[1:-1]) <= 0:
                x00_rec_rescale[x00_rec_rescale != 0] = 0

            xRec[coverStart: end_i, i] = np.array(x00_rec_rescale[coverAdd:end_i-coverStart+coverAdd])
            x0[coverStart: end_i, i] = np.array(x00[coverAdd:end_i - coverStart + coverAdd])
            start_i = start_i_next
            # show predict
            print('RNA signal: ', x00)
            print('RNA recons: ', x00_rec_rescale)
            print('Pred polII series: ', yp1[1:])
            print('Pred polII contri: ', yp0[1:])

            time_step = [x for x in range(len(yp0[1:]))]
            plt.subplot(3, 1, 3)
            plt.plot(time_step, yp0[1:], linestyle='-.')
            plt.ylabel('Pol Contr')
            plt.xlabel('Time Step')

        time_step = [x for x in range(signalSeriesILen)]
        plt.subplot(3, 1, 1)
        plt.plot(time_step, signalSeriesI, color='skyblue')
        # plt.gca().set_prop_cycle(None)
        plt.plot(time_step, xRec[:, i], linestyle='-.', color='r', linewidth=2)
        plt.ylabel('RNA Signal')
        plt.legend(['Original', 'Predict'])
        plt.title(emboryName+' '+str(i))

        plt.subplot(3, 1, 2)
        plt.bar(time_step, yp1G[:, i], color='y')
        plt.ylabel('PolII ini Predict')
        plt.xlabel('Time Step')
        # plt.show(block=True)
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(os.path.join(OutputPath, emboryName + ' ' + str(i) + '.png'))
        plt.savefig(os.path.join(OutputPath, emboryName + ' ' + str(i) + '.svg'))
        plt.clf()

    if len(emboryShortName)>31:
        emboryShortName = emboryShortName[:30]
    polII_contri_predict_dict[emboryShortName] = yp0G
    polII_ini_predict_dict[emboryShortName] = yp1G
    polII_RnaSignal_predict_dict[emboryShortName] = xRec
    polII_RnaSignalRec_predict_dict[emboryShortName] = x0

    np.save(os.path.join(OutputPath, 'polII_contri.npy'), polII_contri_predict_dict)
    np.save(os.path.join(OutputPath, 'polII_ini.npy'), polII_ini_predict_dict)
    np.save(os.path.join(OutputPath, 'RNA_ori.npy'), polII_RnaSignal_predict_dict)
    np.save(os.path.join(OutputPath, 'RNA_rec.npy'), polII_RnaSignalRec_predict_dict)
    combined_dict = {
        'polII_contri': polII_contri_predict_dict,
        'polII_ini': polII_ini_predict_dict,
        'RNA_ori': polII_RnaSignal_predict_dict,
        'RNA_rec': polII_RnaSignalRec_predict_dict
    }
    # 保存为单个 .mat 文件
    savemat(os.path.join(OutputPath, 'AIPredict.mat'), combined_dict)
