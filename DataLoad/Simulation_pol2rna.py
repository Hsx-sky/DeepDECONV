import random
import numpy as np
from transformer_ModelParaSet import (
    intensity_random, noise_random, inputCharacter, outputCharacter2, polcLength0, polcLength1,
    randomPolc, polcLength2, polcMax, randomRiseNum, CatStart, IniNumWithLen, polcat, NormLink)
from DataLoad.waveProduct import wave_with_oscillation_product


def f_cov(i, polC):
    i_l = len(i)
    # i = np.convolve(i,np.array([1,2,2,1]), mode='full')
    # random PolII contribution
    i = np.convolve(i, np.array(polC), mode='full')
    # random noise
    Noise = 0.1 * np.random.normal(0, random.uniform(0, noise_random), len(i))
    # i = i + np.round([(random.random() - 0.5) * noise_random for i in range(i_l)] * i * 2).astype(int)
    i = i + np.round(i * Noise).astype(int)
    # random intensity
    DataIntensity_random = random.randint(intensity_random[0], intensity_random[1])
    i = i * DataIntensity_random
    i[i < 0] = 0
    return i
def get_data(TimeLength):
    # poly contribution
    UpLen = random.randint(polcLength0[0], polcLength0[1])
    DownLen = random.randint(polcLength2[0], polcLength2[1])
    CosLen = random.randint(polcLength1[0], polcLength1[1])
    pcLen = UpLen+DownLen+CosLen
    cat_len = random.randint(max([UpLen, polcat[0]]), polcat[1])
    cat_int = random.randint(0, min(polcLength0[1], cat_len))
    # the number of polIIs allowed binding to gene in one time step
    Nums = [str(i) for i in np.array(range(outputCharacter2)).tolist()]
    # The probability of binding number
    if IniNumWithLen == 1:
        p = [random.randint(1, round(500/(pcLen * (num+1)))) for num in range(outputCharacter2)]
    else:
        p = [random.randint(0, 100) for _ in range(outputCharacter2)]
    p[0] = p[0]+0
    p = np.array(p)
    p = p / p.sum()

    preZero = [0 for _ in range(random.randint(0, TimeLength[1]//2))]
    # random time length
    ncat = random.randint(TimeLength[0], TimeLength[1]-(polcLength0[1]+polcLength0[1]))
    ncatStart = random.randint(polcLength0[1]+polcLength0[1]+1, TimeLength[0])
    if CatStart == 0:
        ncatStart = 0
    n = random.randint(TimeLength[0]+TimeLength[1]+polcLength0[1]+polcLength0[1]+1, 2*TimeLength[1])
    x0 = np.random.choice(Nums, size=n, replace=True, p=p)
    # polII random
    x00 = preZero + [int(i) for i in x0]
    x = x00
    # contribution of PolII

    RiseNum = random.randint(randomRiseNum[0], randomRiseNum[1])
    polcMaxR = random.randint(polcMax[0], polcMax[1])
    RiseNum = max([1, min([RiseNum, (UpLen-2)//2-1])])
    # def pol_contrbution():
    #     PolC = [i + 1 for i in range(UpLen)] + [UpLen for _ in range(CosLen)]
        # PolC=[random.randint(0, 3) for i in range(random.randint(2, 4))]
    #     return PolC
    # polC = [i + 1 for i in range(UpLen)] + [UpLen for _ in range(CosLen)] + [0]

    wave_with_oscillation, trapezoidal_wave = wave_with_oscillation_product(UpLen, CosLen, DownLen, RiseNum, cat_len, cat_int)

    if randomPolc:
        polC = np.array(wave_with_oscillation)
    else:
        polC = np.array(trapezoidal_wave)

    polC = polC/max(polC)*polcMaxR

    InfoAddLen = len(polC)
    # y是对x的变换得到的
    # 字母大写,数字取10以内的互补数

    y00 = f_cov(x00, polC)
    if not max(y00) == 0:
        Normalize = (inputCharacter-1)/max(y00)
        y00 = (np.array(y00)*Normalize).astype(int)
        if NormLink == 1:
            polC = polC*Normalize

    y = [i for i in y00.tolist()]
    polC = [i for i in polC]
    return x[ncatStart:ncatStart+ncat+InfoAddLen], y[ncatStart:ncatStart+ncat+InfoAddLen], polC
