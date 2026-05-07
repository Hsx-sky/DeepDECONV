# InputNum
InputNum = 1
# EmbeddingSize
headNum = 12
LayerNumber = 6
embeddingSize = headNum*8
# Simulation Data
intensity_random = [1, 1]
noise_random = 0.5
NormLink = 1
# Dataset
trainDataSize = 300000
batchSize = 32
# One head output
inputCharacter = 1000 + 1
outputCharacter = 4
# Two head output
outputCharacter1 = 15
outputCharacter2 = 6
Classify = True
if not Classify:
    outputCharacter2 = 1

#Set for 256, timeResolution large
# Model parameter
maxLen = 256+2
addCharacter = 3
# outputCharacter1 polc length
randomPolc = False
randomRiseNum = [1, 1]
polcLength0 = [1, 5]  #up
polcLength1 = [0, 20]
polcLength2 = [0, 0]  #down
polcat = [30, 30]
polcMax = [15, 15]

#Set for 768, timeResolution small
# Model parameter
# maxLen = 768+2
# addCharacter = 3
# # outputCharacter1 polc length
# randomPolc = False
# randomRiseNum = [1, 1]
# polcLength0 = [5, 15]  #up
# polcLength1 = [10, 25]
# polcLength2 = [0, 10]  #down
# polcat = [50, 50]
# polcMax = [15, 15]


input2lag = [0, 10]
CatStart = 1
IniNumWithLen = 0
