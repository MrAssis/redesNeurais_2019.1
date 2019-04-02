import numpy as np
import RBF

# x = np.genfromtxt('train-mod.csv',delimiter=',')[1:,1:]
# alldata = []
# for i in range(len(x)):
#    if np.isnan(x[i,-2]):
#     continue
#    alldata.append(x[i])
#
# feature = alldata
# labels = np.asanyarray([aux[1:] for aux in alldata])
data = 'train-mod.csv'
sigma = 1.2
itergd = 300

dataread = np.genfromtxt(data, delimiter=',')[1:, 1:]
print(len(dataread))

alldata = []
for i in range(len(dataread)):
    if np.isnan(dataread[i, -2]):
        continue
    alldata.append(dataread[i])

alldata = np.asarray(alldata)

# dividing data
trainparam = alldata[:600, 1:]
trainlabel = alldata[:600, 0]

testparam = alldata[600:, 1:]
testlabel = alldata[600:, 0]

###############
# normalization#
###############

std = np.zeros((len(trainparam[0]))).astype('float32')
rata = np.zeros((len(trainparam[0]))).astype('float32')
trainparamnorm = np.zeros(np.shape(trainparam))  # .flatten()
testparamnorm = np.zeros(np.shape(testparam))  # .flatten()

for i in range(len(trainparam[0])):
    std[i] = np.std(trainparam[:, i])
    rata[i] = np.mean(trainparam[:, i])
    trainparamnorm[:, i] = (trainparam[:, i] - rata[i]) / std[i]
    testparamnorm[:, i] = (testparam[:, i] - rata[i]) / std[i]

#k = 3 melhor configuração
rbfnet = RBF.RBFNet(lr=1e-2, k=3, inferStds=True)
rbfnet.fit(trainparamnorm, trainlabel)

NUM_SAMPLES = len(testparamnorm)
y_pred = rbfnet.predict(testparamnorm)

errorabs = abs(testlabel - y_pred)
print('error: ', np.sum(errorabs[0] / NUM_SAMPLES, axis=0))