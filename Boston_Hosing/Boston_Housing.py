import pandas as pd
import numpy as np
import torch
import os
import torch.utils.data as Data

LEARN_RATE = 0.001
EPOCH = 128
BATCH_SIZE = 16


def load_trainData():

    data_csv = pd.read_csv("./Data/train.csv")
    print()
    prices = data_csv['medv']
    features = data_csv.drop(['medv', 'ID'], axis=1)

    # 数据预处理
    mean = features.mean(axis=0)
    features -= mean  # 减去均值
    std = features.std(axis=0)  # 特征标准差
    features /= std

    print(features)
    print(mean)
    return features, prices


def load_testData():

    features = pd.read_csv("./Data/test.csv")
    mean = features.mean(axis=0)
    features -= mean  # 减去均值
    std = features.std(axis=0)  # 特征标准差
    features /= std

    return features


def save_testResult(prediction):
    result = pd.read_csv("./Data/submission_example.csv")
    result['medv'] = prediction['medv']
    result.to_csv('submission', index=False)


class Net(torch.nn.Module):
    def __init__(self, feaNum, hidNum, hidNum_2, outNum):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(feaNum, hidNum)
        self.hidden_2 = torch.nn.Linear(hidNum, hidNum_2)
        self.predict = torch.nn.Linear(hidNum_2, outNum)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = torch.nn.functional.relu(self.hidden_2(x))
        x = self.predict(x)
        return x


if __name__ == "__main__":

    os.chdir("./Boston_Hosing")

    featureTrainTensor = torch.zeros(333, 13)
    targetTrainTensor = torch.zeros(333)
    featureTestTensor = torch.zeros(173, 13)

    net = Net(13, 128, 8, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARN_RATE)
    lossFunc = torch.nn.MSELoss()
    trainFeature, trainTarget = load_trainData()

    for i in range(333):
        featureTrainTensor[i] = torch.Tensor(trainFeature.iloc[i].values[0:13])

    targetTrainTensor = torch.Tensor(trainTarget.values)

    print(featureTrainTensor)

    trainDataset = Data.TensorDataset(featureTrainTensor, targetTrainTensor)
    trainLoader = Data.DataLoader(
        dataset=trainDataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    for i in range(EPOCH):

        for step, (inp, tar) in enumerate(trainLoader):
            prediction = net(inp)
            loss = lossFunc(prediction, tar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(i)
            print('Loss=%.4f' % loss.data.numpy())

    testFeature = load_testData()

    for i in range(173):
        featureTestTensor[i] = torch.Tensor(testFeature.iloc[i].values[1:14])

    prediction = net(featureTestTensor)

    prediction = prediction.data.numpy()
    prediction = pd.DataFrame(prediction, columns=['medv'])
    save_testResult(prediction)
