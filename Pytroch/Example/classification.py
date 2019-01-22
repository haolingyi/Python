import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x = Variable(x)
y = Variable(y)

# method1


class Net(torch.nn.Module):
    def __init__(self, feaNum, hidNum, outNum):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(feaNum, hidNum)
        self.predict = torch.nn.Linear(hidNum, outNum)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

    def save(self):
        torch.save(self, 'classification.pkl')


# method2
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)


net = Net(2, 32, 2)
print(net)
print(net2)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
lossFunc = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)

    loss = lossFunc(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        predY = prediction.data.numpy()
        targetY = y.data.numpy()

        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[
                    :, 1], c=predY, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((predY == targetY).astype(
            int).sum()) / float(targetY.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' %
                 accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

net.save()
print('end')
plt.ioff()
plt.show()
