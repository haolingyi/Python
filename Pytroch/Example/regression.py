import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
# x = Variable(x)
# y = Variable(y)

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

    def save(self, net):
        torch.save(net, 'regression.pkl')


net = Net(1, 10, 1)
print(net)


plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
lossFunc = torch.nn.MSELoss()

for t in range(100):

    prediction = net(x)

    loss = lossFunc(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

net.save(net)
print('end')
plt.ioff()
plt.show()
