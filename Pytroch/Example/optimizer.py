import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.2*torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

torch_dataset = data.TensorDataset(x, y)
loader = data.DataLoader(dataset=torch_dataset,
                         batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net_SGD = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR)
opts = [opt_SGD, opt_RMSprop, opt_Adam]

lossFunc = torch.nn.MSELoss()

lossDatas = [[], [], []]


print(loader)
for epoch in range(EPOCH):
    print(epoch)
    for step, (batchX, batchY) in enumerate(loader):
        a = 1
        for net, opt, lossData in zip(nets, opts, lossDatas):
            output = net(batchX)
            loss = lossFunc(output, batchY)
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lossData.append(loss.data[0])

label = ['SGD', 'RMSprop', 'Adam']
for i, lossData in enumerate(lossDatas):
    plt.plot(lossData, label=label[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
