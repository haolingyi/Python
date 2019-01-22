<<<<<<< HEAD
import numpy as np

x = np.ones([2, 2])
y = np.ones([2, 2])
a = x + y
b = x + y
c = a * b
print(c)
=======
import torch
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)
print(torch.cuda.is_available())

# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))
>>>>>>> hly
