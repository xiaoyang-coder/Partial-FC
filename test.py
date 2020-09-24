import torch
from torch import nn
from torch.nn.parameter import Parameter
import sys
torch.nn.SyncBatchNorm.convert_sync_batchnorm
x = torch.ones(25)
index=[1,2,3,4,5]
y = torch.ones(10000,device=1)*0.1
t=torch.zeros(3,dtype=torch.bool,device=1)
tmp=torch.tensor([1,5,2,3,4,5,5])
tmp[[1,2,3]]-=1
print(tmp)
sys.exit(0)

x.requires_grad=True
y.requires_grad=True

x2=x[index].cuda(1)

loss=torch.sum(x2*y)
loss.backward()
print(y.grad)
print(x.grad)
print(torch.Tensor(1,2,device=1))
