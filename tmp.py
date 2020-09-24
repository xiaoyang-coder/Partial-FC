import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.distributed as dist
import sys
import torchvision
bs=10
fs=10
lable_gather=torch.LongTensor([-1,-1,-1,1,2,3,-1,-1,-1,6])
grad=torch.zeros(bs,fs)
index = torch.where(lable_gather != -1)[0]
one_hot = torch.zeros(index.size()[0], grad.size()[1], device=grad.device)
loss = torch.zeros(grad.size()[0], 1, device=grad.device)
one_hot.scatter_(dim=1, index=lable_gather[index, None],value=0.4)
grad[index]+=one_hot
print(grad)
