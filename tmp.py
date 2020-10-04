import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.distributed as dist
import sys
import torchvision
from collections import defaultdict
bs=10
fs=10
with torch.no_grad():
    state=defaultdict(dict)
    cpuin=torch.rand(5,5).cuda()
    buf=cpuin[0:2,:]
state[cpuin]=buf
p=state[cpuin]
p.mul_(0).add_(1)
# tin[:]=1
print(cpuin)

