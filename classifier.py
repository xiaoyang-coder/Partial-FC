import torch
from torch import nn
from torch.nn import init
from torch.nn import Module
from config import config as cfg
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class classifier(Module):
    @torch.no_grad()
    def __init__(self, in_features, out_features, gpu_storage=True, sample_rate = 1.0):
        super(classifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.bn = nn.BatchNorm1d(in_features).cuda(cfg.local_rank)
        self.weight = torch.Tensor(out_features, in_features)
        self.sample_rate =sample_rate
        if gpu_storage or sample_rate == 1.0:
            self.weight = self.weight.cuda(cfg.local_rank)
        self.sub_num = int(sample_rate*out_features)
        self.reset_parameters()
        if sample_rate == 1.0:
            self.update=lambda : 0
            self.sub_weight = Parameter(self.weight.cuda(cfg.local_rank))
        else:
            self.perm = torch.LongTensor(cfg.num).cuda(cfg.local_rank)
            self.sub_weight = Parameter(self.weight[:self.sub_num].cuda(cfg.local_rank))


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def sample(self, global_label):
        P = (cfg.s <= global_label) & (global_label < cfg.s+cfg.num)
        global_label[~P] = -1
        global_label[P] -= cfg.s
        if self.sample_rate!=1.0:
            positive = torch.unique(global_label[P], sorted=False)
            if self.sub_num-positive.size(0) > 0:
                torch.randperm(cfg.num, out=self.perm)
                start = cfg.num-self.sub_num
                index = torch.cat((positive, self.perm[start:]))
                index = torch.unique(index, sorted=False)
                start = index.size()[0]-self.sub_num
                index = index[start:]
            else:
                index = positive
                
            index = torch.sort(index)[0].long()
            self.index = index
            global_label[P] = torch.searchsorted(index, global_label[P])
            self.sub_weight[:] = self.weight[index.to(self.weight.device)].cuda(cfg.local_rank)

    def forward(self, x, global_label):
        self.sample(global_label)
        return F.linear(x,F.normalize(self.sub_weight))
    
    def update(self,):
        # print('update')
        self.weight[self.index.to(self.weight.device),:]=self.sub_weight.to(self.weight.device)



if __name__ == "__main__":
    cfg.local_rank=0
    clss=classifier(5,25,sample_rate=1.0)