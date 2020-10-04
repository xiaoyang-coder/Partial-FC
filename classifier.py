import torch
from torch import nn
from torch.nn import init
from torch.nn import Module
from config import config as cfg
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import torch.distributed as dist

class classifier(Module):
    @torch.no_grad()
    def __init__(self, in_features, out_features, sample_rate = 1.0):
        super(classifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty([out_features, in_features],device=cfg.local_rank)
        self.momentum = torch.zeros_like(self.weight)
        self.sample_rate = sample_rate
        self.sub_num = int(sample_rate*out_features)
        self.stream = torch.cuda.Stream(cfg.local_rank)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if sample_rate == 1.0:
            self.update= lambda : 0
            self.sub_weight = Parameter(self.weight)
            self.sub_momentum = self.momentum
        else:
            self.sub_weight = Parameter(torch.empty([0,0]).cuda(cfg.local_rank))
            self.perm = torch.LongTensor(cfg.num).cuda(cfg.local_rank)

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
            self.sub_weight = Parameter(self.weight[index])
            self.sub_momentum = self.momentum[index]
    
    def forward(self, x_gather, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = F.linear(x_gather, norm_weight)
        return logits
        
    @torch.no_grad()
    def update(self,):
        pass
        self.momentum[self.index]=self.sub_momentum
        self.weight[self.index]=self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            lable_gather = torch.zeros(label.size()[0]*cfg.world_size, device=cfg.local_rank, dtype=torch.long)
            dist.all_gather(list(lable_gather.chunk(cfg.world_size, dim=0)), label)
            self.sample(lable_gather)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_momentum
            norm_weight = F.normalize(self.sub_weight)
            return lable_gather, norm_weight

if __name__ == "__main__":
    cfg.local_rank=0
    cfg.s=0
    cfg.num=100
    clss=classifier(5,25,sample_rate=0.1)
    print(list(clss.parameters()))
