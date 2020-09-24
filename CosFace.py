import torch
from torch import nn
class CosFace(nn.Module):
    def __init__(self,s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        one_hot = torch.zeros(cosine.size()[0],cosine.size()[1], device=cosine.device)
        one_hot[index] = one_hot[index].scatter(1, label[index,None],1)
        ret = self.s*((one_hot * (cosine-self.m)) + ((1.0 - one_hot) * cosine))
        return ret
