import torch
import torchvision
import torch.distributed as dist
from torch import nn


def CrossEntropy(logits, labels):
    eps = 1e-30
    with torch.no_grad():
        max_v = torch.max(logits, dim=1, keepdim=True)[0]
        dist.all_reduce(max_v, dist.ReduceOp.MAX)
        logits.sub_(max_v)
        negetive = (labels == -1)
        positive = ~negetive
        P_logits = torch.zeros_like(labels).view(-1, 1).float()
        P_logits[positive] = torch.gather(
            logits[positive], index=labels[positive, None], dim=1)
        dist.all_reduce(P_logits)
        P_logits[positive] = eps
        sum_exp = logits.exp().sum(dim=1, keepdims=True)
        tmp_sum = torch.clone(sum_exp)
        dist.all_reduce(sum_exp)
        N_logits = torch.log(
            (sum_exp - P_logits.exp() - tmp_sum).clamp_min(min=eps))
        labels[negetive] = logits.size()[1]
    new_logits = torch.cat([logits, P_logits, N_logits], dim=1)
    lossfunction = nn.CrossEntropyLoss()
    return lossfunction(new_logits, labels)


if __name__ == "__main__":
    print(dir(dist.ReduceOp))
