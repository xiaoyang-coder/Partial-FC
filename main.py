import argparse
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from CosFace import CosFace
from backbones import iresnet50
from classifier import classifier
from config import config as cfg
from config import get_sub_class
from dataset import MXFaceDataset, DataLoaderX
from sgd import SGD

torch.backends.cudnn.benchmark = True


# .......
def main(local_rank, world_size, init_method='tcp://127.0.0.1:23499'):
    dist.init_process_group(backend='nccl', init_method=init_method, rank=local_rank, world_size=world_size)

    # the rank of current process group
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    cfg.rank = dist.get_rank()
    cfg.world_size = world_size

    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoaderX(local_rank=local_rank,
        dataset=trainset, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=True, drop_last=False)
    backbone = iresnet50(False).to(cfg.local_rank)
    backbone.train()
    # backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = torch.nn.parallel.DistributedDataParallel(backbone, broadcast_buffers=False,
        device_ids=[dist.get_rank()])
    backbone.train()
    sub_start, sub_classnum = get_sub_class(cfg.rank, dist.get_world_size())
    print(sub_start, sub_classnum)
    classifier_head = classifier(cfg.embedding_size, sub_classnum, sample_rate=0.4)
    cosface = CosFace(s=64.0, m=0.4)
    optimizer = SGD([
        {'params': backbone.parameters()},
        {'params': classifier_head.parameters()}
    ], 0.1, momentum=0.9, weight_decay=cfg.weight_decay, rescale=cfg.world_size)
    warm_up_with_multistep_lr = lambda epoch: ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [20, 29] if m - 1 <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    n_epochs = 33
    start_epoch = 0

    if cfg.local_rank == 0:
        writer = SummaryWriter(log_dir='logs/shows')
    global_step = 0
    loss_fun = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(trainloader):
            start = time.time()
            lable_gather, norm_weight = classifier_head.prepare(label, optimizer)
            x = F.normalize(backbone(img))
            x_gather = torch.zeros(x.size()[0] * cfg.world_size, cfg.embedding_size, device=cfg.local_rank)
            dist.all_gather(list(x_gather.chunk(cfg.world_size, dim=0)), x.data)
            x_gather.requires_grad = True

            logits = classifier_head(x_gather, norm_weight)

            logits = cosface(logits, lable_gather)

            with torch.no_grad():
                max_v = torch.max(logits, dim=1, keepdim=True)[0]
                dist.all_reduce(max_v, dist.ReduceOp.MAX)
                exp = torch.exp(logits - max_v)
                sum_exp = exp.sum(dim=1, keepdims=True)
                dist.all_reduce(sum_exp, dist.ReduceOp.SUM)
                exp.div_(sum_exp.clamp_min(1e-20))
                grad = exp
                index = torch.where(lable_gather != -1)[0]
                one_hot = torch.zeros(index.size()[0], grad.size()[1], device=grad.device)
                one_hot.scatter_(1, lable_gather[index, None], 1)

                loss = torch.zeros(grad.size()[0], 1, device=grad.device)
                loss[index] = grad[index].gather(1, lable_gather[index, None])
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss_v = loss.clamp_min_(1e-20).log_().mean() * (-1)

                grad[index] -= one_hot
                grad.div_(grad.size()[0])

            logits.backward(grad)
            if x_gather.grad is not None:
                x_gather.grad.detach_()
            x_grad = torch.zeros_like(x)
            dist.reduce_scatter(x_grad, list(x_gather.grad.chunk(cfg.world_size, dim=0)))
            x.backward(x_grad)
            optimizer.step()
            classifier_head.update()
            optimizer.zero_grad()
            if cfg.rank == 0:
                print(x_gather.grad.max(), x_gather.grad.min())
                print('loss_v', loss_v.item(), global_step)
                writer.add_scalar('loss', loss_v, global_step)
                print('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
                print(cfg.batch_size / (time.time() - start))

            global_step += 1
        scheduler.step()
        if cfg.rank == 0:
            torch.save(backbone.module.state_dict(), "models/" + str(epoch) + 'backbone.pth')
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--world_size', type=int, default=8, help='world_size')
    args = parser.parse_args()
    main(args.local_rank, args.world_size)
