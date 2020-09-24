import torch.distributed as dist
import torch.utils.data.distributed
import argparse
import sys
import time
import torchvision.transforms as transforms
import torch
import torchvision
from torch import nn,optim
from torch.distributed.optim import DistributedOptimizer
from classifier import classifier
import itertools
import torch.nn.functional as F
from config import config as cfg
from config import get_sub_class
from dataset import MXFaceDataset,DataLoaderX
from CosFace import CosFace
from backbones import iresnet50
from CrossEntropyLoss import CrossEntropy
import torch.multiprocessing as mp
from  torch.utils.tensorboard  import SummaryWriter
from sgd import SGD
torch.backends.cudnn.benchmark = True

#.......
def main(local_rank,world_size,init_method='tcp://127.0.0.1:23499'):
    dist.init_process_group(backend='nccl', init_method=init_method, rank=local_rank, world_size=world_size)
    cfg.local_rank=local_rank
    cfg.rank=dist.get_rank()
    cfg.world_size=world_size
    print(cfg.rank,dist.get_world_size())
    trainset = MXFaceDataset(root_dir='/root/face_datasets/webface',local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,shuffle=True)
    trainloader = DataLoaderX(
        trainset, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=0,pin_memory=False,drop_last=True)
    backbone = iresnet50(True).to(cfg.local_rank)
    backbone.train()
    # backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    for ps in backbone.parameters():
        dist.all_reduce(ps,dist.ReduceOp.MIN)

    backbone = torch.nn.parallel.DistributedDataParallel(backbone,broadcast_buffers=False,device_ids=[dist.get_rank()])
    backbone.train()
    sub_start,sub_classnum=get_sub_class(cfg.rank,dist.get_world_size())
    print(sub_start,sub_classnum)
    classifier_head=classifier(cfg.embedding_size,sub_classnum,sample_rate=1.0,gpu_storage=True)
    cosface=CosFace(s=64.0,m=0.4)
    optimizer = SGD([
        {'params': backbone.parameters()},
        {'params': classifier_head.parameters()}
    ], 0.1,momentum=0.9,weight_decay=cfg.weight_decay,rescale=cfg.batch_size*cfg.world_size)
    warm_up_with_multistep_lr = lambda epoch: ((epoch+1)/(4+1))**2 if epoch < -1 else 0.1**len([m for m in [20,29] if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    n_epochs=34
    start_epoch=0
    
    # torch.cuda.empty_cache()
    if cfg.local_rank==0:
        writer=SummaryWriter(log_dir='logs/shows')
    global_step=0
    loss_fun=nn.CrossEntropyLoss()
    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        
        for step,(img,label) in enumerate(trainloader):
            start = time.time()
            torch.cuda.synchronize()
            img=img.to(local_rank)
            img.requires_grad=True
            label=label.to(local_rank)
            x=F.normalize(backbone(img))

            lable_gather=torch.zeros(x.size()[0]*cfg.world_size,device=cfg.local_rank,dtype=torch.long)
            x_gather=torch.zeros(x.size()[0]*cfg.world_size,cfg.embedding_size,device=cfg.local_rank)

            dist.all_gather(list(x_gather.chunk(cfg.world_size,dim=0)),x.data)
            dist.all_gather(list(lable_gather.chunk(cfg.world_size,dim=0)),label)

            x_gather.requires_grad=True

            logits = classifier_head(x_gather, lable_gather)
            logits = cosface(logits, lable_gather)

            with torch.no_grad():
                max_v= torch.max(logits,dim=1,keepdim=True)[0]
                dist.all_reduce(max_v,dist.ReduceOp.MAX)
                exp = torch.exp(logits-max_v)
                sum_exp = exp.sum(dim=1, keepdims=True)
                dist.all_reduce(sum_exp, dist.ReduceOp.SUM)
                exp.div_(sum_exp.clamp_min(1e-20))
                grad = exp
                index = torch.where(lable_gather != -1)[0]
                one_hot = torch.zeros(grad.size()[0], grad.size()[1], device=grad.device)
                loss = torch.zeros(grad.size()[0], 1, device=grad.device)
                one_hot[index] = one_hot[index].scatter(1, lable_gather[index, None], 1)
                loss[index] = grad[index].gather(1, lable_gather[index, None])
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss=loss.clamp_min(1e-20)
                loss.log_()
                loss_v=torch.mean(-1*loss)
                grad.sub_(one_hot)
        
            optimizer.zero_grad()
            logits.backward(grad)
            torch.cuda.synchronize()
            if x_gather.grad is not None:
                x_gather.grad.detach_()

            x_grad = x_gather.grad
            dist.all_reduce(x_grad,dist.ReduceOp.SUM)
            x.backward(x_grad.chunk(cfg.world_size,dim=0)[cfg.rank])

            optimizer.step()
            classifier_head.update()
            if cfg.rank==0:
                print(x_gather.grad.max(),x_gather.grad.min())
                print('loss_v',loss_v.item(),global_step)
                writer.add_scalar('loss',loss_v,global_step)
                print('lr',optimizer.state_dict()['param_groups'][0]['lr'],global_step)
                print(cfg.batch_size/(time.time()-start))

            global_step+=1
        scheduler.step()
        if cfg.rank==0:
            torch.save(backbone.module.state_dict(),'backbone.pth')
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank',type=int,default=0,help='local_rank')
    parser.add_argument('--world_size',type=int,default=8,help='world_size')
    args = parser.parse_args()
    # mp.spawn(main, nprocs=8, args=(8,),join=True)
    main(args.local_rank,args.world_size)
