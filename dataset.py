from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision
from torchvision import transforms
import numbers
import numpy as np
import mxnet as mx
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

TFS=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
])

class MXFaceDataset (Dataset):
    def __init__(self, root_dir, local_rank, transform=TFS):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            print('header0 label', header.label)
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            print("Number of Samples:{}". format(len(self.imgidx)))
                

    def __getitem__(self, index):
        # index =0
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
            # img=img.to(local_rank)
            # label=label.to(local_rank)
        # return sample.to(self.local_rank), label.to(self.local_rank)
        return sample,label

    def __len__(self):
        return len(self.imgidx)


if __name__ == "__main__":
    # /root/xy/face/faces_emore
    # /root/face_datasets/webface/
    trainset = MXFaceDataset(root_dir='/root/xy/face/faces_emore',local_rank=0)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,shuffle=True)
    print(trainset[-1][1])
    trainloader = DataLoaderX(
        trainset, batch_size=128, #sampler=train_sampler,
        num_workers=0,pin_memory=False,drop_last=True)
    print(len(trainset))
    
    for step,(img,label) in enumerate(trainloader):
        if step<5:
            continue;
        print(img.max(),img.min())
        print(img.shape)
        img=torchvision.utils.make_grid(img,nrow=8)
        torchvision.utils.save_image(img,'tmp.jpg')
        print(label)
        break;