# encoding=utf-8
import logging
import os
import torch
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


class dep_dataset(data.Dataset):
    def __init__(self, root):

        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]

        label = int(img_path[-5])

        pil_img = Image.open(img_path).convert('RGB')

        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    data1 = dep_dataset(r"/public/home/acw92jjscn/lgl/data_mix_5k/Fold_1/train")
    train_sampler = RandomSampler(data1)
    train_loader = DataLoader(data1,
                              sampler=train_sampler,
                              batch_size=16,
                              num_workers=0,
                              pin_memory=True, drop_last=True)

    data2 = dep_dataset(r"/public/home/acw92jjscn/lgl/data_mix_5k/Fold_1/test")
    test_sampler = RandomSampler(data2)
    test_loader = DataLoader(data2,
                             sampler=test_sampler,
                             batch_size=16,
                             num_workers=0,
                             pin_memory=True, drop_last=True)

    return train_loader, test_loader
