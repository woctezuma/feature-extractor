# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os

import faiss
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import timm
from timm import optim as timm_optim
from timm import scheduler as timm_scheduler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.datasets.folder import default_loader, is_image_file


# Model

def build_backbone(path, name):
    """ Build a pretrained torchvision backbone from its name.
    Args:
        path: path to the checkpoint, can be an URL
        name: "torchscript" or name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        or timm (see https://rwightman.github.io/pytorch-image-models/models/). 
    Returns:
        model: nn.Module
    """
    if name == 'torchscript':
        model = torch.jit.load(path)
        return model
    else:
        if hasattr(models, name):
            model = getattr(models, name)(pretrained=True)
        elif name in timm.list_models():
            model = timm.models.create_model(name, num_classes=0)
        else:
            raise NotImplementedError('Model %s does not exist in torchvision'%name)
        model.head = nn.Identity()
        model.fc = nn.Identity()
        if path is not None:
            if path.startswith("http"):
                checkpoint = torch.hub.load_state_dict_from_url(path, progress=False)
            else:
                checkpoint = torch.load(path)
            state_dict = checkpoint
            for ckpt_key in ['state_dict', 'model_state_dict', 'teacher']:
                if ckpt_key in checkpoint:
                    state_dict = checkpoint[ckpt_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
        return model

# Data loading

@functools.lru_cache()
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    paths = []
    for path, _, files in os.walk(path):
        for filename in files:
            paths.append(os.path.join(path, filename))
    return sorted([fn for fn in paths if is_image_file(fn)])

class ImageFolder:
    """An image folder dataset without classes"""

    def __init__(self, path, transform=None, loader=default_loader):
        self.samples = get_image_paths(path)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):
        assert 0 <= idx < len(self)
        img = self.loader(self.samples[idx])
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    """ Collate function for data loader. Allows to have img of different size"""
    return batch

def get_dataloader(data_dir, transform, batch_size=128, num_workers=8, collate_fn=collate_fn):
    """ Get dataloader for the images in the data_dir. """
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False, pin_memory=True, drop_last=False)
    return dataloader
