# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
from pathlib import Path

from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader, is_image_file

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Data loading


@functools.lru_cache
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    paths = []
    for _dirpath, _dirnames, filenames in os.walk(path):
        paths.extend([Path(path) / filename for filename in filenames])
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
    """Collate function for data loader. Allows to have img of different size"""
    return batch


def get_dataloader(
    data_dir,
    transform,
    batch_size=128,
    num_workers=8,
    collate_fn=collate_fn,
):
    """Get dataloader for the images in the data_dir."""
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader
