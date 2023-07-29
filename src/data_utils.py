# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import DataLoader

from src.image_folder import ImageFolder


def collate_fn(batch):
    """Collate function for data loader. Allows to have img of different size"""
    return batch


def get_dataloader(
    data_dir,
    transform,
    batch_size=128,
    num_workers=2,
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
