# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from pathlib import Path

import torch
import tqdm

from src import utils
from src.transform_utils import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--output_dir",
            type=str,
            default='features',
            help="The path to the output folder where features will be saved.",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="images",
            help="The path to the input folder where images are stored.",
        )
        parser.add_argument(
            "--model_repo",
            type=str,
            default="facebookresearch/dinov2",
            help="A github repo with format `repo_owner/repo_name`, for example ‘pytorch/vision’.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="dinov2_vits14",
            help="The name of a callable (entrypoint) defined in the repo’s hubconf.py.",
        )
        parser.add_argument(
            "--resize_size",
            type=int,
            default=256,
            help="Desired image output size after the resize.",
        )
        parser.add_argument(
            "--keep_ratio",
            type=bool,
            default=True,
            help="Whether to keep the image ratio: the smallest image side will match `resize_size`.",
        )
        parser.add_argument(
            "--crop_size",
            type=int,
            default=224,
            help="Desired image output size after the center-crop.",
        )
        parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")

        return parser

    params = get_parser().parse_args()
    print(f"__log__:{json.dumps(vars(params))}")

    print('>>> Creating output directory...')
    Path(params.output_dir).mkdir(parents=True, exist_ok=True)

    print('>>> Building backbone...')
    model = torch.hub.load(params.model_repo, params.model_name)

    print('>>> Creating dataloader...')
    img_loader = utils.get_dataloader(
        params.data_dir,
        get_transform(params.resize_size, params.keep_ratio, params.crop_size),
        batch_size=params.batch_size,
        collate_fn=None,
    )

    print('>>> Extracting features...')
    features = []
    with (Path(params.output_dir) / "filenames.txt").open('w') as f, torch.no_grad():
        for ii, imgs in enumerate(tqdm.tqdm(img_loader)):
            fts = model(imgs.to(device))
            features.append(fts.cpu())
            for jj in range(fts.shape[0]):
                sample_fname = img_loader.dataset.samples[ii * params.batch_size + jj]
                f.write(sample_fname + "\n")

    print('>>> Saving features...')
    features = torch.concat(features, dim=0)
    torch.save(features, Path(params.output_dir) / 'fts.pth')
