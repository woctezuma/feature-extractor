# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import torch
import tqdm

from src import utils
from src.parser_utils import get_parser
from src.transform_utils import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
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
