# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import numpy as np
import torch

from src.dataloader_utils import get_dataloader
from src.device_utils import get_device
from src.feature_utils import extract_features
from src.numpy_utils import save_as_numpy_file
from src.parser_utils import get_parser
from src.transform_utils import get_transform


def main():
    params = get_parser().parse_args()
    print(f"__log__:{json.dumps(vars(params))}")

    print('>>> Creating output directory...')
    Path(params.output_dir).mkdir(parents=True, exist_ok=True)

    print('>>> Building backbone...')
    model = torch.hub.load(params.model_repo, params.model_name)
    model.to(get_device())

    print('>>> Creating dataloader...')
    img_loader = get_dataloader(
        params.data_dir,
        get_transform(params.resize_size, params.keep_ratio, params.crop_size),
        batch_size=params.batch_size,
        collate_fn=None,
    )

    print('>>> Extracting features...')
    features, sample_fnames = extract_features(
        model,
        img_loader,
        params.batch_size,
        verbose=params.verbose,
    )

    print('>>> Saving features...')
    torch.save(features, Path(params.output_dir) / params.torch_features)

    fname = Path(params.output_dir) / params.numpy_features
    save_as_numpy_file(fname, features, np.float16)

    print('>>> Saving sample names...')
    with (Path(params.output_dir) / params.img_list).open('w') as f:
        json.dump(sample_fnames, f)


if __name__ == '__main__':
    main()
