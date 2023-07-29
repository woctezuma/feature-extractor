import functools
import logging
import os
from pathlib import Path

from PIL import ImageFile
from torchvision.datasets.folder import default_loader, is_image_file

ImageFile.LOAD_TRUNCATED_IMAGES = True


@functools.lru_cache
def get_image_paths(path):
    logging.info(f"Resolving files in: {path}")
    paths = []
    for _dirpath, _dirnames, filenames in os.walk(path):
        paths.extend([str(Path(_dirpath) / filename) for filename in filenames])
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
