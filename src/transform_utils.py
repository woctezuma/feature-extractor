import torch
from torchvision import transforms

# Reference: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def get_target_image_size(resize_size=256, keep_ratio=True):
    return resize_size if keep_ratio else (resize_size, resize_size)


def get_transform(
    resize_size=256,
    keep_ratio=True,
    crop_size=224,
    interpolation=transforms.InterpolationMode.BICUBIC,
):
    transforms_list = [
        transforms.Resize(
            get_target_image_size(resize_size, keep_ratio),
            interpolation=interpolation,
        ),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
    return transforms.Compose(transforms_list)
