import argparse


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
        action="store_true",
        help="Whether to keep the image ratio: the smallest image side will match `resize_size`.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="Desired image output size after the center-crop.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity.",
    )

    return parser
