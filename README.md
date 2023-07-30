# :pushpin: Feature Extractor

This repository contains Python code to map images to representation vectors, which is useful for image retrieval.

## Requirements

-   Install the latest version of [Python 3.X][python-download-url].
-   Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Given a Github repository which includes `hubconf.py`, run:
```bash
!python extract_fts.py \
 --output_dir features --data_dir images --batch_size 256 \
 --model_repo "facebookresearch/dinov2" --model_name dinov2_vits14 \
 --resize_size 256 --keep_ratio --crop_size 224 \
 --torch_features fts.pth --numpy_features fts.npy \
 --img_list img_list.json \
 --verbose
```
The following files will be saved in the `--output_dir` folder: 
- `fts.pth`: the features in a PyTorch file,
- `fts.npy`: the features (as `np.float16`) in a NumPy file, 
- `img_list.json`: the list of image paths corresponding to the features.

## Example

For instance, to extract features for images in the `balloon` dataset:

```bash
%cd /content
!git clone https://github.com/woctezuma/feature-extractor.git
%cd feature-extractor
%pip install --quiet -r requirements.txt

!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
!unzip -q balloon_dataset.zip

!python extract_fts.py --data_dir balloon
```

## References

-   A [feature matcher][feature-matcher] based on the `faiss` library.
-   [`match-steam-banners`][github-match-steam-banners]: retrieve games with similar store banners.
-   [`steam-DINOv2`][github-match-with-dinov2]: retrieve games with similar store banners, using Meta AI's DINOv2.
-   Meta AI's *Active Image Indexing*:
    - [Official blog post][active-image-indexing-blog]
    - [Official Github repository][active-image-indexing-github]
    - [Fernandez, Pierre, et al. *Active image indexing*. ICLR 2023.][active-image-indexing-arxiv] 

<!-- Definitions -->

[python-download-url]: <https://www.python.org/downloads/>
[feature-matcher]: <https://github.com/woctezuma/feature-matcher>
[github-match-steam-banners]: <https://github.com/woctezuma/match-steam-banners>
[github-match-with-dinov2]: <https://github.com/woctezuma/steam-DINOv2>
[active-image-indexing-blog]: <https://pierrefdz.github.io/publications/activeindexing/>
[active-image-indexing-github]: <https://github.com/facebookresearch/active_indexing>
[active-image-indexing-arxiv]: <https://arxiv.org/abs/2210.10620>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
