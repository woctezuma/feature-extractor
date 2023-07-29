# :pushpin: Feature Extractor

This repository contains Python code to map images to representation vectors, which is useful for image retrieval.

## Requirements

-   Install the latest version of [Python 3.X][python-download-url].
-   Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run:
```bash
!python extract_fts.py \
 --output_dir features --data_dir images --batch_size 256 \
 --model_repo "facebookresearch/dinov2" --model_name dinov2_vits14 \
 --resize_size 256 --keep_ratio --crop_size 224
```
The following files will be saved in the `--output_dir` folder: 
- `fts.pt`: the features in a PyTorch file,
- `fts.npy`: the features in a NumPy file, 
- `filenames.txt`: the list of image names corresponding to the features.

## Example

To extract features from the `balloon` dataset:

```bash
%cd /content
!git clone https://github.com/woctezuma/feature-extractor.git
%cd feature-extractor
%pip install -r requirements.txt

!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
!unzip -q balloon_dataset.zip

!python extract_fts.py \
 --output_dir features --data_dir balloon --batch_size 256 \
 --model_repo "facebookresearch/dinov2" --model_name dinov2_vits14 \
 --resize_size 256 --keep_ratio --crop_size 224
```

## References

-   Meta AI's *Active Image Indexing*:
    - [Official blog post][active-image-indexing-blog]
    - [Official Github repository][active-image-indexing-github]
    - [Fernandez, Pierre, et al. *Active image indexing*. ICLR 2023.][active-image-indexing-arxiv] 

<!-- Definitions -->

[python-download-url]: <https://www.python.org/downloads/>
[active-image-indexing-blog]: <https://pierrefdz.github.io/publications/activeindexing/>
[active-image-indexing-github]: <https://github.com/facebookresearch/active_indexing>
[active-image-indexing-arxiv]: <https://arxiv.org/abs/2210.10620>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
