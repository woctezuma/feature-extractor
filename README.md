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
!python extract_fts \
 --output_dir features --data_dir images --batch_size 256 \
 --model_repo "facebookresearch/dinov2" --model_name dinov2_vits14 \
 --resize_size 256 --keep_ratio --crop_size 224
```
This will save in the `--output_dir` folder: 
- `fts.pt`: the features in a torch file, 
- `filenames.txt`: the list of image names corresponding to the features.

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
