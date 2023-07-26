# :pushpin: Feature Extractor

This repository contains Python code to maps images to representation vectors, which is useful for image retrieval.

## Usage

### Requirements

-   Install the latest version of [Python 3.X][python-download-url].
-   Install the required packages:

```bash
pip install -r requirements.txt
```

### Feature extraction

We provide a simple script to extract features from a given model and a given image folder.
The features are extracted from the last layer of the model.
```
python extract_fts --model_name torchscript --model_path path/to/model --data_dir path/to/folder --output_dir path/to/output
```
This will save in the `--output_dir` folder: 
- `fts.pt`: the features in a torch file, 
- `filenames.txt`: a file containing the list of filenames corresponding to the features.

By default, images are resized to $288 \times 288$ (it can be changed with the `--resize_size` argument). 

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
