# Multimodal CLIP Applications

Prototype for multimodal image/text applications using [OpenAI's CLIP preprocessing architecture](https://openai.com/blog/clip/). This application uses [FAISS](https://ai.facebook.com/tools/faiss)'s Inner Product Nearest Neighbours (NNs) approximations to search (text-to-image) or classify (image-to-text) images.

---

## Prerequisites

To run this application you'll need `wget`, Python 3.6+, and the following Python dependencies, installed from the PyPI using `conda` or `pip` (as appropriate).

- `faiss-cpu`/`faiss-gpu`
- `flask`+`flask-cors` (for deployment)
- `ftfy`
- `regex`
- `torch`, `torchvision` [(with CUDA preferably)](https://pytorch.org/get-started/locally/)
- `tqdm`

A GPU is also preferred.

### [CLIP](https://github.com/openai/CLIP)

You can install it globally.

```sh
pip install git+https://github.com/openai/CLIP.git
```

Or install it locally from submodule.

```sh
git submodule update --init
```

## Setup

Before running any scripts, review `lib/hparams.py` and change any hyperparameters as necessary.

- This is where the path to the dataset should be defined. By default, it uses the `tiny-imagenet-200` dataset.

### Data

- Run `scripts/init.sh` to prepare the workspace. This creates folders for data, and generated indexes and encodings.
- If you want to use the default dataset (`tiny-imagenet-200`), run `scripts/tiny-imagenet-200.sh`.
- Make sure that any datasets (AKA repositories) you want to use are accessible with a relative/absolute Unix path from the base of the dataset/repository).

### Indexing (Manual)

The indexes will be stored in `indexes_*` and have a filename indicating dataset and number of features (which may vary given the CLIP model and FAISS compression method used).

#### Generating Image Indexes

To generate an image index, run `python build_image_index.py <dataset_name> <dataset_path> <n_components>`, where

- `dataset_name` is the name of the dataset/repository you would like to index.
- `dataset_path` is the relative/absolute filepath to the dataset. The Dataloader will recursively include all images under this directory.
- `n_components` is the number of components the feature vectors will contain. PCA compression is coming soon.
- These values have default values in `lib/hparams.py`.

After each dataset generates an index, its (`dataset_name`, `dataset_path`) are added to `collection_<datatype>.txt` if they weren't already there. This provides an easy reference to reconstruct an ordered compound index and dataloader.

The file tree should look something like this:

```
indexes_images/
   | tiny-imagenet-200-train_1024.index
   | tiny-imagenet-200-train_512.index
   | tiny-imagenet-200-test_1024.index
   | tiny-imagenet-200-test_512.index
   | tiny-imagenet-200-val_1024.index
   | tiny-imagenet-200-val_512.index
```

#### Generating Text Indexes

1. Review the hyperparameters in the `vocabulary` section of `lib/hparams.py`. Give the vocabulary a name and define the URL from which it can be retrieved.
2. Run `python build_text_index.py` and review the indexes in `indexes_text/`. The default configuration should create the following file subtree.

   - Text indexes might take a while because they're not partitioned yet.

   ```
   indexes_text/
      | text_aidemos_1024.index
      | text_aidemos_512.index
   ```

## Retrieval API (Public)

Run `sh scripts/retrieval.sh` to start the publical Retrieval API. Run this on a separete port and GPU, for example. The API hasn't yet been fully documented so please see `retrieval.py` for now.

```sh
CUDA_VISIBLE_DEVICES=0 FLASK_RUN_PORT=5020 sh scripts/public.sh
```

## Indexing API (Private)

Run `BLOCKING= sh scripts/indexing.sh` to start the private Indexing API. Make sure this API is not publicly exposed as it can be blocked by indexing function calls. Run this on a separete port and GPU, for example.

```sh
BLOCKING= CUDA_VISIBLE_DEVICES=1 FLASK_RUN_PORT=5021 sh scripts/index.sh
```

### Adding Image Repos: `/api/add-image-repo`

Example in Python.

```python
import requests

url = "http://0.0.0.0:5021/api/add-text-repo"
payload = {
    "modelName": "clip",
    "name": "tiny-imagenet-200-train",
    "path": "data/tiny-imagenet-200/train",
}

r = requests.post(url, json=payload)
```

### Adding Text Repos: `/api/add-text-repo`

Example in Python.

```python
import requests

url = "http://0.0.0.0:5021/api/add-text-repo"
payload = {
    "modelName": "clip",
    "name": "cities",
    "vocab": {
        "london": {},
        "st. petersberg": {},
        "paris": {},
    }
}

r = requests.post(url, json=payload)
```
