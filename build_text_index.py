from CLIP_FAISS_NNs.data.dataset import load_text
from CLIP_FAISS_NNs.hparams import n_components
from CLIP_FAISS_NNs.index import build_txt_index_faiss
from CLIP_FAISS_NNs.data.collection import update_collection_text

from json import load
from sys import argv
from time import time

if len(argv) <= 2:
    print("Please specify dataset/repository name.")
    exit(0)
dataset_name, dataset_path = argv[1], argv[2]
if len(argv) >= 4:
    n_components = int(argv[3])

start_time = time()
with open(dataset_path) as fp:
    data = load(fp)
text_values = load_text(data)

build_txt_index_faiss(dataset_name,
                      text_values,
                      "../indexes_text",
                      n_components=n_components,
                      verbose=True)
print("Text indexing for {} values completed in {}.".format(
    len(text_values),
    time() - start_time))
update_collection_text(dataset_name, dataset_path, "../collection_text.txt")
