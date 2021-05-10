from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from json import dump
from os import environ

from CLIP_FAISS_NNs.data import *
from CLIP_FAISS_NNs.index.build import build_img_index_faiss, build_txt_index_faiss
from CLIP_FAISS_NNs.data.collection import update_collection_images, update_collection_text
from CLIP_FAISS_NNs.index.query import classify_img, search_sim, search_txt

app = Flask("Multimodal CLIP Application Indexing")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

collection_file_images = environ["COLLECTION_IMAGES"]
collection_file_text = environ["COLLECTION_TEXT"]
indexes_images = environ["INDEXES_IMAGES"]
indexes_text = environ["INDEXES_TEXT"]
vocab_dir = environ["VOCAB_DIR"]


@app.route("/api/add-image-repo", methods=["POST"])
def add_image_repo():
    if "BLOCKING" not in environ:
        return jsonify({}), 403

    data = request.json
    name, dataset_path = data["name"], data["path"]

    n_total = build_img_index_faiss(name,
                                    dataset_path,
                                    indexes_images,
                                    verbose=True)
    update_collection_images(name, dataset_path, collection_file_images)
    return jsonify({"size": n_total}), 200


@app.route("/api/add-text-repo", methods=["POST"])
def add_text_repo():
    if "BLOCKING" not in environ:
        return jsonify({}), 403

    data = request.json
    name, vocab = data["name"], data["vocab"]

    text_values = load_text(vocab)

    build_txt_index_faiss(name,
                          text_values,
                          indexes_text,
                          n_components=n_components,
                          verbose=True)

    filepath = "{}/{}.json".format(vocab_dir, name)
    update_collection_text(name, filepath, collection_file_text)
    with open(filepath, "w") as outfile:
        dump(vocab, outfile)

    return jsonify({"size": len(text_values)})
