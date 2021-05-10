from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from os import environ
from random import sample

from CLIP_FAISS_NNs.data import *
from CLIP_FAISS_NNs.index.query import classify_img, search_sim, search_txt

app = Flask("Multimodal CLIP Application Demo")
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

collection_file_images = environ["COLLECTION_IMAGES"]
collection_file_text = environ["COLLECTION_TEXT"]
indexes_images = environ["INDEXES_IMAGES"]
indexes_text = environ["INDEXES_TEXT"]

global img_repos, txt_repos
img_repos = build_img_repo_map(indexes_images, collection_file_images)
txt_repos = build_txt_repo_map(indexes_text, collection_file_text)
subset_preview_length = 6


@app.route("/api/index", methods=["GET", "POST"])
def index():
    global img_repos, txt_repos
    img_repos = build_img_repo_map(indexes_images, collection_file_images)
    txt_repos = build_txt_repo_map(indexes_text, collection_file_text)
    return jsonify({}), 200


@app.route("/api/repos-images", methods=["POST"])
def get_img_repos():
    global img_repos
    img_repos = build_img_repo_map(indexes_images, collection_file_images)
    return jsonify({"repos": sorted(list(img_repos.keys()))})


@app.route("/api/repos-text", methods=["POST"])
def get_txt_repos():
    global txt_repos
    txt_repos = build_txt_repo_map(indexes_text, collection_file_text)
    return jsonify({"repos": sorted(list(txt_repos.keys()))})


@app.route("/api/gallery", methods=["POST"])
def get_gallery():
    data = request.json
    mode = data["mode"]["id"]
    selected_repos_img = data["repos"]

    subsets, subset_size = build_img_data_subset(img_repos, selected_repos_img)
    subset_indices = sample(range(subset_size), subset_preview_length)
    filepaths = [(int(i), index_into_img_subsets(subsets, i))
                 for i in subset_indices]
    print(filepaths)

    return jsonify({"filepaths": filepaths})


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.json
    selected_repos_img = data["repos"]
    selected_repos_txt = data["txt_repos"]
    index = data["index"]
    nnn = data["n_neighbours"]

    subsets, _ = build_txt_data_subset(txt_repos, selected_repos_txt)
    result_indices = classify_img(selected_repos_img, selected_repos_txt,
                                  indexes_images, indexes_text, index, nnn)
    text = [(int(i), index_into_txt_subsets(subsets, i))
            for i in result_indices]

    return jsonify({"classified": text})


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    selected_repos_img = data["repos"]
    query = "a picture of {}".format(data["query"])
    nnn = data["n_neighbours"]

    subsets, _ = build_img_data_subset(img_repos, selected_repos_img)
    result_indices = search_txt(selected_repos_img, indexes_images, query, nnn)
    filepaths = [(int(i), index_into_img_subsets(subsets, i))
                 for i in result_indices]

    return jsonify({"filepaths": filepaths})


@app.route("/api/similar", methods=["POST"])
def similar():
    data = request.json
    selected_repos_img = data["repos"]
    index = data["index"]
    nnn = data["n_neighbours"]

    subsets, _ = build_img_data_subset(img_repos, selected_repos_img)
    result_indices = search_sim(selected_repos_img, indexes_images, index, nnn)
    filepaths = [(int(i), index_into_img_subsets(subsets, i))
                 for i in result_indices]

    return jsonify({"filepaths": filepaths})
