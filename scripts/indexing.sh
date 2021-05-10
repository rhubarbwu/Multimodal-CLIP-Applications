#!/bin/sh

export COLLECTION_IMAGES=../collection_images.txt
export COLLECTION_TEXT=../collection_text.txt
export INDEXES_IMAGES=../indexes_images
export INDEXES_TEXT=../indexes_text
export VOCAB_DIR=../vocab

export FLASK_APP=indexing
export FLASK_ENV=development
export FLASK_RUN_HOST=0.0.0.0
flask run
