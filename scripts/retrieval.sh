#!/bin/sh

export COLLECTION_IMAGES=../collection_images.txt
export COLLECTION_TEXT=../collection_text.txt
export INDEXES_IMAGES=../indexes_images
export INDEXES_TEXT=../indexes_text

export FLASK_APP=retrieval
export FLASK_ENV=development
export FLASK_RUN_HOST=0.0.0.0
flask run
