#!/bin/bash
# docker pull jupyter/base-notebook
# docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/base-notebook
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m nltk.downloader stopwords