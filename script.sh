#!/bin/bash
mkdir -p .kaggle
mv ~/Download/kaggle.json ~/.kaggle/

chmod 600 ~/.kaggle/kaggle.json


pip install kaggle



kaggle datasets download -d snap/amazon-fine-food-reviews


unzip amazon-fine-food-reviews.zip



