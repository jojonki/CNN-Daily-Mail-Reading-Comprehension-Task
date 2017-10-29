#!/bin/sh

DATA_DIR="./dataset/"
mkdir -p $DATA_DIR

download () {
  URL=$1
  FILE_NAME=$2

  if [ ! -f "$DATA_DIR$FILE_NAME" ]; then
    wget $URL$FILE_NAME -O $DATA_DIR/$FILE_NAME
  else
    echo "You've already downloaded $FILE_NAME dataset"
  fi
}


download "http://cs.stanford.edu/~danqi/data/" "cnn.tar.gz"
download "http://cs.stanford.edu/~danqi/data/" "dailymail.tar.gz"
download "http://nlp.stanford.edu/data/" "glove.6B.zip"
