#!/usr/bin/env bash
echo "Downloading data..."
wget http://data.cslt.org/uych180/uych180.tgz
tar -xzvf uych180.tgz
mkdir data
sed -n '1,180612p' uych180/chinese.txt > data/train.src
sed -n '180613,182597p' uych180/chinese.txt > data/dev.src
sed -n '182598,183589p' uych180/chinese.txt > data/test.src
sed -n '1,180612p' uych180/uyghur.txt > data/train.trg
sed -n '180613,182597p' uych180/uyghur.txt > data/dev.trg
sed -n '182598,183589p' uych180/uyghur.txt > data/test.trg
echo "Preparing data..."
python data_utils.py
echo "Downloading models..."
wget http://data.cslt.org/uych180/models.tgz
tar -xzvf models.tgz
echo "Downloading word alignment file..."
wget http://data.cslt.org/uych180/aligns
mv aligns data/
echo "Making directories to save model checkpoints..."
mkdir MNMT/train
mkdir NMT/train