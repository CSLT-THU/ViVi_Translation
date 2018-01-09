#!/usr/bin/env bash
if [ $1 == "nmt" ]
then
    if [ $2 == "train" ]
    then
        echo "Training a NMT model... Checkpoints are saved in NMT/train/"
        python ./NMT/translate.py
    elif [ $2 == "test" ]
    then
        echo "Testing a NMT checkpoint 'translate.ckpt-nmt'..."
        cp ./models/translate.ckpt-nmt ./NMT/train/
        python ./NMT/translate.py --model translate.ckpt-nmt  --decode --beam_size 12 < ./data/test.src > res
        perl multi-bleu.perl ./data/test.trg < res
    fi
elif [ $1 == "mnmt" ]
then
    echo "Generating memory files and preparing a trained NMT model..."
    python mem.py
    cp ./models/translate.ckpt-nmt ./MNMT/train/
    if [ $2 == "train" ]
    then
        echo "Training a MNMT model... Checkpoints are saved in MNMT/train/"
        python ./MNMT/translate.py
    elif [ $2 == "test" ]
    then
        echo "Testing a MNMT checkpoint 'translate.ckpt-mnmt'..."
        cp ./models/translate.ckpt-mnmt ./MNMT/train/
        python ./MNMT/translate.py --model2 translate.ckpt-mnmt --decode --beam_size 12 < ./data/test.src > res
        perl multi-bleu.perl ./data/test.trg < res
    fi
fi

