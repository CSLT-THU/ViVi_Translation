#!/usr/bin/env bash
read -p "Please input model type (nmt or mnmt)：" model
if [ $model != "nmt" and $model != "mnmt" ]
then
    echo "Wrong input!"
else
    read -p "Please input operation type (train or test)：" operation
    if [ $operation != "test" and $operation != "train" ]
    then
        echo "Wrong input!"
    else
        if [ $model == "nmt" ]
        then
            if [ $operation == "train" ]
            then
                echo "Training a NMT model... Checkpoints are saved in NMT/train/"
                python ./NMT/translate.py
            elif [ $operation == "test" ]
            then
                echo "Testing a NMT checkpoint 'translate.ckpt-nmt'..."
                cp ./models/translate.ckpt-nmt ./NMT/train/
                python ./NMT/translate.py --model translate.ckpt-nmt  --decode --beam_size 12 < ./data/test.src > res
                perl multi-bleu.perl ./data/test.trg < res
            fi
        elif [ $model == "mnmt" ]
        then
            echo "Generating memory files and preparing a trained NMT model..."
            python mem.py
            cp ./models/translate.ckpt-nmt ./MNMT/train/
            if [ $operation == "train" ]
            then
                echo "Training a MNMT model... Checkpoints are saved in MNMT/train/"
                python ./MNMT/translate.py
            elif [ $operation == "test" ]
            then
                echo "Testing a MNMT checkpoint 'translate.ckpt-mnmt'..."
                cp ./models/translate.ckpt-mnmt ./MNMT/train/
                python ./MNMT/translate.py --model2 translate.ckpt-mnmt --decode --beam_size 12 < ./data/test.src > res
                perl multi-bleu.perl ./data/test.trg < res
            fi
        fi
    fi
fi


