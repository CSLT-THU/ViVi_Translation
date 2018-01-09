wget http://data.cslt.org/uych180/zh_uy_data.tar.gz
tar -xzvf zh_uy_data.tgz
python ./zh_uy_data/data_utils.py
if [ $1 == "nmt" ]
then
    if [ $2 == "train" ]
    then
        echo "Training a NMT model... Checkpoints are saved in NMT/train/"
        python ./NMT/translate.py
    elif [ $2 == "test" ]
    then
        echo "Testing a NMT checkpoint 'translate.ckpt-orig'..."
        cp ./zh_uy_data/translate.ckpt-nmt ./NMT/train/
        python ./NMT/translate.py --model translate.ckpt-nmt  --decode --beam_size 12 < ./zh_uy_data/test.src > res
        perl ./zh_uy_data/multi-bleu.perl ./zh_uy_data/test.trg < res
    fi
elif [ $1 == "mnmt" ]
then
    echo "Generating memory files and preparing a trained NMT model..."
    python ./zh_uy_data/mem.py
    cp ./zh_uy_data/translate.ckpt-nmt ./MNMT/train/
    if [ $2 == "train" ]
    then
        echo "Training a MNMT model... Checkpoints are saved in MNMT/train/"
        python ./MNMT/translate.py
    elif [ $2 == "test" ]
    then
        echo "Testing a MNMT checkpoint 'translate.ckpt-mnmt'..."
        cp ./zh_uy_data/translate.ckpt-mnmt ./MNMT/train/
        python ./MNMT/translate.py --model2 translate.ckpt-mnmt  --decode --beam_size 12 < ./zh_uy_data/test.src > res
        perl ./zh_uy_data/multi-bleu.perl ./zh_uy_data/test.trg < res
    fi
fi

