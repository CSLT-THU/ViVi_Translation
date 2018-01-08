#!/usr/bin/env bash
ckpt=30000
for((i=1;i<200;i++))
do
    echo ${ckpt} >> test/test_eval.txt
    python ./translate.py --model translate.ckpt-${ckpt} --decode --beam_size 12 < ../zh_uy_data/test.en > test/prediction.txt
    perl ./multi-bleu.perl ../zh_uy_data/test.fr < test/prediction.txt >> test/test_eval.txt
    ckpt=$[30000 + $i*1000]
done
