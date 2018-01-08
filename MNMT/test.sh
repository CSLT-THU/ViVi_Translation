ckpt=30000
for((i=1;i<160;i++))
do
    echo ${ckpt} >> test/test_mem.txt
    python ./translate.py --model2 translate.ckpt-${ckpt} --decode --beam_size 12 < ../zh_uy_data/test.src > test/prediction.txt
    perl ./multi-bleu.perl ../zh_uy_data/test.trg < test/prediction.txt >> test/test_mem.txt
    ckpt=$[84000 + $i*1000]
done
