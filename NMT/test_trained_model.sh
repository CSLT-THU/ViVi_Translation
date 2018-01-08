python ./translate.py --model translate.ckpt-376000-35.24  --decode --beam_size 12 < ../zh_uy_data/test.src > res
perl ./multi-bleu.perl ../zh_uy_data/test.trg < res
