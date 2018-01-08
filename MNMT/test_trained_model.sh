python ./translate.py --model2 translate.ckpt-155000-36.88 --decode --beam_size 12 < ../zh_uy_data/test.src > res
perl ./multi-bleu.perl ../zh_uy_data/test.trg < res
