# coding: utf-8
import pickle as pkl
from collections import Counter

import sys
sys.path.append("../zh_uy_data")

import data_utils

def get_mem_s2t():
    slines = open("../zh_uy_data/train.ids30000.src")
    tlines = open("../zh_uy_data/train.ids30000.trg")
    mlines = open("../zh_uy_data/aligns")
    mem = {}
    for sline, tline, mline in zip(slines, tlines, mlines):
        zh_words = sline.strip().split(' ')
        en_words = tline.strip().split(' ')
        maps = mline.strip().split(' ')
        for m in maps:
            zhid, enid = m.split('-')
            zh_word = zh_words[int(zhid)]
            if int(zh_word) == 3:
                continue
            en_word = en_words[int(enid)]
            if int(en_word) == 3:
                continue
            if int(zh_word) not in mem:
                mem[int(zh_word)] = []
            mem[int(zh_word)].append(int(en_word))

    for m in mem:
        l = len(mem[m])
        words = Counter(mem[m])
        words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        mem[m] = map(lambda x: (x[0], x[1] / float(l)), words)

    print(len(mem))

    del slines
    del tlines
    del mlines

    en_vocab_path = "../zh_uy_data/vocab30000.src"
    fr_vocab_path = "../zh_uy_data/vocab30000.trg"
    en_vocab, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)
    fr_vocab, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    for i, word in enumerate(rev_en_vocab):
        if i not in mem:
            if word in fr_vocab:
                mem[i] = [(fr_vocab[word], 1.0), (fr_vocab['_NULL'], 0.0)]
            else:
                mem[i] = [(fr_vocab['_NULL'], 0.0), (fr_vocab['_NULL'], 0.0)]

    f = open("mems2t.pkl", 'wb')
    pkl.dump(mem, f)

    print(len(mem))


def get_mem_t2s():
    slines = open("../zh_uy_data/train.ids30000.src")
    tlines = open("../zh_uy_data/train.ids30000.trg")
    mlines = open("../zh_uy_data/aligns")
    mem = {}
    for sline, tline, mline in zip(slines, tlines, mlines):
        zh_words = sline.strip().split(' ')
        en_words = tline.strip().split(' ')
        maps = mline.strip().split(' ')
        for m in maps:
            zhid, enid = m.split('-')
            zh_word = zh_words[int(zhid)]
            if int(zh_word) == 3:
                continue
            en_word = en_words[int(enid)]
            if int(en_word) == 3:
                continue
            if int(en_word) not in mem:
                mem[int(en_word)] = []
            mem[int(en_word)].append(int(zh_word))

    for m in mem:
        l = float(len(mem[m]))
        words = Counter(mem[m])
        mem[m] = {w: words[w] / l for w in words}

    f = open("memt2s.pkl", 'wb')
    pkl.dump(mem, f)

    print(len(mem))

if __name__ == '__main__':
    get_mem_s2t()
    get_mem_t2s()
