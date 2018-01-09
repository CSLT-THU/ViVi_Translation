# ViVi_Translation

## Introduction

ViVi_Translation is the source code of our previous papers:

[Memory-augmented Chinese-Uyghur Neural Machine Translation. (APSIAP'17)](https://arxiv.org/abs/1706.08683)

[Memory-augmented Neural Machine Translation. (EMNLP'17)](https://arxiv.org/abs/1708.02005)

NMT is the attention-based NMT (RNNsearch), we reproduced [RNNsearch](https://arxiv.org/abs/1409.0473) on tensorflow. 

MNMT is the proposed memory-augmented NMT in our previous works.

## User Manual

### Installation

#### System Requirements

* Linux or MacOS
* Python 2.7

We recommand to use GPUs:

* NVIDIA GPUs 
* cuda 7.5

#### Installing Prerequisites

##### CUDA 7.5 environment
Assume CUDA 7.5 has been installed in "/usr/local/cuda-7.5/", then environment variables need to be set:

```
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH 
```
##### Tensorflow 0.10
To have tensorflow 0.10 installed, serval methods can be applied. Here, we only introduce the installation through virtualenv. And we install the tensorflow-gpu, if you choose to use CPU, please install tensorflow of cpu.

```
pip install virtualenv --user
virtualenv --system-site-packages tf0.10  
source tf0.10/bin/activate
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```
##### Test installation
Get into python console, and import tensorflow. If no error is encountered, the installation is successful.

```
Python 2.7.5 (default, Nov  6 2016, 00:28:07) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-11)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow 
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
>>> 
```

### Quick Start

```
git clone https://github.com/CSLT-THU/ViVi_Translation.git
cd ViVi_Translation
sh prepare.sh    
```

To train a NMT model:
```
sh run.sh nmt train
```

To test a trained NMT model:
```
sh run.sh nmt test

[zhangsy@wolf08 ViVi_Translation]$ sh run.sh nmt test
Testing a NMT checkpoint 'translate.ckpt-nmt'...
.......
Reading model parameters from ./train/translate.ckpt-nmt
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 41295 get requests, put_count=41280 evicted_count=1000 eviction_rate=0.0242248 and unsatisfied allocation rate=0.0270008
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
BLEU = 35.24, 57.7/39.8/31.9/27.0 (BP=0.939, ratio=0.941, hyp_len=14535, ref_len=15446)
```

To train a MNMT model:
```
sh run.sh mnmt train
```

To test a trained MNMT model:
```
sh run.sh mnmt test

[zhangsy@wolf08 ViVi_Translation]$ sh run.sh mnmt test
Testing a MNMT checkpoint 'translate.ckpt-mnmt'...
......
Reading model parameters from ./train_mem/translate.ckpt-nmt and ./train_mem/translate.ckpt-mnmt
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:244] PoolAllocator: After 46089 get requests, put_count=46012 evicted_count=1000 eviction_rate=0.0217335 and unsatisfied allocation rate=0.0255375
I tensorflow/core/common_runtime/gpu/pool_allocator.cc:256] Raising pool_size_limit_ from 100 to 110
BLEU = 36.88, 58.8/40.8/32.4/27.1 (BP=0.968, ratio=0.968, hyp_len=14957, ref_len=15446)
```


### Train

#### NMT

```
python NMT/translate.py 
```

Model parameters and training settings can be set by command-line arguments, as follows:

```
--learning_rate: The initial learning rate of optimizer, default is 0.0005.
--learning_rate_decay_factor: Learning rate decays by this value, default is 0.99
--max_gradient_norm: Clip gradients to this norm, default is 1.0.
--batch_size: Batch size to use during training, default is 80.
--hidden_units: Size of hidden units for each layer, default is 1000.
--hidden_edim: Dimension of word embedding, default is 500.
--num_layers: Number of layers of RNN, default is 1.
--keep_prob: The keep probability used for dropout, default is 0.8.
--src_vocab_size: Vocabulary size of source language, default is 30000.
--trg_vocab_size: Vocabulary size of target language, default is 30000.
--data_dir: Data directory, default is './data'. 
--train_dir: Training directory, default is './NMT/train/.
--steps_per_checkpoint: How many training steps to do per checkpoint, default is 1000.
```


#### MNMT
To train the MNMT model, a NMT model need to be trained first. Assume we already have a trained NMT model "translate.ckpt-nmt" in "./MNMT/train"

```
python MNMT/translate.py 
```

Model parameters and training settings can be set by command-line arguments, as follows:

```
--learning_rate: The initial learning rate of optimizer, default is 0.0005.
--learning_rate_decay_factor: Learning rate decays by this value, default is 0.99
--max_gradient_norm: Clip gradients to this norm, default is 1.0.
--batch_size: Batch size to use during training, default is 80.
--hidden_units: Size of hidden units for each layer, default is 1000.
--hidden_edim: Dimension of word embedding, default is 500.
--num_layers: Number of layers of RNN, default is 1.
--keep_prob: The keep probability used for dropout, default is 0.8.
--src_vocab_size: Vocabulary size of source language, default is 30000.
--trg_vocab_size: Vocabulary size of target language, default is 30000.
--data_dir: Data directory, default is './data'. 
--train_dir: Training directory, default is './MNMT/train.
--model: The trained NMT model to load.
--steps_per_checkpoint: How many training steps to do per checkpoint, default is 1000.
```

### Test
#### NMT
To test the 10000th checkpoint, run the command below.

```
python ./NMT/translate.py --model translate.ckpt-10000  --decode --beam_size 12 < ./data/test.src > res
perl multi-bleu.perl ./data/test.trg < res
```

Model parameters should be the same settings when training, and other parameters for decoding are as follows.

```
--decode: True or False. Set to True for interactive decoding, default is False.
--model: The NMT model to load.
--beam_size: The size of beam search, default is 5.
```


#### MNMT
To test the 10000th checkpoint, run the command below. 

```
python ./MNMT/translate.py --model2 translate.ckpt-10000 --decode --beam_size 12 < ./data/test.src > res
perl multi-bleu.perl ./data/test.trg < res
```

Model parameters should be the same settings when training, and other parameters for decoding are as follows.

```
--decode: True or False. Set to True for interactive decoding, default is False.
--model2: The MNMT model to load.
--beam_size: The size of beam search, default is 5.
```

### Apply to other datasets
#### NMT
To apply the NMT model to other datasets is easy. You only need to format your own data as the data in "./data". 

#### MNMT
To apply the MNMT model to other datasets needs more operations. 

First, you also need to format your own data as the data in "./data". 

Second, you need to acquire the word aligments between "train.src" and "train.trg", just as the downloaded "aligns" file.
You can get it via [Giza++](https://github.com/moses-smt/giza-pp) or other toolkits. 

Third, run "mem.py" to generate the "mems2t.pkl" and "memt2s.pkl". These two files will be used in the training of MNMT.

"mems2t.pkl" is the mappings from source words to target words. 
For example, the following shows a list of target words for a source word (source word id = 10). 
Each item in the list is (target word id, probablity). The probability is the probablity of translating the source word into the target word. 
The list is decending sorted by the probability.

```
>>> f = open("mems2t.pkl", 'rb')
>>> mem = pkl.load(f)
>>> mem[10][:10]
[(804, 0.020487682252388135), (57, 0.01282051282051282), (8, 0.01124937154348919), (50, 0.009112619406737054), (311, 0.008861236802413273), (511, 0.008735545500251383), (77, 0.008169934640522876), (951, 0.007101558572146807), (2050, 0.006598793363499246), (156, 0.00641025641025641)]
```

"memt2s.pkl" is the mappings from target words to source words.
For example, the following shows a map of source words for a target word (target word id = 10). 
Each item in the map is "source word id: probablity". The probability is the probablity of translating the target word into the source word. 
```
>>> f = open("memt2s.pkl", 'rb')
>>> mem = pkl.load(f)
>>> mem[10]
{28677: 0.0002770850651149903, 6: 0.0002770850651149903, 7: 9.236168837166343e-05, 10: 0.001293063637203288, 12: 9.236168837166343e-05,...}
```

Note that "mems2t.pkl" and "memt2s.pkl" do not need to be generated from training set and alignments. 
It can be derived from any source-to-target dictionary. 
As long as these two files were formatted according to the descriptions above, our model could perform correctly.

## License
Open source licensing is under the Apache License 2.0, which allows free use for research purposes. 
For commercial licensing, please email byryuer@gmail.com.

## Development Team

Project leaders: Dong Wang, Feng Yang, Askar Hamdulla

Project members: Shiyue Zhang, Gulnigar Mahmut, Andi Zhang, Shipan Ren

## Contact

If you have questions, suggestions and bug reports, please email [byryuer@gmail.com](mailto:byryuer@gmail.com).
