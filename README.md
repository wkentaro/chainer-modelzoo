# chainer-modelzoo

This repository is to list reusable models for [Chainer](https://github.com/pfnet/chainer)
as [Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) does for [Caffe](https://github.com/BVLC/caffe).

There is already [list of external chainer projects](https://github.com/pfnet/chainer/wiki/External-examples),
but it has only links to external GitHub projects and reusing pretrained model is a bit difficult.

This repository contains:

- **Pretrained model** (downloaded by `download.py` to `data`): can be loaded by serializer.
- **Model definition** (`model.py`): with Python class which defines the model for loading pretrained model.
- **Inference example** (`infer.py`): using the pretrained model.


## Usage

You need to install requirements with following commands.

```bash
pip install -r requirements.txt
cd <model>  # ex) cd alexnet
./download.py
./infer.py
```


## Models

name | paper
--- | ---
[alexnet](alexnet) | https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
[fcn](fcn) | https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
