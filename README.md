# chainer-modelzoo

This repository is to list reusable models for [Chainer](https://github.com/pfnet/chainer)
as [Modelzoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) does for [Caffe](https://github.com/BVLC/caffe).

There is already [list of external chainer projects](https://github.com/pfnet/chainer/wiki/External-examples),
but it has only links to external GitHub projects and reusing pretrained model is a bit difficult.

This repository contains:

- **Pretrained model** (downloaded by `download.py`): can be loaded by serializer.
- **Model definition** (`model.py`): Contains Python class which defines the model used to load pretrained model.
- **Example** (`infer.py`): inference example using the pretrained model.
