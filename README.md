# chainer-modelzoo

This repository is to list reusable models for [Chainer](https://github.com/pfnet/chainer)
as [Modelzoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) does for [Caffe](https://github.com/BVLC/caffe).

There is already [list of external chainer projects](https://github.com/pfnet/chainer/wiki/External-examples),
but it has only links to external GitHub projects and reusing pretrained model is a bit difficult.

This repository contains:

- **Pretrained model**: can be loaded by `npz` or `hdf5` serializer.
- **Model definition**: `Chain` Python class which defines the model used to load `chainermodel`.
- **Example**: inference example using the pretrained model.