#!/usr/bin/env python

import os.path as osp
import sys

import chainer
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from download import IMAGE_PATH
from download import MODEL_PATH
from download import SYNSET_PATH
from model import VGG16

this_dir = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, osp.join(this_dir, '..'))

from _lib import draw_image_classification_top5  # NOQA


def main():
    # load model
    model = VGG16()
    print('Loading pretrained model from {0}'.format(MODEL_PATH))
    chainer.serializers.load_hdf5(MODEL_PATH, model)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    # prepare net input

    print('Loading image from {0}'.format(IMAGE_PATH))
    img = scipy.misc.imread(IMAGE_PATH, mode='RGB')
    img = scipy.misc.imresize(img, (224, 224))
    img_in = img.copy()

    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    mean_bgr = np.array([104, 117, 123], dtype=np.float32)
    img -= mean_bgr

    x_data = np.array([img.transpose(2, 0, 1)])
    x = chainer.Variable(x_data)

    # infer
    model(x)
    score = model.score.data[0]

    # visualize result

    likelihood = np.exp(score) / np.sum(np.exp(score))
    argsort = np.argsort(score)

    print('Loading label_names from {0}'.format(SYNSET_PATH))
    with open(SYNSET_PATH, 'r') as f:
        label_names = np.array([line.strip() for line in f.readlines()])

    print('Likelihood of top5:')
    top5 = argsort[::-1][:5]
    for index in top5:
        print('  {0:5.1f}%: {1}'
              .format(likelihood[index] * 100, label_names[index]))

    img_viz = draw_image_classification_top5(
        img_in, label_names[top5], likelihood[top5])
    plt.imshow(img_viz)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
