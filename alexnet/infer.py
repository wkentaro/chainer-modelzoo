#!/usr/bin/env python

import chainer
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from download import IMAGE_PATH
from download import MODEL_PATH
from download import SYNSET_PATH
from model import AlexNet


def main():
    # load model
    model = AlexNet()
    print('Loading pretrained model from {0}'.format(MODEL_PATH))
    chainer.serializers.load_hdf5(MODEL_PATH, model)

    # prepare net input

    print('Loading image from {0}'.format(IMAGE_PATH))
    img = scipy.misc.imread(IMAGE_PATH, mode='RGB')
    img = scipy.misc.imresize(img, (227, 227))
    img_in = img.copy()

    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    mean_bgr = np.array([104, 117, 123], dtype=np.float32)
    img -= mean_bgr

    x_data = np.array([img.transpose(2, 0, 1)])
    x = chainer.Variable(x_data, volatile='ON')

    # infer
    model(x)
    score = model.h_fc8.data[0]

    # visualize result

    likelihood = np.exp(score) / np.sum(np.exp(score))
    argsort = np.argsort(score)
    label_id = argsort[-1]

    print('Loading label_names from {0}'.format(SYNSET_PATH))
    with open(SYNSET_PATH, 'r') as f:
        label_names = [line.strip() for line in f.readlines()]

    print('Likelihood of top5:')
    for index in argsort[::-1][:5]:
        print('  {0:5.1f}%: {1}'
              .format(likelihood[index]*100, label_names[index]))

    plt.title('{0:5.1f}%: {1}'
              .format(likelihood[label_id]*100, label_names[label_id]))
    plt.imshow(img_in)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
