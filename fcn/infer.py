#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from download import IMAGE_PATH
from download import MODEL_PATH
from model import FCN8s


LABEL_NAMES = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'tv/monitor',
]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='gpu id')
    args = parser.parse_args()

    # load model
    model = FCN8s()
    print('Loading pretrained model from: {0}'.format(MODEL_PATH))
    chainer.serializers.load_hdf5(MODEL_PATH, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    chainer.config.train = False
    chainer.config.enable_backprop = False

    # prepare net input

    print('Loading image from: {0}'.format(IMAGE_PATH))
    img = scipy.misc.imread(IMAGE_PATH, mode='RGB')
    img_in = img.copy()

    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    mean_bgr = np.array([104, 117, 123], dtype=np.float32)
    img -= mean_bgr

    x_data = np.array([img.transpose(2, 0, 1)])
    if args.gpu >= 0:
        x_data = chainer.cuda.to_gpu(x_data)
    x = chainer.Variable(x_data)

    # infer
    model(x)
    score = model.score.data[0]
    score = chainer.cuda.to_cpu(score)

    # visualize result

    label = np.argmax(score, axis=0)
    n_labels = score.shape[0]
    colormap = matplotlib.cm.prism(np.arange(n_labels))[:, :3]
    colormap = np.vstack(([0, 0, 0], colormap))  # bg color
    label_viz = colormap[label]

    # network input
    plt.subplot(121)
    plt.imshow(img_in)
    plt.axis('off')

    # network output
    plt.subplot(122)
    plt.imshow(label_viz)
    plt.axis('off')
    plt_handlers = []
    plt_titles = []
    for label_value in np.unique(label):
        if (label == label_value).sum() < 0.01 * label.size:
            continue  # skip small region
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append('%d: %s' % (label_value, LABEL_NAMES[label_value]))
    plt.legend(plt_handlers, plt_titles, loc='upper right',
               framealpha=0.5)

    out_file = osp.join(osp.dirname(IMAGE_PATH), 'result.jpg')
    plt.savefig(out_file)
    print('Saved as: {0}'.format(out_file))


if __name__ == '__main__':
    main()
