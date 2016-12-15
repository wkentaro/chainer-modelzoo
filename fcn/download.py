#!/usr/bin/env python

import argparse
import os.path as osp
import sys

this_dir = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, osp.join(this_dir, '..'))

from _lib import cached_download


data_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

MODEL_PATH = osp.join(data_dir, 'pascal2012_fcn8s.chainermodel')
IMAGE_PATH = osp.join(data_dir, '2008_002123.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vTXU0QzUwSkVwOFk',
        path=MODEL_PATH,
        md5='a1083db5a47643b112af69bfa59954f9',
        quiet=args.quiet,
    )

    cached_download(
        url='http://www.cs.stanford.edu/~roozbeh/pascal-context/2008_002123.jpg',  # NOQA
        path=IMAGE_PATH,
        md5='',
        quiet=args.quiet,
    )
