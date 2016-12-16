#!/usr/bin/env python

import argparse
import os.path as osp
import sys

this_dir = osp.dirname(osp.realpath(__file__))
sys.path.insert(0, osp.join(this_dir, '..'))

from _lib import cached_download  # NOQA


data_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

MODEL_PATH = osp.join(data_dir, 'bvlc_alexnet.chainermodel')
SYNSET_PATH = osp.join(data_dir, 'synset_words.txt')
IMAGE_PATH = osp.join(data_dir, 'leopard.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    cached_download(
        url='https://drive.google.com/uc?id=0B5DV6gwLHtyJZkd1ZTRiNUdrUXM',
        path=MODEL_PATH,
        md5='2175620a2237bbd33e35bf38867d84b2',
        quiet=args.quiet,
    )

    cached_download(
        url='https://drive.google.com/uc?id=0B9P1L--7Wd2vTEF0d1RQaC1SMmM',
        path=SYNSET_PATH,
        md5='4d234b5833aca44928065a180db3016a',
        quiet=args.quiet,
    )

    cached_download(
        url='http://rocknrollnerd.github.io/assets/article_images/2015-05-27-leopard-sofa/leopard.jpg',  # NOQA
        path=IMAGE_PATH,
        md5='6092e23e0fc3e6fdd7e971b8a8c220d6',
        quiet=args.quiet,
    )
