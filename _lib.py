from __future__ import division
from __future__ import print_function

import hashlib
import os.path as osp
import shlex
import subprocess
import sys

import numpy as np

try:
    import cv2
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False

    def _warn_opencv_unavailable():
        print('cv2 is not available. Please install python-opencv.',
              file=sys.stderr)


def download(url, path, quiet=False):
    client = 'gdown'  # to support GDrive public links
    cmd = '{client} {url} -O {path}'.format(client=client, url=url, path=path)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))

    return path


def cached_download(url, path, md5=None, quiet=False):

    def check_md5(path, md5, quiet=False):
        if md5 and len(md5) != 32:
            raise ValueError('md5 must be 32 charactors.\n'
                             'actual: {} ({})'.format(md5, len(md5)))
        if not quiet:
            print('Checking md5 of file: {}'.format(path))
        is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
        return is_same

    if osp.exists(path) and not md5:
        if not quiet:
            print('{} already exists'.format(path))
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        return download(url, path, quiet=quiet)


def centerize(src, dst_shape):
    """Centerize image for specified image size

    Parameters
    ----------
    src: numpy.ndarray (height, width, channel)
        image to centerize.
    dst_shape: tuple of int
        image shape (height, width) or (height, width, channel).
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical+h,
               pad_horizontal:pad_horizontal+w] = src
    return centerized


def draw_image_classification_top5(img, label_names, proba):
    assert len(label_names) == len(proba)
    square_size = min(img.shape[:2])
    img = centerize(img, dst_shape=(square_size, square_size))
    # draw bars
    bars = np.zeros((square_size // 2, square_size, 3), dtype=np.uint8)
    bars.fill(255)
    step = square_size // (2 * 5)
    for i in range(5):
        y1 = step * i
        y2 = y1 + step
        x1 = 0
        x2 = int(square_size * proba[i])
        color = np.array((proba[i], 0, 1 - proba[i]))
        bars[y1:y2, x1:x2] = (color * 255).astype(np.uint8)
        bars[y1:y1+1, :] = 0
        bars[y2-1:y2, :] = 0
        label_name = label_names[i].split(' ')[1].strip(',')
        if _OPENCV_AVAILABLE:
            cv2.putText(bars, label_name,
                        (x1 + square_size // 2, y1 + step // 2),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        else:
            _warn_opencv_unavailable()
    bars = bars[:y2, :]
    img = np.vstack((img, bars))
    return img
