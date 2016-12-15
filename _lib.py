import hashlib
import os.path as osp
import re
import shlex
import subprocess


def download(url, path, quiet=False):

    def is_google_drive_url(url):
        m = re.match('^https?://drive.google.com/uc\?id=.*$', url)
        return m is not None

    if is_google_drive_url(url):
        client = 'gdown'
    else:
        client = 'wget'

    cmd = '{client} {url} -O {path}'.format(client=client, url=url, path=path)
    if quiet:
        cmd += ' --quiet'
    subprocess.call(shlex.split(cmd))

    return path


def cached_download(url, path, md5=None, quiet=False):

    def check_md5(path, md5, quiet=False):
        if not quiet:
            print('Checking md5 of file: {}'.format(path))
        is_same = hashlib.md5(open(path, 'rb').read()).hexdigest() == md5
        return is_same

    if osp.exists(path) and not md5:
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        return download(url, path, quiet=quiet)
