import hashlib
import os.path as osp
import shlex
import subprocess


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
        print('{} already exists'.format(path))
        return path
    elif osp.exists(path) and md5 and check_md5(path, md5):
        return path
    else:
        return download(url, path, quiet=quiet)
