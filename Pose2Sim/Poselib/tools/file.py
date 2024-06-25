import hashlib
import os
import re
import sys
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse  # noqa: F401
from urllib.request import Request, urlopen
import zipfile
import shutil

from tqdm import tqdm


def _get_rtmhub_dir():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'rtmlib')))
    return os.path.join(torch_home, 'hub')

def extract_zip(zip_file_path, extract_to_path):
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """Download object at the given URL to a local path.

    Modified from `torch.hub.download_url_to_file`.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, for example,
            ``/tmp/temporary_file``.
        hash_prefix (str, optional): If not None, the SHA256 downloaded
            file should start with ``hash_prefix``. Defaults to None.
        progress (bool): whether or not to display a progress
            bar to stderr Defaults to True.
    """
    file_size = None
    req = Request(url, headers={'User-Agent': 'mmlmtools'})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders('Content-Length')
    else:
        content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = Path(dst).expanduser().absolute()
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst.parent)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size,
                  disable=not progress,
                  unit='B',
                  unit_scale=True,
                  unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest))
        Path(f.name).rename(dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def download_checkpoint(url: str,
                        dst_dir: Optional[str] = None,
                        progress: bool = True,
                        check_hash: bool = False,
                        filename: Optional[str] = None) -> str:
    """Download the checkpoint from the given URL.

    Modified from `torch.hub.load_state_dict_from_url`.

    If the object is already present in `dst_dir`, it will be returned
    directly.
    The default value of ``dst_dir`` is the same as the checkpoint cache
    path of PyTorch hub.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to
            stderr. Defaults to True.
        check_hash(bool, optional): If True, the filename part of the URL
            should follow the naming convention ``filename-<sha256>.ext`` where
            ``<sha256>`` is the first eight or more digits of the SHA256 hash
            of the contents of the file. The hash is used to ensure unique
            names and to verify the contents of the file. Defaults to False.
        filename (str, optional): name for the downloaded file.
            Filename from ``url`` will be used if not set.

    Returns:
        str: The path of the downloaded file.
    """
    if dst_dir is None:
        dst_dir = os.path.join(_get_rtmhub_dir(), 'checkpoints')

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    parts = urlparse(url)
    filename = filename or os.path.basename(parts.path)
    cached_file = dst_dir / filename
    onnx_name = Path(dst_dir,str(filename).split('.')[0]+".onnx")

    if not cached_file.exists():
        if os.path.exists(onnx_name):
            return onnx_name

        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if str(cached_file).split('.')[-1] == 'zip':
        # os.system(f'unzip -d {dst_dir}/tmp {cached_file}')
        tmp_dir = Path(Path(cached_file).parent, "tmp")
        extract_zip(cached_file, tmp_dir)
        cached_list = glob(f'{dst_dir}/**', recursive=True)

        for each in cached_list:
            if each[-12:] == 'end2end.onnx':
                cached_onnx = each
                break
        # os.system(f'mv {cached_onnx} {onnx_name}')
        # os.system(f'rm -rf {cached_file}')
        # os.system(f'rm -rf {dst_dir}/tmp')
        shutil.move(cached_onnx, onnx_name)
        os.remove(cached_file)
        shutil.rmtree(tmp_dir)
        
        cached_file = onnx_name

    return str(cached_file)
