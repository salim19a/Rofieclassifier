import math
import os
from pathlib import Path
import torch
import sys
import logging
import glob
import urllib
from typing import Union, List

LOGGER = logging.getLogger('rofie')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # rofie root directory

def check_suffix(file: Union[str, List[str]] = 'rofie.pt', suffix: Union[str, List[str]] = ('.pt',), msg: str = '') -> None:
    """
    Check if the file(s) have an acceptable suffix.

    Args:
        file (Union[str, List[str]], optional): The file or list of files to check. Defaults to 'rofie.pt'.
        suffix (Union[str, List[str]], optional): The acceptable suffix or list of acceptable suffixes. 
            Defaults to ('.pt',).
        msg (str, optional): Additional message for the assertion error. Defaults to ''.

    Raises:
        AssertionError: If the file does not have an acceptable suffix.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def check_yaml(file: str, suffix: Union[str, List[str]] = ('.yaml', '.yml')) -> str:
    """
    Search/download YAML file (if necessary) and return its path, checking suffix.

    Args:
        file (str): The path or URL to the YAML file.
        suffix (Union[str, List[str]], optional): The acceptable suffix or list of acceptable suffixes. 
            Defaults to ('.yaml', '.yml').

    Returns:
        str: The path to the YAML file.

    Raises:
        AssertionError: If the file does not have an acceptable suffix.
    """
    return check_file(file, suffix)

def check_file(file: str, suffix: str = '') -> str:
    """
    Search/download file (if necessary) and return its path.

    Args:
        file (str): The path or URL to the file.
        suffix (str, optional): The acceptable suffix. Defaults to ''.

    Returns:
        str: The path to the file.

    Raises:
        AssertionError: If the file does not exist or the file's suffix is not acceptable.
    """
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if os.path.isfile(file):
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = []
        for d in os.listdir(ROOT):  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

def check_img_size(imgsz: Union[int, List[int]], s: int = 32, floor: int = 0) -> Union[int, List[int]]:
    """
    Verify that the image size is a multiple of the stride s in each dimension.

    Args:
        imgsz (Union[int, List[int]]): The image size as an integer or list of integers.
        s (int, optional): The stride size. Defaults to 32.
        floor (int, optional): The floor value. Defaults to 0.

    Returns:
        Union[int, List[int]]: The new image size(s) that are multiples of the stride.

    Example:
        >>> check_img_size(640, s=32, floor=0)
        640
        >>> check_img_size([640, 480], s=32, floor=0)
        [640, 480]
    """
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def make_divisible(x: Union[int, torch.Tensor], divisor: int) -> int:
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (Union[int, torch.Tensor]): The number or tensor to make divisible.
        divisor (int): The divisor.

    Returns:
        int: The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
