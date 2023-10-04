from typing import List

import gzip
import os
import lzma as xz


def compression_ratio(
        path: str,
        data: List[str],
        algorithm: str = 'gzip',
        verbose: bool = False
) -> float:
    """ Calculates the compression ratio for a collection of text.

    Args:
        path (str): Path to store temporarily zipped files.
        data (List[str]): Strings to compress.
        algorithm (str, optional): Either 'gzip' or 'xz'. Defaults to 'gzip'.
        verbose (bool, optional): Print out the original and compressed size separately. Defaults to False.

    Returns:
        float: Compression ratio (original size / compressed size)
    """
    
    with open(path+'original.txt', 'w+') as f:
        f.write(' '.join(data))

    original_size = os.path.getsize(os.path.join(path, "original.txt"))

    if algorithm == 'gzip':

        with gzip.GzipFile(path+'compressed.gz', 'w+') as f:
            f.write(gzip.compress(' '.join(data).encode('utf-8')))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))

    elif algorithm == 'xz': 

        with xz.open(path+'compressed.gz', 'wb') as f:
            f.write(' '.join(data).encode('utf-8'))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))

    if verbose: 
        print(f"Original Size: {original_size}\nCompressed Size: {compressed_size}")

    return original_size / compressed_size
    
