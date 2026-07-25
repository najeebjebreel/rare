"""One-off: convert the CINIC-10 train/test image folders to LMDB databases."""

import _bootstrap  # noqa: F401

import pickle

import lmdb
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def build_lmdb(src_dir, lmdb_path):
    ds = ImageFolder(src_dir)

    env = lmdb.open(
        lmdb_path,
        map_size=50 * 1024 ** 3,  # 50 GB upper bound
        subdir=False,
        meminit=False,
        map_async=True,
    )

    with env.begin(write=True) as txn:
        txn.put(b"__len__", pickle.dumps(len(ds)))
        for idx, (path, label) in enumerate(tqdm(ds.samples)):
            with open(path, "rb") as f:
                img_bytes = f.read()
            txn.put(
                f"{idx}".encode(),
                pickle.dumps((img_bytes, label)),
            )

    env.sync()
    env.close()


if __name__ == "__main__":
    build_lmdb("data/cinic10/train", "data/cinic10/train.lmdb")
    build_lmdb("data/cinic10/test", "data/cinic10/test.lmdb")
