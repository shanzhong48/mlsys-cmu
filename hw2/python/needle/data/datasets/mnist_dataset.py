from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
            assert(magic_num == 2051)
            tot_pixels = row * col
            X = np.vstack([np.array(struct.unpack(f"{tot_pixels}B", img_file.read(tot_pixels)), dtype=np.float32) for _ in range(img_num)])
            X -= np.min(X)
            X /= np.max(X)

        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            assert(magic_num == 2049)
            y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)
        
        self.images = X
        self.labels = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        
        imgs = self.images[index]
        labels = self.labels[index]
        if len(imgs.shape) > 1:
            imgs = np.dstack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
        return (imgs.transpose().reshape(-1,28*28), labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION