import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = p.load(fo, encoding='bytes')
    return dict
# 3, 8, 8, 0, 6, 6, 1, 6,
if __name__ == "__main__":
    dict = unpickle('D:\\datasets\\cifar-10-python\\cifar-10-batches-py\\data_batch_1')
    print(dict)

