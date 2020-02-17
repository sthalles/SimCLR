import cv2
import numpy as np


class GaussianBlur(object):

    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        sigma = (self.max - self.min) * np.random.random_sample() + self.min
        sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         sample = sample.transpose((2, 0, 1))
#         return torch.tensor(sample)
