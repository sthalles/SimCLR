import cv2
import numpy as np
import torch

np.random.seed(0)


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return negative_mask


class GaussianBlur(object):

    def __init__(self, min=0.1, max=2.0, kernel_size=9):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

# if use_cosine_similarity:
#     cos1d = torch.nn.CosineSimilarity(dim=1)
#     cos2d = torch.nn.CosineSimilarity(dim=2)
#     similarity_dim1 = lambda x, y: cos1d(x, y.unsqueeze(0))
#     similarity_dim2 = lambda x, y: cos2d(x, y.unsqueeze(0))
# else:
#     similarity_dim1 = lambda x, y: torch.bmm(x.unsqueeze(1), y.unsqueeze(2))
#     similarity_dim2 = lambda x, y: torch.tensordot(x, y.T.unsqueeze(0), dims=2)
