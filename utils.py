import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

np.random.seed(0)
cos1d = torch.nn.CosineSimilarity(dim=1)
cos2d = torch.nn.CosineSimilarity(dim=2)


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return negative_mask


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def get_augmentation_transform(s, crop_size):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_aug_ope = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomResizedCrop(size=crop_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomApply([color_jitter], p=0.8),
                                       transforms.RandomGrayscale(p=0.2),
                                       GaussianBlur(kernel_size=int(0.1 * crop_size)),
                                       transforms.ToTensor()])
    return data_aug_ope


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = torch.bmm(x.unsqueeze(1), y.unsqueeze(2))  #
    return v


def _dot_simililarity_dim2(x, y):
    v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


def _cosine_simililarity_dim1(x, y):
    v = cos1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cos2d(x.unsqueeze(1), y.unsqueeze(0))
    return v


def get_similarity_function(use_cosine_similarity):
    if use_cosine_similarity:
        return _cosine_simililarity_dim1, _cosine_simililarity_dim2
    else:
        return _dot_simililarity_dim1, _dot_simililarity_dim2
