import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

np.random.seed(0)


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


def get_augmentation_transform(s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_aug_ope = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomResizedCrop(96),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomApply([color_jitter], p=0.8),
                                       transforms.RandomGrayscale(p=0.2),
                                       GaussianBlur(),
                                       transforms.ToTensor()])
    return data_aug_ope

# if use_cosine_similarity:
#     cos1d = torch.nn.CosineSimilarity(dim=1)
#     cos2d = torch.nn.CosineSimilarity(dim=2)
#     similarity_dim1 = lambda x, y: cos1d(x, y.unsqueeze(0))
#     similarity_dim2 = lambda x, y: cos2d(x, y.unsqueeze(0))
# else:
#     similarity_dim1 = lambda x, y: torch.bmm(x.unsqueeze(1), y.unsqueeze(2))
#     similarity_dim2 = lambda x, y: torch.tensordot(x, y.T.unsqueeze(0), dims=2)
