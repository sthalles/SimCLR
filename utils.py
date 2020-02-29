import numpy as np
import torch

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
