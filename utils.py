import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

np.random.seed(0)
cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


def get_train_validation_data_loaders(train_dataset, config):
    # obtain training indices that will be used for validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(config['valid_size'] * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
                              num_workers=config['num_workers'], drop_last=True, shuffle=False)

    valid_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=valid_sampler,
                              num_workers=config['num_workers'],
                              drop_last=True)
    return train_loader, valid_loader


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
    v = cosine_similarity(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
    return v


def get_similarity_function(use_cosine_similarity):
    if use_cosine_similarity:
        return _cosine_simililarity_dim1, _cosine_simililarity_dim2
    else:
        return _dot_simililarity_dim1, _dot_simililarity_dim2
