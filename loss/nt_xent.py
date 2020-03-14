import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask()
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.labels = self._get_labels()

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_labels(self):
        labels = (np.eye((2 * self.batch_size), 2 * self.batch_size - 1, k=-self.batch_size) + np.eye(
            (2 * self.batch_size),
            2 * self.batch_size - 1,
            k=self.batch_size - 1)).astype(np.int)
        labels = torch.from_numpy(labels)
        labels = labels.to(self.device)
        return labels

    def _get_correlated_mask(self):
        mask_samples_from_same_repr = (1 - torch.eye(2 * self.batch_size)).type(torch.bool)
        return mask_samples_from_same_repr

    def _dot_simililarity(self, x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        negatives = torch.cat([zjs, zis], dim=0)

        logits = self.similarity_function(negatives, negatives)
        logits = logits[self.mask_samples_from_same_repr.type(torch.bool)].view(2 * self.batch_size, -1)
        logits /= self.temperature
        assert logits.shape == (2 * self.batch_size, 2 * self.batch_size - 1), "Shape of negatives not expected." + str(
            logits.shape)

        probs = self.softmax(logits)
        loss = torch.mean(-torch.sum(self.labels * torch.log(probs), dim=-1))
        return loss
