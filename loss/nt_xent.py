import torch
import numpy as np


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.5, use_cosine_similarity: bool = True):
        super().__init__()
        self.temperature = temperature

        if use_cosine_similarity:
            self._similarity_function = self._cosine_simililarity
        else:
            self._similarity_function = self._dot_simililarity

    def forward(self, zis, zjs):
        representations = torch.stack([zjs, zis], dim=1)
        representations = representations.reshape(-1, representations.shape[-1])
        score = torch.nn.functional.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )
        score /= self.temperature
        # mask self cosine in the diagonal
        score[range(len(score)), range(len(score))] = float("-inf")
        # generate target
        target = score.new_tensor(
            [i + (-1) ** (i % 2) for i in range(len(score))], dtype=torch.long
        )
        loss = torch.nn.functional.cross_entropy(score, target)

        return loss

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (2N, 2N)
        return v

    @staticmethod
    def _cosine_simililarity(x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        v = torch.nn.functional.cosine_similarity(
            x.unsqueeze(1), y.unsqueeze(0), dim=-1
        )
        return v
