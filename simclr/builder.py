import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):

    def __init__(self, base_encoder, dim=128, mlp_dim=2048, T=0.1):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimCLR, self).__init__()

        self.T = T
        self.criterion = nn.CrossEntropyLoss()

        # build encoder
        self.encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    def nt_xent(self, features):

        feat_a, feat_b = torch.chunk(features, 2)

        batch_size, _ = feat_a.shape

        feat_a_large = concat_all_gather(feat_a)
        feat_b_large = concat_all_gather(feat_b)

        enlarged_batch_size = feat_a_large.shape[0]

        masks = torch.eye(
            batch_size, enlarged_batch_size, dtype=torch.bool, device=feat_a.device
        )

        logits_aa = torch.matmul(feat_a, feat_a_large.t()) / self.T
        logits_aa.masked_fill_(
            masks, value=torch.finfo(torch.float16).min
        )  # mask out main diagonal (dot product between the same view)

        logits_bb = torch.matmul(feat_b, feat_b_large.t()) / self.T
        logits_bb.masked_fill_(
            masks, value=torch.finfo(torch.float16).min
        )  # mask out main diagonal (dot product between the same view)

        logits_ab = torch.matmul(feat_a, feat_b_large.t()) / self.T
        logits_ba = torch.matmul(feat_b, feat_a_large.t()) / self.T

        targets = torch.arange(batch_size, dtype=torch.long).cuda()
        loss_a = self.criterion(torch.cat([logits_ab, logits_aa], dim=-1), targets)
        loss_b = self.criterion(torch.cat([logits_ba, logits_bb], dim=-1), targets)
        loss = loss_a + loss_b
        return loss

    def forward(self, images):
        """
        Input:
            images: Input containing two views of an image, shape (2 * BS, dim)
        Output:
            loss
        """
        # compute features
        feats = self.encoder(images)
        feats = F.normalize(feats, dim=1)

        return self.nt_xent(feats)


class SimCLR_ResNet(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder.fc.weight.shape[1]

        # projectors
        self.encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)


class SimCLR_ViT(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder.head.weight.shape[1]

        # projectors
        self.encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
