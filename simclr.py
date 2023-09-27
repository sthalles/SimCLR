import logging
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class InfoNCELoss(nn.Module):

    @staticmethod
    def loss_forward(features: torch.Tensor, batch_size: int, n_views: int, temperature: float):
        labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0).to(features.device)
        # noinspection PyUnresolvedReferences
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0] * (n_views - 1), -1)

        # select only the negatives
        # change: copy if n_views > 2 for other positive pairs of img
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1).repeat(n_views - 1, 1)

        logits = torch.cat([positives, negatives], dim=1)
        # the idx-0 corresponding to similarity between same img from different views (positive pairs) while the
        # other columns correspond to similarity between negative pairs.
        # the objective is to get the feature representation such that the positive pairs have higher similarity
        # (0-th column in logits) while the negative pairs (the rest of columns) have lower similairty.
        # therefore the label is set to 0 and crossentropy loss is applied afterward between label and logits.
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / temperature
        return logits, labels

    def __init__(self, batch_size, n_views, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature

    def forward(self, features):
        return InfoNCELoss.loss_forward(features, self.batch_size, self.n_views, self.temperature)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.info_nce_loss = InfoNCELoss(self.args.batch_size, self.args.n_views, self.args.temperature)

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
