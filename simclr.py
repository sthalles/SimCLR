import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from torchvision import datasets, transforms
# torch.manual_seed(0)



class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=f"./runs/{self.args.dataset_name}_TreeLevel{self.args.level_number}_lossAtAll{self.args.loss_at_all_level}_reg{self.args.regularization}reg_att_all{self.args.regularization_at_all_level}_pernode{self.args.per_node}_perlevel{self.args.per_level}")
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels
    
    
    def probability_vec_with_level(self, feature, level):
        prob_vec = torch.tensor([], device=self.args.device, requires_grad=True)
        for u in torch.arange(2**level-1, 2**(level+1) - 1, dtype=torch.long):
            # print(torch.arange(2**level-1, 2**(level+1) - 1, dtype=torch.long))
            probability_u = torch.ones_like(feature[:, 0], device=self.args.device, dtype=torch.float32)
            while(u > 0):
                if u/2 > torch.floor(u/2):
                    # print(f'{u} : {torch.floor(u/2)}')
                    # Poszlismy w lewo
                    u = torch.floor(u/2) 
                    u = u.long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # Poszlismy w prawo
                    # print(f' 1 - {u} : {torch.floor(u/2) - 1}') 
                    u = torch.floor(u/2) - 1
                    u = u.long()
                    probability_u *=  1 - feature[:, u]
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
        return prob_vec
    
    def probability_vec(self, feature):
        prob_vec = torch.tensor([], device=self.args.device, requires_grad=True)
        for u in torch.arange(2**self.args.level_number-1, 2**(self.args.level_number+1) - 1, dtype=torch.long):
            probability_u = torch.ones_like(feature[:, 0], device=self.args.device, dtype=torch.float32)
            while(u > 0):
                if u/2 > torch.floor(u/2):
                    # Poszlismy w lewo
                    u = torch.floor(u/2) 
                    u = u.long()
                    probability_u *= feature[:, u]
                elif u/2 == torch.floor(u/2):
                    # Poszlismy w prawo
                    u = torch.floor(u/2) - 1 
                    u = u.long()
                    probability_u *=  1 - feature[:, u]
            prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
        return prob_vec
            
            
    def binary_tree_loss(self, features):
        loss_value = torch.tensor([0], device=self.args.device, dtype=torch.float32)
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        
        # probability_leaves = []
        # for leave in range(((2**(self.args.level_number+1))-1) - 2**self.args.level_number):
        #      probability_leaves.append(torch.mean(features[:, leave]))
        # # for leave in range(((2**(self.args.level_number+1))-1) - 2**self.args.level_number):
        # loss_value += torch.mean(torch.log(probability_leaves))
        # discard the main diagonal from both: labels - there are not positive classes
        labels = labels * ~mask
        # Calculate probability vec based upon binary tree Approach
        if self.args.loss_at_all_level:
            for level in range(1, self.args.level_number + 1):
                prob_features = self.probability_vec_with_level(features, level)
                # print(prob_features.shape)
                # loss_value += torch.nn.KLDivLoss(reduction='mean')(torch.log((prob_features[torch.where(labels > 0)[0]]+  1e-8)), (prob_features[torch.where(labels > 0)[1]] + 1e-8))
                # loss_value -= torch.sum((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8)))) / prob_features[torch.where(labels > 0)[0]].shape[0]
                loss_value -= torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))

                # Calculate loss on negative classes
                # loss_value -=torch.nn.KLDivLoss(reduction='mean')(torch.log((prob_features[torch.where(labels == 0)[0]] + 1e-8)), (prob_features[torch.where(labels == 0)[1]] + 1e-8))
                # loss_value += torch.sum((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8)))) / prob_features[torch.where(labels == 0)[0]].shape[0]
                loss_value += torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))
            # return loss_value
        else:
            # return loss_value
            prob_features = self.probability_vec(features)
            
            # Calculate loss on positive classes
            # To avoid nan while calculating sqrt https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702  https://github.com/richzhang/PerceptualSimilarity/issues/69
            loss_value -= torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8), torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8))))
            # Calculate loss on negative classes
            loss_value += torch.mean((torch.bmm(torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8), torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8))))

        # probability_leaves = []
        # for leave in range(0, 2**self.args.level_number):
        #      probability_leaves.append(torch.mean(prob_features[:, leave]))
        # loss_value += torch.nn.L1Loss()(torch.tensor(probability_leaves), (1/(2**self.args.level_number) * torch.ones_like(torch.tensor(probability_leaves))))
        # for leave in range(((2**(self.args.level_number+1))-1) - 2**self.args.level_number):
        # loss_value += torch.mean(torch.log(torch.tensor(probability_leaves)))        
        return loss_value
        

    def train(self, train_loader):
        torch.autograd.set_detect_anomaly(True)
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for i, (images, label) in enumerate(tqdm(train_loader)):

                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)
                loss_ce = torch.tensor([0], device=self.args.device, dtype=torch.float32)
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    features2 = self.model.backbone(images)
                    logits2, labels2 = self.info_nce_loss(features2)
                    loss = self.binary_tree_loss(features)
                    loss2 = self.criterion(logits2, labels2)
                    loss_sum = loss + loss2
                
                    # loss to have unifrom dist on leaves
                    if self.args.regularization:
                        loss_reg = torch.tensor([0], device=self.args.device, dtype=torch.float32)
                        if self.args.regularization_at_all_level:
                            for level in range(1, self.args.level_number+1):
                                prob_features = self.probability_vec_with_level(features, level)
                                probability_leaves = torch.mean(prob_features, dim=0)
                                if self.args.per_level:
                                    loss_reg += (-torch.sum((1/(2**level)) * torch.log(probability_leaves)))
                                if self.args.per_node:
                                    for leftnode in range(0,int((2**level)/2)):
                                        loss_reg -=  (0.5 * torch.log(probability_leaves[2*leftnode]) + 0.5 * torch.log(probability_leaves[2*leftnode+1]))
                        else:
                            loss_reg = torch.tensor([0], device=self.args.device, dtype=torch.float32)
                            prob_features = self.probability_vec_with_level(features, self.args.level_number)
                            probability_leaves = torch.mean(prob_features, dim=0)
                            if self.args.per_level:
                                loss_reg += (-torch.sum((1/(2**self.args.level_number)) * torch.log(probability_leaves)))
                            if self.args.per_node:
                                for leftnode in range(0,int((2**self.args.level_number)/2)):
                                        loss_reg -=  (0.5 * torch.log(probability_leaves[2*leftnode]) + 0.5 * torch.log(probability_leaves[2*leftnode+1]))
                        loss_sum += (1/(2**self.args.level_number)) * loss_reg
                
                self.optimizer.zero_grad()
                scaler.scale(loss_sum).backward()
                scaler.step(self.optimizer)
                scaler.update() 
                
                if n_iter % self.args.log_every_n_steps == 0:
                    # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss tree', loss, global_step=n_iter)
                    self.writer.add_scalar('loss simclr', loss2, global_step=n_iter)
                    if self.args.regularization:
                        self.writer.add_scalar('loss reg', loss_reg, global_step=n_iter)
                    # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                    self.writer.add_histogram('histogram of the outputs', features, epoch_counter)
                    
                    
                n_iter += 1
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tTree Loss: {loss}\t")
            logging.debug(f"Epoch: {epoch_counter}\t SimCLR Loss {loss2}")
            if self.args.regularization:
                logging.debug(f"Epoch: {epoch_counter}\t Regularization Loss: {loss_reg}\t")

            for n, p in self.model.named_parameters():
                if 'bias' not in n:
                    self.writer.add_histogram('{}'.format(n), p, epoch_counter)
                    if p.requires_grad:
                        self.writer.add_histogram('{}.grad'.format(n), p.grad, epoch_counter)
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
        