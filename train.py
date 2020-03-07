import torch
import yaml

print(torch.__version__)
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.resnet_simclr import ResNetSimCLR
from utils import get_negative_mask, get_similarity_function
from data_aug.data_transform import DataTransform, get_data_transform_opes

torch.manual_seed(0)

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

batch_size = config['batch_size']
out_dim = config['out_dim']
temperature = config['temperature']
use_cosine_similarity = config['use_cosine_similarity']

data_augment = get_data_transform_opes(s=config['s'], crop_size=96)

train_dataset = datasets.STL10('./data', split='unlabeled', download=True, transform=DataTransform(data_augment))

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=config['num_workers'], drop_last=True,
                          shuffle=True)

# model = Encoder(out_dim=out_dim)
model = ResNetSimCLR(base_model=config["base_convnet"], out_dim=out_dim)

train_gpu = torch.cuda.is_available()
print("Is gpu available:", train_gpu)

# moves the model parameters to gpu
if train_gpu:
    model.cuda()

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), 3e-4)

train_writer = SummaryWriter()

sim_func_dim1, sim_func_dim2 = get_similarity_function(use_cosine_similarity)

# Mask to remove positive examples from the batch of negative samples
negative_mask = get_negative_mask(batch_size)

n_iter = 0
for e in range(config['epochs']):
    for step, ((xis, xjs), _) in enumerate(train_loader):

        optimizer.zero_grad()

        if train_gpu:
            xis = xis.cuda()
            xjs = xjs.cuda()

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        train_writer.add_histogram("xi_repr", ris, global_step=n_iter)
        train_writer.add_histogram("xi_latent", zis, global_step=n_iter)

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]
        train_writer.add_histogram("xj_repr", rjs, global_step=n_iter)
        train_writer.add_histogram("xj_latent", zjs, global_step=n_iter)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        l_pos = sim_func_dim1(zis, zjs).view(batch_size, 1)
        l_pos /= temperature

        negatives = torch.cat([zjs, zis], dim=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = torch.zeros(batch_size, dtype=torch.long)
            if train_gpu:
                labels = labels.cuda()

            l_neg = l_neg[negative_mask].view(l_neg.shape[0], -1)
            l_neg /= temperature

            logits = torch.cat([l_pos, l_neg], dim=1)  # [N,K+1]
            loss += criterion(logits, labels)

        loss = loss / (2 * batch_size)
        train_writer.add_scalar('loss', loss, global_step=n_iter)

        loss.backward()
        optimizer.step()
        n_iter += 1

model_checkpoints_folder = os.path.join(train_writer.log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder)

torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
