import shutil

import torch
import yaml

print(torch.__version__)
import torch.optim as optim
import os

from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from models.resnet_simclr import ResNetSimCLR
from utils import get_similarity_function, get_train_validation_data_loaders
from data_aug.data_transform import DataTransform, get_simclr_data_transform

torch.manual_seed(0)
np.random.seed(0)

config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

batch_size = config['batch_size']
out_dim = config['out_dim']
temperature = config['temperature']
use_cosine_similarity = config['use_cosine_similarity']

data_augment = get_simclr_data_transform(s=config['s'], crop_size=96)

train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True, transform=DataTransform(data_augment))

train_loader, valid_loader = get_train_validation_data_loaders(train_dataset, **config)

# model = Encoder(out_dim=out_dim)
model = ResNetSimCLR(base_model=config["base_convnet"], out_dim=out_dim)

if config['continue_training']:
    checkpoints_folder = os.path.join('./runs', config['continue_training'], 'checkpoints')
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
    model.load_state_dict(state_dict)
    print("Loaded pre-trained model with success.")

train_gpu = torch.cuda.is_available()
print("Is gpu available:", train_gpu)

# moves the model parameters to gpu
if train_gpu:
    model = model.cuda()

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), 3e-4)

train_writer = SummaryWriter()

similarity_func = get_similarity_function(use_cosine_similarity)

megative_mask = (1 - torch.eye(2 * batch_size)).type(torch.bool)
labels = (np.eye((2 * batch_size), 2 * batch_size - 1, k=-batch_size) + np.eye((2 * batch_size), 2 * batch_size - 1,
                                                                               k=batch_size - 1)).astype(np.int)
labels = torch.from_numpy(labels)
softmax = torch.nn.Softmax(dim=-1)

if train_gpu:
    labels = labels.cuda()


def step(xis, xjs):
    # get the representations and the projections
    ris, zis = model(xis)  # [N,C]

    # get the representations and the projections
    rjs, zjs = model(xjs)  # [N,C]

    if n_iter % config['log_every_n_steps'] == 0:
        train_writer.add_histogram("xi_repr", ris, global_step=n_iter)
        train_writer.add_histogram("xi_latent", zis, global_step=n_iter)
        train_writer.add_histogram("xj_repr", rjs, global_step=n_iter)
        train_writer.add_histogram("xj_latent", zjs, global_step=n_iter)

    # normalize projection feature vectors
    zis = F.normalize(zis, dim=1)
    zjs = F.normalize(zjs, dim=1)

    negatives = torch.cat([zjs, zis], dim=0)

    logits = similarity_func(negatives, negatives)
    logits = logits[megative_mask.type(torch.bool)].view(2 * batch_size, -1)
    logits /= temperature
    # assert logits.shape == (2 * batch_size, 2 * batch_size - 1), "Shape of negatives not expected." + str(
    #     logits.shape)

    probs = softmax(logits)
    loss = torch.mean(-torch.sum(labels * torch.log(probs), dim=-1))

    return loss


model_checkpoints_folder = os.path.join(train_writer.log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder)
    shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

n_iter = 0
valid_n_iter = 0
best_valid_loss = np.inf

for epoch_counter in range(config['epochs']):
    for (xis, xjs), _ in train_loader:
        optimizer.zero_grad()

        if train_gpu:
            xis = xis.cuda()
            xjs = xjs.cuda()

        loss = step(xis, xjs)

        train_writer.add_scalar('train_loss', loss, global_step=n_iter)
        loss.backward()
        optimizer.step()
        n_iter += 1

    if epoch_counter % config['eval_every_n_epochs'] == 0:

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            for counter, ((xis, xjs), _) in enumerate(valid_loader):

                if train_gpu:
                    xis = xis.cuda()
                    xjs = xjs.cuda()
                loss = (step(xis, xjs))
                valid_loss += loss.item()

            valid_loss /= counter

            if valid_loss < best_valid_loss:
                # save the model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

            train_writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
            valid_n_iter += 1

        model.train()
