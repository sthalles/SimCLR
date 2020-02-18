import numpy as np
import torch

print(torch.__version__)
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from model import Encoder
from utils import GaussianBlur

batch_size = 3
out_dim = 4
s = 1
temperature = 0.5

color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

data_augment = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomResizedCrop(96),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomApply([color_jitter], p=0.8),
                                   transforms.RandomGrayscale(p=0.2),
                                   GaussianBlur(),
                                   transforms.ToTensor()])

train_dataset = datasets.STL10('data', split='train', download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, drop_last=True, shuffle=True)

model = Encoder(out_dim=out_dim)
print(model)

train_gpu = torch.cuda.is_available()
print("Is gpu available:", train_gpu)
# moves the model paramemeters to gpu
if train_gpu:
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 3e-4)

train_writer = SummaryWriter()

similarity_dim1 = torch.nn.CosineSimilarity(dim=1)
similarity_dim2 = torch.nn.CosineSimilarity(dim=2)

# This mask can only be created once
# Mask to remove positive examples from the batch of negative samples
# negative_mask = torch.ones((batch_size, 2*batch_size), dtype=bool)
# for i in range(batch_size):
#     negative_mask[i, i] = 0
#     negative_mask[i, i + batch_size] = 0

n_iter = 0
for e in range(40):
    for step, (batch_x, _) in enumerate(train_loader):
        # print("Input batch:", batch_x.shape, torch.min(batch_x), torch.max(batch_x))
        optimizer.zero_grad()

        xis = []
        xjs = []
        for k in range(len(batch_x)):
            xis.append(data_augment(batch_x[k]))
            xjs.append(data_augment(batch_x[k]))

        # fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=False)
        # for i_ in range(3):
        #     axs[i_, 0].imshow(xis[i_].permute(1, 2, 0))
        #     axs[i_, 1].imshow(xjs[i_].permute(1, 2, 0))
        # plt.show()

        xis = torch.stack(xis)
        xjs = torch.stack(xjs)
        if train_gpu:
            xis = xis.cuda()
            xjs = xjs.cuda()
        # print("Transformed input stats:", torch.min(xis), torch.max(xjs))

        ris, zis = model(xis)  # [N,C]
        train_writer.add_histogram("xi_repr", ris, global_step=n_iter)
        train_writer.add_histogram("xi_latent", zis, global_step=n_iter)
        # print(his.shape, zis.shape)

        rjs, zjs = model(xjs)  # [N,C]
        train_writer.add_histogram("xj_repr", rjs, global_step=n_iter)
        train_writer.add_histogram("xj_latent", zjs, global_step=n_iter)
        # print(hjs.shape, zjs.shape)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        assert zis.shape == (batch_size, out_dim), "Shape not expected."
        assert zis.shape == (batch_size, out_dim), "Shape not expected."

        # positive pairs
        # l_pos = torch.bmm(zis.view(batch_size, 1, out_dim), zjs.view(batch_size, out_dim, 1)).view(batch_size, 1)
        l_pos = similarity_dim1(zis.view(batch_size, out_dim), zjs.view(batch_size, out_dim)).view(batch_size,
                                                                                                   1) / temperature

        assert l_pos.shape == (batch_size, 1)  # [N,1]
        l_neg = []

        #############
        #############
        # negatives = torch.cat([zjs, zis], dim=0)
        #############
        #############

        for i in range(zis.shape[0]):
            mask = np.ones(zjs.shape[0], dtype=bool)
            mask[i] = False
            negs = torch.cat([zjs[mask], zis[mask]], dim=0)  # [2*(N-1), C]
            # l_neg.append(torch.mm(zis[i].view(1, zis.shape[-1]), negs.permute(1, 0)))
            l_neg.append(similarity_dim1(zis[i].view(1, zis.shape[-1]), negs).flatten())

        l_neg = torch.stack(l_neg)  # [N, 2*(N-1)]
        l_neg /= temperature

        assert l_neg.shape == (batch_size, 2 * (batch_size - 1)), "Shape of negatives not expected." + str(l_neg.shape)
        # print("l_neg.shape -->", l_neg.shape)

        #############
        #############
        # l_negs = similarity_dim2(zis.view(batch_size, 1, out_dim), negatives.view(1, (2*batch_size), out_dim))
        # l_negs = l_negs[negative_mask].view(l_negs.shape[0], -1)
        # l_negs /= temperature
        #############
        #############

        logits = torch.cat([l_pos, l_neg], dim=1)  # [N,K+1]
        # print("logits.shape -->",logits.shape)

        labels = torch.zeros(batch_size, dtype=torch.long)

        if train_gpu:
            labels = labels.cuda()

        loss = criterion(logits, labels)
        train_writer.add_scalar('loss', loss, global_step=n_iter)

        loss.backward()
        optimizer.step()
        n_iter += 1
        # print("Step {}, Loss {}".format(step, loss))

torch.save(model.state_dict(), './model/checkpoint.pth')
