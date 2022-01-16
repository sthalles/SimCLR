import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from models.resnet_simclr import ResNetSimCLR
import argparse
import torch
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from tqdm import tqdm
from torchvision import models
import os
import yaml

def _load_pre_trained_weights(model):
        try:
            file = open(r'config.yaml')
            cfg = yaml.load(file, Loader=yaml.FullLoader)
            checkpoints_folder = os.path.join('./runs', cfg['folder'])
            checkpoint = torch.load(os.path.join(checkpoints_folder, cfg['model']))
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')





def plot():
        
    def _plot(v, labels, centroids, fname):
        fig = plt.figure(figsize=(10, 10))
        plt.axis('off')
        sns.set_style("darkgrid")
        sns.scatterplot(x=v[:, 0], y=v[:, 1], hue=labels, legend='full', palette=sns.color_palette("bright", 10))
        sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], marker='x')
        plt.legend(list(range(10)))
        plt.savefig(fname)
        plt.close()
        
        
            
    with torch.no_grad():
        args = parser.parse_args()  
        dataset = ContrastiveLearningDataset(args.data)

        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
            
        ResnetsimCLR = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).cpu()
            
        ResnetsimCLR = _load_pre_trained_weights(ResnetsimCLR)
        
        kmeans = KMeans(n_clusters=10)
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)

        ResnetsimCLR.eval()
        x = []  
        y_ = []
        for (xis, xjs), target in tqdm(train_loader):
            ris = ResnetsimCLR(xjs)  # [N,C]
            x.append(ris.cpu())
            y_.append(target.cpu())
        y = torch.stack(y_).cpu().view(-1)
        x = torch.stack(x).cpu().view(-1, args.out_dim)
        x = tsne.fit_transform(x)
        y_tsne = kmeans.fit_transform(pca.fit_transform(x))
        _plot(x, y, kmeans.cluster_centers_, 'cluster.png')
            
if __name__ == "__main__":
    print("Calling Main()...")
    plot()
    print("Plot saved...")