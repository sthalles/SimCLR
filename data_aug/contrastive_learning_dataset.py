from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
class CIFAROnlyKClasses(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform, classes):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.original_cifar = datasets.CIFAR10('./', train=True)
        self.labels = [label for elem,label in self.original_cifar if label in classes]
        self.photos = [elem for elem,label in self.original_cifar if label in classes]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform:
             transformed_photo  = self.transform(self.photos[idx])
        return transformed_photo, self.labels[idx]
    
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),
                          'cifar10kclasses': lambda: CIFAROnlyKClasses(self.root_folder,transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),classes=(1,3,8)),
                          
                          'svhn': lambda: datasets.SVHN(self.root_folder, split='train',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),
                          

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          
                          'mnist': lambda: datasets.MNIST(self.root_folder, train=True,
                                                         transform=ContrastiveLearningViewGenerator(
                                                              transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
                                                                    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                                                                    transforms.RandomAffine(degrees=15,
                                                                                            translate=[0.1, 0.1],
                                                                                            scale=[0.9, 1.1],
                                                                                            shear=15),
                                                                ]),
                                                                                                    n_views),
                                                              download=True),
                          'fmnist': lambda: datasets.FashionMNIST(self.root_folder, train=True,
                                                         transform=ContrastiveLearningViewGenerator(
                                                              transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
                                                                    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                                                                    transforms.RandomAffine(degrees=15,
                                                                                            translate=[0.1, 0.1],
                                                                                            scale=[0.9, 1.1],
                                                                                            shear=15),
                                                                ]),
                                                                                                    n_views),
                                                              download=True)
                          }
                        

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
