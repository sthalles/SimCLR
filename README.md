# PyTorch implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

### Blog post with full documentation: [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sthalles.github.io/simple-self-supervised-learning/)

#### For a Tensorflow 2.0 Implementation: [Tensorflow SimCLR](https://github.com/sthalles/SimCLR-tensorflow)

![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)


## Dependencies

- pytorch
- opencv

## Config file

Before running SimCLR, make sure you choose the correct running configurations on the ```config.yaml``` file.

```yaml
batch_size: 256 # A batch size of N, produces 2 * (N-1) negative samples. Original implementation uses a batch size of 8192
out_dim: 64 # Output dimensionality of the embedding vector z. Original implementation uses 2048
s: 1
temperature: 0.5 # Temperature parameter for the contrastive objective
base_convnet: "resnet18" # The ConvNet base model. Choose one of: "resnet18 or resnet50". Original implementation uses resnet50
use_cosine_similarity: True # Distance metric for contrastive loss. If False, uses dot product
epochs: 40 # Number of epochs to train
num_workers: 4 # Number of workers for the data loader
valid_size: 0.05 # validation set size
eval_every_n_epochs: 2 # frequency to eval the feature representations' quality 
```

## Feature Evaluation

Feature evaluation is done using a linear model protocol. Feature are learned using the ```STL10 unsupervised``` set and evaluated in the train/test splits;

Check the ```feature_eval/linear_feature_eval.ipynb``` notebook for reproducebility.
|  Feature Extractor  |    Method    | Architecture | Top 1 |
|:-------------------:|:------------:|:------------:|:-----:|
| Logistic Regression | PCA Features |       -      | 36.0% |
|         KNN         | PCA Features |       -      | 31.8% |
| Logistic Regression |    SimCLR    |   ResNet-18  | 75.0% |
|         KNN         |    SimCLR    |   ResNet-18  | 70.0% |

## Download pre-trained model 

- [ResNet-18](https://drive.google.com/open?id=1zDHvk1oE1k3b4ApsK1AdMREIAcpkG-We) Trained using ```STl10 unsupervised``` set.
