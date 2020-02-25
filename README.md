# PyTorch implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

Check out the Blog post with full documentation: [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sthalles.github.io/simple-self-supervised-learning/)

## Config file

Before runing SimCLR, make sure you choose the correct running configurations on the ```config.yaml``` file.

```
batch_size: 256 # A batch size of N, produces 2 * (N-1) negative samples. Original implementation uses a batch size of 8192
out_dim: 64 # Output dimensionality of the embedding vector z. Original implementation uses 2048
s: 1
temperature: 0.5 # Temperature parameter for the contrastive objective
base_convnet: "resnet18" # The ConvNet base model. Choose one of: "resnet18 or resnet50". Original implementation uses resnet50
use_cosine_similarity: True # Distance metric for contrastive loss. If False, uses dot product
epochs: 40 # Number of epochs to train
num_workers: 4 # Number of workers for the data loader
```
