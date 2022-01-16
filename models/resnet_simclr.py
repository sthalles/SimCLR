import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, args=None):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        if args.dataset_name == 'mnist':
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        
        self.tree_model = nn.Sequential(nn.Linear(args.out_dim, ((2**(args.level_number+1))-1) - 2**args.level_number), nn.Sigmoid())
        # self.tree_model_new = nn.Sequential(nn.Linear(args.out_dim, ((2**(args.level_number+1))-1) - 2**args.level_number), nn.Softmax())
        self.tree_model = self.tree_model.to(args.device)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.backbone(x)
        return self.tree_model(x)
