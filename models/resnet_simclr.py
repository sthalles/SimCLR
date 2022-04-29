import torch.nn as nn
import torchvision.models as models
import torch
from torch import Tensor


from exceptions.exceptions import InvalidBackboneError

class GumbelSigmoid(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    
    # Math:
    # Sigmoid is a softmax of two logits: a and 0
    # e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    # Gumbel-sigmoid is a gumbel-softmax for same logits:
    # gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(0 + gumbel2/t)]
    # where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    # gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    # gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    # For computation reasons:
    # gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    # gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    
    def __init__(self, temp : float = 0.4, eps:float =1e-10) -> None:
        super(GumbelSigmoid, self).__init__()
        self.temp = temp
        self.eps = eps
        
    def forward(self, input: Tensor) -> Tensor:
        # 
        if self.training:
            # input ma shape (batch,ilosc ilosc_wezlow_wew) -  (ilosc wezlow wew)
            # print(input.shape)
            # print(input.shape[1])
            uniform1 = torch.rand(input.shape[1]).cuda()
            uniform2 = torch.rand(input.shape[1]).cuda()
            
            noise = -torch.log(torch.log(uniform2 + self.eps)/torch.log(uniform1 + self.eps) +self.eps)
            
            #draw a sample from the Gumbel-Sigmoid distribution
            return torch.sigmoid((input + noise) / self.temp)
        else:
            return torch.sigmoid(input)
            

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, args=None):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        if args.dataset_name == 'mnist' or args.dataset_name == 'fmnist':
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        
        sigmoidType = GumbelSigmoid(temp=args.temp) if args.gumbel else nn.Sigmoid() 
        self.tree_model = nn.Sequential(nn.Linear(args.out_dim, ((2**(args.level_number+1))-1) - 2**args.level_number), sigmoidType)
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
