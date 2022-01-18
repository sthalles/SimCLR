import os
import shutil

import torch
import yaml
import gdown

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def load_model_to_steal(folder_name, model, victim_mlp, device, discard_mlp=False):
    def get_file_id_by_model(folder_name):
        file_id = {'resnet18_100-epochs_stl10': '14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF',
                   'resnet18_100-epochs_cifar10': '1lc2aoVtrAetGn0PnTkOyFzPCIucOJq7C',
                   'resnet50_50-epochs_stl10': '1ByTKAUsdm_X7tLcii6oAEl5qFRqRMZSu'}
        return file_id.get(folder_name, "Model not found.")
    
    # file_id = get_file_id_by_model(folder_name)
    print("Stealing model: ", folder_name)
    
    # download and extract model files
    # url = 'https://drive.google.com/uc?id={}'.format(file_id)
    # output = 'checkpoint_0100.pth.tar'
    # gdown.download(url, output, quiet=False)
    checkpoint = torch.load('/ssd003/home/nikita/SimCLR/runs/{}/checkpoint_0200.pth.tar'.format(folder_name), map_location=device)
    state_dict = checkpoint['state_dict']
    mlp_state_dict = checkpoint['mlp_state_dict']

    if discard_mlp:
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]
        
    log = model.load_state_dict(state_dict, strict=False)
    victim_mlp.load_state_dict(mlp_state_dict)
    return model, victim_mlp
