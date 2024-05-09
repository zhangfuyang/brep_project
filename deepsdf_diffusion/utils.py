import yaml
import torch
import glob


def load_model(model_class, config_path, pretrained_model_path=None, 
               exclude_prefix='vae_model.', remove_keys=()):
    if config_path is None:
        return None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = model_class(**config['model_params'])
    
    if pretrained_model_path is not None:
        pretrained_model_path = glob.glob(pretrained_model_path)[0]
        state_dict = torch.load(pretrained_model_path, map_location='cpu')['state_dict']
        # modify the key names
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in remove_keys:
                continue
            if k.startswith(exclude_prefix):
                new_state_dict[k[len(exclude_prefix):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    
    return model
