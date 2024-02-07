import yaml
import torch


def load_model(model_class, config_path, pretrained_model_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = model_class(**config['model_params'])
    if pretrained_model_path is not None:
        model.load_state_dict(
            torch.load(
                pretrained_model_path, 
                map_location='cpu')['state_dict'], strict=True)
    
    return model