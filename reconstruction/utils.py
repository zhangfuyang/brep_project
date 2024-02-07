import yaml
import torch

def collate_fn(batch):
    sdf = torch.stack([item['sdf'] for item in batch])
    face_dist = torch.cat([item['face_dist'] for item in batch], dim=0)
    face_num = torch.tensor([item['face_num'] for item in batch])
    filename = [item['filename'] for item in batch]
    return {'sdf': sdf, 'face_dist': face_dist, 'face_num': face_num, 'filename': filename}


def load_model(model_class, config_path, pretrained_model_path=None, exclude_prefix='vae_model.'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = model_class(**config['model_params'])
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location='cpu')['state_dict']
        # modify the key names
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(exclude_prefix):
                new_state_dict[k[len(exclude_prefix):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    
    return model