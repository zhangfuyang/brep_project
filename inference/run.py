import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from dataset import VoxelDataset
from experiment import VAEExperiment, VAEExperiment2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_model import VQVAE3D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/vae.yaml')
parser.add_argument('--pretrained_weight', type=str, default='checkpoints/last.ckpt')
parser.add_argument('--output_dir', type=str, default='output')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

model = VQVAE3D(**config['model_params'])
# load pretrained model
state_dict = torch.load(args.pretrained_weight, map_location='cpu')['state_dict']
# modify the key names
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('vae_model.'):
        new_state_dict[k[10:]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=True)

experiment = VAEExperiment2(config['exp_params'], model)

val_dataset = VoxelDataset(config['data_params'], 'val')
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['data_params']['val_batch_size'], 
    shuffle=False, num_workers=config['data_params']['num_workers'])

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1, 
    default_root_dir=args.output_dir)

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.validate(experiment, val_dataloader)

