import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from dataset import VoxelDataset
from experiment import ReconExperiment
from utils import load_model, collate_fn
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_model import VQVAE3D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/vae.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

face_model = load_model(VQVAE3D, 
                        config['exp_params']['face_model']['config'], 
                        config['exp_params']['face_model']['pretrained_path'],
                        exclude_prefix='vae_model.')
sdf_model = load_model(VQVAE3D, 
                        config['exp_params']['sdf_model']['config'], 
                        config['exp_params']['sdf_model']['pretrained_path'],
                        exclude_prefix='vae_model.')
experiment = ReconExperiment(config['exp_params'], face_model, sdf_model)

val_dataset = VoxelDataset(config['data_params'], 'val')
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['data_params']['val_batch_size'], 
    shuffle=False, num_workers=config['data_params']['num_workers'],
    collate_fn=collate_fn)

trainer = pl.Trainer(
    accelerator=config['trainer_params']['accelerator'], max_epochs=config['trainer_params']['max_epochs'],
    precision=config['trainer_params']['precision'], num_nodes=config['trainer_params']['num_nodes'],
    devices=config['trainer_params']['devices'], strategy=config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    default_root_dir=config['trainer_params']['default_root_dir'])

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.validate(experiment, val_dataloader)


