import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from dataset import TestDataset, LatentDataset
from experiment import DiffusionExperiment
from diffusion_model import Solid3DModel
from utils import load_model
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_model import VQVAE3D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/vae.yaml')
parser.add_argument('--pretrained_weight', type=str, default='model.ckpt')
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
if 'generate_solid' in config['exp_params']:
    config['model_params']['gen_solid'] = config['exp_params']['generate_solid']
model = Solid3DModel(**config['model_params'])
experiment = DiffusionExperiment(config['exp_params'], model, face_model, sdf_model)
# load pretrained model
experiment.load_state_dict(
    torch.load(args.pretrained_weight, 
               map_location='cpu')['state_dict'], strict=True)

config['data_params']['max_faces'] = 30
val_dataset = LatentDataset(config['data_params'], 'val' if config['data_params']['debug'] else 'train')
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['data_params']['val_batch_size'], 
    shuffle=True, num_workers=config['data_params']['num_workers'])

trainer = pl.Trainer(
    accelerator=config['trainer_params']['accelerator'], max_epochs=config['trainer_params']['max_epochs'],
    precision=config['trainer_params']['precision'], num_nodes=config['trainer_params']['num_nodes'],
    devices=1, strategy=config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    limit_val_batches=config['trainer_params']['limit_val_batches'],
    val_check_interval=config['trainer_params']['val_check_interval'],
    num_sanity_val_steps=config['trainer_params']['num_sanity_val_steps'],
    detect_anomaly=config['trainer_params']['detect_anomaly'],
    default_root_dir=config['trainer_params']['default_root_dir'],
    limit_test_batches=2
    )

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.test(experiment, val_dataloader)


