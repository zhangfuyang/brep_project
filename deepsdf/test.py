import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import SDFDataset
from experiment import Experiment
from model import SDFModel
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='deepsdf/configs/sdf.yaml')
parser.add_argument('--pretrained_weight', type=str, default='model.ckpt')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['trainer_params']['manual_seed'], True)

val_dataset = SDFDataset(config['data_params']['val'])
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['data_params']['val']['batch_size'], 
    shuffle=True, num_workers=config['data_params']['val']['num_workers'])

model = SDFModel(**config['model_params'])

# load pretrained model
checkpoint = torch.load(args.pretrained_weight, 
               map_location='cpu')['state_dict']

code_length = checkpoint['latent_code.weight'].shape[0]
code_dim = checkpoint['latent_code.weight'].shape[1]

experiment = Experiment(config['exp_params'], model, code_length, code_dim)
experiment.load_state_dict(checkpoint, strict=True)

trainer = pl.Trainer(
    accelerator='cpu',#config['trainer_params']['accelerator'], 
    precision=config['trainer_params']['precision'], 
    num_nodes=config['trainer_params']['num_nodes'],
    devices=config['trainer_params']['devices'], 
    strategy=config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    limit_val_batches=config['trainer_params']['limit_val_batches'],
    val_check_interval=config['trainer_params']['val_check_interval'],
    limit_test_batches=1,
    num_sanity_val_steps=config['trainer_params']['num_sanity_val_steps'],
    detect_anomaly=config['trainer_params']['detect_anomaly'],
    default_root_dir=config['trainer_params']['default_root_dir'])

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.test(experiment, val_dataloader)


