import os
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import LatentDataset, PartialPCLatentDataset
from experiment import DiffusionExperiment
from diffusion_model import Solid3DModel, Solid3DModel_v2
from utils import load_model
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_model import VQVAE3D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='cond_face_diffusion/configs/config.yaml')
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
if config['model_params']['class_type'] == 'Solid3DModel':
    ModelClass = Solid3DModel
elif config['model_params']['class_type'] == 'Solid3DModel_v2':
    ModelClass = Solid3DModel_v2
model = ModelClass(config['model_params'])

experiment = DiffusionExperiment(config['exp_params'], model, face_model, sdf_model)
# load pretrained model
experiment.load_state_dict(
    torch.load(args.pretrained_weight, 
               map_location='cpu')['state_dict'], strict=True)

if config['data_params']['class_type'] == 'LatentDataset_temp':
    DataClass = LatentDataset_temp
elif config['data_params']['class_type'] == 'LatentDataset':
    DataClass = LatentDataset

val_config = config['data_params']['val']
val_config['max_faces'] = 50
val_config['batch_size'] = 15
val_dataset = DataClass(val_config)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=val_config['batch_size'], 
    shuffle=True, num_workers=val_config['num_workers'])

seed_everything(54321, True)
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
    limit_test_batches=3
    )

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.validate(experiment, val_dataloader)


