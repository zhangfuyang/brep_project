import os
import torch
torch.set_float32_matmul_precision('medium')
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataset import LatentDataset
from experiment import DiffusionExperiment
from diffusion_model import Solid3DModel
from utils import load_model, load_model_v2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vqvae')))
from vae_model import VQVAE3D


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/vae.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['trainer_params']['manual_seed'], True)

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

if config['exp_params']['face_diffusion_model']['pretrained_path'] is not None:
    load_model_v2(model.face_unet, config['exp_params']['face_diffusion_model']['pretrained_path'],
                    exclude_prefix='diffusion_model.', remove_prefix='vqvae_model.')

experiment = DiffusionExperiment(config['exp_params'], model, face_model, sdf_model)
# load pretrained model
if config['exp_params']['pretrained_model_path'] is not None:
    experiment.load_state_dict(
        torch.load(config['exp_params']['pretrained_model_path'], 
                   map_location='cpu')['state_dict'], strict=True)

train_dataset = LatentDataset(config['data_params'], 'val' if config['data_params']['debug'] else 'train')
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['data_params']['train_batch_size'], 
    shuffle=True, num_workers=config['data_params']['num_workers'])

val_dataset = LatentDataset(config['data_params'], 'val')
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['data_params']['val_batch_size'], 
    shuffle=True, num_workers=config['data_params']['num_workers'])

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1, monitor='train_loss', mode='min', 
    save_last=True, filename='{epoch}-loss={train_loss:.4f}')

checkpoint_callback_last = pl.callbacks.ModelCheckpoint(
    filename='last-model-{epoch:02d}',
    save_last=True,
)

lr_monitor = pl.callbacks.LearningRateMonitor()

trainer = pl.Trainer(
    accelerator=config['trainer_params']['accelerator'], max_epochs=config['trainer_params']['max_epochs'],
    precision=config['trainer_params']['precision'], num_nodes=config['trainer_params']['num_nodes'],
    devices=config['trainer_params']['devices'], strategy=config['trainer_params']['strategy'],
    log_every_n_steps=config['trainer_params']['log_every_n_steps'], 
    limit_val_batches=config['trainer_params']['limit_val_batches'],
    val_check_interval=config['trainer_params']['val_check_interval'],
    num_sanity_val_steps=config['trainer_params']['num_sanity_val_steps'],
    detect_anomaly=config['trainer_params']['detect_anomaly'],
    default_root_dir=config['trainer_params']['default_root_dir'],
    callbacks=[checkpoint_callback, checkpoint_callback_last, lr_monitor],
    accumulate_grad_batches=config['trainer_params']['accumulate_grad_batches'],
    gradient_clip_val=config['trainer_params']['gradient_clip_val'],
    check_val_every_n_epoch=None
    )

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.fit(experiment, train_dataloader, val_dataloader)


