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
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

seed_everything(config['trainer_params']['manual_seed'], True)

train_dataset = SDFDataset(config['data_params']['train'])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['data_params']['train']['batch_size'], 
    shuffle=True, num_workers=config['data_params']['train']['num_workers'])

val_dataset = SDFDataset(config['data_params']['val'])
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['data_params']['val']['batch_size'], 
    shuffle=True, num_workers=config['data_params']['val']['num_workers'])

model = SDFModel(**config['model_params'])

experiment = Experiment(config['exp_params'], model, len(train_dataset), config['model_params']['latent_code_size'])
# load pretrained model
if config['exp_params']['pretrained_model_path'] is not None:
    experiment.load_state_dict(
        torch.load(config['exp_params']['pretrained_model_path'], 
                   map_location='cpu')['state_dict'], strict=True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1, monitor='train_loss', mode='min', 
    save_last=True, filename='{epoch}-loss={train_loss:.4f}')

checkpoint_callback_last = pl.callbacks.ModelCheckpoint(
    filename='last-model-{epoch:02d}',
    save_last=True,
)

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
    callbacks=[checkpoint_callback, checkpoint_callback_last])

# save training code corresponding to data id
code_book_dict = {}
for data_idx, data in enumerate(train_dataset.data_list):
    key = str(data[0]) + '_' + str(data[1])
    code_book_dict[key] = data_idx
os.makedirs(trainer.log_dir, exist_ok=True)
np.save(os.path.join(trainer.log_dir, 'code_idx.npy'), code_book_dict)

# cp yaml file
if trainer.is_global_zero:
    os.makedirs(trainer.log_dir, exist_ok=True)
    os.system(f'cp {args.config} {trainer.log_dir}/config.yaml')

trainer.fit(experiment, train_dataloader, val_dataloader)

