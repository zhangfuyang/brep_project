import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import trimesh

class Experiment(pl.LightningModule):
    def __init__(self, config, model, data_size, code_dim):
        super(Experiment, self).__init__()
        self.config = config
        self.model = model
        self.data_size = data_size
        self.latent_code = torch.nn.Embedding(self.data_size, code_dim)
        torch.nn.init.normal_(
            self.latent_code.weight.data,
            0.0,
            1.0 / math.sqrt(code_dim),
        )
        self.latent_code.requires_grad = True
    
    def training_step(self, batch, batch_idx):
        points, dist, code_id = batch
        # points: BxNx3   dist: BxN   code_id: B
        N = points.shape[1]

        latent_code = self.latent_code(code_id) # Bxlatent_code_size
        latent_code = latent_code.unsqueeze(1).expand(-1, points.shape[1], -1) # BxNxlatent_code_size

        points = points.reshape(-1, 3) # BNx3
        dist = dist.reshape(-1) # BN
        latent_code = latent_code.reshape(-1, latent_code.shape[-1]) # BNxlatent_code_size

        pred = self.model(points, latent_code) # BNx1

        recon_loss = F.mse_loss(pred[:,0], dist, reduction='sum') / N
        self.log('recon_loss', recon_loss, rank_zero_only=True, prog_bar=True)
        
        norm_loss = torch.sum(torch.norm(latent_code, dim=1)) / N
        self.log('norm_loss', norm_loss, rank_zero_only=True, prog_bar=True)

        loss = recon_loss + self.config['norm_weight'] * min(1, self.current_epoch / 50) * norm_loss
        self.log('train_loss', loss, rank_zero_only=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        points, dist, code_id = batch
        B = points.shape[0]
        N = points.shape[1]
        # points: BxNx3   dist: BxN   code_id: B

        latent_code = self.latent_code(code_id) # Bxlatent_code_size
        latent_code = latent_code.unsqueeze(1).expand(-1, points.shape[1], -1) # BxNxlatent_code_size

        points = points.reshape(-1, 3) # BNx3
        latent_code = latent_code.reshape(-1, latent_code.shape[-1]) # BNxlatent_code_size

        pred = self.model(points, latent_code) # BNx1
        pred = pred[:,0].reshape(dist.shape) # BxN
        points = points.reshape(B, N, 3)
        
        if batch_idx == 0:
            for i in range(B):
                save_name = os.path.join(self.logger.log_dir, 'images', 
                                         f'{self.global_step}_{i}_gt.obj')
                self.render_(points[i], dist[i], save_name)

                save_name = os.path.join(self.logger.log_dir, 'images',
                                            f'{self.global_step}_{i}_pred.obj')
                self.render_(points[i], pred[i], save_name)

    def render_(self, points, sdf, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        points = points.cpu().numpy()
        sdf = sdf.cpu().numpy()

        colors = []
        for i in range(len(points)):
            dist = sdf[i]
            if dist < -0.05:
                colors.append([0, 255, 0, 255])
            elif dist > 0.05:
                colors.append([0, 0, 255, 255])
            else:
                colors.append([255, 0, 0, 255])
        
        colors = np.array(colors).astype(np.uint8)

        pc = trimesh.points.PointCloud(points, colors=colors)
        # save
        pc.export(filename, include_color=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.model.parameters(), 'lr': self.config['lr']},
                {'params': self.latent_code.parameters(), 'lr': self.config['lr']}
            ], 
            lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_decay'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

            
    