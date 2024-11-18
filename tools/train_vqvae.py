import pytorch_lightning as pl
import torch
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
import torch

from projects.mmdet3d_plugin.datasets import SemanticKittiLabelDataModule
from projects.mmdet3d_plugin.sgn.modules.latent_v5 import vqvae_2

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from projects.configs.config import CONF

class LitVQVAE(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()

        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target_1_2 = batch['target_1_2'].long()
        x = target_1_2
        _, vq_loss, recons_loss = self.model(x)

        loss = recons_loss + vq_loss
        self.log("train_loss", loss)
        self.log("train_recons_loss", recons_loss)
        self.log("train_vq_loss", vq_loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        target_1_2 = batch['target_1_2'].long()
        x = target_1_2
        _, vq_loss, recons_loss = self.model(x)

        loss = recons_loss + vq_loss
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_recons_loss", recons_loss)
        self.log("val_vq_loss", vq_loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

if __name__ == "__main__":
    output_dir = os.path.join(CONF.PATH.OUTPUT, 'output_VQVAE')
    os.makedirs(output_dir, exist_ok=True)
    
    # Init Model
    model = vqvae_2(
        init_size=CONF.SSD.init_size, 
        num_classes=CONF.SSD.num_classes, 
        vq_size=CONF.SSD.vq_size, 
        l_size=CONF.SSD.l_size, 
        l_attention=CONF.SSD.l_attention
        )
    
    # Lit Model
    lit_model = LitVQVAE(model=model, lr=1e-4)

    # Data Module
    data_module = SemanticKittiLabelDataModule(
        data_root=CONF.PATH.DATA_DATASETS, 
        batch_size=10, 
        num_workers=4
        )

    # Early stop
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  
        patience=5, 
        verbose=True,  
        mode='min',  
    )
    
    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='vqvae-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss', 
        mode='min',
        save_top_k=1, 
        every_n_epochs=1 
    )

    # log
    logger = TensorBoardLogger(
        save_dir=output_dir,  
        name='logs'  
    )
    
    # GPU Device
    device_stats = DeviceStatsMonitor()
    
    # Trainer
    trainer = pl.Trainer(
        benchmark=True,
        accelerator="gpu",  
        devices=1,          
        strategy="ddp",     
        max_epochs=100,      
        callbacks=[early_stop_callback, checkpoint_callback, device_stats],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(lit_model, data_module)