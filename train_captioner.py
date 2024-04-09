# using https://github.com/rmokady/CLIP_prefix_caption
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from dataset import PrefixingDataset
from models.mapping_networks import MLP, TransformerMapper

import yaml
import argparse

from typing import Optional


class CaptionModel(pl.LightningModule):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 prefix_length: int,
                 config: dict
                 ):
        super(CaptionModel, self).__init__()
        self.save_hyperparameters(ignore=['model', 'optimizer', 'scheduler'])
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prefix_length = prefix_length
        self.config = config
        
    def forward(self, tokens, prefix, mask):
        return self.model(tokens, prefix.float(), mask.float())

    def loss_fn(self, outputs, tokens):
        logits = outputs.logits[:, self.prefix_length - 1: -1]
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
    
    def training_step(self, batch, batch_idx):
        tokens, mask, prefix = batch
        output = self(tokens, prefix, mask)
        
        loss = self.loss_fn(output, tokens)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tokens, mask, prefix = batch
        output = self(tokens, prefix, mask)
        
        loss = self.loss_fn(output, tokens)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]
    

class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: str = 'mlp', gpt2_type: str = 'gpt2'):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_type)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == 'mlp':
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

def main(config, savename):
    logger = WandbLogger(
        entity='slavaheroes',
        project='brain-captioner',
        name=f'captioner_{config["gpt2_type"]}_prefix_{config["prefix_length"]}_{savename}',
    )
    
    # load dataset
    train_ds = PrefixingDataset(
        data_path=config['data_path'],
        captions_path=config['captions_path'],
        idx_path=config['train_idx_path'],
        prefix_length=config['prefix_length'],
        gpt2_type=config['gpt2_type'],
        normalize_prefix=config['normalize_prefix']
    )
    
    test_ds = PrefixingDataset(
        data_path=config['data_path'],
        captions_path=config['captions_path'],
        idx_path=config['test_idx_path'],
        prefix_length=config['prefix_length'],
        gpt2_type=config['gpt2_type'],
        normalize_prefix=config['normalize_prefix']
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    print("Length of train loader is ", len(train_loader))
    print("Length of test loader is ", len(test_loader))
    
    config['mapping_network']['prefix_length'] = config['prefix_length']
    # load mapping network
    if config['mapping_network']['only_prefix']:
        # train only mapping network
        model = ClipCaptionPrefix(
            prefix_length=config['mapping_network']['prefix_length'],
            clip_length=config['mapping_network']['clip_length'],
            prefix_size=config['mapping_network']['prefix_size'],
            num_layers=config['mapping_network']['num_layers'],
            mapping_type=config['mapping_network']['mapping_type'],
            gpt2_type=config['gpt2_type']
        )
        
    else:
        # train both captioner and mapping network
        model = ClipCaptionModel(
            prefix_length=config['mapping_network']['prefix_length'],
            clip_length=config['mapping_network']['clip_length'],
            prefix_size=config['mapping_network']['prefix_size'],
            num_layers=config['mapping_network']['num_layers'],
            mapping_type=config['mapping_network']['mapping_type'],
            gpt2_type=config['gpt2_type']
        )
    
    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=config['num_warmup_steps'], 
                                               num_training_steps=config['epochs']*len(train_loader))
    
    print("Number of training steps: ", config['epochs']*len(train_loader))
    
    pl_module = CaptionModel(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prefix_length=config['prefix_length'],
        config=config
    )
    
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=f'./checkpoints/captioner_{config["gpt2_type"]}_prefix_{config["prefix_length"]}_{savename}',
            filename='orig_dinov2_captioner_{epoch:02d}_{val_loss:.5f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            verbose=True
        )
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[6],
        max_epochs=config['epochs'],
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        val_check_interval=0.25,
    )
    
    trainer.fit(pl_module, train_loader, test_loader)
    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train a captioner')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--savename', type=str, default='captioner', help='Name of the model')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    pl.seed_everything(config['seed'])
    
    main(config, args.savename)
    
