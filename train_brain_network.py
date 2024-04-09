import os
import argparse
import yaml
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset import fMRI_Stimuli_Embed
from models.vit import BrainTransformer
from models.cnn import BrainConv3DNet

class BrainModel(pl.LightningModule):
    def __init__(self,
                 model_params,
                 hparams,
                 loss_type='mse+clip',
                 config=None
                 ):
        super(BrainModel, self).__init__()
        self.save_hyperparameters()
        
        self.hyperparams = hparams
        self.model_params = model_params
        
        if config['model_type'] == 'cnn':
            self.model = BrainConv3DNet(**model_params)
        elif config['model_type'] == 'transformer':
            self.model = BrainTransformer(**model_params)
        
        print("Model is ready: ", config['model_type'])
        
        if 'mse' in loss_type:
            self.metric_loss = torch.nn.MSELoss()
            print("Using MSE Loss")
        elif 'l1' in loss_type:
            self.metric_loss = torch.nn.L1Loss()
            print("Using L1 Loss")
        else:
            self.metric_loss = None
            
        if 'clip' in loss_type:
            self.clip_loss = True
            self.temperature = 1.0
            print("Using CLIP Loss")
        else:
            self.clip_loss = False
        
        if self.clip_loss==False and self.metric_loss is None:
            raise ValueError('No valid loss function specified')
            
    
    def loss_fn(self, x, y):
        if self.metric_loss is not None:
            loss = self.metric_loss(x, y)
        
        if self.clip_loss:
            # contrative loss as in CLIP
            # x: fmri embed, y: clip vision embed
            logits = (y @ x.T) / self.temperature
            fmri_similarity = (x @ x.T) / self.temperature
            clip_similarity = (y @ y.T) / self.temperature
            targets = torch.nn.functional.softmax((fmri_similarity + clip_similarity) / 2 * self.temperature, dim=-1)
            
            fmri_loss = (-targets * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=1)
            
            if self.metric_loss is not None:
                loss += fmri_loss.mean()
            else:
                # only contrastive loss
                loss = fmri_loss.mean()
            
        return loss
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        clip_vision_emb, fmri_data = batch
        output = self.model(fmri_data)
        loss = self.loss_fn(output, clip_vision_emb)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        clip_vision_emb, fmri_data = batch
        output = self.model(fmri_data)
        loss = self.loss_fn(output, clip_vision_emb)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hyperparams['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hyperparams['max_epochs'])
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    

def train(args):
    pl.seed_everything(args.seed)
    
    # load config
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    
    logger = WandbLogger(
        entity='slavaheroes',
        project='brain-transformer',
        name=f'sub{args.sub}_seed{args.seed}_{args.loss}',
    )
    
    # dataset
    train_ds = fMRI_Stimuli_Embed(
        pickle_dict='processed_data/subj{:02d}/sig_train_sub{}.pkl'.format(args.sub, args.sub),
        fmri_dir='processed_data/subj{:02d}/fmri'.format(args.sub),
        vision_path='processed_data/stimuli_original_dino_vision.pkl',
        sub=args.sub,
        use_mask=args.use_mask,
        do_resize=False,
    )
    
    test_ds = fMRI_Stimuli_Embed(
        pickle_dict='processed_data/subj{:02d}/sig_test_sub{}.pkl'.format(args.sub, args.sub),
        fmri_dir='processed_data/subj{:02d}/fmri'.format(args.sub),
        vision_path='processed_data/stimuli_original_dino_vision.pkl',
        sub=args.sub,
        use_mask=args.use_mask,
        do_resize=False,
    )
    
    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    print("Train and Test Data Loaders are ready: ", len(train_loader), len(test_loader))
    
    # training params
    hparams = config['training']
    
    # model
    if args.model_type == 'transformer':
        model_params = {
            'img_size': (81, 104, 83),
            'in_chans': 2 if args.use_mask else 1,
            'embed_dim': config['pred_dim'],
            'depth': config['model']['depth'],
            'n_heads': config['model']['n_heads'],
            'mlp_ratio': 4.,
            'qkv_bias': True,
            'p': 0.,
            'attn_p': 0.,
            'drop_path_rate': 0.1,
        }
    elif args.model_type == 'cnn':
        model_params = {
            'in_chans': 2 if args.use_mask else 1,
            'planes': config['model']['planes'],
            'num_classes': config['pred_dim']
        }
    else:
        raise ValueError('Invalid model type')
    
    config['model_type'] = args.model_type
    config['sub'] = args.sub
    config['seed'] = args.seed
    config['loss'] = args.loss
    model = BrainModel(model_params, hparams, loss_type=args.loss, config=config)
    
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=f'./checkpoints/brain_network_{args.loss}_sub{args.sub}',
            filename='brain_network_{epoch:02d}_{val_loss:.5f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            mode='min',
            verbose=True
        )
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[6],
        max_epochs=hparams['max_epochs'],
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        val_check_interval=0.25,
    )
    
    trainer.fit(model, train_loader, test_loader)
    
    
    

if __name__=="__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Train Brain Model')
    parser.add_argument("-sub", "--sub", help="Subject Number", type=int, default=1)
    parser.add_argument("-seed", "--seed", help="Seed ", type=int, default=42)
    parser.add_argument("-config_path", "--config_path", help="Config Path", type=str, default='config.yaml')
    parser.add_argument("-loss", "--loss", help="Loss name + savename", type=str, default='mse')
    parser.add_argument("-epochs", "--epochs", help="Epochs ", type=int, default=200)
    parser.add_argument("--model_type", help="Model Type", type=str, default='transformer')
    parser.add_argument("--use_mask", help="Use Mask", action="store_true")
    args = parser.parse_args()
    
    
    train(args)

