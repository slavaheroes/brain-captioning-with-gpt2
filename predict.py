import argparse
import yaml
from tqdm import tqdm
import pickle

import torch
import pytorch_lightning as pl

from transformers import GPT2Tokenizer
import evaluate

from dataset import fMRI_Stimuli_Embed, ModifiedPrefixingDataset
from train_brain_network import BrainModel
from train_captioner import CaptionModel, ClipCaptionModel, ClipCaptionPrefix
from utils import generate2, generate_beam


device = "cuda:6"


def main_predict(captioner, tokenizer, brain_model, test_ds, args, from_precomputed):
    
    if brain_model is None and not from_precomputed:
        raise ValueError("Brain model is None & Precomputed embeddings are not used")
    elif brain_model is not None and from_precomputed:
        raise ValueError("Brain model and Precomputed both provided")
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    gt_captions = pickle.load(open('processed_data/stimuli_original_captions.pkl', 'rb'))
    # select only the test captions
    gt_captions = [gt_captions[idx] for idx in test_ds.image_ids]
    print("Len of GT Captions is ", len(gt_captions))
        
    
    fmri_captions = []
    dinov2_captions = []
    
    captioner.to(device)
    if brain_model is not None:
        brain_model.to(device)
    
    for batch in tqdm(test_loader):
        if from_precomputed:
            _, _, pred_embed = batch
            pred_embed = pred_embed.float().to(device)
            orig_embed = None
        else:
            orig_embed, fmri_data = batch
            orig_embed, fmri_data = orig_embed.to(device), fmri_data.to(device)
            
            with torch.no_grad():
                pred_embed = brain_model(fmri_data)
        
        projection = captioner.clip_project(pred_embed)
        projection = projection.reshape(1, test_ds.prefix_length, -1)
        
        
        if orig_embed is not None:
            orig_proj = captioner.clip_project(orig_embed)
            orig_proj = orig_proj.reshape(1, test_ds.prefix_length, -1)
        
        if args.use_beam:
            if orig_embed is not None:
                orig_caption = generate_beam(
                    captioner,
                    tokenizer,
                    embed=orig_proj,
                )[0] # select first beam
            
            fmri_caption = generate_beam(
                captioner,
                tokenizer,
                embed=projection,
            )[0]
        else:
            if orig_embed is not None:
                orig_caption = generate2(
                    captioner,
                    tokenizer,
                    embed=orig_proj,
                )
            
            fmri_caption = generate2(
                captioner,
                tokenizer,
                embed=projection,
            )
        
        fmri_captions.append(fmri_caption)
        if orig_embed is not None:
            dinov2_captions.append(orig_caption)
    
    # meteor
    # https://github.com/MedARC-AI/MindEyeV2/blob/main/src/final_evaluations.ipynb
    
    meteor = evaluate.load('meteor')
    meteor_fmri_gt = meteor.compute(predictions=fmri_captions, references=gt_captions)
    
    print("METEOR Score for FMRI & GT Captions is ", meteor_fmri_gt['meteor'])
    
    if orig_embed is not None:
        meteor_fmri_dinov2 = meteor.compute(predictions=fmri_captions, references=dinov2_captions)
        print("METEOR Score for FMRI & DINOv2 Captions is ", meteor_fmri_dinov2['meteor'])
    
    with open('results/sub0{}_{}_fmri_captions.pkl'.format(args.sub, args.savename), 'wb') as f:
        pickle.dump(fmri_captions, f)
    
    if orig_embed is not None:
        # same for all subjs
        with open('results/sub0{}_{}_dinov2_captions.pkl'.format(args.sub, args.savename), 'wb') as f:
            pickle.dump(dinov2_captions, f)
    
    print("Sub ", args.sub, " Results saved: ", args.savename)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Inference on a captioner')
    parser.add_argument('--brain_net', type=str, default='ckpt.ckpt', help='Path to the brain network')
    parser.add_argument('--captioner', type=str, default='ckpt.ckpt', help='Path to the captioner')
    parser.add_argument("--model_type", help="Brain Model Type", type=str, default='transformer')
    parser.add_argument("--model_config", help="Brain Model Config", type=str, default='config.yaml')
    parser.add_argument("--use_mask", help="Use Mask", action="store_true")
    parser.add_argument('--captioner_config', type=str, default='config.yaml', help='Path to the captioner config')
    parser.add_argument("--use_beam", help="Use Beam Search", action="store_true")
    parser.add_argument('--savename', type=str, default='exp_00_mlp_40', help='Save results')
    parser.add_argument('--sub', type=int, default=1, help='subject')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    # load config
    captioner_config = yaml.load(open(args.captioner_config, 'r'), Loader=yaml.FullLoader)
    
    if args.brain_net.endswith('.ckpt'):
        from_precomputed = False
        
        brain_config = yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader)
        test_ds = fMRI_Stimuli_Embed(
            pickle_dict='processed_data/subj{:02d}/sig_test_sub{}.pkl'.format(args.sub, args.sub),
            fmri_dir='processed_data/subj{:02d}/fmri'.format(args.sub),
            vision_path='processed_data/stimuli_original_dino_vision.pkl',
            sub=args.sub,
            use_mask=args.use_mask,
            do_resize=False
        )
        
        # model
        if args.model_type == 'transformer':
            model_params = {
                'img_size': (81, 104, 83),
                'in_chans': 2 if args.use_mask else 1,
                'embed_dim': brain_config['pred_dim'],
                'depth': brain_config['model']['depth'],
                'n_heads': brain_config['model']['n_heads'],
                'mlp_ratio': 4.,
                'qkv_bias': True,
                'p': 0.,
                'attn_p': 0.,
                'drop_path_rate': 0.1,
            }
        elif args.model_type == 'cnn':
            model_params = {
                'in_chans': 2 if args.use_mask else 1,
                'planes': brain_config['model']['planes'],
                'num_classes': brain_config['pred_dim']
            }
        else:
            raise ValueError('Invalid model type')
        
        brain_config['model_type'] = args.model_type
        brain_model = BrainModel(
            model_params=model_params,
            hparams=brain_config['training'],
            config=brain_config,
        )
        
        brain_model.load_state_dict(
            torch.load(args.brain_net, map_location='cpu')['state_dict']
        )
        
        brain_model = brain_model.model 
        brain_model.eval()
    elif args.brain_net.endswith('.pkl'):
        from_precomputed = True
        
        # using precomputed predictions
        print("Brain Network is not a checkpoint")
        print("Loading precomputed predictions")
        
        test_ds = ModifiedPrefixingDataset(
            data_path=args.brain_net,
            captions_path=captioner_config['captions_path'],
            idx_path=captioner_config['test_idx_path'],
            prefix_length=captioner_config['prefix_length'],
            gpt2_type=captioner_config['gpt2_type'],
            normalize_prefix=captioner_config['normalize_prefix']
        )
        
        brain_model = None
        
    else:
        raise ValueError("Invalid Brain Network")
    
    test_ds.prefix_length = captioner_config['prefix_length']
    # load captioner
    if captioner_config['mapping_network']['only_prefix']:
        caption_model = ClipCaptionPrefix(
            prefix_length=captioner_config['mapping_network']['prefix_length'],
            clip_length=captioner_config['mapping_network']['clip_length'],
            prefix_size=captioner_config['mapping_network']['prefix_size'],
            num_layers=captioner_config['mapping_network']['num_layers'],
            mapping_type=captioner_config['mapping_network']['mapping_type'],
            gpt2_type=captioner_config['gpt2_type']
        )
    else:
        caption_model = ClipCaptionModel(
            prefix_length=captioner_config['mapping_network']['prefix_length'],
            clip_length=captioner_config['mapping_network']['clip_length'],
            prefix_size=captioner_config['mapping_network']['prefix_size'],
            num_layers=captioner_config['mapping_network']['num_layers'],
            mapping_type=captioner_config['mapping_network']['mapping_type'],
            gpt2_type=captioner_config['gpt2_type']
        )
        
    captioner = CaptionModel(
        model=caption_model,
        optimizer=None,
        scheduler=None,
        prefix_length=captioner_config['prefix_length'],
        config=captioner_config,
    )
    captioner.load_state_dict(
        torch.load(args.captioner, map_location='cpu')['state_dict']
    )
    captioner = captioner.model
    captioner.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained(captioner_config['gpt2_type'])
    
    main_predict(captioner, tokenizer, brain_model, test_ds, args, from_precomputed)
    