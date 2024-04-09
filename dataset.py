import pickle
from torch.utils.data import Dataset
import os

import numpy as np
import torch

from skimage.transform import resize

from transformers import GPT2Tokenizer
from typing import Tuple

from tqdm import tqdm

class fMRI_Stimuli_Embed(Dataset):
    def __init__(self, pickle_dict, fmri_dir, vision_path, sub=1, use_mask=False, resize_shape=(80, 80, 80), do_resize=False):
        self.dict = pickle.load(open(pickle_dict, 'rb'))
        self.fmri_dir = fmri_dir
        
        with open(vision_path, 'rb') as f:
            self.vision_embeds = pickle.load(f)
        
        print("Visual Embeddings size is ", self.vision_embeds.shape)
        
        # load mask
        self.use_mask = use_mask
        if self.use_mask:
            print("Using Mask")
            
        self.mask = np.load(f'processed_data/subj0{sub}/nsd_mask_sub{sub}.npy')
        self.mask = torch.tensor(self.mask).float()
        
        self.dict_keys = list(self.dict.keys())
        self.image_ids = self.dict_keys
        self.sub = sub
        
        self.do_resize = do_resize
        self.resize_shape = resize_shape
    
    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self, idx):
        stimId = self.dict_keys[idx]
        fmriIds = self.dict[stimId]
        vision_emb = self.vision_embeds[stimId]
        
        fmri_data = []
        for fmriId in fmriIds:
            arr = np.load(os.path.join(self.fmri_dir, 'nsd_fmri_sub_{}_trial{}.npy'.format(self.sub, fmriId)))
            if self.do_resize:
                arr = resize(arr, self.resize_shape, anti_aliasing=True)
                
            fmri_data.append(arr)
        
        # average over trials
        fmri_data = np.array(fmri_data).mean(0)
        # -1 and 1 scaling
        fmri_data = ((fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min())) * 2 - 1
        
        fmri_data = torch.from_numpy(fmri_data).float().squeeze(0) 
        if self.use_mask:
            fmri_data = torch.stack((self.mask, fmri_data), dim=0) 
        else:
            fmri_data = fmri_data.unsqueeze(0) # to ensure channel dimension

        return vision_emb.float(), fmri_data
    
    
class PrefixingDataset(Dataset):
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        caption = self.captions_tokens[item]
        
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            # self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            # self.captions_tokens[item] = tokens
            
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix = self.all_data[self.caption2embedding[item]]
        
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix

    def __init__(self, data_path: str, captions_path: str, idx_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)
        print("Vision embeds size is ", self.all_data.shape)
        
        with open(captions_path, 'rb') as f:
            all_data_captions = pickle.load(f)
        print("Captions size is ", len(all_data_captions))
        
        with open(idx_path, 'rb') as f:
            ids = pickle.load(f)
        
        self.image_ids = list(ids.keys())
        
        del ids
        
        if 'train' in idx_path:
            split = 'train'
        else:
            split = 'test'
        
        print("Split is ", split)

        self.captions_tokens = []
        self.caption2embedding = []

        
        for idx in tqdm(self.image_ids):
            
            for caption in all_data_captions[idx]:
                # each valid caption for a given image
                
                self.captions_tokens.append(caption)
                self.caption2embedding.append(idx)
        
        del all_data_captions    
        
        self.max_seq_len = 38 
        print("Max sequence length is ", self.max_seq_len)
        

class ModifiedPrefixingDataset(PrefixingDataset):
    def __init__(self, data_path: str, captions_path: str, idx_path: str, prefix_length: int, gpt2_type: str = "gpt2", normalize_prefix=False):
        super().__init__(data_path, captions_path, idx_path, prefix_length, gpt2_type, normalize_prefix)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)
        print("Vision embeds size is ", self.all_data.shape)
        
        with open(captions_path, 'rb') as f:
            all_data_captions = pickle.load(f)
        print("Captions size is ", len(all_data_captions))
        
        with open(idx_path, 'rb') as f:
            ids = pickle.load(f)
        
        self.image_ids = list(ids.keys())
        
        del ids
        
        if 'train' in idx_path:
            split = 'train'
        else:
            split = 'test'
        
        print("Split is ", split)
        
        self.captions_tokens = []
        self.caption2embedding = []

        for i, idx in tqdm(enumerate(self.image_ids)):
            
            self.captions_tokens.append(all_data_captions[idx][0])
            self.caption2embedding.append(i)

        del all_data_captions    
        
        self.max_seq_len = 38 
        print("Max sequence length is ", self.max_seq_len)