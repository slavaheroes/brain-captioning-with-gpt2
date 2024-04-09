import torch
from torchvision import transforms

from PIL import Image
import h5py
from tqdm import tqdm
import numpy as np

import pickle

annots_cur = np.load('../data/annots/COCO_73k_annots_curated.npy')
print("Annotations are loaded.", annots_cur.shape)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

f_stim = h5py.File('../data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick'][:]

print("Stimuli are loaded.", stim.shape)

device = "cuda:6" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')

model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    
])

all_embeddings = []
all_captions = []

for i in tqdm(range(stim.shape[0])):
    image = stim[i].astype(np.uint8)
    
    captions = annots_cur[i] 
    captions = [c.strip() for c in captions if c.strip() != '']

    with torch.no_grad():
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        prefix = model.backbone.forward_features(image)['x_norm_clstoken']
    
    all_embeddings.append(prefix.cpu())
    all_captions.append(captions)
        
    if (i+1)%1000 or i == stim.shape[0]-1:
        # Save the embeddings and captions
        with open('../processed_data/stimuli_original_dino_vision.pkl', 'wb') as f:
            pickle.dump(torch.cat(all_embeddings, dim=0), f)
        
        with open('../processed_data/stimuli_original_captions.pkl', 'wb') as f:
            pickle.dump(all_captions, f)
                   
print(len(all_embeddings), len(all_captions))
print("All done.")