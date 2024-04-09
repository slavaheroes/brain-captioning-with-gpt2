# https://github.com/ozcelikfu/brain-diffuser
import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import pickle
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

os.makedirs('processed_data/subj{:02d}'.format(sub), exist_ok=True)

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


stim_order_f = '../data/nsddata/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)

## Selecting ids for training and test data

sig_train = {}
sig_test = {}
num_trials = 37*750
for idx in tqdm(range(num_trials)):
    ''' nsdId as in design csv files'''
    nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx]>1000:
        if nsdId not in sig_train:
            sig_train[nsdId] = []
        sig_train[nsdId].append(idx)
    else:
        if nsdId not in sig_test:
            sig_test[nsdId] = []
        sig_test[nsdId].append(idx)

# save sig train and test to pickle
with open('../processed_data/subj{:02d}/sig_train_sub{}.pkl'.format(sub,sub), 'wb') as f:
    pickle.dump(sig_train, f)

with open('../processed_data/subj{:02d}/sig_test_sub{}.pkl'.format(sub,sub), 'wb') as f:
    pickle.dump(sig_test, f)


train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())


roi_dir = '../data/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
betas_dir = '../data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)

mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()

print(mask.shape)
np.save('../processed_data/subj{:02d}/nsd_mask_sub{}.npy'.format(sub,sub), mask) # save mask

indices = np.argwhere(mask > 0)
positions = np.array([indices[:, 0], indices[:, 1], indices[:, 2]])

print(positions.shape)
np.save('../processed_data/subj{:02d}/nsd_mask_positions_sub{}.npy'.format(sub,sub), positions)

os.makedirs('../processed_data/subj{:02d}/fmri'.format(sub), exist_ok=True)

for i in tqdm(range(37)):
    beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
    beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)
    
    beta_f_transposed = np.transpose(beta_f, axes=(3, 0, 1, 2))
    # print(i, ' Beta shape:', beta_f_transposed.shape)
    # (time, x, y, z)
    for t in range(750):
        fmri = beta_f_transposed[t]
        np.save('../processed_data/subj{:02d}/fmri/nsd_fmri_sub_{}_trial{}.npy'.format(sub, sub, (i*750)+t), fmri)
    