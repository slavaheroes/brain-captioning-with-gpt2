import numpy as np
import pickle

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import torch
import pytorch_lightning as pl
import argparse

def main(X_train, Y_train, X_test, Y_test):
    alphas = [0.1, 1, 10, 100, 1000, 10000, 60000, 100000]
    best_alpha = None
    best_mse = float('inf')
    best_model = None
    best_preds = None

    for alpha in alphas:
        # Initialize Ridge Regression with current alpha
        model = Ridge(alpha=alpha, max_iter=50000, fit_intercept=True)

        # Fit the model on training data
        model.fit(X_train, Y_train)

        # Predict on validation data
        Y_pred = model.predict(X_test)

        # Calculate MSE
        mse = mean_squared_error(Y_test, Y_pred)

        print("Alpha: ", alpha, " MSE: ", mse)
        # Update best model if current model is better
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_model = model
            best_preds = Y_pred
    
    print("Best alpha is: ", best_alpha)
    
    return best_preds, best_model


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Inference on a captioner')
    parser.add_argument('--train_fmri', type=str, default='nsd_train.npy', help='Path to linearized train fMRI signals')
    parser.add_argument('--test_fmri', type=str, default='nsd_test.npy', help='Path to linearized test fMRI signals')
    parser.add_argument('--z_normalize', action='store_true')
    parser.add_argument('--embeds', type=str, default='vision.pkl', help='Path to Vision embeddings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sub', type=int, default=1)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    # load train & test
    train_fmri = np.load(args.train_fmri)
    test_fmri = np.load(args.test_fmri)
    
    print("Train size: ", train_fmri.shape)
    print("Test size: ", test_fmri.shape)
    
    if args.z_normalize:
        print('Normalizing...')
        train_fmri = train_fmri/300
        test_fmri = test_fmri/300
        
        norm_mean_train = np.mean(train_fmri, axis=0)
        norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
        train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
        test_fmri = (test_fmri - norm_mean_train) / norm_scale_train
        
    print(np.mean(train_fmri),np.std(train_fmri))
    print(np.mean(test_fmri),np.std(test_fmri))

    print(np.max(train_fmri),np.min(train_fmri))
    print(np.max(test_fmri),np.min(test_fmri))
    
    # load embds
    vis_embeds = pickle.load(open(args.embeds, 'rb'))
    nsd_test_indices = pickle.load(open(f'processed_data/subj0{args.sub}/sig_test_sub{args.sub}.pkl', 'rb'))
    nsd_train_indices = pickle.load(open(f'processed_data/subj0{args.sub}/sig_train_sub{args.sub}.pkl', 'rb'))
    
    train_embeds = []
    test_embeds = []
    
    for idx in nsd_train_indices.keys():
        train_embeds.append(vis_embeds[idx, :].numpy())
    
    for idx in nsd_test_indices.keys():
        test_embeds.append(vis_embeds[idx].numpy())
    
    train_embeds = np.array(train_embeds)
    test_embeds = np.array(test_embeds)
    
    print("Train & Test embeds shape: ", train_embeds.shape, test_embeds.shape)
    
    best_preds, _ = main(train_fmri, train_embeds, test_fmri, test_embeds)
    
    best_preds = torch.tensor(best_preds)
    print("Best preds: ", best_preds.shape, type(best_preds))
    
    with open(f'results/linear_regression_sub0{args.sub}_test_dinov2_preds.pkl', 'wb') as f:
        pickle.dump(best_preds, f)
