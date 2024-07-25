# Brain Captioning with GPT-2

[**Preprint**](./BrainCaptioning_preprint.pdf)

## Examples of Captioning Results

| True COCO Caption | Caption from fMRI |
|---|---|
| ['Two giraffes that are standing in the grass.', 'Two giraffes are standing together in the field.'] | A couple of giraffes standing in a field. |
| ['Group of people standing on the side of a busy city street.'] | A group of people walking on a city street. |
| ['A person who is riding a surfboard in the ocean.'] | A person on a surfboard in the water. |


## Data preparation

We followed the data processing instruction of [Brain Diffuser](https://github.com/ozcelikfu/brain-diffuser).

Go to `data_scripts` folder, and then:

1. Run `python download_nsd_data.py` & place downloaded data to `data` folder. It should look like this: 

```
├── ccn24_brain_captioning
│   ├── data/ 
│   └── data_scripts
│       └── download_nsd_data.py
```
2. Download "COCO_73k_annots_curated.npy" file from [HuggingFace NSD](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main), and place it inside `data/annots/` folder.

3. Run `python prepare_nsd_data.py --sub [NUM]` 

    - It splits the data to train/test as in `Brain Diffuser` and other works.
    
    - ### Data Contamination

    In `nsd_contamination.ipynb` notebook, we show that all of test images in NSD train/test split are present in [COCO Train](https://cocodataset.org/#download). 

4. Run `python parse_dinov2_embeds.py`

## Training

### Brain Module: fMRI to Dinov2 

Every subject has own different brain module. Below are examples of how to train brain networks:

> Note: Training without `--use_mask` showed better performance than training with it.

- #### CNNs

    Example run:
    ```bash
    python train_brain_network.py --sub 1 --seed 42 --loss  cnn_mse_2  --model_type cnn --config_path configs/cnn_brain_network.yaml 
    ```

- #### Linear model as in `Brain Diffuser`

    First, run `python parse_nsd_for_linear.py` script inside `data_scripts` folder, to linearize fMRI voxels.

    Then, run: 
    ```bash
    python train_linear_ridge.py --sub 1 --seed 42 --z_normalize  --train_fmri processed_data/subj01/nsd_train_fmriavg_nsdgeneral_sub1.npy  --test_fmri processed_data/subj01/nsd_test_fmriavg_nsdgeneral_sub1.npy  --embeds processed_data/stimuli_original_dino_vision.pkl 
    ```

    Best value for $\alpha$ is `60000`

### Captioning module: From Dinov2 embeddings

We used the codebase of [ClipCap paper](https://github.com/rmokady/CLIP_prefix_caption). 

We need only one captioning model for all subjects:
```bash
python train_captioner.py --config  ./configs/captioner_orig_dinov2.yaml
```

## Prediction

- For prediction from outputs of Ridge Regression:
    ```
    python predict.py  --brain_net results/linear_regression_sub01_test_dinov2_preds.pkl --captioner checkpoints/captioner_gpt2_prefix_10_captioner/orig_dinov2_captioner_epoch=04_val_loss=2.37030.ckpt --model_type linear --model_config configs/linear.yaml --use_mask  --captioner_config ./configs/captioner_orig_dinov2.yaml --use_beam  --savename linear_w_beam --sub 1 --seed 42
    ```

- For prediction with CNNs brain networks:
    ```
    python predict.py  --brain_net checkpoints/brain_network_cnn_mse_1/brain_network_epoch=21_val_loss=1.61380.ckpt --captioner checkpoints/captioner_gpt2_prefix_10_captioner/orig_dinov2_captioner_epoch=04_val_loss=2.37030.ckpt --model_type cnn --model_config configs/shallow_cnn_brain_network.yaml   --captioner_config ./configs/captioner_orig_dinov2.yaml --use_beam  --savename shallowcnn_w_beam --sub 1 --seed 42
    ```

## Evaluation

The results in the paper are obtained by running `final_evaluation.ipynb` notebook. There you can find scores and best/worse captioning examples.

Below are the ablation table comparing different brain networks.


|          	|   	|       fMRI vs COCO         	|                	|
|:----------:	|:--------------:	|:--------------:	|:--------------:	|
| Metrics  	| Ridge          	| Shallow CNN    	| Wide CNN       	|
| METEOR   	|  0.263 ± 0.007 	|  0.267 ± 0.009 	|  0.273 ± 0.008 	|
| ROUGE-1  	|  0.331 ± 0.009 	|  0.340 ± 0.009 	|  0.346 ± 0.008 	|
| ROUGE-L  	|  0.300 ± 0.008 	|  0.312 ± 0.009 	|  0.317 ± 0.007 	|
| Sentence 	| 34.92\%±1.52\% 	| 36.71\%±2.54\% 	| 38.91\%±2.20\% 	|
| CLIP-B   	| 66.73\%±0.61\% 	| 67.22\%±1.32\% 	| 67.79\%±1.16\% 	|
| CLIP-L   	| 55.72\%±0.80\% 	| 56.65\%±1.67\% 	| 57.59\%±1.31\% 	|




