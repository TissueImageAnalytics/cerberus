# One Model is All You Need: Multi-Task Learning Enables Simultaneous Histology Image Segmentation and Classification 

This repository contains code for using Cerberus, our multi-task model outlined in our [preprint](https://arxiv.org/abs/2203.00077).

**WARNING:** THE REPOSITORY IS DUE FOR COMPLETION BY NOV 19TH 2022.

## Set Up Environment

```
conda env create -f environment.yml
conda activate cerberus
pip install torch==1.6.0 torchvision==0.7.0
```

Above, we install PyTorch version 1.6 with CUDA 10.2. 

## Repository Structure

Below we outline the contents of the directories in the repository.

- `infer`: Inference scripts
- `loader`: Data loading, augmentation and post processing scripts
- `misc`: Miscellaneous scripts and functions
- `models`: Scripts relating to model definition and hyperparameters
- `run_utils`: Model engine and callbacks

The purpose of the main scripts in the repository:

- `run.py`: Script for triggering multi-task training
- `run_train.py`: Main training script - triggered via `run.py` 
- `run_infer_tile.py`: Run inference on image tiles
- `run_infer_wsi.py`: Run inference on whole-slide images
- `extract_patches.py`: Extract patches from image tiles for multi-task learning
- `dataset.yml`: Defines the dataset paths and information

# Training 

Before unleashing training, you need to ensure patches are appropriately extracted using `extract_patches.py`. This is only applicable for our segmentation tasks (gland, lumen and nucleus segmentation).

Cerberus uses an input patch size of 448x448 for segmentation. For this, we extract a patch double the size (996x996) [set by `win_size` in the script] and then perform a central crop after augmentation. `step_size` determines the stride used during patch extraction. We use 448 for gland/lumen segmentation tasks and 224 for the nucleus segmentation task. All other information, including the image paths, annotation file extension and fold information (`split_info`) should be populated in `dataset.yml`.

Upon extracting patches, it's time to unleash training. For this, modify the command line arugments in `run.py` enter `python run.py -h` for a full description.

In particular, you can toggle different tasks, set the pretrained weights, determine the batch type and set the paths to the input data. 

For example, if performing single task nuclear segmentation on a single GPU with a mixed batch, use:

```
python run.py --gpu="0" --nuclei --nuclei_dir=<path> --mix_target_in_batch --log_dir=<path>
```

If performing multi-task gland, lumen and nuclei segmentation, along with patch classification use:

```
python run.py --gpu="0" --gland --gland_dir=<path> --lumen --lumen_dir=<path> --nuclei --nuclei_dir=<path> --pclass --pclass_dir=<path> --mix_target_in_batch --log_dir=<path>
```

Alongside the above, other model details and hyperparameters must be set in `paramset.yml`, including batch size, learning rate, and target output. Also ensure the utilised decoder kwargs, along with the appopriate number of channels are defined - this must align with the tasks specified in the CLIs above!

## Inference
### Tiles
To process large image tiles, run:

```
python run_infer_tile.py --gpu="0"
```

### WSIs
To process whole-slide images, run:

```
python run_infer_wsi.py --gpu="0"
```







