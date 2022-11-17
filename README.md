<p align="center">
  <img src="doc/cerberus.png",
  width="600",
  height="146" >
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
  <a href="#cite-this-repository"><img src="https://img.shields.io/badge/Cite%20this%20repository-BibTeX-brightgreen" alt="DOI"></a> <a href="https://doi.org/10.1016/j.media.2022.102685"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.media.2022.102685-blue" alt="DOI"></a>
<br>


# One Model is All You Need: Multi-Task Learning Enables Simultaneous Histology Image Segmentation and Classification 

This repository contains code for using Cerberus, our multi-task model outlined in our [Medical Image Analysis paper](https://doi.org/10.1016/j.media.2022.102685).

Scroll down to the bottom to find instructions on downloading our [pretrained weights](#download-weights) and [WSI-level results](#download-tcga-results).

## Set Up Environment

```
# create base conda environment
conda env create -f environment.yml

# activate environment
conda activate cerberus

# install PyTorch with pip
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

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

## Training 

Before initialising training, ensure patches are appropriately extracted using `extract_patches.py`. This is only applicable for our segmentation tasks (gland, lumen and nucleus segmentation).

Cerberus uses input patches of size of 448x448 for segmentation. For this, we extract a patches double the size (996x996) [set by `win_size` in the script] and then perform a central crop after augmentation. `step_size` determines the stride used during patch extraction. We use 448 for gland/lumen segmentation tasks and 224 for the nucleus segmentation task. All other information, including the image paths, annotation file extension and fold information (`split_info`) should be populated in `dataset.yml`. The script also converts data for patch classification into the appropriate format. 

We have provided a sample dataset to help with data curation and MTL debugging. Here, data from the `original_data` folder is converted into data in the `patches` folder for training. Download the sample data at [this webpage](https://warwick.ac.uk/fac/cross_fac/tia/software/cerberus/).

Upon extracting patches, it's time to unleash training. For this, modify the command line arugments in `run.py` enter `python run.py -h` for a full description.

In particular, you can toggle different tasks, set the pretrained weights, determine the batch type and set the paths to the input data. 

For example, if performing single task nuclear instance segmentation on a single GPU with a mixed batch, use:

```
python run.py --gpu="0" --nuclei_inst --nuclei_dir=<path> --mix_target_in_batch --log_dir=<path>
```

If performing multi-task gland, lumen and nuclei instance segmentation, along with patch classification use:

```
python run.py --gpu="0" --gland_inst --gland_dir=<path> --lumen_inst --lumen_dir=<path> --nuclei_inst --nuclei_dir=<path> --pclass --pclass_dir=<path> --mix_target_in_batch --log_dir=<path>
```

You can also simultaneously perform nuclear and gland classification by adding the `gland_class` and `nuclei_class` arguments. However, in our paper we perform the object subtyping in a second step after freezing the initially trained weights of the network. To perform nuclei subtyping using our strategy, use the following command:

```
python run.py --gpu="0" --gland_inst --gland_dir=<path> --lumen_inst --lumen_dir=<path> --nuclei_inst --nuclei_class --nuclei_subtype --nuclei_dir=<path> --pclass --pclass_dir=<path> --mix_target_in_batch --log_dir=<path> --pretrained=<str>
```

Using the `nuclei_subtype` argument ensures that the other parts of the network are frozen. Otherwise, the entire network will be trained. When using the two-stage strategy for subtyping, you will also need to provide the weights from the first stage using the `pretrained` argument. For this, you can explicitly include the path to the weights or you can add the paths in `models/pretrained.yml`. If adding paths to the yaml file, ensure that the correct argument is specified to utilise the appropriate weights. We have included stage one weights obtained from each fold after training with multi-task learning. Links are provided [later](#download-weights) in this file.

Alongside the above, other model details and hyperparameters must be set in `paramset.yml`, including batch size, learning rate, and target output. Also ensure the utilised decoder kwargs, along with the appopriate number of channels are defined - this must align with the tasks specified in the command line arguments above!

## Inference
### Tiles
To process large image tiles, run:

```
python run_infer_tile.py --gpu=<gpu_id> --batch_size=<n> --model=<path> --input_dir=<path> --output_dir=<path> 
```

For convenience, we have also included a bash script, where you can populate command line arguments. To make this script executable, run `chmod +x run_tile.sh`. Then use the command `./run_tile.sh`.

### WSIs
To process whole-slide images, run:

```
python run_infer_wsi.py --gpu=<gpu_id> --batch_size=<n> --model=<path> --input_dir=<path>  mask_dir=<path> --output_dir=<path> 
```

Similar to the tile mode, we have included an example bash script (`run_wsi.sh`) that can be used to run the command, without having to always re-enter the arguments.

For both tile and WSI inference, the model path should point to a directory containing the settings file and the weights (`.tar` file). You will see from the above command that there is a `mask_dir` argument. In this repo, we assume that tissue masks have been automatically generated. You should include masks - otherwise it will lead to significantly longer processing times.

## Download Weights

In this repository, we enable the download of:

- Cerberus model for simultaneous:
    - Gland instance segmentation 
    - Gland semantic segmentation (classification)
    - Nuclear instance segmentation
    - Nuclear semantic segmentation (classification)
    - Lumen instance segmentation
    - Tissue type patch classification
- Pretrained ResNet weights (torchvision compatible) for transfer learning
- Pretrained weights obtained from training each fold using:
    - ImageNet weights and MTL
    - ImageNet weights and MTL (with patch classification)

Download all of the above weights by visiting [this page](https://warwick.ac.uk/fac/cross_fac/tia/software/cerberus/).

Note, the pretrained weights are designed for weight initialisation - not for model inference.
  
All weights are under a non-commercial license. See the [License section](#license) for more details.

## Download TCGA Results

Download results from processing 599 CRC WSIs using Cerberus at [this page](https://warwick.ac.uk/fac/cross_fac/tia/software/cerberus/).

## License

Code is under a GPL-3.0 license. See the [LICENSE](https://github.com/TissueImageAnalytics/cerberus/blob/master/LICENSE) file for further details.

Model weights are licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider the implications of using the weights under this license. 

## Cite this repository

```
@article{graham2022one,
  title={One model is all you need: multi-task learning enables simultaneous histology image segmentation and classification},
  author={Graham, Simon and Vu, Quoc Dang and Jahanifar, Mostafa and Raza, Shan E Ahmed and Minhas, Fayyaz and Snead, David and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={102685},
  year={2022},
  publisher={Elsevier}
}
```




