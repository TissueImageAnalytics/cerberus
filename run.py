"""run.py 

Usage:
  run.py [--gpu=<id>] [--fold=<n>] [--logdir=<path>] [--exp_code=<str>] [--param_path=<path>] \
      [--mix_target_in_batch] [--gland] [--gland_dir=<path>] [--lumen] [--lumen_dir=<path>] [--nuclei] \
          [--nuclei_dir=<path>] [--pclass] [--pclass_dir=<path>] [--pretrained=<str>] [--pretrained_info=<path>] 
  run.py (-h | --help)
  run.py --version

Options:
  -h --help                     Show this string.
  --version                     Show version.
  --gpu=<id>                    Comma separated GPU list. [default:0]       
  --fold=<n>                    Fold number to train with. If not provided - assumes no fold info!
  --log_dir=<path>              Path to directory where logs will be saved. [default:logs/]
  --exp_code=<str>              Code given to training run. [default:v1.0]
  --param_path=<path>           Path to file containing run setup. [default:models/paramset.yml]
  --mix_target_in_batch         Whether to use mixed or fixed batch. Toggle on if want to mix!
  --gland                       Whether gland segmentation task considered.
  --gland_dir=<path>            Patch to where gland segmentation data is located. 
  --lumen                       Whether lumen segmentation task is considered.
  --lumen_dir=<path>            Patch to where lumen segmentation data is located.
  --nuclei                      Whether nuclei segmentation task is considered.
  --nuclei_dir=<path>           Patch to where gland segmentation data is located.
  --pclass                      Whether patch classification task is considered.
  --pclass_dir=<path>           Patch to where gland segmentation data is located.
  --pretrained=<str>            String determining what pretrained weights (if any) to use. [default:imagenet]
  --pretrained_info=<path>      Path to file denoting the paths to pretrained weights. [default:models/pretrained.yml]
  
"""

from docopt import docopt
import os
import importlib
import yaml
import random
from collections import OrderedDict

from loader.train_loader import MyConcatDataset
from loader.train_loader import (
    SingleTaskBatchSampler,
    MixedTaskBatchSampler,
    MixedTaskBatchSampler2,
    PatchSegClassDataset
)
from misc.utils import mkdir, recur_find_ext


if __name__ == "__main__":
    args = docopt(__doc__, version="Main trigger script")
    print(args)

    seed = 5
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
    nr_gpus = len(args["--gpu"].split(","))

    root_logdir = args["--log_dir"]
    
    considered_data = []
    if args["--gland"]:
        considered_data.append("Gland")
    if args["--lumen"]:
        considered_data.append("Lumen")
    if args["--nuclei"]:
        considered_data.append("Nuclei")
    if args["--pclass"]:
        considered_data.append("Patch-Class")

    #! raise an error if task selected, but path to data not given
    if args["--gland"] and args["--gland_dir"] is None:
        raise ValueError("Gland task selected, but path to data directory not provided!")
    if args["--lumen"] and args["--lumen_dir"] is None:
        raise ValueError("Lumen task selected, but path to data directory not provided!")
    if args["--nuclei"] and args["--nuclei_dir"] is None:
        raise ValueError("Nuclei task selected, but path to data directory not provided!")
    if args["--pclass"] and args["--pclass_dir"] is None:
        raise ValueError("Patch classification task selected, but path to data directory not provided!")

    def run_one_fold_with_param_set(fold_data, paramset, pretrained_path, save_path):
        mkdir(save_path)

        settings = paramset.copy()
        settings["seed"] = seed

        with open("%s/settings.yml" % save_path, "w") as fptr:
            yaml.dump(settings, fptr, default_flow_style=False)

        train_loader_list = [v for v in fold_data.keys() if "train" in v]
        infer_loader_list = [v for v in fold_data.keys() if not ("train" in v)]

        cfg_module = importlib.import_module("models.opt")
        cfg_getter = getattr(cfg_module, "get_config")
        model_config = cfg_getter(
            train_loader_list, infer_loader_list, pretrained_path, **settings)

        def create_dataset(run_mode=None, subset_name=None, setup_augmentor=None):
            sub_ds_dict = fold_data[subset_name]

            seg_ds_list = []
            ds_list = []
            # TODO: each sub_ds_name should be of a tissue type
            for sub_ds_name, file_path_list in sub_ds_dict.items():
                # select task-dependent datasets
                if sub_ds_name in considered_data:
                    if run_mode == 'infer':
                        # only consider small proportion for validation (faster)
                        file_path_list = random.sample(
                            file_path_list, int(len(file_path_list) / 8))
                    if sub_ds_name != 'Patch-Class':
                        seg_ds = PatchSegClassDataset(
                            file_path_list, 'seg',
                            run_mode=run_mode,
                            **paramset["dataset_kwargs"])
                        ds_list.append(seg_ds)
            # ensure at the end of ds_list
            if 'Patch-Class' in considered_data:
                pclass_ds = PatchSegClassDataset(
                    file_path_list, 'class',
                    run_mode=run_mode,
                    **paramset["dataset_kwargs"])
                ds_list.append(pclass_ds)

            # combine datasets
            input_dataset = MyConcatDataset(ds_list)

            # will replace the loader sampler with this sampler and overwrite batch aggregation
            # such that each batch coming entirely from a single dataset
            loader_kwargs = paramset["loader_kwargs"]
            sampler_batch_size = loader_kwargs[subset_name]["batch_size"] * nr_gpus
            # fixed batch (single task per batch) & not patch classfication
            if not args["--mix_target_in_batch"] and not args["--pclass"]:
                batch_sampler = SingleTaskBatchSampler(
                    input_dataset, sampler_batch_size, run_mode)
            # mixed batch (multiple tasks per batch) & not patch classfication
            elif args["--mix_target_in_batch"] and not args["--pclass"]:
                batch_sampler = MixedTaskBatchSampler(
                    input_dataset, sampler_batch_size, run_mode)
            ## mixed batch (multiple tasks per batch) & patch classfication
            elif args["--mix_target_in_batch"] and args["--pclass"]:
                batch_sampler = MixedTaskBatchSampler2(
                    input_dataset, sampler_batch_size, run_mode)

            return input_dataset, batch_sampler

        run_kwargs = {
            "seed": seed,
            "debug": False,
            "logging": True,
            "log_dir": save_path + "/model/",
            "create_dataset": create_dataset,
            "model_config": model_config,
        }

        from run_train import RunManager

        trainer = RunManager(**run_kwargs)
        trainer.run()
        return

    # determine how to group together data splits to form the folds
    fold_info = {
        1: {"train": 1, "valid": 2},
        2: {"train": 2, "valid": 3},
        3: {"train": 3, "valid": 1},
    }

    # if multiple folds are provided, then run one after the other
    for fold in args["--fold"].split(","):
        fold_info_train = fold_info[int(fold)]["train"]
        fold_info_valid = fold_info[int(fold)]["valid"]

        fold_data = {
            "train": OrderedDict([
                ["Gland",       recur_find_ext(f"{args["--gland_dir"]}/split_{fold_info_train}/996_448", ".dat")],
                ["Lumen",       recur_find_ext(f"{args["--lumen_dir"]}/split_{fold_info_train}/996_448", ".dat")],
                ["Nuclei",      recur_find_ext(f"{args["--nuclei_dir"]}/split_{fold_info_train}/996_224", ".dat")],
                ["Patch-Class", recur_find_ext(f"{args["--pclass_dir"]}/split_{fold_info_train}", ".dat")],
            ]),
            "valid": OrderedDict([
                ["Gland",       recur_find_ext(f"{args["--gland_dir"]}/split_{fold_info_valid}/996_448", ".dat")],
                ["Lumen",       recur_find_ext(f"{args["--lumen_dir"]}/split_{fold_info_valid}/996_448", ".dat")],
                ["Nuclei",      recur_find_ext(f"{args["--nuclei_dir"]}/split_{fold_info_valid}/996_224", ".dat")],
                ["Patch-Class", recur_find_ext(f"{args["--pclass_dir"]}/split_{fold_info_valid}", ".dat")],
            ]),
        }

        with open(args["--param_path"]) as fptr:
            paramset = yaml.full_load(fptr)

            # get encoder name and target information to use in logdir output
            backbone = paramset["model_kwargs"]["encoder_backbone_name"]
            targets = list(paramset["model_kwargs"]["decoder_kwargs"].keys())
            separator = "_"
            targets = separator.join(targets)

            # get encoder name
            backbone = paramset["model_kwargs"]["encoder_backbone_name"]

            pretrained = args["--pretrained"].lower()
            if pretrained == 'imagenet':
                paramset["model_kwargs"]["backbone_imagenet_pretrained"] = True
                paramset["model_kwargs"]["fullnet_custom_pretrained"] = False
                pretrained_path = None
            elif pretrained in ['random', 'mtl', 'imagenet_mtl', 'imagenet_mtl_class', 'custom']:
                paramset["model_kwargs"]["backbone_imagenet_pretrained"] = False
                if pretrained == 'random':
                    paramset["model_kwargs"]["fullnet_custom_pretrained"] = False
                    pretrained_path = None
                else:
                    paramset["model_kwargs"]["fullnet_custom_pretrained"] = True
                    with open(args.pretrained_info) as fptr:
                        pretrained_info = yaml.full_load(fptr)
                        pretrained_path = pretrained_info[backbone][f"fold{fold_info_train}"][pretrained]
            else:
                raise ValueError(
                    "`pretrained` argument not recognised. Provide one of `random`, `imagenet`, `mtl`, `imagenet_mtl` or `imagenet_mtl_class`!")

        save_path = "%s/%s/%s/fold%s/%s/" % (
            root_logdir,
            backbone,
            targets,
            fold,
            args["--exp_code"],
        )

        mkdir(save_path)
        logging.basicConfig(
            level=logging.INFO,
            format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d|%H:%M:%S",
            handlers=[
                logging.FileHandler("%s/debug.log" % save_path),
                logging.StreamHandler(),
            ],
        )
        run_one_fold_with_param_set(
            fold_data, paramset, pretrained_path, save_path)
