"""extract_patches.py

Patch extraction script. This replicates the strategy that was used for the original paper, using
3-fold cross validation. The script can be modified for your own needs.

Usage:
  extract_patches.py [--output_dir=<path>] [--win_size=<n>] [--step_size=<n>] [--extract_type=<str>]
  extract_patches.py (-h | --help)
  extract_patches.py --version

Options:
  -h --help               Show this string.
  --version               Show version.
  --output_dir=<path>     Root path where patches will be saved.
  --win_size=<n>          Size of patches to extract. [default: 996]
  --step_size=<n>         Stride used during patch extraction. [default: 448]
  --extract_type=<str>    Whether to use `mirror` or `valid` padding at boundary during patch extraction. [default: valid]

"""

from docopt import docopt
import logging
import os
import pathlib
import scipy.io as sio

import cv2
import joblib
import numpy as np
import pandas as pd
import tqdm
import yaml
from termcolor import colored

from misc.patch_extractor import PatchExtractor
from misc.utils import log_info, recur_find_ext, rm_n_mkdir


def load_msk(base_name, ds_info):
    # ! wont work if different img shares
    # ! same base_name but has different content.
    mask_present = False
    if "msk_dir" in ds_info:
        msk_dir = ds_info["msk_dir"]
        msk_ext = ds_info["msk_ext"]
        file_path = "%s/%s%s" % (msk_dir, base_name, msk_ext)
        if os.path.exists(file_path):
            msk = cv2.imread(file_path)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
            msk[msk > 0] = 1
            msk = np.expand_dims(msk, -1)
            mask_present = True
    if not mask_present:
        img_dir = ds_info["img_dir"]
        img_ext = ds_info["img_ext"]
        file_path = "%s/%s%s" % (img_dir, base_name, img_ext)
        img = cv2.imread(file_path)
        msk = np.full(img.shape[:2], 1, dtype=np.uint8)
        msk = msk[..., None]
    return msk


def load_img(base_name, ds_info):
    """Load image file."""
    img_dir = ds_info["img_dir"]
    img_ext = ds_info["img_ext"]
    file_path = "%s/%s%s" % (img_dir, base_name, img_ext)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
    return img


def load_ann(basename, ds_info, ann_type, use_channel_list):
    """Load annotation file."""
    ann_type = ann_type.capitalize()
    ann_info_dict = ds_info["ann_info"]
    ann_list = []
    ch_code_list = []
    ann_dir = ann_info_dict[ann_type]["ann_dir"]
    ann_ext = ann_info_dict[ann_type]["ann_ext"]
    ann_channel_code = ann_info_dict[ann_type]["channel_code"]
    file_path = "%s/%s%s" % (ann_dir, basename, ann_ext)
    
    # hard assumption on structure and type of annotation file!
    ann = sio.loadmat(file_path)
    ann_inst_map = ann['inst_map'] 
    ann_id = np.squeeze(ann['id']).tolist()
    if isinstance(ann_id, int):
        ann_id = [ann_id]

    if "TYPE" in use_channel_list and "TYPE" in ann_channel_code:
        # list of classes per instance
        ann_class = np.squeeze(ann['class']).tolist()
        if isinstance(ann_class, int):
            ann_class = [ann_class]
        ann_class_map = np.zeros([ann_inst_map.shape[0], ann_inst_map.shape[1]])
        for i, val in enumerate(ann_id):
            tmp = ann_inst_map == val
            class_val = ann_class[i]
            ann_class_map[tmp] = class_val

        # HxWx2 array - first channel instance map, second channel classification map
        ann = np.dstack([ann_inst_map, ann_class_map])
    
    else:
        ann = ann_inst_map

    if len(ann.shape) == 2:
        ann = ann[..., None]  # to NHWC
    ch_indices = [
        ch_idx
        for ch_idx, ch_code in enumerate(ann_channel_code)
        if ch_code in use_channel_list
    ]
    if len(ch_indices) == 0:
        assert False, "Request channel `%s` but `%s` has `%s`" % (
            use_channel_list,
            file_path,
            ann_channel_code,
        )
    ann = ann[..., ch_indices]
    sub_ch_code_list = np.array(ann_channel_code)[ch_indices]
    sub_ch_code_list = [
        "%s-%s" % (ann_type, ch_code) for ch_code in sub_ch_code_list
    ]
    ann_list.append(ann)
    ch_code_list.extend(sub_ch_code_list)
    if len(ann_list) == 0:
        return None
    ann_list = np.concatenate(ann_list, axis=-1)
    return ann_list, ch_code_list


# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = docopt(__doc__)

    logging.basicConfig(
        level=logging.DEBUG,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    win_size = int(args['--win_size']) # double the size of segmentation patch input size
    step_size = int(args['--step_size']) # 448 for glands / lumen. 224 for nuclei
    extract_type = args["--extract_type"] # `valid`` or `mirror``. `valid` doesn't perform mirror padding at boundary
    save_root = args["--output_dir"] # where the data will be saved!

    warn = colored('WARNING:', 'red')
    warn_win_size = colored('win_size', 'green')
    warn_step_size = colored('step_size', 'green')
    print(f"{warn} The selected {warn_win_size} and {warn_step_size} will need to be incorporated into the `fold_data` dict in run.py!")

    use_channel_list = ["INST", "TYPE"]

    # extract patches for these datasets
    ds_list = ["gland", "lumen", "nuclei", "tissue-type"]
    xtractor = PatchExtractor(win_size, step_size)

    with open("dataset.yml") as fptr:
        ds_info_dict = yaml.full_load(fptr)

    # sanity check
    for ds in ds_list:
        if ds not in ds_info_dict:
            assert False, "Dataset `%s` is not defined in yml."

    for ds_name in ds_list:
        ds_info = ds_info_dict[ds_name]
        img_dir = ds_info["img_dir"]
        img_ext = ds_info["img_ext"]
        split_info = pd.read_csv(ds_info["split_info"])
        if ds_name == "tissue-type":
            type_names = ds_info["type_names"]

        out_dir_root = "%s/%s" % (save_root, ds_name)
        # extract patches in separate directories according to the dataset split
        nr_splits = ds_info["nr_splits"] 

        for split_nr in range(nr_splits):
            if ds_name is not "tissue-type":
                out_dir_tmp = out_dir_root + "/split_%d/%d_%d" % (
                    split_nr + 1,
                    win_size,
                    step_size,
                )
            else:
                out_dir_tmp = out_dir_root + "/split_%d" % (split_nr + 1)
            rm_n_mkdir(out_dir_tmp)

        file_path_list = recur_find_ext(img_dir, img_ext)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_path_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_path_list):
            basename = pathlib.Path(file_path).stem
            split_nr = split_info.loc[split_info["Filename"] == basename, "Split"].iloc[
                0
            ]
            if ds_name is not "tissue-type":
                # if split_nr <= nr_splits and split_nr > 0:
                if split_nr > 0:
                    out_dir = out_dir_root + "/split_%d/%d_%d" % (
                        split_nr,
                        win_size,
                        step_size,
                    )
                    img = load_img(basename, ds_info)
                    msk = load_msk(basename, ds_info)
                    ann = load_ann(basename, ds_info, ds_name, use_channel_list)
                    if ann is None:
                        # no annotation detected, skip
                        log_info("`%s` has no annotation." % basename)
                        continue
                    ann, ch_code = ann

                    img = np.concatenate([img, msk, ann], axis=-1)
                    sub_patches = xtractor.extract(img, extract_type)

                    pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
                    pbar = tqdm.tqdm(
                        total=len(sub_patches),
                        leave=False,
                        bar_format=pbar_format,
                        ascii=True,
                        position=1,
                    )

                    for idx, patch in enumerate(sub_patches):
                        patch_img = patch[..., :3]
                        patch_msk = patch[..., 3]
                        patch_ann = patch[..., 4:]

                        if np.sum(patch_msk) <= 0:
                            continue

                        joblib.dump(
                            {
                                "img": patch_img.astype(np.uint8),
                                "ann": patch_ann.astype(np.int32),
                                "channel_code": ch_code,
                            },
                            "%s/%s-%04d.dat" % (out_dir, basename, idx),
                        )
                        assert patch.shape[0] == win_size
                        assert patch.shape[1] == win_size
                pbar.update()
                pbar.close()

            else:
                class_name = file_path.split("/")[-2] # folder name indicates class name
                ann = type_names.index(class_name)
                if split_nr > 0:
                    out_dir = out_dir_root + "/split_%d" % split_nr
                    patch_img = cv2.imread(file_path)
                    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB) # BGR to RGB
                    joblib.dump(
                        {
                            "img": patch_img.astype(np.uint8),
                            "ann": int(ann),
                            "channel_code": ["Patch-Class"],
                        },
                        "%s/%s.dat" % (out_dir, basename),
                    )

            pbarx.update()
        pbarx.close()
