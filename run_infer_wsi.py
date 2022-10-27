"""run_infer_wsi.py

Usage:
  run_infer_wsi.py [--gpu=<id>] [--model_path=<path>] [--nr_inference_workers=<n>] \
            [--nr_post_proc_workers=<n>] [--batch_size=<n>] [--tile_shape=<n>] [--chunk_shape=<n>] \
            [--ambiguous_size=<int>] [--wsi_proc_mag=<n>] [--cache_path=<path>] [--input_dir=<path>] \
            [--msk_dir=<path>] [--output_dir=<path>] [--patch_input_shape=<n>] [--patch_output_shape=<n>] \
            [--wsi_bulk_idx=<n>] [--wsi_num_bulk=<n>] [--save_thumb] [--save_mask]
  run_infer_wsi.py (-h | --help)
  run_infer_wsi.py --version

Options:
  -h --help                   Show this string.
  --version                   Show version.
  --gpu=<id>                  GPU list. [default: 0]
  --model_path=<path>         Path to saved checkpoint.
  --nr_inference_workers=<n>  Number of workers during inference. [default: 0]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 0]
  --batch_size=<n>            Batch size. [default: 100]
  --tile_shape=<n>            Shape of tile for processing. [default: 2048]
  --chunk_shape=<n>           Shape of tile for processing. [default: 15000]
  --ambiguous_size=<int>      Define ambiguous region along tiling grid to perform re-post processing. [default: 64]
  --wsi_proc_mag=<n>          Objective magnification used for WSI processing. [default: 0.5]
  --cache_path=<path>         Path for cache. Should be placed on SSD with at least 100GB. [default: cache/]
  --input_dir=<path>          Path to input data directory. Assumes the files are not nested within directory.
  --msk_dir=<path>            Path to directory containing tissue masks. Should have the same name as corresponding WSIs. [default: '']
  --output_dir=<path>         Path to output data directory. Will create automtically if doesn't exist. [default: output/]
  --patch_input_shape=<n>     Shape of input patch to the network- Assume square shape. [default: 448]
  --patch_output_shape=<n>    Shape of network output- Assume square shape. [default: 144]
  --wsi_bulk_idx=<n>          Index for batch processing. Indexing is from 0 to n-1. [default: 0]
  --wsi_num_bulk=<n>          Number of batches for processing. [default: 0]
  --save_thumb                Whether to save the slide thumbnail
  --save_mask                 Whether to save the slide mask

"""

import os
import yaml
import glob
import numpy as np
from docopt import docopt

from misc.utils import rm_n_mkdir

# -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__, version="CoBi Gland Inference")

    method_root_dir = "logs/"

    args["--cache_path"] = "/root/dgx_workspace/cache/"

    if args["--gpu"]:
       os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]

    input_dir =  args["--input_dir"]
    output_dir =  args["--output_dir"]
    cache_path = args["--cache_path"] + args["--wsi_bulk_idx"]

    # create output directory
    if not os.path.exists(output_dir):
        rm_n_mkdir(output_dir)

    wsi_file_list = glob.glob(input_dir + '*.mrxs')
    wsi_file_list.sort()

    wsi_list = []
    mask_list = []
    for wsi_filename in wsi_file_list:
        wsi_basename = os.path.basename(wsi_filename)
        wsi_basename = wsi_basename[:-5]
        # only consider if mask exists for now!
        if os.path.isfile(args["--msk_dir"] + wsi_basename + ".png"):
            wsi_list.append(wsi_filename)
            mask_list.append(args["--msk_dir"] + wsi_basename + ".png")
    
    proc_list = []
    for idx, entry in enumerate(wsi_list):
        proc_list.append(idx)

    step = 5
    start_idx = (int(args["--wsi_bulk_idx"]) - 1) * step
    end_idx = int(args["--wsi_bulk_idx"]) * step

    proc_list = proc_list[start_idx:end_idx]
    wsi_list = np.array(wsi_list)[proc_list].tolist()
    mask_list = np.array(mask_list)[proc_list].tolist()
    
    print('NR WSIs', len(wsi_list))

    run_root_dir = "/root/lsf_workspace/pretrained/cerberus/resnet34_cerberus"
    checkpoint_path = "%s/resnet34_cerberus_cobi.tar" % run_root_dir
    with open("%s/settings.yml" % (run_root_dir)) as fptr:
        run_paramset = yaml.full_load(fptr)

    target_list = ["gland", "lumen", "nuclei", "patch-class"]

    run_args = {
        "nr_inference_workers": int(args["--nr_inference_workers"]),
        "nr_post_proc_workers": int(args["--nr_post_proc_workers"]),
        "batch_size": int(args["--batch_size"]),
        "input_list": wsi_list,
        "mask_list": mask_list,
        "output_dir": output_dir,
        "patch_input_shape": int(args["--patch_input_shape"]),
        "patch_output_shape": int(args["--patch_output_shape"]),
        "save_thumb": args["--save_thumb"],
        "save_mask": args["--save_mask"],
        "mask_dir": args["--msk_dir"],
        "postproc_list": target_list,
    }

    wsi_run_args = {
        "msk_dir": args["--msk_dir"],
        "tile_shape": int(args["--tile_shape"]),
        "chunk_shape": int(args["--chunk_shape"]),
        "ambiguous_size": int(args["--ambiguous_size"]),
        "cache_path": cache_path,
        "wsi_proc_mag": float(args["--wsi_proc_mag"]),
    }
    run_args.update(wsi_run_args)

    from infer.wsi import InferManager

    infer = InferManager(
        checkpoint_path=checkpoint_path,
        decoder_dict=run_paramset["dataset_kwargs"]["req_target_code"],
        model_args=run_paramset["model_kwargs"],
    )
    infer.process_wsi_list(run_args)
