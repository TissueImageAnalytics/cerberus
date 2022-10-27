"""run_infer_tile.py

Usage:
  run_infer_tile.py [--gpu=<id>] [--model_path=<path>] [--nr_inference_workers=<n>] \
            [--nr_post_proc_workers=<n>] [--batch_size=<n>] [--tile_shape=<n>] \
            [--input_dir=<path>] [--output_dir=<path>] [--patch_input_shape=<n>] [--patch_output_shape=<n>]
  run_infer_wsi.py (-h | --help)
  run_infer_wsi.py --version

Options:
  -h --help                   Show this string.
  --version                   Show version.
  --gpu=<id>                  GPU list. [default: 0]
  --model_path=<path>         Path to saved checkpoint.
  --nr_inference_workers=<n>  Number of workers during inference. [default: 0]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 0]
  --batch_size=<n>            Batch size. [default: 10]
  --tile_shape=<n>            Shape of tile for processing. [default: 2048]
  --input_dir=<path>          Path to input data directory. Assumes the files are not nested within directory.
  --output_dir=<path>         Path to output data directory. Will create automtically if doesn't exist. [default: output/]
  --patch_input_shape=<n>     Shape of input patch to the network- Assume square shape. [default: 448]
  --patch_output_shape=<n>    Shape of network output- Assume square shape. [default: 144]

"""

import os
import yaml
from docopt import docopt

from misc.utils import rm_n_mkdir

# -------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__, version="CoBi Gland Inference")

    if args["--gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]

    input_dir = args["--input_dir"]
    output_dir = args["--output_dir"]

    # create output directory 
    if not os.path.exists(output_dir):
        rm_n_mkdir(output_dir)

    run_root_dir = "pretrained_weights/resnet34_cerberus"
    checkpoint_path = "%s/resnet34_cerberus_cobi.tar" % run_root_dir
    with open("%s/settings.yml" % (run_root_dir)) as fptr:
        run_paramset = yaml.full_load(fptr)
    
    target_list = ['gland', 'lumen', 'nuclei', 'patch-class']

    run_args = {
        "nr_inference_workers": int(args["--nr_inference_workers"]),
        "nr_post_proc_workers": int(args["--nr_post_proc_workers"]),
        "batch_size": int(args["--batch_size"]),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "patch_input_shape": int(args["--patch_input_shape"]),
        "patch_output_shape": int(args["--patch_output_shape"]),
        "patch_output_overlap": 0,
        "postproc_list": target_list,
    }

    from infer.tile import InferManager

    infer = InferManager(
        checkpoint_path=checkpoint_path,
        decoder_dict=run_paramset["dataset_kwargs"]["req_target_code"],
        model_args=run_paramset["model_kwargs"],
    )
    infer.process_file_list(run_args)
