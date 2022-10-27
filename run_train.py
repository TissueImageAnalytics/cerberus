"""run_train.py

Usage:
  run_train.py [--gpu=<id>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list. 
"""

import cv2

cv2.setNumThreads(0)
import inspect
import logging
import os
import pathlib

import numpy as np
import torch
import yaml
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
from torch.utils.data import DataLoader

from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine
from run_utils.utils import (
    check_manual_seed, 
    colored, 
    convert_pytorch_checkpoint
    )

#### have to move outside because of spawn
# * must initialize augmentor per worker, else duplicated rng generators may happen
def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


def collate(sample_list):
    # to enforce
    # key_A: batch
    # key_B: batch
    batch_dict = {}
    for sample in sample_list:
        for k, v in sample.items():
            if k not in batch_dict:
                batch_dict[k] = [v[None]]
            else:
                batch_dict[k].append(v[None])

    batch_dict = {k: np.concatenate(v, axis=0) for k, v in batch_dict.items()}
    batch_dict = {k: torch.from_numpy(v) if k != 'dummy_target' else v 
                  for k, v in batch_dict.items()}

    return batch_dict


####
class RunManager(object):
    """Initialise the main training loop."""

    def __init__(self, **kwargs):
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        return

    ####
    def _get_datagen(self, batch_size, run_mode, subset_name, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0

        input_dataset, sampler = self.create_dataset(
            run_mode=run_mode, subset_name=subset_name
        )
        print("Dataset %s - %s : %d" % (run_mode, subset_name, len(input_dataset)))

        # ! HOTFIX: DataLoader `batch_size` argument may not play well with sampler that
        # ! sample its whole batch, this may lead to unintended behavior
        # ! atm, unknown how such will interact with the worker

        dataloader = DataLoader(
            input_dataset,
            sampler=sampler,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train" and sampler is None,
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return dataloader

    ####
    def _run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """
        Simply run the defined run_step of the related method once
        """
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            # rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            yaml_log_file = log_dir + "/stats.yaml"
            with open(yaml_log_file, "w") as fptr:
                yaml.dump({}, fptr, default_flow_style=False)
            log_info = {
                "yaml_file": yaml_log_file,
                "tfwriter": tfwriter,
            }

        # ! create list of data loader
        def create_loader_dict(run_mode, loader_name_list):
            loader_dict = {}
            for loader_name in loader_name_list:
                loader_opt = opt["loader"][loader_name]
                loader_dict[loader_name] = self._get_datagen(
                    loader_opt["batch_size"],
                    run_mode,
                    loader_name,
                    nr_procs=loader_opt["nr_procs"],
                    fold_idx=fold_idx,
                )
            return loader_dict

        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.yaml"
            with open(stat_file_path) as stat_file:
                info = yaml.full_load(stat_file)
            tracker_code_list = [v for v in info.keys()]
            tracker_uidx = [int(v.split('-')[1]) for v in tracker_code_list]
            last_tracker_code = tracker_code_list[np.argmax(tracker_uidx)]
            last_chkpts_path = "%s/net_%s.tar" % (
                prev_phase_dir,
                last_tracker_code,
            )
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # parsing the network and optimizer information
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = pathlib.Path(pretrained_path).suffix
                    if chkpt_ext == ".npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == ".tar" or chkpt_ext == ".pth":  # ! assume same saving format we desire
                        net_state_dict = torch.load(pretrained_path)["desc"]
                    else:
                        raise ValueError("Checkpoint path `%s` is invalid" % pretrained_path)

                colored_word = colored(net_name, color="red", attrs=["bold"])
                logging.info(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                logging.info("Missing Variables: %s \n" % load_feedback[0])
                logging.info("Detected Unknown Variables: %s \n" % load_feedback[1])

            # net_desc = torch.jit.script(net_desc)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of trainable parameters:', count_parameters(net_desc))

            # print(net_desc) # * dump network definition or not?
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)

            nr_iter = opt[
                "nr_epochs"
            ]  # ! may want to change scheduler trigger to per step
            scheduler = net_info["lr_scheduler"](optimizer, nr_iter)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # * all engine shared the same network info declaration
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_loader_dict = create_loader_dict(runner_name, runner_opt["loader"])
            runner_dict[runner_name] = RunEngine(
                loader_dict=runner_loader_dict,
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        main_runner = runner_dict["train"]
        main_runner.separate_loader_output = False
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        main_runner.run(opt["nr_epochs"])

        logging.info("\n")
        logging.info("########################################################")
        logging.info("\n")
        return

    ####
    def run(self):
        """
        Define multi-stage run or cross-validation or whatever in here
        """
        self.nr_gpus = torch.cuda.device_count()
        print("Number of GPUs detected: %d" % self.nr_gpus)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            pretrained = phase_info['run_info']['net']['pretrained']
            if pretrained == -1 and phase_idx == 0:
                prev_save_path = self.prev_log_dir

            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                save_path = self.log_dir + "/%02d" % (phase_idx)

            self._run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path



if __name__ == "__main__":
    args = docopt(__doc__, version="MTL v1.0")
    trainer = RunManager()

    os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
    trainer.run()
