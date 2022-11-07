import warnings
warnings.filterwarnings("ignore")

from importlib import import_module
import torch
from termcolor import colored


class InferManager(object):
    def __init__(self, **kwargs):
        self.run_step = None
        for variable, value in kwargs.items():
            self.__setattr__(variable, value)
        self.__load_model()
        return

    def __load_model(self):
        """Create the model, load the checkpoint and define
        associated run steps to process each data batch

        """
        model_desc = import_module("models.net_desc")
        model_creator = getattr(model_desc, "create_model")

        # TODO: deal with parsing multi level model desc
        net = model_creator(**self.model_args)
        # pytorch_total_params = sum(p.numel() for p in net.parameters())
        saved_state_dict = torch.load(self.checkpoint_path)["desc"]

        variable_name_list = list(saved_state_dict.keys())
        is_in_parallel_mode = all(
            v.split(".")[0] == "module" for v in variable_name_list
        )
        if is_in_parallel_mode:
            colored_word = colored("WARNING", color="red", attrs=["bold"])
            print(
                (
                    "%s: Detect checkpoint saved in data-parallel mode."
                    " Start converting saved model to single GPU mode." % colored_word
                ).rjust(80)
            )
            saved_state_dict = {
                ".".join(k.split(".")[1:]): v for k, v in saved_state_dict.items()
            }
        net.load_state_dict(saved_state_dict, strict=True)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_step = lambda input_batch, output_shape: run_step(
            input_batch, net, output_shape, self.model_args["considered_tasks"]
        )
        return
