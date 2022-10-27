
import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from termcolor import colored

from .base import BaseCallbacks
from .serialize import fig2data, serialize


class LoggingGradient(BaseCallbacks):
    """Will log per each training step."""
    def _pyplot_grad_flow(self, named_parameters):
        """Plots the gradients flowing through different layers in the net during training.
        "_pyplot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

        ! Very slow if triggered per steps because of CPU <=> GPU.

        """
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        fig = plt.figure(figsize=(10, 10))
        plt.bar(np.arange(len(max_grads)), max_grads,
                alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads,
                alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)],
                   ['max-gradient', 'mean-gradient', 'zero-gradient'])
        fig = np.transpose(fig2data(fig), axes=[2, 0, 1])  # HWC => CHW
        plt.close()
        return fig

    def run(self, state, event):
        if random.random() > 0.05: return
        curr_step = state.curr_global_step

        # logging the grad of all trainable parameters
        tfwriter = state.log_info['tfwriter']
        run_info = state.run_info
        for net_name, net_info in run_info.items():
            netdesc = net_info['desc'].module
            for param_name, param in netdesc.named_parameters():
                param_grad = param.grad
                # TODO: sync test None or epislon for pytorch 1.4 vs 1.5
                if param_grad is None: continue
                tfwriter.add_histogram(
                    "%s_grad/%s" % (net_name, param_name),
                    param_grad.detach().cpu().numpy().flatten(),
                    global_step=curr_step) # ditribute into 10 bins (np default)
                tfwriter.add_histogram(
                    "%s_para/%s" % (net_name, param_name),
                    param.detach().cpu().numpy().flatten(),
                    global_step=curr_step) # ditribute into 10 bins (np default)
        return


class LoggingOutput(BaseCallbacks):
    """Must declare save dir first in the shared global state 
    of the attached engine.
    
    """
    def __init__(self, per_n_epoch=1, per_n_step=None):
        super().__init__()
        assert (per_n_epoch is None and per_n_step is not None) or \
                (per_n_epoch is not None and per_n_step is None)
        self.per_n_step = per_n_step
        self.per_n_epoch = per_n_epoch

    def run(self, state, event):
        # get the main engine state, not the spawned child
        # only logging every n epochs or so
        global_state = state.get_top_parent_state()

        if self.per_n_epoch is not None:
            if (global_state.curr_epoch % self.per_n_epoch != 0):
                return
            current_tracker = global_state.curr_epoch
            run_affix = 'epoch'
            # current_tracker = 'epoch-%06d' % current_tracker

        # only logging every n global step
        if self.per_n_step is not None:
            if (global_state.curr_global_step % self.per_n_step != 0) or \
                (global_state.curr_global_step == 0):
                return
            current_tracker = global_state.curr_global_step
            run_affix = 'step'
            # current_tracker = 'step-%06d' % current_tracker
        #---

        output = state.tracked_step_output

        def get_serializable_values(output_format):
            log_dict = {}
            # get type and variable that is serializable
            # to console or other logging format (json, tensorboard)
            for variable_type, variable_dict in output.items():
                for value_name, value in variable_dict.items():
                    code_list = [run_affix, state.attached_engine_name, value_name]
                    if state.loader_name is not None:
                        code_list.insert(1, state.loader_name)
                    value_name = '-'.join(code_list)
                    new_format = serialize(value, variable_type, output_format)
                    if new_format is not None:
                        log_dict[value_name] = new_format
            return log_dict

        # * Serialize to Console
        # align the console print output
        formatted_values = get_serializable_values('console')
        max_length = len(max(formatted_values.keys(), key=len))
        for value_name, value_text in formatted_values.items():
            value_name = value_name.replace('#', '_') # strip special char
            value_name = colored(value_name.ljust(max_length), 'green')
            logging.info('%s : %s' % (value_name, value_text))

        if global_state.logging is None or \
            global_state.logging is False:
            return

        # * Serialize to JSON file
        stat_dict = get_serializable_values('json')
        # json stat log file, update and overwrite
        with open(state.log_info['yaml_file']) as fptr:
            prev_data = yaml.full_load(fptr) # a dict

        tracker_code = '%s-%06d' % (run_affix, current_tracker)
        if tracker_code in prev_data:
            old_stat_dict = prev_data[tracker_code]
            stat_dict.update(old_stat_dict)
        current_tracker_dict = {tracker_code: stat_dict}
        prev_data.update(current_tracker_dict)

        # TODO: may corrupt
        with open(state.log_info['yaml_file'], "w") as fptr:
            yaml.dump(prev_data, fptr, default_flow_style=False)

        # * Serialize to Tensorboard
        tfwriter = state.log_info['tfwriter']
        formatted_values = get_serializable_values('tensorboard')
        # ! may need to flush to force update
        for value_name, value in formatted_values.items():
            value_name = value_name.replace('#', '_') # strip special char
            # TODO: dynamically call this
            if value[0] == 'scalar':
                tfwriter.add_scalar(value_name, value[1], current_tracker)
            elif value[0] == 'image':
                tfwriter.add_image(value_name, value[1], current_tracker,
                                   dataformats='HWC')
        tfwriter.flush()

        return
