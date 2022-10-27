
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import mode as major_value
from sklearn.metrics import confusion_matrix
import operator

from misc.utils import center_pad_to_shape, cropping_center



class BaseCallbacks(object):
    def __init__(self):
        self.engine_trigger = False

    def reset(self):
        pass

    def run(self, state, event):
        pass


class TrackLr(BaseCallbacks):
    """Add learning rate to tracking."""
    def __init__(self, per_n_epoch=1, per_n_step=None):
        super().__init__()
        self.per_n_epoch = per_n_epoch
        self.per_n_step = per_n_step

    def run(self, state, event):
        # logging learning rate, decouple into another callback?
        run_info = state.run_info
        for net_name, net_info in run_info.items():
            lr = net_info['optimizer'].param_groups[0]['lr']
            state.tracked_step_output['scalar']['lr-%s' % net_name] = lr
        return


class ScheduleLr(BaseCallbacks):
    """Trigger all scheduler."""
    def __init__(self):
        super().__init__()

    def run(self, state, event):
        # logging learning rate, decouple into another callback?
        run_info = state.run_info
        for net_name, net_info in run_info.items():
            net_info['lr_scheduler'].step()
        return


class TriggerEngine(BaseCallbacks):
    def __init__(self, triggered_engine_name, nr_epoch=1, 
                 per_n_epoch=1, per_n_step=None):
        assert (per_n_epoch is None and per_n_step is not None) or \
                (per_n_epoch is not None and per_n_step is None)
        self.per_n_step = per_n_step
        self.per_n_epoch = per_n_epoch

        self.nr_epoch = nr_epoch
        self.engine_trigger = True
        self.triggered_engine_name = triggered_engine_name
        self.triggered_engine = None
        return

    def run(self, state, event):

        global_state = state.get_top_parent_state()
        if self.per_n_epoch is not None:
            if (global_state.curr_epoch % self.per_n_epoch != 0):
                return
        if self.per_n_step is not None:
            if (global_state.curr_global_step % self.per_n_step != 0) or \
                (global_state.curr_global_step == 0):
                return

        self.triggered_engine.run(chained=True,
                                  nr_epoch=self.nr_epoch,
                                  shared_state=state)
        return


class PeriodicSaver(BaseCallbacks):
    """ Must declare save dir first in the shared global state of the
    attached engine.

    """
    def __init__(self, per_n_epoch=1, per_n_step=None):
        super().__init__()
        assert (per_n_epoch is None and per_n_step is not None) or \
                (per_n_epoch is not None and per_n_step is None)
        self.per_n_step = per_n_step
        self.per_n_epoch = per_n_epoch
        
    def run(self, state, event):
        # get the main engine state, not the spawned child

        # -----
        # only logging every n epochs or so
        global_state = state.get_top_parent_state()
        if global_state.logging is None or \
            global_state.logging is False:
            return

        if self.per_n_epoch is not None:
            if (global_state.curr_epoch % self.per_n_epoch != 0):
                return
            current_tracker = global_state.curr_epoch
            current_tracker = 'epoch-%06d' % current_tracker

        # only logging every n global step
        if self.per_n_step is not None:
            if (global_state.curr_global_step % self.per_n_step != 0) or \
                (global_state.curr_global_step == 0):
                return
            current_tracker = global_state.curr_global_step
            current_tracker = 'step-%06d' % current_tracker
        # -----

        for net_name, net_info in state.run_info.items():
            net_checkpoint = {}
            for key, value in net_info.items():
                if key != 'extra_info': 
                    net_checkpoint[key] = value.state_dict()
            torch.save(net_checkpoint, '%s/%s_%s.tar' %
                       (state.log_dir, net_name, current_tracker))
        return


class ConditionalSaver(BaseCallbacks):
    """
    Must declare save dir first in the shared global state of the
    attached engine.

    """
    def __init__(self, metric_name, comparator='>='):
        super().__init__()
        self.metric_name = metric_name
        self.comparator = comparator

    def run(self, state, event):
        if not state.logging:
            return

        ops = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,            
        }
        op_func = ops[self.comparator]
        if self.comparator == '>' or self.comparator == '>=':
            best_value  = -float("inf")
        else:
            best_value  = +float("inf")

        # json stat log file, update and overwrite
        with open(state.log_info['json_file']) as json_file:
            json_data = json.load(json_file)

        for epoch, epoch_stat in json_data.items():
            epoch_value = epoch_stat[self.metric_name]
            if op_func(epoch_value, best_value):
                best_value  = epoch_value

        current_value = json_data[str(state.curr_epoch)][self.metric_name]
        if not op_func(current_value, best_value):
            return # simply return because not satisfy

        print(state.curr_epoch) # TODO: better way to track which optimal epoch is saved
        for net_name, net_info in state.run_info.items():
            net_checkpoint = {}
            for key, value in net_info.items():
                if key != 'extra_info': 
                    net_checkpoint[key] = value.state_dict()
            torch.save(net_checkpoint, '%s/%s_best=[%s].tar' %
                       (state.log_dir, net_name, self.metric_name))
        return


class AccumulateRawOutput(BaseCallbacks):
    def run(self, state, event):
        step_output = state.step_output['raw']
        accumulated_output = state.epoch_accumulated_output
        accumulated_output.append(step_output)
        return


class ScalarMovingAverage(BaseCallbacks):
    """
    Calculate the running average for all scalar output of 
    each runstep of the attached RunEngine.

    """
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def run(self, state, event):
        # TODO: protocol for dynamic key retrieval for EMA
        step_output = state.step_output['EMA']

        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # calculate the exponential moving average
                new_ema_value = old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                self.tracking_dict[key] = new_ema_value
            else:  # init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value

        state.tracked_step_output['scalar'] = self.tracking_dict
        return


class ProcessAccumulatedEpochOutput(BaseCallbacks):
    def __init__(self, proc_func, per_n_epoch=1):
        # TODO: allow dynamically attach specific procesing for `type`
        super().__init__()
        self.per_n_epoch = per_n_epoch
        self.proc_func = proc_func

    def run(self, state, event):
        current_epoch = state.curr_epoch
        # if current_epoch % self.per_n_epoch != 0: return
        raw_data = state.epoch_accumulated_output
        # TODO: allow full access ?
        track_dict = self.proc_func(state.loader_name, raw_data)
        # update global shared states
        state.tracked_step_output = track_dict
        return


class VisualizeOutput(BaseCallbacks):
    def __init__(self, proc_func, per_n_epoch=1, per_n_step=None):
        super().__init__()
        assert (per_n_epoch is None and per_n_step is not None) or \
                (per_n_epoch is not None and per_n_step is None)

        self.per_n_epoch = per_n_epoch
        self.per_n_step = per_n_step
        self.proc_func = proc_func

    def run(self, state, event):
        # ! TODO: seperate visual logging and tensorboard logging ?
        global_state = state.get_top_parent_state()
        if self.per_n_epoch is not None:
            if (global_state.curr_epoch % self.per_n_epoch != 0):
                return
        if self.per_n_step is not None:
            if (global_state.curr_global_step % self.per_n_step != 0) or \
                (global_state.curr_global_step == 0):
                return

        if self.per_n_epoch is not None:
            current_epoch = state.curr_epoch
            raw_output = state.step_output['raw']
            viz_image = self.proc_func(raw_output)
            if viz_image is None:
                return
            state.tracked_step_output['image']['output'] = viz_image
        else: 
            current_step = global_state.curr_global_step
            raw_output = state.step_output['raw']
            viz_image = self.proc_func(raw_output)
            if viz_image is None:
                return
            tfwriter = state.log_info['tfwriter']
            tfwriter.add_image('step-viz', 
                               viz_image, 
                               current_step, 
                               dataformats='HWC')
        return
