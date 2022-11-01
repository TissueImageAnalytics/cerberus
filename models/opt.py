import torch.optim as optim
from run_utils.callbacks.base import (
    PeriodicSaver,
    ProcessAccumulatedEpochOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    TriggerEngine,
    VisualizeOutput,
)
from run_utils.callbacks.logging import LoggingOutput
from run_utils.engine import Events

from .net_desc import create_model
from .run_desc import (
    ProcStepRawOutput,
    proc_cum_epoch_step_output,
    train_step,
    valid_step,
    viz_step_output,
)

# __PER_N_STEPS = 3000 # mtl
# __PER_N_STEPS = 800 # nuc-class

__PER_N_STEPS = 3 

def get_config(
    train_loader_list,
    infer_loader_list,
    pretrained_path,
    loader_kwargs={},
    model_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    **kwargs
):

    config = {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        "phase_list": [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(**model_kwargs),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-3,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        "lr_scheduler": lambda opt, n_iter: optim.lr_scheduler.StepLR(
                            # opt, 75000 #mtl
                            opt, 16000 #nuc-class
                            # opt, 9000
                        ),
                        # 'lr_scheduler': lambda opt, n_iter: \
                        #     optim.lr_scheduler.CosineAnnealingLR(opt, 50),
                        "extra_info": {
                            "loss": loss_kwargs
                            # OUTPUT_HEAD_NAME(TARGET_NAME)
                            #   weight # weight loss for learning this target
                            #   loss # loss func to calculate on this head and its weight
                            # overall_loss = sum target_loss * weightloss * loss
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        "pretrained": pretrained_path,
                    },
                },
                "loader": loader_kwargs,
                "nr_epochs": 140,
                # "nr_epochs": 800,
            },
        ],
        # ------------------------------------------------------------------
        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        "run_engine": {
            "train": {
                "loader": train_loader_list,
                "run_step": train_step,
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        ScalarMovingAverage(alpha=0.95),
                        TrackLr(),
                        PeriodicSaver(per_n_epoch=None, per_n_step=__PER_N_STEPS),
                        VisualizeOutput(
                            viz_step_output, per_n_epoch=None, per_n_step=__PER_N_STEPS
                        ),
                        LoggingOutput(per_n_epoch=None, per_n_step=__PER_N_STEPS),
                        TriggerEngine(
                            "infer", per_n_epoch=None, per_n_step=__PER_N_STEPS
                        ),
                        ScheduleLr(),
                    ],
                },
            },
            "infer": {
                "loader": infer_loader_list,
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        ProcStepRawOutput()
                    ],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedEpochOutput(
                            lambda a, b: proc_cum_epoch_step_output(a, b)
                        ),
                        LoggingOutput(per_n_epoch=None, per_n_step=__PER_N_STEPS),
                    ],
                },
            },
        },
    }

    return config
