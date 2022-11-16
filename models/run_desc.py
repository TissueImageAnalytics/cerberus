import numpy as np
import copy
import operator
import matplotlib.pyplot as plt
from functools import reduce

from collections import OrderedDict
    
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.loss_utils import dice_loss, xentropy_loss
from misc.utils import center_pad_to_shape, cropping_center
from run_utils.callbacks import BaseCallbacks


def get_class_wmap(wmap, weights):
    wmap_copy = torch.clone(wmap).detach()
    for class_val, class_weight in weights.items():
        wmap_copy[wmap == class_val] = class_weight
    return wmap_copy


def train_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    loss_func_dict = {"ce": xentropy_loss, "dice": dice_loss}
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    model = run_info["net"]["desc"]
    optimizer = run_info["net"]["optimizer"]

    #* assume batch data is of the form
    # {
    #   'img' : Tensor, NxHWC
    #   'dummy_target'  : Tensor, NxB   # indicate which GT each sample has and at the
    #                             # location that doesnt contain GT, the corresponding GT in the branch is dummy 0 array
    #                             # branch is one-hot encoded, and the coded position is indicate from external
    #   'ouput_head_A': Tensor, NxHWC # GT for ouput_head_A
    #   'ouput_head_B': Tensor, NxHWC # GT for ouput_head_B
    #   'Branch etc' : Tensor
    # }

    img_list = batch_data.pop("img")
    input_dims = img_list.shape[1:3]
    # 0 value is not a target but is a dummy array
    has_target_list = batch_data.pop("dummy_target")
    true_dict = batch_data

    batch_size = img_list.shape[0]

    img_list = img_list.to("cuda").type(torch.float32)  # to NCHW
    img_list = img_list.permute(0, 3, 1, 2).contiguous()

    true_dict = OrderedDict(
        [[k, v.to("cuda").type(torch.float32)] for k, v in true_dict.items()]
    )
    true_dict = OrderedDict(
        [[k, v.permute(0, 3, 1, 2).contiguous()] for k, v in true_dict.items()]
    )

    #### *
    tgt_name_list = np.unique(has_target_list[has_target_list != None])
    dec_head_list = list(model.module.decoder_head.keys())
    
    # check which branch should be trained basing on the GT available within the batch
    train_dec_list = [
        dec_head_name
        for dec_head_name in dec_head_list
        if np.any([dec_head_name in x for x in tgt_name_list])
    ]
    tgt_name_list = list(tgt_name_list) # convert back to list for index retrieval later

    #### *
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    # ! freeze model weights for object subtyping (see net_desc.py)
    if model.module.subtype_gland or model.module.subtype_nuclei:
        model.module._freeze_weight()

    pred_dict = model(img_list, train_dec_list)

    #### *
    loss_opts = run_info["net"]["extra_info"]["loss"]

    all_loss = 0
    for head_name, head_pred in pred_dict.items():
        # ! we know the protocol is that each col correspond to 1 target,
        # ! so any check will show which sample has target or not
        head_flag = np.any(has_target_list == head_name, axis=-1)
        head_flag = torch.from_numpy(head_flag.astype(np.float32)).to("cuda")

        if head_name == "Gland-TYPE":
            nr_classes = model.module.decoder_info_list["Gland#TYPE"]["TYPE"]
        elif head_name == "Nuclei-TYPE":
            nr_classes = model.module.decoder_info_list["Nuclei#TYPE"]["TYPE"]

        sample_true_list = true_dict[head_name]
        sample_pred_list = pred_dict[head_name]

        # make sure targets and predictions have same spatial dims
        if (
            sample_pred_list.shape[2] == sample_true_list.shape[2]
            and sample_pred_list.shape[3] == sample_true_list.shape[3]
        ):
            # retrieve sample weight map if exists
            sample_wmap_name = head_name.split("#")[0] + "#WEIGHT-MAP"
            if sample_wmap_name in true_dict:
                sample_wmap = true_dict[sample_wmap_name]
            else:
                # ! may have problem between NCHW and NHW
                sample_wmap = torch.ones_like(sample_true_list)  

            if head_name == "Nuclei-TYPE" or head_name == "Gland-TYPE":
                class_weights = loss_opts["class_weight"][head_name]
                # only compute loss within nuclei/gland for type classification
                binary_map = torch.clone(sample_true_list).detach()
                binary_map = (binary_map > 0).type(torch.float32) * 1.0  # binarize
                sample_wmap = get_class_wmap(sample_true_list, class_weights)

            if head_name == "Patch-Class":
                sample_true_list = torch.squeeze(sample_true_list)
                sample_pred_list = torch.squeeze(sample_pred_list)

            head_all_loss = 0
            head_loss_dict = loss_opts["loss_info"][head_name]["loss"]
            head_loss_weight = loss_opts["loss_info"][head_name]["weight"]
            for loss_name, loss_weight in head_loss_dict.items():
                loss_func = loss_func_dict[loss_name]
                if loss_name == "dice":
                    sample_true_list_ = F.one_hot(
                        torch.squeeze(sample_true_list.to(torch.int64)), num_classes=nr_classes
                    )
                    sample_true_list_ = sample_true_list_.permute(0, 3, 1, 2)
                    sample_pred_list_ = torch.softmax(sample_pred_list, 1)
                    # only compute for positive classes
                    loss_args = [
                        sample_true_list_[:, 1:, :, :],
                        sample_pred_list_[:, 1:, :, :],
                    ]
                    term_loss = loss_func(*loss_args, reduction=False, mask=binary_map)
                else:
                    sample_true_list_ = sample_true_list
                    sample_pred_list_ = sample_pred_list
                    loss_args = [sample_true_list_, sample_pred_list_]
                    sample_loss = loss_func(*loss_args, reduction=False)
                    # ! may not behave correctly when different loss dont like the sample wmap
                    sample_loss = sample_loss * sample_wmap[:, 0]  # still NHW
                    # reduce pixel-wise loss to sample wise loss
                    sample_loss = torch.mean(sample_loss, dim=(1, 2))
                    # now multiply with mask so exclude dummy target out and reduce
                    term_loss = torch.sum((sample_loss * head_flag)) / (
                        torch.sum(head_flag) + 1.0e-8
                    )
                # multiply weight loss if needed
                head_all_loss += term_loss * loss_weight
            track_value(
                "%s_loss" % head_name, head_all_loss.cpu().item() * head_loss_weight
            )
            all_loss += head_all_loss * head_loss_weight

    track_value("overall_loss", all_loss.cpu().item())

    # * gradient update
    all_loss.backward()
    optimizer.step()

    proc_func_dict = {
        "Lumen-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Gland-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Gland-TYPE": lambda x: torch.softmax(x, -1),
        "Nuclei-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Nuclei-TYPE": lambda x: torch.softmax(x, -1),
        "Patch-Class": lambda x: torch.argmax(
            torch.softmax(x, -1), dim=-1, keepdim=True
        ),
    }

    # pick 2 random sample from the batch for visualization
    sample_indices = torch.randint(0, batch_size, (2,))

    img_list = (img_list[sample_indices]).byte()  # to uint8
    img_list = img_list.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    true_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in true_dict.items()]
    )
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )

    sub_pred_dict = OrderedDict()
    sub_true_dict = OrderedDict()
    for head_name, head_output in pred_dict.items():
        head_output = head_output[sample_indices]
        head_output_pred = proc_func_dict[head_name](head_output)
        if head_name == "Patch-Class" or "Patch-Class" in tgt_name_list:
            if head_name != "Patch-Class":
                head_output_pred = torch.permute(head_output_pred, (0, 3, 1, 2))
            head_output_pred = F.interpolate(
                head_output_pred.type(torch.float32), size=input_dims, mode="nearest"
            )
            head_output_pred = torch.permute(head_output_pred, (0, 2, 3, 1))
        head_output_pred = torch.squeeze(head_output_pred)
        if "TYPE" in head_name:
            head_output_pred = torch.argmax(head_output_pred, dim=-1, keepdim=False)
        sub_pred_dict[head_name] = head_output_pred.detach().cpu().numpy()

        head_output_true = true_dict[head_name][sample_indices]
        if head_name == "Patch-Class" or "Patch-Class" in tgt_name_list:
            head_output_true = torch.permute(head_output_true, (0, 3, 1, 2))
            head_output_true = F.interpolate(
                head_output_true.type(torch.float32), size=input_dims, mode="nearest"
            )
            head_output_true = torch.permute(head_output_true, (0, 2, 3, 1))
        head_output_true = torch.squeeze(head_output_true)
        sub_true_dict[head_name] = head_output_true.detach().cpu().numpy()

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": img_list,
        "true": sub_true_dict,
        "pred": sub_pred_dict,
    }
    return result_dict


def viz_step_output(raw_data, value_range=None, align_mode="max"):
    """
    `raw_data` will be implicitly provided in the similar format as the
    return dict from train/valid step, but may have been accumulated across N running step
    """
    # a recur func to find all shape with raw_data
    def list_shape(nested_arr_dict):
        all_shape = []
        for k, v in nested_arr_dict.items():
            if isinstance(v, dict):
                all_shape += list_shape(v)
            else:
                # assume to be shape NHWC
                all_shape += [v.shape[:3]]  # exclude the channel dim out
        return all_shape

    shape_list = list_shape(raw_data)
    if align_mode == "max":
        aligned_shape = np.max(np.array(shape_list), axis=0)[1:3]
    else:
        aligned_shape = np.min(np.array(shape_list), axis=0)[1:3]

    def colorize(ch, vmin, vmax, cmap):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32"))
        vmin = vmin if vmin is not None else ch.min()
        vmax = vmax if vmax is not None else ch.max()
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        return ch_cmap

    def make_border(v, margin=3):
        v[:margin] = 0
        v[-margin:] = 0
        v[:, :margin] = 0
        v[:, -margin:] = 0
        return v

    def make_row(arr_dict, sample_idx, viz_info, map_type="pred", nr_inst_classes=3):
        col_list = []
        for k, v in arr_dict.items():
            # don't perform viz for patch classification
            if k != "Patch-Class":
                v = v[sample_idx]
                if align_mode == "max":
                    v = center_pad_to_shape(v, aligned_shape)
                else:
                    v = cropping_center(v, aligned_shape)
                if k == "img":
                    v = make_border(v)
                    col_list = [v] + col_list  # prepend
                    continue

                v_range = viz_info[k]["range"]
                cmap = viz_info[k]["cmap"]

                if v.ndim == 2:
                    for val in range(nr_inst_classes - 1):
                        v_tmp = v == val + 1
                        v_tmp = colorize(v_tmp, v_range[0], v_range[1], cmap)
                        v_tmp = make_border(v_tmp)
                        col_list.append(v_tmp)
                else:
                    for idx in range(v.shape[-1]):
                        v_tmp = v[..., idx]
                        v_tmp = colorize(v_tmp, v_range[0], v_range[1], cmap)
                        v_tmp = make_border(v_tmp)
                        col_list.append(v_tmp)
        col_list = np.concatenate(col_list, axis=1)
        return col_list

    pred_dict = raw_data["pred"]
    true_dict = raw_data["true"]
    pred_dict["img"] = raw_data["img"]
    true_dict["img"] = raw_data["img"]

    viz_info_dict = {
        "Lumen-INST": {"range": (0, 1), "cmap": plt.get_cmap("jet")},
        "Gland-INST": {"range": (0, 1), "cmap": plt.get_cmap("jet")},
        "Gland-TYPE": {"range": (0, 2), "cmap": plt.get_cmap("nipy_spectral")},
        "Nuclei-INST": {"range": (0, 1), "cmap": plt.get_cmap("jet")},
        "Nuclei-TYPE": {"range": (0, 6), "cmap": plt.get_cmap("nipy_spectral")},
    }

    row_list = []
    nr_sample = raw_data["img"].shape[0]
    for sample_idx in range(nr_sample):
        pred_row = make_row(pred_dict, sample_idx, viz_info_dict, map_type="pred")
        true_row = make_row(true_dict, sample_idx, viz_info_dict, map_type="true")
        row_list.extend([true_row, pred_row])
    viz_list = np.concatenate(row_list, axis=0)
    return viz_list


def valid_step(batch_data, run_info):
    # TODO: synchronize the attach protocol
    run_info, state_info = run_info
    model = run_info["net"]["desc"]

    #* assume batch data is of the form
    # {
    #   'img' : Tensor, NxHWC
    #   'dummy_target'  : Tensor, NxB   # indicate which GT each sample has and at the
    #                             # location that doesnt contain GT, the corresponding GT in the branch is dummy 0 array
    #                             # branch is one-hot encoded, and the coded position is indicate from external
    #   'ouput_head_A': Tensor, NxHWC # GT for ouput_head_A
    #   'ouput_head_B': Tensor, NxHWC # GT for ouput_head_B
    #   'Branch etc' : Tensor
    # }

    img_list = batch_data.pop("img")
    input_dims = img_list.shape[1:3]
    # 0 value is not a target but is a dummy array
    has_target_list = batch_data.pop("dummy_target")
    true_dict = batch_data

    img_list = img_list.to("cuda").type(torch.float32)  # to NCHW
    img_list = img_list.permute(0, 3, 1, 2).contiguous()

    true_dict = OrderedDict(
        [[k, v.to("cuda").type(torch.float32)] for k, v in true_dict.items()]
    )
    true_dict = OrderedDict(
        [[k, v.permute(0, 3, 1, 2).contiguous()] for k, v in true_dict.items()]
    )

    tgt_name_list = np.unique(has_target_list[has_target_list != None])
    tgt_name_list = list(tgt_name_list) # convert back to list for index retrieval later
    
    #### *
    model.eval()
    
    with torch.no_grad():
        pred_dict = model(img_list)

    proc_func_dict = {
        "Lumen-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Gland-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Gland-TYPE": lambda x: torch.softmax(x, -1),
        "Nuclei-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Nuclei-TYPE": lambda x: torch.softmax(x, -1),
        "Patch-Class": lambda x: torch.argmax(
            torch.softmax(x, -1), dim=-1, keepdim=True
        ),
    }

    #### *
    img_list = (img_list).byte()  # to uint8
    img_list = img_list.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    true_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in true_dict.items()]
    )
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )

    sub_pred_dict = {}
    sub_true_dict = {}
    for head_name, head_output in pred_dict.items():

        head_output_pred = proc_func_dict[head_name](head_output)
        if head_name == "Patch-Class" or "Patch-Class" in tgt_name_list:
            if head_name != "Patch-Class":
                head_output_pred = torch.permute(head_output_pred, (0, 3, 1, 2))
            head_output_pred = F.interpolate(
                head_output_pred.type(torch.float32), size=input_dims, mode="nearest"
            )
            head_output_pred = torch.permute(head_output_pred, (0, 2, 3, 1))
        head_output_pred = torch.squeeze(head_output_pred)
        if "TYPE" in head_name:
            head_output_pred = torch.argmax(head_output_pred, dim=-1, keepdim=False)
        sub_pred_dict[head_name] = head_output_pred.detach().cpu().numpy()

        head_output_true = true_dict[head_name]

        if head_name == "Patch-Class" or "Patch-Class" in tgt_name_list:
            if head_name == "Patch-Class":
                head_output_true = torch.permute(head_output_true, (0, 3, 1, 2))
            head_output_true = F.interpolate(
                head_output_true.type(torch.float32), size=input_dims, mode="nearest"
            )
        head_output_true = torch.squeeze(head_output_true)
        sub_true_dict[head_name] = head_output_true.detach().cpu().numpy()

    # number of output channels per task- defined in paramset.yml
    channel_info = model.module.decoder_info_list

    # * up to user to define the protocol to process the raw output per step!
    result_dict = {
        "raw": {  # protocol for contents exchange within `raw`
            "img": img_list,
            "true": sub_true_dict,
            "pred": sub_pred_dict,
            "dummy": has_target_list,
            "channel_info": channel_info,
        }
    }
    return result_dict


def infer_step(img_list, model, output_shape, head_name_list):
    img_list = img_list.to("cuda").type(torch.float32)  # to NCHW
    img_list = img_list.permute(0, 3, 1, 2).contiguous()

    if not isinstance(output_shape, list):
        output_shape = [output_shape, output_shape]

    #### *
    model.eval()
    with torch.no_grad():
        pred_dict = model(img_list)
    
    proc_func_dict = {
        "Lumen-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Gland-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Gland-TYPE": lambda x: torch.softmax(x, -1),
        "Nuclei-INST": lambda x: torch.softmax(x, -1)[..., 1:],
        "Nuclei-TYPE": lambda x: torch.softmax(x, -1),
        "Patch-Class": lambda x: torch.argmax(
            torch.softmax(x, -1), dim=-1, keepdim=True
        ),
    }

    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )

    head_name_map = {
        "Gland" : "Gland-INST",
        "Gland#TYPE" : "Gland-TYPE",
        "Lumen" : "Lumen-INST",
        "Nuclei" : "Nuclei-INST",
        "Nuclei#TYPE" : "Nuclei-TYPE",
        "Patch-Class" : "Patch-Class",
    }

    sub_pred_dict = OrderedDict()
    for head_name_ in head_name_list:
        head_name = head_name_map[head_name_]
        head_output = pred_dict[head_name]
        head_output = proc_func_dict[head_name](head_output)
        if head_name == "Patch-Class":
            head_output = F.interpolate(
                head_output.type(torch.float32), size=output_shape, mode="nearest"
            )
            head_output = torch.squeeze(head_output)
            # fix when batch size is 1
            if len(list(head_output.shape)) == 2:
                head_output = torch.unsqueeze(head_output, 0)
        else:
            head_output = cropping_center(head_output, output_shape, batch=True)
        if "TYPE" in head_name:
            head_output = torch.argmax(head_output, dim=-1, keepdim=False)
        sub_pred_dict[head_name] = head_output.detach().cpu().numpy()

    batch_output_list = []
    for sample_idx in range(img_list.shape[0]):
        sample_output_dict = {
            head_name: head_pred[sample_idx]
            for head_name, head_pred in sub_pred_dict.items()
        }
        batch_output_list.append(sample_output_dict)

    return batch_output_list


def proc_cum_epoch_step_output(runner_name, epoch_data):

    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def get_from_nested_dict(nested_dict, nested_key_list):
        return reduce(operator.getitem, nested_key_list, nested_dict)

    def flatten_dict_hierarchy(nested_key_list, raw_data):
        output_list = []
        for step_output in raw_data:
            step_output = get_from_nested_dict(step_output, nested_key_list)
            step_output = np.split(step_output, step_output.shape[0], axis=0)
            output_list.extend(step_output)
        output_list = [np.squeeze(v) for v in output_list]
        return output_list

    ####
    def summarize_stats(cum_stats):
        accuracy = (cum_stats["over_correct"] + 1.0e-8) / (
            cum_stats["nr_pixels"] + 1.0e-8
        )
        dice_score = 2 * cum_stats["over_inter"] / (cum_stats["over_total"] + 1.0e-8)
        return accuracy, dice_score

    ####
    cum_stat_dict = epoch_data[1]

    for target_name, cum_stat in cum_stat_dict.items():
        if "INST" in target_name:
            for k, v in cum_stat.items():
                accu_list, dice_list = summarize_stats(v)
                track_value(f"{target_name}-{k}-accu", accu_list, "scalar")
                track_value(f"{target_name}-{k}-dice", dice_list, "scalar")
        elif "TYPE" in target_name:
            accu_list = []
            dice_list = []
            for k, v in cum_stat.items():
                accu, dice = summarize_stats(v)
                accu_list.append(accu)
                dice_list.append(dice)
                track_value(f"{target_name}-{k}-dice", dice, "scalar")
            track_value(f"{target_name}-avg-accu", np.mean(accu_list), "scalar")
            track_value(f"{target_name}-avg-dice", np.mean(dice_list), "scalar")
        else:
            accu_list = []
            dice_list = []
            for k, v in cum_stat.items():
                accu, dice = summarize_stats(v)
                accu_list.append(accu)
                dice_list.append(dice)
                track_value(f"{target_name}-{k}-dice", dice, "scalar")
            track_value(f"{target_name}-avg-accu", np.mean(accu_list), "scalar")
            track_value(f"{target_name}-avg-dice", np.mean(dice_list), "scalar")
            
        print()
        print(f"{target_name}-avg-accu", np.mean(accu_list))
        print(f"{target_name}-avg-dice", np.mean(dice_list))

    ####
    sampled_raw_data = epoch_data[0]
    # ! dont do image tracking because no raw data were stored (percentage base)
    if len(sampled_raw_data) == 0:  # also assume min batch size = 8
        return track_dict

    target_name_list = list(sampled_raw_data[0]["pred"].keys())

    pred_dict = OrderedDict()
    true_dict = OrderedDict()
    img_list = flatten_dict_hierarchy(["img"], sampled_raw_data)
    for target_name in target_name_list:
        pred_list = flatten_dict_hierarchy(["pred", target_name], sampled_raw_data)
        true_list = flatten_dict_hierarchy(["true", target_name], sampled_raw_data)
        pred_dict[target_name] = pred_list
        true_dict[target_name] = true_list

    # * pick 1 random sample from the batch for visualization
    nr_sel_sample = 1
    nr_all_sample = len(img_list)
    sampled_indices = np.random.randint(0, nr_all_sample, nr_sel_sample).tolist()

    sub_img_list = np.array([img_list[idx] for idx in sampled_indices])

    sub_pred_dict = OrderedDict()
    sub_true_dict = OrderedDict()
    for head_name in target_name_list:
        sub_pred = pred_dict[head_name]
        sub_true = true_dict[head_name]
        sub_pred_dict[head_name] = np.array([sub_pred[idx] for idx in sampled_indices])
        sub_true_dict[head_name] = np.array([sub_true[idx] for idx in sampled_indices])
    viz_raw_data = {"img": sub_img_list, "pred": sub_pred_dict, "true": sub_true_dict}
    viz_fig = viz_step_output(viz_raw_data)
    track_dict["image"]["output"] = viz_fig

    return track_dict


####
class ProcStepRawOutput(BaseCallbacks):
    def run(self, state, event):
        def _dice_info(true, pred, label, mask=None):
            true = np.array(true == label, np.int32)
            pred = np.array(pred == label, np.int32)
            if mask is None:
                inter = np.sum(pred * true, axis=(1, 2))  # collapse HW
                total = np.sum(pred + true, axis=(1, 2))  # collapse HW
            else:
                inter = np.sum(mask * (pred * true), axis=(1, 2))  # collapse HW
                total = np.sum(mask * (pred + true), axis=(1, 2))  # collapse HW
            return inter, total

        def _batch_stats(
            true,
            pred,
            cum_dict,
            patch_target_flag,
            patch_size,
            label_value=1,
            mask=None,
        ):
            inter, total = _dice_info(true, pred, label_value, mask)
            correct = np.sum(true == pred, axis=(1, 2))
            cum_dict["over_inter"] += np.sum(patch_target_flag * inter)
            cum_dict["over_total"] += np.sum(patch_target_flag * total)
            cum_dict["over_correct"] += np.sum(patch_target_flag * correct)
            cum_dict["nr_pixels"] += np.sum(patch_target_flag * patch_size)

            return cum_dict

        def get_batch_stat(patch_true, patch_pred, cum_dict, target_name, channel_info):
            patch_true = np.squeeze(patch_true)  # ! may be wrong for n=1
            patch_pred = np.squeeze(patch_pred)
            n, h, w = patch_pred.shape[:3]
            patch_size = np.array([h * w for i in range(n)])
            patch_target_flag = np.any(step_dummy == target_name, axis=-1).astype(
                np.float32
            )
            target_split = target_name.split("-")
            if target_split[-1] == "INST":
                nr_inst_types = channel_info[target_split[0]]["INST"]
                for idx in range(1, nr_inst_types):
                    patch_pred_ = np.array(
                        patch_pred[..., idx - 1] > 0.5, dtype=np.int32
                    )
                    patch_pred_ *= idx
                    cum_dict[idx] = _batch_stats(
                        patch_true,
                        patch_pred_,
                        cum_dict[idx],
                        patch_target_flag,
                        patch_size,
                        label_value=idx,
                    )
            elif target_split[-1] == "TYPE":
                mask = patch_true > 0
                nr_types = channel_info[f"{target_split[0]}#TYPE"]["TYPE"]
                # don't consider background
                for idx in range(1, nr_types):
                    cum_dict[idx] = _batch_stats(
                        patch_true,
                        patch_pred,
                        cum_dict[idx],
                        patch_target_flag,
                        patch_size,
                        label_value=idx,
                        mask=mask,
                    )
            else:
                # patch classification
                nr_types = channel_info[f"{target_split[0]}-Class"]["OUT"]
                for idx in range(nr_types):
                    cum_dict[idx] = _batch_stats(
                        patch_true,
                        patch_pred,
                        cum_dict[idx],
                        patch_target_flag,
                        patch_size,
                        label_value=idx,
                    )
                    
            return cum_dict

        step_output = state.step_output["raw"]
        step_pred_output = step_output["pred"]
        step_true_output = step_output["true"]
        step_dummy = step_output["dummy"]
        channel_info = step_output["channel_info"]
        target_name_list = list(step_pred_output.keys())
        # ! assume that target_name_list and step_dummy
        # ! are in the same order !

        state_cum_output = state.epoch_accumulated_output
        # custom init and protocol
        if state.curr_epoch_step == 0:
            stat_list = ["over_inter", "over_total", "over_correct", "nr_pixels"]
            template_stat_dict = {s: 0 for s in stat_list}

            step_cum_stat_dict = {}
            for target_name in target_name_list:
                target_base = target_name.split('-')[0]
                if "INST" in target_name:
                    NUM_CLASSES = channel_info[target_base]["INST"]
                    step_cum_stat_dict[target_name] = {
                        type_id: copy.deepcopy(template_stat_dict)
                        for type_id in range(1, NUM_CLASSES)
                    }
                elif "TYPE" in target_name:
                    NUM_CLASSES = channel_info[f"{target_base}#TYPE"]["TYPE"]
                    step_cum_stat_dict[target_name] = {
                        type_id: copy.deepcopy(template_stat_dict)
                        for type_id in range(1, NUM_CLASSES)
                    }
                elif "Patch-Class" in target_name:
                    NUM_CLASSES = channel_info[f"{target_base}-Class"]["OUT"]
                    step_cum_stat_dict[target_name] = {
                        type_id: copy.deepcopy(template_stat_dict)
                        for type_id in range(NUM_CLASSES)
                    }

            state_cum_output = [[], step_cum_stat_dict]
            state.epoch_accumulated_output = state_cum_output
        state_cum_output = state.epoch_accumulated_output

        # edit by reference also, variable is a reference, not a deep copy
        step_cum_stat_dict = state_cum_output[1]
        for target_name in target_name_list:
            new_cum_dict = get_batch_stat(
                step_true_output[target_name],
                step_pred_output[target_name],
                step_cum_stat_dict[target_name],
                target_name,
                channel_info
            )
            step_cum_stat_dict[target_name] = new_cum_dict

        #! for debugging
        state_cum_output[0].append(step_output)
        state_cum_output[1] = step_cum_stat_dict

        return
