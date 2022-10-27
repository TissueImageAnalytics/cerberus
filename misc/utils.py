import glob
import inspect
import logging
import os
import json
import pathlib
import shutil
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import skimage
from scipy import ndimage
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_erosion,
)
from skimage.filters import rank, threshold_otsu
from skimage.morphology import disk, remove_small_holes, remove_small_objects
from tqdm import tqdm


def get_overlap(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {"x1": bb1[2], "y1": bb1[0], "x2": bb1[3], "y2": bb1[1]}
    bb2 = {"x1": bb2[2], "y1": bb2[0], "x2": bb2[3], "y2": bb2[1]}
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def get_files(data_dir_list, data_ext):
    """Given a list of directories containing data with extention 'data_ext',
    generate a list of paths for all files within these directories

    """
    data_files = []
    for sub_dir in data_dir_list:
        files_list = glob.glob(sub_dir + "/*" + data_ext)
        files_list.sort()  # ensure same order
        data_files.extend(files_list)

    return data_files


def remap_label(pred, by_size=False, ds_factor=None):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1

    return new_pred


def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def center_pad_to_shape(img, size, cval=255):
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


def stain_entropy_otsu(img):
    """Include docstring"""
    img_copy = img.copy()
    hed = skimage.color.rgb2hed(img_copy)  # convert colour space
    hed = (hed * 255).astype(np.uint8)
    h = hed[:, :, 0]
    e = hed[:, :, 1]
    d = hed[:, :, 2]
    selem = disk(4)  # structuring element
    # calculate entropy for each colour channel
    h_entropy = rank.entropy(h, selem)
    e_entropy = rank.entropy(e, selem)
    d_entropy = rank.entropy(d, selem)
    entropy = np.sum([h_entropy, e_entropy], axis=0) - d_entropy
    # otsu threshold
    threshold_global_otsu = threshold_otsu(entropy)
    mask = entropy > threshold_global_otsu

    return mask


def morphology(mask):
    """Apply morphological operation to refine tissue mask"""

    # Remove thin structures
    selem1 = disk(int(3))
    mask = binary_erosion(mask, selem1)

    # Remove small disconnected objects
    mask = remove_small_holes(mask, area_threshold=2000, connectivity=1,)

    mask = remove_small_objects(mask, min_size=2000, connectivity=1,)

    mask = binary_dilation(mask, selem1)

    mask = remove_small_holes(mask, area_threshold=2000, connectivity=1,)

    # Fill holes in mask
    mask = ndimage.binary_fill_holes(mask)

    return mask


def get_tissue_mask(img):
    """Description"""
    mask = stain_entropy_otsu(img)
    mask = morphology(mask)
    mask = mask.astype("uint8")

    return mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext`
    such as `ext='.png'`.

    return list is sorted.

    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def log_debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


def log_info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def multiproc_dispatcher(data_list, nr_worker=0, show_pbar=True):
    """
    data_list is alist of [[func, arg1, arg2, etc.]]
    Resutls are alway sorted according to source position
    """
    if nr_worker > 0:
        proc_pool = ProcessPoolExecutor(nr_worker)

    result_list = []
    future_list = []

    if show_pbar:
        pbar = tqdm(total=len(data_list), ascii=True, position=0)
    for run_idx, dat in enumerate(data_list):
        func = dat[0]
        args = dat[1:]
        if nr_worker > 0:
            future = proc_pool.submit(func, run_idx, *args)
            future_list.append(future)
        else:
            # ! assume 1st return is alwasy run_id
            result = func(run_idx, *args)
            result_list.append(result)
            if show_pbar:
                pbar.update()
    if nr_worker > 0:
        for future in as_completed(future_list):
            if future.exception() is not None:
                logging.info(future.exception())
            else:
                result = future.result()
                result_list.append(result)
            if show_pbar:
                pbar.update()
        proc_pool.shutdown()
    if show_pbar:
        pbar.close()

    result_list = sorted(result_list, key=lambda k: k[0])
    result_list = [v[1:] for v in result_list]
    return result_list


def save_json(path, old_dict, mag=None):
    new_dict = {}
    for target, old_dict_target in old_dict.items():
        new_dict_tmp = {}
        for inst_id, inst_info in old_dict_target.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to jsonable
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            new_dict_tmp[inst_id] = new_inst_info
        new_dict[target] = new_dict_tmp

    json_dict = {"mag": mag, "instances": new_dict}  # to sync the format protocol
    with open(path, "w") as handle:
        json.dump(json_dict, handle)


def to_wasabi(save_path, inst_info_dict, viz_info, mode, scale_factor, annotator):

    line_width = viz_info["line_width"]

    ann_list_all = []
    type_list_all = []
    for idx, inst_info in inst_info_dict.items():
        ann_list_all.append(inst_info[mode])
        if "type" in inst_info.keys():
            type_list_all.append(inst_info["type"])
        else:
            type_list_all.append(-1)

    def gen_wasabi_dict(id, coords, type_name, type_color, mode, line_width):
        new_dict = {
            "fillColor": "rgba({0},{1},{2},{3})".format(*type_color),
            "id": "{:024d}".format(id),
            "label": {"value": "nuclei"},
            "group": type_name,
        }
        if mode == "centroid":
            update_dict = {
                "lineColor": "rgb(0, 0, 0)",
                "type": "point",
                "center": coords,
                "lineWidth": line_width,
            }
            new_dict.update(update_dict)
        elif mode == "contour":
            update_dict = {
                "lineColor": "rgb({0},{1},{2})".format(*type_color),
                "type": "polyline",
                "closed": True,
                "points": coords,
                "lineWidth": line_width,
            }
            new_dict.update(update_dict)

        return new_dict

    format_obj_list = []
    for i, ann in enumerate(ann_list_all):
        lab = type_list_all[i]
        if mode == "contour":
            pts_list = np.ceil(np.array(ann) * scale_factor)
            pts_list = [[int(v[0]), int(v[1]), 0] for v in pts_list]
        elif mode == "centroid":
            pos = ann * scale_factor
            pts_list = [int(pos[0]), int(pos[1]), 0]

        type_name = viz_info["type_names"][lab]
        if lab == -1:
            type_colour = viz_info["inst_colour"]
            type_name = viz_info["type_names"][1]
        else:
            type_colour = viz_info["type_colour"][lab]
        obj_dict = gen_wasabi_dict(i, pts_list, type_name, type_colour, mode, line_width)
        format_obj_list.append(obj_dict)
    output_dict = {
        "annotation": {
            "description": "",
            "elements": format_obj_list,
            "name": annotator,
        }
    }
    with open(save_path, "w") as handle:
        json.dump(output_dict, handle)
    return
