import collections

import cv2
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import distance_transform_edt

from misc.utils import cropping_center, get_bounding_box
from .augs import fix_mirror_padding


def unet_weight_map(ann, inst_list, w0=10.0, sigma=3.0, decay_margin=10):
    """
    `decay_margin` denote when how far the `w0 * np.exp(-(weight_map ** 2) / 2)`
    will move to zero (as in < epsilon) at that distance
    """
    if len(inst_list) <= 1:  # 1 instance only
        return np.zeros(ann.shape[:2])
    stacked_inst_bgd_dst = np.full(
        ann.shape[:2] + (len(inst_list),),
        1000,  # infinite far anyways
        dtype=np.float32,
    )

    # HW x HW
    # -> pairwise distance between contour pixels locations

    hw = np.array(ann.shape[:2])
    for idx, inst_id in enumerate(inst_list):
        inst_fgd_map = np.array(ann == inst_id, dtype=np.uint8)

        rmin, rmax, cmin, cmax = get_bounding_box(inst_fgd_map)
        inst_tl = np.array([rmin, cmin])
        inst_br = np.array([rmax, cmax])
        inst_tl -= decay_margin
        inst_br += decay_margin
        inst_tl[inst_tl < 0] = 0
        inst_br[inst_br > hw] = hw[inst_br > hw]

        inst_bgd_map = inst_fgd_map[inst_tl[0] : inst_br[0], inst_tl[1] : inst_br[1]]
        inst_bgd_map = np.array(inst_bgd_map == 0, dtype=np.uint8)
        inst_bgd_dst = distance_transform_edt(inst_bgd_map)
        stacked_inst_bgd_dst[
            inst_tl[0] : inst_br[0], inst_tl[1] : inst_br[1], idx
        ] = inst_bgd_dst

    # ! will only work up to 2nd closest
    near_nth = np.partition(stacked_inst_bgd_dst, 1, axis=-1)[..., 0:2]
    near_1 = near_nth[..., 0]
    near_2 = near_nth[..., 1]

    #
    pix_dst = near_1 + near_2
    weight_map = pix_dst / sigma
    weight_map = w0 * np.exp(-(weight_map ** 2) / 2)
    weight_map[ann > 0] = 0  # inner instances zero
    return weight_map


class InstPixelMap(object):
    output_ch_code = [""]

    def __call__(self, ann, *args, **kwargs):
        return [(ann > 0).astype(np.int32)]


class InstErodedMap(object):
    output_ch_code = ["", "", "#WEIGHT-MAP"]

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, ann, crop_shape, gen_unet_weight_map=True, *args, **kwargs):
        """Input annotation must be of original shape"""
        orig_ann = ann.copy()  # instance ID map
        fixed_ann = fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, crop_shape)

        # setting 1 boundary pix of each instance to background
        inner_map = np.zeros(fixed_ann.shape[:2], np.uint8)

        inst_list = list(np.unique(crop_ann))
        if 0 in inst_list:
            inst_list.remove(0)  # 0 is background

        # get structuring element
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ksize, self.ksize))

        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inner = cv2.erode(inst_map, k_disk, iterations=1)
            inner_map += inner

        if gen_unet_weight_map:
            inner_label = measurements.label(inner_map)[0]
            inst_list = np.unique(inner_label).tolist()[1:]
            weight_map = unet_weight_map(inner_label, inst_list, sigma=self.ksize)
        else:
            weight_map = np.zeros([ann.shape[0], ann.shape[1]])
        weight_map += 1

        inner_map[inner_map > 0] = 1  # binarize
        bg_map = 1 - inner_map

        return [bg_map, inner_map, weight_map]


class InstErodedContourMap(object):
    output_ch_code = ["", "", "#WEIGHT-MAP"]

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, ann, crop_shape, gen_unet_weight_map=True, *args, **kwargs):
        """Input annotation must be of original shape"""
        orig_ann = ann.copy()  # instance ID map
        fixed_ann = fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, crop_shape)

        bin_map = fixed_ann.copy()
        bin_map[bin_map > 0] = 1

        # setting 1 boundary pix of each instance to background
        inner_map = np.zeros(fixed_ann.shape[:2], np.uint8)
        contour_map = np.zeros(fixed_ann.shape[:2], np.uint8)

        inst_list = list(np.unique(crop_ann))
        if 0 in inst_list:
            inst_list.remove(0)  # 0 is background

        # get structuring element
        k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ksize, self.ksize))

        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inner = cv2.erode(inst_map, k_disk, iterations=1)
            outer = cv2.dilate(inst_map, k_disk, iterations=1)
            inner_map += inner
            contour_map += (outer- inner)


        if gen_unet_weight_map:
            inner_label = measurements.label(inner_map)[0]
            inst_list = np.unique(inner_label).tolist()[1:]
            weight_map = unet_weight_map(inner_label, inst_list, sigma=self.ksize)
        else:
            weight_map = np.zeros([ann.shape[0], ann.shape[1]])
        weight_map += 1

        inner_map[inner_map > 0] = 1  # binarize
        contour_map[contour_map>0] = 1 # binarize
        bg_map = 1 - (inner_map + contour_map)

        positive_map = inner_map+(2*contour_map)
        positive_map = np.round(positive_map).astype('int')

        return [bg_map, positive_map, weight_map]


class TypePixelMap(object):
    output_ch_code = [""]

    def __call__(self, ann, *args, **kwargs):
        return [ann.astype(np.int32)]


class NucleiPixelMap(object):
    output_ch_code = [""]

    def __call__(self, ann, *args, **kwargs):
        # binarize
        ann[ann>0] = 1 
        return [ann.astype(np.int32)]


class PatchClass(object):
    output_ch_code = [""]

    def __call__(self, ann, *args, **kwargs):
        return [ann.astype(np.int32)]


def gen_targets(ann, channel, channel_to_target, crop_shape, task_mode, **kwargs):
    """Generate the targets for the network by decoding each channel info. 
    Return a dict with keys denotes the names of each target map, and 
    a vector to indicate which target is dummy (0 value).

    A target map may one or many channel. However, each target map name should
    be paired to one of the target output name from the neural network.

    `channel` denote the information encode within the channel at the same index within `ann`

    if requested source channel in `channel_to_target` doesnt exist within `channel`, 
    a dummy filled is inserted, If `return_dict=True`, return a dictionary with key is 
    the target code

    Each target map must maintain shape HWC.

    """
    target_getter_dict = {
        "IP": InstPixelMap(),
        "IP-ERODED-3": InstErodedMap(ksize=3),
        "IP-ERODED-11": InstErodedMap(ksize=11),
        "IP-ERODED-CONTOUR-3": InstErodedContourMap(ksize=3),
        "IP-ERODED-CONTOUR-11": InstErodedContourMap(ksize=11),
        "NP": NucleiPixelMap(),
        "TP": TypePixelMap(),
        "PC": PatchClass()
    }

    has_flag = []
    new_ch_list = []
    new_ch_code = []
    for ch_code, tg_code in channel_to_target.items():
        target_getter = target_getter_dict[tg_code]
        sub_ch_code = target_getter.output_ch_code
        sub_ch_code = [ch_code + c for c in sub_ch_code]
        if ch_code not in channel:
            ann_ch = [np.zeros(list(ann.shape[:2])) for i in range(len(sub_ch_code))]
            has_flag.extend([None for i in range(len(sub_ch_code))])
        else:
            if task_mode == 'seg':
                ch_idx = channel.index(ch_code)
                ann_ch = (ann[..., ch_idx]).copy()
            else:
                ann_ch = ann
            ann_ch = target_getter(ann_ch, crop_shape, **kwargs)

            has_flag.extend(sub_ch_code)
        new_ch_list.extend(ann_ch)
        new_ch_code.extend(sub_ch_code)


    assert len(new_ch_list) == len(new_ch_code)
    new_ch_list = [cropping_center(ch, crop_shape) for ch in new_ch_list]
    # reshape to ensure HWC always
    new_ch_list = [v[..., None] for v in new_ch_list if len(v.shape) == 2]

    target_dict = list(zip(new_ch_code, new_ch_list))
    target_dict = collections.OrderedDict(target_dict)

    return target_dict, has_flag
