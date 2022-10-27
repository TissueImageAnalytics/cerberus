import math
import numpy as np
import scipy.io as sio
import joblib
from collections import OrderedDict

import torch
import torch.utils.data as data

from misc.utils import cropping_center


class SerializeFileList(data.IterableDataset):
    """Read a single file as multiple patches of same shape, perform the padding beforehand."""

    def __init__(self, img_list, patch_info_list, patch_size, preproc=None):
        super().__init__()
        self.patch_size = patch_size

        self.img_list = img_list
        self.patch_info_list = patch_info_list

        self.worker_start_img_idx = 0
        # * for internal worker state
        self.curr_img_idx = 0
        self.stop_img_idx = 0
        self.curr_patch_idx = 0
        self.stop_patch_idx = 0
        self.preproc = preproc
        return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.stop_img_idx = len(self.img_list)
            self.stop_patch_idx = len(self.patch_info_list)
            return self
        else:  # in a worker process so split workload, return a reduced copy of self
            per_worker = len(self.patch_info_list) / float(worker_info.num_workers)
            per_worker = int(math.ceil(per_worker))

            global_curr_patch_idx = worker_info.id * per_worker
            global_stop_patch_idx = global_curr_patch_idx + per_worker
            self.patch_info_list = self.patch_info_list[
                global_curr_patch_idx:global_stop_patch_idx
            ]
            self.curr_patch_idx = 0
            self.stop_patch_idx = len(self.patch_info_list)
            # * check img indexer, implicit protocol in infer.py
            global_curr_img_idx = self.patch_info_list[0][-1]
            global_stop_img_idx = self.patch_info_list[-1][-1] + 1
            self.worker_start_img_idx = global_curr_img_idx
            self.img_list = self.img_list[global_curr_img_idx:global_stop_img_idx]
            self.curr_img_idx = 0
            self.stop_img_idx = len(self.img_list)
            return self  # does it mutate source copy?

    def __next__(self):

        if self.curr_patch_idx >= self.stop_patch_idx:
            raise StopIteration # when there is nothing more to yield 
        # `img_idx` wrt to original img_list before being split into worker
        (input_info, output_info), img_idx = self.patch_info_list[self.curr_patch_idx]
        input_tl, input_br = input_info

        img_ptr = self.img_list[img_idx - self.worker_start_img_idx]
        patch_data = img_ptr[input_tl[0] : input_br[0],
                             input_tl[1] : input_br[1]]
        self.curr_patch_idx += 1
        return patch_data, output_info, img_idx

####
class SerializeArray(data.Dataset):
    def __init__(self, mmap_array_path, patch_info_list, patch_size, preproc=None):
        super().__init__()
        self.patch_size = patch_size

        # use mmap as intermediate sharing, else variable will be duplicated
        # accross torch worker => OOM error, open in read only mode
        self.image = np.load(mmap_array_path, mmap_mode="r")

        self.patch_info_list = patch_info_list
        self.preproc = preproc
        return

    def __len__(self):
        return len(self.patch_info_list)

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        patch_data = self.image[
            patch_info[0] : patch_info[0] + self.patch_size[0],
            patch_info[1] : patch_info[1] + self.patch_size[1],
        ]
        patch_data = np.array(patch_data)

        if self.preproc is not None:
            patch_data = self.preproc(patch_data)
        return patch_data, patch_info



class PatchDataset(torch.utils.data.Dataset):
    """Loads images from a file list for inference.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    def __init__(
        self,
        file_list,
        input_shape=None
    ):
        assert input_shape is not None

        # aggregate the data as [{'image_path': inst_id}]
        self.info_list = []
        self.total_objs = 0
        for filename in file_list:
            info = sio.loadmat(filename)
            nr_objs = len(info['class'])
            for id in range(nr_objs):
                inst_info = [info['class'][id], info['centroid'][id]]
                self.info_list.append({filename: inst_info})
                self.total_objs += 1

        self.input_shape = [input_shape, input_shape]
        self.id = 0

        return

    def __len__(self):
        return len(self.info_list)

    def _extract_patch(self, img, centroid, shape):
        """Extract patch of given size from image centred at a provided
        coordinate. Also perform reflective padding if needed.
        
        """
        left_ = centroid[0] - (shape[1] // 2)
        if left_ < 0:
            padl = abs(left_)
            left = 0
        else:
            padl = 0
            left = left_
        right = left_ + shape[1]
        if right > img.shape[1]:
            padr = right - img.shape[1]
            right = img.shape[1]
        else:
            padr = 0
        top_ = centroid[1] - (shape[0] // 2)
        if top_ < 0:
            padt = abs(top_)
            top = 0
        else:
            padt = 0
            top = top_
        bottom = top_ + shape[0]
        if bottom > img.shape[0]:
            padb = bottom - img.shape[0]
            bottom = img.shape[0]
        else:
            padb = 0
        
        patch = img[int(top):int(bottom), int(left):int(right)]
        # perform reflective padding if needed
        if padl > 0 or padr > 0 or padt > 0 or padb > 0:
            patch = np.lib.pad(patch, ((int(padt), int(padb)), (int(padl), int(padr)), (0, 0)), 'reflect')
        return patch 


    def __getitem__(self, idx):
        info_dict = self.info_list[idx]
        filename = list(info_dict.keys())[0]
        # read image
        img = sio.loadmat(filename)["img"]
        # get centroid and class values
        inst_info = list(info_dict.values())[0]
        class_ = inst_info[0] - 1 # class range [0, C-1]
        centroid = np.round(inst_info[1]) # ensure integer values

        # extract a patch centred at the centroid of a fixed size
        # extract a patch double the size and then crop after augmentation
        extract_shape = [self.input_shape[0], self.input_shape[1]]
        img_patch = self._extract_patch(img, centroid, extract_shape)

        return  img_patch, class_


class PatchDataset2(torch.utils.data.Dataset):
    """Loads images from a file list for inference.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    def __init__(
        self,
        file_list,
        input_shape=None
    ):
        assert input_shape is not None
        self.info_list = file_list
        self.input_shape = input_shape

        self.id = 0

        return

    def __len__(self):
        return len(self.info_list)


    def __getitem__(self, idx):
        filename = self.info_list[idx]
        # read image
        info = joblib.load(filename)
        patch = info["img"]
        class_ = info["ann"]

        patch = cropping_center(patch, self.input_shape)

        return  patch, class_

