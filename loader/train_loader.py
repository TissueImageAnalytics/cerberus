import math
from collections import OrderedDict

import joblib
import numpy as np
import torch.utils.data
from imgaug import augmenters as iaa

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)
from .targets import gen_targets


class PatchSegDataset(torch.utils.data.Dataset):
    """Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        output_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    def __init__(
        self,
        file_list,
        input_shape=None,
        output_shape=None,
        class_input_shape=None,
        run_mode="train",
        setup_augmentor=True,
        req_target_code=None,
    ):
        assert input_shape is not None and output_shape is not None
        self.run_mode = run_mode
        self.info_list = file_list
        self.output_shape = [output_shape, output_shape]
        self.input_shape = [input_shape, input_shape]
        self.id = 0
        self.req_target_code = req_target_code
        if setup_augmentor:
            self.setup_augmentor(0, 0)

        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = joblib.load(path)
        # split stacked channel into image and label
        img = data["img"]
        ann = data["ann"]
        ann_ch_code = data["channel_code"]

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        feed_dict = OrderedDict([["img", img]])

        # TODO: document hard coded assumption about #
        # TODO: expose weight to feed augmentention or GT gen opt
        target_dict, has_flag = gen_targets(
            ann,
            ann_ch_code,
            self.req_target_code,
            self.output_shape,
            gen_unet_weight_map=self.run_mode == "train",
        )
        feed_dict.update(target_dict)
        feed_dict["dummy_target"] = np.array(has_flag)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    mode="reflect",  # padding type at border
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf([
                    iaa.Lambda(seed=rng, func_images=lambda *args: gaussian_blur(*args, max_ksize=3)),
                    iaa.Lambda(seed=rng, func_images=lambda *args: median_blur(*args, max_ksize=3)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
                    ]),
                # apply colour augmentation 90% of time
                iaa.Sometimes(0.90,iaa.Sequential([
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_hue(*args, range=(-8, 8))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_saturation(*args, range=(-0.2, 0.2))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_brightness(*args, range=(-26, 26))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_contrast(*args, range=(0.75, 1.25)))
                    ],random_order=True),
                ),
            ]
        elif mode == "valid":
            # set position to 'center' for center crop
            # else 'uniform' for random crop
            shape_augs = [
                iaa.CropToFixedSize(
                    self.input_shape[0],
                    self.input_shape[1], position="center")
            ]
            input_augs = []

        return shape_augs, input_augs
    

class PatchClassDataset(torch.utils.data.Dataset):
    """Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        output_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    def __init__(
        self,
        file_list,
        input_shape=None,
        output_shape=None,
        class_input_shape=None,
        run_mode="train",
        setup_augmentor=True,
        req_target_code=None,
    ):
        # assert input_shape is not None and output_shape is not None
        self.run_mode = run_mode
        self.info_list = file_list
        self.input_shape = [input_shape, input_shape]
        self.class_input_shape = [class_input_shape, class_input_shape]
        self.id = 0
        self.req_target_code = req_target_code
        if setup_augmentor:
            self.setup_augmentor(0, 0)

        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = joblib.load(path)
        # split stacked channel into image and label
        img = data["img"]
        lab = data["ann"]
        # increase dims to make it work seamlessly with segmentation pipeline
        lab = np.expand_dims(lab, -1)

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        feed_dict = OrderedDict([["img", img], ['lab', lab]])

        feed_dict["dummy_target"] = np.array(['PatchClass']*9)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    mode="reflect",  # padding type at border
                    seed=rng,
                ),
                iaa.Rot90((0,3)),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.class_input_shape[0], self.class_input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf([
                    iaa.Lambda(seed=rng, func_images=lambda *args: gaussian_blur(*args, max_ksize=3)),
                    iaa.Lambda(seed=rng, func_images=lambda *args: median_blur(*args, max_ksize=3)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
                    ]),
                # apply color augmentation 90% of time
                iaa.Sometimes(0.90,iaa.Sequential([
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_hue(*args, range=(-8, 8))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_saturation(*args, range=(-0.2, 0.2))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_brightness(*args, range=(-26, 26))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_contrast(*args, range=(0.75, 1.25)))
                    ],random_order=True),
                ),
            ]

        elif mode == "valid":
            # set position to 'center' for center crop
            # else 'uniform' for random crop
            shape_augs = [
                iaa.CropToFixedSize(self.class_input_shape[0], self.class_input_shape[1], position="center")
            ]
            input_augs = []

        return shape_augs, input_augs


class PatchSegClassDataset(torch.utils.data.Dataset):
    """Loads images from a file list and
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        output_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'

    """

    def __init__(
        self,
        file_list,
        task_mode=None,
        input_shape=None,
        output_shape=None,
        class_input_shape=None,
        run_mode="train",
        setup_augmentor=True,
        req_target_code=None,
    ):
        assert input_shape is not None and output_shape is not None
        self.run_mode = run_mode
        self.task_mode = task_mode
        self.info_list = file_list

        self.output_shape = [output_shape, output_shape]
        if self.task_mode == 'seg':
            self.input_shape = [input_shape, input_shape]
            self.output_shape = [output_shape, output_shape]
        else:
            self.input_shape = [class_input_shape, class_input_shape]
            self.output_shape = [1, 1]
        self.id = 0
        self.req_target_code = req_target_code
        if setup_augmentor:
            self.setup_augmentor(0, 0)

        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, self.task_mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = joblib.load(path)
        # split stacked channel into image and label
        img = data["img"]
        ann = data["ann"]
        ann_ch_code = data["channel_code"]
        
        if self.task_mode == 'class':
            ann = np.reshape(np.array(ann), [1, 1])

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            if self.task_mode == 'seg':
                ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        feed_dict = OrderedDict([["img", img]])

        target_dict, has_flag = gen_targets(
            ann,
            ann_ch_code,
            self.req_target_code,
            self.output_shape,
            self.task_mode,
            gen_unet_weight_map=self.run_mode == "train",
        )
        feed_dict.update(target_dict)
        feed_dict["dummy_target"] = np.array(has_flag)
        
        return feed_dict

    def __get_augmentation(self, mode, task_mode, rng):
        if mode == "train":
            if task_mode == 'seg':
                shape_augs = [
                    # * order = ``0`` -> ``cv2.INTER_NEAREST``
                    # * order = ``1`` -> ``cv2.INTER_LINEAR``
                    # * order = ``2`` -> ``cv2.INTER_CUBIC``
                    # * order = ``3`` -> ``cv2.INTER_CUBIC``
                    # * order = ``4`` -> ``cv2.INTER_CUBIC``
                    iaa.Affine(
                        # scale images to 80-120% of their size, individually per axis
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                        # translate by -A to +A percent (per axis)
                        translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                        shear=(-5, 5),  # shear by -5 to +5 degrees
                        rotate=(-179, 179),  # rotate by -179 to +179 degrees
                        order=0,  # use nearest neighbour
                        backend="cv2",  # opencv for fast processing
                        mode="reflect",  # padding type at border
                        seed=rng,
                    ),
                    # set position to 'center' for center crop
                    # else 'uniform' for random crop
                    iaa.CropToFixedSize(
                        self.input_shape[0], self.input_shape[1], position="center"
                    ),
                    iaa.Fliplr(0.5, seed=rng),
                    iaa.Flipud(0.5, seed=rng),
                ]
            else:
                shape_augs = [
                    # * order = ``0`` -> ``cv2.INTER_NEAREST``
                    # * order = ``1`` -> ``cv2.INTER_LINEAR``
                    # * order = ``2`` -> ``cv2.INTER_CUBIC``
                    # * order = ``3`` -> ``cv2.INTER_CUBIC``
                    # * order = ``4`` -> ``cv2.INTER_CUBIC``
                    iaa.Affine(
                        # scale images to 80-120% of their size, individually per axis
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                        # translate by -A to +A percent (per axis)
                        shear=(-5, 5),  # shear by -5 to +5 degrees
                        order=0,  # use nearest neighbour
                        backend="cv2",  # opencv for fast processing
                        mode="reflect",  # padding type at border
                        seed=rng,
                    ),
                    iaa.Rot90((0,3)),
                    iaa.CropToFixedSize(
                        self.input_shape[0], self.input_shape[1], position="center"
                    ),
                    iaa.Fliplr(0.5, seed=rng),
                    iaa.Flipud(0.5, seed=rng),
                ]

            input_augs = [
                iaa.OneOf([
                    iaa.Lambda(seed=rng, func_images=lambda *args: gaussian_blur(*args, max_ksize=3)),
                    iaa.Lambda(seed=rng, func_images=lambda *args: median_blur(*args, max_ksize=3)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
                    ]),
                # apply colour augmentation 90% of time
                iaa.Sometimes(0.90,iaa.Sequential([
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_hue(*args, range=(-8, 8))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_saturation(*args, range=(-0.2, 0.2))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_brightness(*args, range=(-26, 26))),
                    iaa.Lambda(seed=rng, func_images=lambda *args: add_to_contrast(*args, range=(0.75, 1.25)))
                    ],random_order=True),
                ),
            ]
        else:
            if task_mode == 'seg':
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                shape_augs = [
                    iaa.CropToFixedSize(
                        self.input_shape[0], self.input_shape[1], position="center"
                    ),]
            else:
                shape_augs = [
                    iaa.CropToFixedSize(
                        self.input_shape[0], self.input_shape[1], position="center")
                ]
            input_augs = []

        return shape_augs, input_augs

class MyConcatDataset(torch.utils.data.dataset.ConcatDataset):
    def setup_augmentor(self, worker_id, seed):
        for sub_ds in self.datasets:
            # will modify the reference so dont need to reassign
            sub_ds.setup_augmentor(worker_id, seed)
        return

class SingleTaskBatchSampler(torch.utils.data.sampler.Sampler):
    """Iterate over tasks and provide a random batch per task in each mini-batch."""

    def __init__(self, dataset, batch_size, run_mode):
        self.run_mode = run_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max(
            [len(cur_dataset.info_list) for cur_dataset in dataset.datasets]
        )

    def __len__(self):
        return (
            self.batch_size
            * math.ceil(self.largest_dataset_size / self.batch_size)
            * len(self.dataset.datasets)
        )

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if self.run_mode == "train":
                sampler = torch.utils.data.sampler.RandomSampler(cur_dataset)
            else:
                sampler = torch.utils.data.sampler.SequentialSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class MixedTaskBatchSampler(torch.utils.data.sampler.Sampler):
    """Randomly select a sample from a randomly selected task in each mini-batch."""

    def __init__(self, dataset, batch_size, run_mode):
        self.run_mode = run_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max(
            [len(cur_dataset.info_list) for cur_dataset in dataset.datasets]
        )
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        self.epoch_samples = self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __len__(self):
        return self.epoch_samples
        
    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if self.run_mode == "train":
                sampler = torch.utils.data.sampler.RandomSampler(cur_dataset)
            else:
                sampler = torch.utils.data.sampler.SequentialSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        #epoch_samples = self.largest_dataset_size * self.number_of_datasets
        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, self.epoch_samples, samples_to_grab):
            cur_samples = []
            counter = 0
            for _ in range(samples_to_grab):
                sel_task = counter % self.number_of_datasets
                cur_batch_sampler = sampler_iterators[sel_task]
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[sel_task]
                    cur_samples.append(cur_sample)
                except StopIteration:
                    # got to the end of iterator - restart the iterator and continue to get samples
                    # until reaching "epoch_samples"
                    sampler_iterators[sel_task] = samplers_list[sel_task].__iter__()
                    cur_batch_sampler = sampler_iterators[sel_task]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[sel_task]
                    cur_samples.append(cur_sample)
                counter += 1
            final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

class MixedTaskBatchSampler2(torch.utils.data.sampler.Sampler):
    """Randomly selects Mixed MTL or patch classification and then extracts random samples."""
    def __init__(self, dataset, batch_size,run_mode):
        self.run_mode = run_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max(
            [len(cur_dataset.info_list) for cur_dataset in dataset.datasets][:-1]
        )
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        self.epoch_samples = self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __len__(self):
        return self.epoch_samples

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            if self.run_mode == "train":
                sampler = torch.utils.data.sampler.RandomSampler(cur_dataset)
            else:
                sampler = torch.utils.data.sampler.SequentialSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        samples_to_grab = self.batch_size
        
        final_samples_list = []  # this is a list of indexes from the combined dataset
        task_counter = 0
        for _ in range(0, self.epoch_samples, samples_to_grab):
            cur_samples = []
            counter = 0
            sel_task = task_counter % (self.number_of_datasets)
            for _ in range(samples_to_grab):
                try:
                    if sel_task > 0:
                        sel_mtl_task = counter % (self.number_of_datasets - 1)
                        cur_batch_sampler = sampler_iterators[sel_mtl_task]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[sel_mtl_task]
                    else:
                        # patch classification batch sampler
                        cur_batch_sampler = sampler_iterators[-1]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[-1]
                    cur_samples.append(cur_sample)
                except StopIteration:
                    # got to the end of iterator - restart the iterator and continue to get samples
                    # until reaching "epoch_samples"
                    if sel_task > 0:
                        sel_mtl_task = counter % (self.number_of_datasets - 1)
                        sampler_iterators[sel_mtl_task] = samplers_list[sel_mtl_task].__iter__()
                        cur_batch_sampler = sampler_iterators[sel_mtl_task]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[sel_mtl_task]
                    else:
                        sampler_iterators[-1] = samplers_list[-1].__iter__()
                        cur_batch_sampler = sampler_iterators[-1]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[-1]

                    cur_samples.append(cur_sample)
                counter += 1
            final_samples_list.extend(cur_samples)
            task_counter += 1
        return iter(final_samples_list)
