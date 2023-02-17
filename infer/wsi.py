import multiprocessing as mp
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed
from multiprocessing import Lock

mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging

import os
import pathlib
import sys
import time
import logging
import uuid
import joblib
from collections import OrderedDict
from typing import List, Tuple, Union

import cv2
import scipy.io as sio
import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.utils.data as torch_data
import tqdm
import yaml
from scipy.ndimage import measurements
from datetime import datetime

from loader.postproc import (
    PostProcInstErodedContourMap,
    PostProcInstErodedMap,
    get_inst_info_dict,
)
from misc.utils import get_bounding_box, rm_n_mkdir

from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.models import (
    IOSegmentorConfig,
    NucleusInstanceSegmentor,
    WSIStreamDataset,
)
from tiatoolbox.models.architecture.hovernet import HoVerNet
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader

from . import base

# ! Change this to match with the targets.py gen code
# target key gen code : post proc class
_postproc_func_dict = {
    "IP-ERODED-3": PostProcInstErodedMap,
    "IP-ERODED-11": PostProcInstErodedMap,
    "IP-ERODED-CONTOUR-3": PostProcInstErodedContourMap,
    "IP-ERODED-CONTOUR-11": PostProcInstErodedContourMap,
}

thread_lock = Lock()

def _init_worker_child(lock_):
    global lock
    lock = lock_

def get_postproc_info(decoder_info):
    # get the postproc functions used for each task

    postproc_dict = {}
    for decoder_name, _ in decoder_info.items():
        task_name, task_type = decoder_name.split("-")
        if task_type == "INST":
            postproc_key = decoder_info[decoder_name]
            proc_func = _postproc_func_dict[postproc_key]
            postproc_dict[task_name] = proc_func

    return postproc_dict


# Python is yet to be able to natively pickle Object method/static method.
# Only top-level function is passable to multi-processing as caller.
# May need 3rd party libraries to use method/static method otherwise.
def _process_tile_predictions(
    ioconfig,
    tile_bounds,
    tile_flag,
    tile_mode,
    ref_inst_dict,
    cache_inst_path,
    cache_type_path,
    postproc,
):
    """Function to merge new tile prediction with existing prediction.

    Args:
        ioconfig (:class:`IOSegmentorConfig`): Object defines information
            about input and output placement of patches.
        tile_bounds (:class:`numpy.array`): Boundary of the current tile, defined as
            (top_left_x, top_left_y, bottom_x, bottom_y).
        tile_flag (list): A list of flag to indicate if instances within
            an area extended from each side (by `ioconfig.margin`) of
            the tile should be replaced by those within the same spatial
            region in the accumulated output this run. The format is
            [top, bottom, left, right], 1 indicates removal while 0 is not.
            For example, [1, 1, 0, 0] denotes replacing top and bottom instances
            within `ref_inst_dict` with new ones after this processing.
        tile_mode (int): A flag to indicate the type of this tile. There
            are 4 flags:
            - 0: A tile from tile grid without any overlapping, it is not
                an overlapping tile from tile generation. The predicted
                instances are immediately added to accumulated output.
            - 1: Vertical tile strip that stands between two normal tiles
                (flag 0). It has the the same height as normal tile but
                less width (hence vertical strip).
            - 2: Horizontal tile strip that stands between two normal tiles
                (flag 0). It has the the same width as normal tile but
                less height (hence vertical strip).
            - 3: tile strip stands at the cross section of four normal tiles
                (flag 0).
        cache_inst_path (str): Path to location storing cached raw predictions.
        cache_type_path (str): Path to location storing cached raw predictions.
        ref_inst_dict (dict): Dictionary contains accumulated output. The
            expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        postproc (callable): Function to post-process the raw assembled tile.
        merge_predictions (callable): Function to merge the `tile_output` into
            raw tile prediction.

    Returns:
        new_inst_dict (dict): A dictionary contain new instances to be accumulated.
            The expected format is {instance_id: {type: int,
            contour: List[List[int]], centroid:List[float], box:List[int]}.
        remove_insts_in_orig (list): List of instance id within `ref_inst_dict`
            to be removed to prevent overlapping predictions. These instances
            are those get cutoff at the boundary due to the tiling process.

    """

    # convert from WSI space to tile space, locations in XY
    tile_tl = tile_bounds[:2]
    tile_br = tile_bounds[2:]

    tile_shape = tile_br - tile_tl  # in width height

    inst_ptr = np.load(cache_inst_path, mmap_mode="r")
    type_ptr = np.load(cache_type_path, mmap_mode="r")
    inst_ptr = inst_ptr[tile_tl[1] : tile_br[1], tile_tl[0] : tile_br[0]]
    type_ptr = type_ptr[tile_tl[1] : tile_br[1], tile_tl[0] : tile_br[0]]
    idx_dict = {"Nuclei-INST": [0, 2], "Nuclei-TYPE": [2, 4]}  # channel starting idx
    raw_map = np.concatenate([np.array(inst_ptr), np.array(type_ptr)], axis=-1)
    inst_map, type_map = postproc.post_process(raw_map, idx_dict, "Nuclei")
    inst_dict = HoVerNet.get_instance_info(inst_map, type_map)
    # should be rare, no nuclei detected in input images
    if len(inst_dict) == 0:
        return {}, []

    m = ioconfig.margin
    w, h = tile_shape
    inst_boxes = [v["box"] for v in inst_dict.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    # An auxiliary dictionary to actually query the index within the source list
    index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
    tile_rtree = STRtree(geometries)

    # create margin bounding box, ordering should match with
    # created tile info flag (top, bottom, left, right)
    boundary_lines = [
        shapely_box(0, 0, w, 1),  # noqa top edge
        shapely_box(0, h - 1, w, h),  # noqa bottom edge
        shapely_box(0, 0, 1, h),  # noqa left
        shapely_box(w - 1, 0, w, h),  # noqa right
    ]
    margin_boxes = [
        shapely_box(0, 0, w, m),  # noqa top edge
        shapely_box(0, h - m, w, h),  # noqa bottom edge
        shapely_box(0, 0, m, h),  # noqa left
        shapely_box(w - m, 0, w, h),  # noqa right
    ]
    # ! this is wrt to WSI coordinate space, not tile
    margin_lines = [
        [[m, m], [w - m, m]],  # noqa top edge
        [[m, h - m], [w - m, h - m]],  # noqa bottom edge
        [[m, m], [m, h - m]],  # noqa left
        [[w - m, m], [w - m, h - m]],  # noqa right
    ]
    margin_lines = np.array(margin_lines) + tile_tl[None, None]
    margin_lines = [shapely_box(*v.flatten().tolist()) for v in margin_lines]

    # the ids within this match with those within `inst_map`, not UUID
    sel_indices = []
    if tile_mode in [0, 3]:
        # for `full grid` tiles `cross section` tiles
        # -- extend from the boundary by the margin size, remove
        #    nuclei whose entire contours lie within the margin area
        sel_boxes = [
            box
            for idx, box in enumerate(margin_boxes)
            if tile_flag[idx] or tile_mode == 3
        ]

        sel_indices = [
            index_by_id[id(geo)]
            for bounds in sel_boxes
            for geo in tile_rtree.query(bounds)
            if bounds.contains(geo)
        ]
    elif tile_mode in [1, 2]:
        # for `horizontal/vertical strip` tiles
        # -- extend from the marked edges (top/bot or left/right) by
        #    the margin size, remove all nuclei lie within the margin
        #    area (including on the margin line)
        # -- remove all nuclei on the boundary also

        sel_boxes = [
            margin_boxes[idx] if flag else boundary_lines[idx]
            for idx, flag in enumerate(tile_flag)
        ]

        sel_indices = [
            index_by_id[id(geo)]
            for bounds in sel_boxes
            for geo in tile_rtree.query(bounds)
        ]
    else:
        raise ValueError(f"Unknown tile mode {tile_mode}.")

    def retrieve_sel_uids(sel_indices, inst_dict):
        """Helper function to retrieve selected instance uids."""
        sel_uids = []
        if len(sel_indices) > 0:
            # not sure how costly this is in large dict
            inst_uids = list(inst_dict.keys())
            sel_uids = [inst_uids[idx] for idx in sel_indices]
        return sel_uids

    remove_insts_in_tile = retrieve_sel_uids(sel_indices, inst_dict)

    # external removal only for tile at cross sections
    # this one should contain UUID with the reference database
    remove_insts_in_orig = []
    if tile_mode == 3:
        inst_boxes = [v["box"] for v in ref_inst_dict.values()]
        inst_boxes = np.array(inst_boxes)

        geometries = [shapely_box(*bounds) for bounds in inst_boxes]
        # an auxiliary dictionary to actually query the index within the source list
        index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
        ref_inst_rtree = STRtree(geometries)
        sel_indices = [
            index_by_id[id(geo)]
            for bounds in margin_lines
            for geo in ref_inst_rtree.query(bounds)
        ]

        remove_insts_in_orig = retrieve_sel_uids(sel_indices, ref_inst_dict)

    # move inst position from tile space back to WSI space
    # an also generate universal uid as replacement for storage
    new_inst_dict = {}
    for inst_uid, inst_info in inst_dict.items():
        if inst_uid not in remove_insts_in_tile:
            inst_info["box"] += np.concatenate([tile_tl] * 2)
            inst_info["centroid"] += tile_tl
            inst_info["contour"] += tile_tl
            inst_uuid = uuid.uuid4().hex
            new_inst_dict[inst_uuid] = inst_info

    return new_inst_dict, remove_insts_in_orig


class InferManager(base.InferManager):
    @staticmethod
    def get_coordinates(
        image_shape: Union[List[int], np.ndarray], ioconfig: IOSegmentorConfig
    ):
        return NucleusInstanceSegmentor.get_coordinates(image_shape, ioconfig)

    @staticmethod
    def filter_coordinates(
        mask_reader: VirtualWSIReader,
        bounds: np.ndarray,
        resolution: Union[float, int] = None,
        units: str = None,
    ):
        return NucleusInstanceSegmentor.filter_coordinates(
            mask_reader, bounds, resolution, units
        )

    @staticmethod
    def _get_tile_info(
        image_shape: Union[List[int], np.ndarray], ioconfig: IOSegmentorConfig,
    ):
        """Generating tile information.

        To avoid out of memory problem when processing WSI-scale in general,
        the predictor will perform the inference and assemble on a large
        image tiles (each may have size of 4000x4000 compared to patch
        output of 256x256) first before stitching every tiles by the end
        to complete the WSI output. For nuclei instance segmentation,
        the stiching process will require removal of predictions within
        some bounding areas. This function generates both the tile placement
        as well as the flag to indicate how the removal should be done to
        achieve the above goal.

        Args:
            image_shape (:class:`numpy.ndarray`, list(int)): The shape of WSI
                to extract the tile from, assumed to be in [width, height].
            ioconfig (:obj:IOSegmentorConfig): The input and output
                configuration objects.
        Returns:
            grid_tiles, removal_flags: ndarray
            vertical_strip_tiles, removal_flags: ndarray
            horizontal_strip_tiles, removal_flags: ndarray
            cross_section_tiles, removal_flags: ndarray

        """
        return NucleusInstanceSegmentor._get_tile_info(image_shape, ioconfig)

    def _to_shared_space(self, wsi_idx, patch_inputs, patch_outputs):
        """Helper functions to transfer variable to shared space.

        We modify the shared space so that we can update worker info without
        needing to re-create the worker. There should be no race-condition
        because only by looping `self._loader` in main thread will trigger querying
        new data from each worker, and this portion should still be in sequential
        execution order in the main thread.

        Args:
            wsi_idx (int): The index of the WSI to be processed. This is used
                to retrieve the file path.
            patch_inputs (list): A list of corrdinates in
                [start_x, start_y, end_x, end_y] format indicating the read location
                of the patch in the WSI image. The coordinates are in the highest
                resolution defined in `self.ioconfig`.
            patch_outputs (list): A list of corrdinates in
                [start_x, start_y, end_x, end_y] format indicating the write location
                of the patch in the WSI image. The coordinates are in the highest
                resolution defined in `self.ioconfig`.

        """
        patch_inputs = torch.from_numpy(patch_inputs).share_memory_()
        patch_outputs = torch.from_numpy(patch_outputs).share_memory_()
        self._mp_shared_space.patch_inputs = patch_inputs
        self._mp_shared_space.patch_outputs = patch_outputs
        self._mp_shared_space.wsi_idx = torch.Tensor([wsi_idx]).share_memory_()

    def _infer_once(self):
        """Running the inference only once for the currently active dataloder."""
        num_steps = len(self._loader)

        pbar_desc = "Process Batch: "
        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(num_steps),
            ncols=80,
            ascii=True,
            position=0,
        )

        cum_output = []
        for _, batch_data in enumerate(self._loader):
            sample_datas, sample_infos = batch_data
            batch_size = sample_infos.shape[0]
            # ! depending on the protocol of the output within infer_batch
            # ! this may change, how to enforce/document/expose this in a
            # ! sensible way?

            sample_outputs = self.run_step(sample_datas, self.patch_output_shape)

            # tensor to numpy, costly?
            sample_infos = sample_infos.numpy()
            sample_infos = np.split(sample_infos, batch_size, axis=0)

            sample_outputs = list(zip(sample_infos, sample_outputs))
            cum_output.extend(sample_outputs)
            pbar.update()
        pbar.close()
        return cum_output

    def _get_tissue_info(self):
        tissue_info = []
        mask_list = np.unique(self.wsi_mask_lab).tolist()
        if len(mask_list) > 1:
            for tissue_region_id in mask_list[1:]:
                tissue_region = self.wsi_mask_lab == tissue_region_id
                rmin, rmax, cmin, cmax = get_bounding_box(tissue_region)
                tissue_info.append([rmin, rmax, cmin, cmax])
        else:
            tissue_info.append([0, self.wsi_mask_lab.shape[0], 0, self.wsi_mask_lab.shape[1]])
        return tissue_info

    def _get_resized_map(self, mmap_path, ds_factor, read_size):
        """Resize an image tile-by-tile and then merge together. Convenient for large
        arrays that otherwise may result in memory errors when resizing the entire array."""

        read_size = [read_size, read_size]

        # ! the coordinates are in XY not YX
        (tile_coords, _) = PatchExtractor.get_coordinates(
            image_shape=self.wsi_proc_shape,
            patch_input_shape=read_size,
            patch_output_shape=read_size,
            stride_shape=read_size,
        )

        _
        read_size = (np.array(read_size)).astype(np.int64)
        new_shape = [int(round(v * ds_factor)) for v in self.wsi_proc_shape]

        # create empty array to populate
        resized_pred_map = np.zeros(new_shape)
        for tile_coord in tile_coords:
            tile_pred_map = wsi_map_ptr[
                tile_coord[1] : tile_coord[3], tile_coord[0] : tile_coord[2],
            ]
            if 0 not in list(tile_pred_map.shape):
                tile_pred_map = np.array(tile_pred_map)  # from mmap to ram
                resize_tile_pred_map = cv2.resize(
                    tile_pred_map,
                    (0, 0),
                    fx=ds_factor,
                    fy=ds_factor,
                    interpolation=cv2.INTER_NEAREST,
                )
                new_coord = (tile_coord * ds_factor).astype(np.int32)

                resized_tmp = resized_pred_map[
                    new_coord[1] : new_coord[3], new_coord[0] : new_coord[2],
                ]
                # make sure nothing breaks!
                if resized_tmp.shape != resize_tile_pred_map.shape:
                    resize_tile_pred_map = cv2.resize(
                        resize_tile_pred_map,
                        (resized_tmp.shape[1], resized_tmp.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                resized_pred_map[
                    new_coord[1] : new_coord[3], new_coord[0] : new_coord[2],
                ] = resize_tile_pred_map

        return resized_pred_map

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        self.chunk_shape = [self.chunk_shape, self.chunk_shape]
        self.tile_shape = [self.tile_shape, self.tile_shape]
        self.patch_input_shape = [self.patch_input_shape, self.patch_input_shape]
        self.patch_output_shape = [self.patch_output_shape, self.patch_output_shape]
        return

    @staticmethod
    def merge_prediction(
        canvas_shape: Union[Tuple[int], List[int], np.ndarray],
        predictions: List[np.ndarray],
        locations: Union[List, np.ndarray],
        save_path: Union[str, pathlib.Path] = None,
        cache_count_path: Union[str, pathlib.Path] = None,
    ):
        return NucleusInstanceSegmentor.merge_prediction(
            canvas_shape,
            predictions,
            locations,
            save_path,
            cache_count_path,
        )

    def _merge_inst_results(self, inst_dict, futures, has_workers=False):
        """Helper to aggregate results from parallel workers."""

        def callback(new_inst_dict, remove_uuid_list):
            """Helper to aggregate worker's results."""
            # ! DEPRECATION:
            # ! will be deprecated upon finalization of SQL annotation store
            inst_dict.update(new_inst_dict)
            for inst_uuid in remove_uuid_list:
                inst_dict.pop(inst_uuid, None)

        for future in futures:
            #  not actually future but the results
            if not has_workers:
                callback(*future)
                continue

            # some errors happen, log it and propagate exception
            # ! this will lead to discard a whole bunch of
            # ! inferred tiles within this current WSI
            if future.exception() is not None:
                raise future.exception()

            # aggregate the result via callback
            result = future.result()
            # manually call the callback rather than
            # attaching it when receiving/creating the future
            callback(*result)

        return inst_dict

    def process_single_file(
        self, ioconfig, ioconfig_pp, wsi_idx, wsi_basename, output_dir
    ):
        """Process a single whole-slide image and save the results.

        Args:
            wsi_path: path to input whole-slide image
            msk_path: path to input mask. If not supplied, mask will be automatically generated.
            output_dir: path where output will be saved

        """
        rm_n_mkdir(self.cache_path) # create caching location

        wsi_path = self.imgs[wsi_idx]
        mask_path = self.masks[wsi_idx]
        start = time.perf_counter()
        # this is in XY
        # assume ioconfig has already been
        # converted to `baseline` for `tile` mode
        resolution = ioconfig.highest_input_resolution
        self.wsi_handler = WSIReader.open(input_img=wsi_path)

        # in XY
        self.wsi_proc_shape = self.wsi_handler.slide_dimensions(**resolution)
        # to YX
        self.wsi_proc_shape = self.wsi_proc_shape[::-1]
        
        self.wsi_base_mag = self.wsi_handler.info.mpp # get scan resolution of WSI
        self.wsi_base_shape = self.wsi_handler.slide_dimensions(self.wsi_base_mag, "mpp")
        self.wsi_base_shape = self.wsi_base_shape[::-1] # to YX

        if mask_path is not None and os.path.isfile(mask_path):
            wsi_mask = cv2.imread(mask_path)
            wsi_mask = cv2.cvtColor(wsi_mask, cv2.COLOR_BGR2GRAY)
            wsi_mask[wsi_mask > 0] = 1
        else:
            wsi_mask = np.ones(self.wsi_proc_shape, dtype=np.uint8)
        mask_downsample_ratio = wsi_mask.shape[0] / self.wsi_proc_shape[0]
        
        if self.save_mask:
            cv2.imwrite(f"{self.output_dir}/mask/{wsi_basename}.png", self.wsi_mask*255)
        if self.save_thumb:
            # thumbnail at 1.25x objective magnification
            wsi_thumb = self.wsi_handler.slide_thumbnail(resolution=1.25, units="power") 
            cv2.imwrite(f"{self.output_dir}/thumb/{wsi_basename}.png", wsi_thumb)
            
        # warning, the value within this is uninitialized
        # cache merging map for each  head output
        fx_list = [v["resolution"] for v in ioconfig.output_resolutions]
        cache_raw_paths = [
            f"{self.cache_path}/raw.{idx}.npy" for idx, _ in enumerate(fx_list)
        ]
        cache_count_paths = [
            f"{self.cache_path}/count.{idx}.npy" for idx, _ in enumerate(fx_list)
        ]

        # * retrieve patch placement
        mask_reader = VirtualWSIReader(wsi_mask, mode="bool")

        # ! input shape is assumed to be in width x height
        (patch_inputs, patch_outputs) = self.get_coordinates(
            self.wsi_proc_shape[::-1], ioconfig
        )
        if mask_reader is not None:
            mask_reader.info = self.wsi_handler.info
            sel = self.filter_coordinates(mask_reader, patch_outputs, **resolution)
            patch_outputs = patch_outputs[sel]
            patch_inputs = patch_inputs[sel]

        # assumed to be [top_left_x, top_left_y, bot_right_x, bot_right_y]
        geometries = [shapely_box(*bounds) for bounds in patch_outputs]
        # An auxiliary dictionary to actually query the index within the source list
        index_by_id = {id(geo): idx for idx, geo in enumerate(geometries)}
        spatial_indexer = STRtree(geometries)

        # * retrieve tile placement and tile info flag
        # tile shape will always be corrected to be multiple of output
        tile_info_sets = self._get_tile_info(self.wsi_proc_shape[::-1], ioconfig)
        # get the raw prediction, given info of inference tiles and patches

        end = time.perf_counter()
        self.logger.info("Preparing Input Output Placement: {0}".format(end - start))

        # ! inference part, commented out for testing, no parallel assembling
        # * raw prediction
        start = time.perf_counter()
        set_bounds, set_flags = tile_info_sets[0]
        for tile_idx, tile_bounds in enumerate(set_bounds):
            tile_flag = set_flags[tile_idx]

            # select any patches that have their output
            # within the current tile
            sel_box = shapely_box(*tile_bounds)
            sel_indices = [
                index_by_id[id(geo)] for geo in spatial_indexer.query(sel_box)
            ]
            if len(sel_indices) == 0:
                continue

            tile_patch_inputs = patch_inputs[sel_indices]
            tile_patch_outputs = patch_outputs[sel_indices]

            self._to_shared_space(wsi_idx, tile_patch_inputs, tile_patch_outputs)

            tile_infer_output = self._infer_once()
            if len(tile_infer_output) == 0:
                continue
            (patch_locations, patch_predictions) = list(zip(*tile_infer_output))
            patch_predictions = [list(v.values()) for v in patch_predictions]
            patch_locations = [np.squeeze(v) for v in patch_locations]
            # merging for each head
            for idx, _ in enumerate(fx_list):
                head_predictions = [v[idx] for v in patch_predictions]
                self.merge_prediction(
                    self.wsi_proc_shape,
                    head_predictions,
                    patch_locations,
                    save_path=cache_raw_paths[idx],
                    cache_count_path=cache_count_paths[idx],
                )

        end = time.perf_counter()
        self.logger.info("Inference Time: {0}".format(end - start))

        head_names = [
            "Nuclei-INST",
            "Nuclei-TYPE",
            "Gland-INST",
            "Gland-TYPE",
            "Lumen-INST",
            "Patch-Class",
        ]
        head_caches = list(zip(head_names, cache_raw_paths))
        head_caches = OrderedDict(head_caches)

        # # * ==== Post processing Nuclei
        wsi_inst_info = {}
        with open("dataset.yml") as fptr:
            dataset_info = yaml.full_load(fptr)

        start = time.perf_counter()
        tile_info_sets = self._get_tile_info(self.wsi_proc_shape[::-1], ioconfig_pp)
        nuclei_inst_info = {}
        for set_idx, (set_bounds, set_flags) in enumerate(tile_info_sets):
            futures = []
            for tile_idx, tile_bounds in enumerate(set_bounds):
                # select any patches that have their output
                # within the current tile
                sel_box = shapely_box(*tile_bounds)
                sel_indices = [
                    index_by_id[id(geo)] for geo in spatial_indexer.query(sel_box)
                ]

                if len(sel_indices) == 0:
                    continue

                tile_flag = set_flags[tile_idx]
                args = [
                    ioconfig_pp,
                    tile_bounds,
                    tile_flag,
                    set_idx,
                    nuclei_inst_info,
                    head_caches["Nuclei-INST"],
                    head_caches["Nuclei-TYPE"],
                    self.postproc_info["Nuclei"],
                ]
                if self._postproc_workers is not None:
                    future = self._postproc_workers.submit(
                        _process_tile_predictions, *args
                    )
                else:
                    future = _process_tile_predictions(*args)
                futures.append(future)
            nuclei_inst_info = self._merge_inst_results(
                nuclei_inst_info,
                futures,
                has_workers=self._postproc_workers is not None,
            )

        wsi_inst_info["Nuclei"] = nuclei_inst_info
        end = time.perf_counter()
        self.logger.info("Nuclei Post Proc Time: {0}".format(end - start))

        # * ==== Post processing Tissue Region Classification

        start = time.perf_counter()
        ds_factor = 0.25
        if "Patch-Class" in self.model_args["decoder_kwargs"].keys():
            # resize the map tile by tile rather than loading entirely in memory
            map_idx = head_names.index("Patch-Class")
            mmap_path = "%s/raw.%d.npy" % (self.cache_path, map_idx)
            pclass_map_ptr = np.load(mmap_path, mmap_mode="r")
            pclass_map = np.array(pclass_map_ptr)  # load to ram
            pclass_map = cv2.resize(
                pclass_map,
                (0, 0),
                fx=ds_factor,
                fy=ds_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            # pclass_map = self._get_resized_map(mmap_path, ds_factor, 5000)

            # multiply by tissue masks
            lores_wsimask = cv2.resize(
                wsi_mask,
                (pclass_map.shape[1], pclass_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            pclass_map *= lores_wsimask
            del lores_wsimask # free memory
            sio.savemat(
                "%s/tissue/%s.mat" % (output_dir, wsi_basename),
                {"pclass": pclass_map}
                )

        end = time.perf_counter()
        self.logger.info("Tissue Region Post Proc Time: {0}".format(end - start))

        # * ==== Post processing Gland and Lumen
        start = time.perf_counter()
        # get labelled mask
        self.wsi_mask_lab = measurements.label(wsi_mask)[0]
        tissue_info_list = self._get_tissue_info()

        target_list = ["gland", "lumen"]
        gland_inst_info = {}
        lumen_inst_info = {}
        for idx, tissue_info in enumerate(tissue_info_list):
            sys.stdout.write(
                "\rPost Proc Gland & Lumen (%d/%d)" % (idx, len(tissue_info_list))
            )
            sys.stdout.flush()
            rmin = int(round(tissue_info[0] / mask_downsample_ratio))
            rmax = int(round(tissue_info[1] / mask_downsample_ratio))
            cmin = int(round(tissue_info[2] / mask_downsample_ratio))
            cmax = int(round(tissue_info[3] / mask_downsample_ratio))
            tissue_topleft = [cmin, rmin]

            # get the mask segment that is currently being considered
            mask_idx = self.wsi_mask_lab[
                tissue_info[0] : tissue_info[1], tissue_info[2] : tissue_info[3]
            ]
            mask_idx = mask_idx == idx + 1

            pred_inst_map_dict = {}
            pred_type_map_dict = {}
            new_idx_dict = {}
            for tissue_code in target_list:
                tissue_code = tissue_code.capitalize()
                tile_pred_list = []
                ch_count = 0
                for output_type in ["INST", "TYPE"]:
                    if tissue_code + "-%s" % output_type in head_names:
                        ch_idx = head_names.index(tissue_code + "-%s" % output_type)
                        tile_pred_map_ptr = np.load(
                            "%s/raw.%d.npy" % (self.cache_path, ch_idx), mmap_mode="r"
                        )
                        tile_pred_map = tile_pred_map_ptr[rmin:rmax, cmin:cmax]
                        tile_pred_map = np.array(tile_pred_map)  # from mmap to ram
                        # only consider probability map in the considered tissue segment
                        if tile_pred_map.shape[0] != mask_idx.shape[0] and tile_pred_map.shape[1] != mask_idx.shape[1]:
                            # resize tissue mask 
                            mask_idx = cv2.resize(
                                mask_idx.astype("uint8"),
                                (tile_pred_map.shape[1], tile_pred_map.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        if mask_idx.ndim == 2:
                            mask_idx = np.expand_dims(mask_idx, -1)
         
                        tile_pred_map *= mask_idx
                        tile_pred_list.append(tile_pred_map)
                        new_idx_dict[tissue_code + "-%s" % output_type] = [
                            ch_count,
                            ch_count + tile_pred_map.shape[-1],
                        ]
                        ch_count += tile_pred_map.shape[-1]
                tile_pred_map = np.concatenate(tile_pred_list, axis=-1)  # combine maps
                # free up memory
                del tile_pred_list

                # resize to make post processing quicker
                ds_factor = 0.5
                tile_pred_map = cv2.resize(
                    tile_pred_map, (0, 0), fx=ds_factor, fy=ds_factor
                )

                inst_info_code = self.decoder_dict[tissue_code + "-INST"]
                proc_func = _postproc_func_dict[inst_info_code]
                pred_inst_map, pred_type_map = proc_func.post_process(
                    tile_pred_map, new_idx_dict, tissue_code, ds_factor
                )
                pred_inst_map_dict[tissue_code] = pred_inst_map
                pred_type_map_dict[tissue_code] = pred_type_map

            # remove lumen predictions not inside glands!
            if "lumen" in target_list and "gland" in target_list:
                binary_gland = pred_inst_map_dict["Gland"].copy()
                binary_gland[binary_gland > 0] = 1
                pred_lumen = binary_gland * pred_inst_map_dict["Lumen"]
                # replace with updated dictionary
                pred_inst_map_dict["Lumen"] = pred_lumen

            for tissue_code in target_list:
                tissue_code = tissue_code.capitalize()
                pred_inst_info = get_inst_info_dict(
                    pred_inst_map_dict[tissue_code],
                    pred_type_map_dict[tissue_code],
                    ds_factor,
                )

                for inst_id, inst_info in pred_inst_info.items():
                    # now correct the coordinate wrt to wsi
                    inst_info["box"] += tissue_topleft
                    inst_info["contour"] += tissue_topleft
                    inst_info["centroid"] += tissue_topleft

                    # flatten
                    inst_bbox = inst_info["box"]
                    inst_info["box"] = np.array(
                        [
                            inst_bbox[0][1],
                            inst_bbox[0][0],
                            inst_bbox[1][1],
                            inst_bbox[1][0],
                        ]
                    )

                    inst_uid = uuid.uuid4().hex
                    if tissue_code == "Gland":
                        gland_inst_info[inst_uid] = inst_info
                        wsi_inst_info[tissue_code] = gland_inst_info
                    else:
                        lumen_inst_info[inst_uid] = inst_info
                        wsi_inst_info[tissue_code] = lumen_inst_info

        sys.stdout.write(
            "\rPost Proc Gland & Lumen (%d/%d)" % (idx + 1, len(tissue_info_list))
        )
        print()
        
        output_path = "%s/dat/%s" % (output_dir, wsi_basename)

        # add resolution information
        wsi_inst_info["proc_resolution"] = {"resolution": self.wsi_proc_mag, "units": "mpp"}
        wsi_inst_info["base_resolution"] = {"resolution": self.wsi_base_mag, "units": "mpp"}
        wsi_inst_info["proc_dimensions"] = self.wsi_proc_shape # YX
        wsi_inst_info["base_dimensions"] = self.wsi_base_shape # YX
        
        # save dictionary as dat file
        joblib.dump(wsi_inst_info, f"{output_path}.dat")

        end = time.perf_counter()
        self.logger.info("Gland & Lumen Post Proc Time: {0}".format(end - start))


    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py

        """
        self._parse_args(run_args)

        self.postproc_info = get_postproc_info(self.decoder_dict)

        if not os.path.exists(self.cache_path):
            rm_n_mkdir(self.cache_path)

        if not os.path.exists(self.output_dir + "/dat/"):
            rm_n_mkdir(self.output_dir + "/dat/")
        if not os.path.exists(self.output_dir + "/tissue/"):
            rm_n_mkdir(self.output_dir + "/tissue/")
        if self.save_thumb:
            if not os.path.exists(self.output_dir + "/thumb/"):
                rm_n_mkdir(self.output_dir + "/thumb/")
        if self.save_mask:
            if not os.path.exists(self.output_dir + "/mask/"):
                rm_n_mkdir(self.output_dir + "/mask/")
        

        self.num_loader_workers = 12
        self.num_postproc_workers = 6

        ioconfig = IOSegmentorConfig(
            input_resolutions=[{"units": "mpp", "resolution": self.wsi_proc_mag}],
            output_resolutions=[
                {"units": "mpp", "resolution": self.wsi_proc_mag},
                {"units": "mpp", "resolution": self.wsi_proc_mag},
                {"units": "mpp", "resolution": self.wsi_proc_mag},
                {"units": "mpp", "resolution": self.wsi_proc_mag},
                {"units": "mpp", "resolution": self.wsi_proc_mag},
                {"units": "mpp", "resolution": self.wsi_proc_mag},
            ],
            margin=64,
            tile_shape=[15000, 15000],
            patch_input_shape=[448, 448],
            patch_output_shape=[144, 144],
            stride_shape=[144, 144],
            save_resolution={"units": "mpp", "resolution": self.wsi_proc_mag},
        )

        ioconfig_pp = IOSegmentorConfig(
            input_resolutions=[{"units": "mpp", "resolution": self.wsi_proc_mag}],
            output_resolutions=[{"units": "mpp", "resolution": self.wsi_proc_mag},],
            margin=64,
            tile_shape=[4096, 4096],
            patch_input_shape=[448, 448],
            patch_output_shape=[144, 144],
            stride_shape=[144, 144],
            save_resolution={"units": "mpp", "resolution": self.wsi_proc_mag},
        )
    
        # workers should be > 0 else Value Error will be thrown
        self._postproc_workers = None
        self.num_postproc_workers = (
            None
            if (self.num_postproc_workers < 1 or self.num_postproc_workers is None)
            else self.num_postproc_workers
        )
        if self.num_postproc_workers is not None:
            self._postproc_workers = ProcessPoolExecutor(
                max_workers=self.num_postproc_workers
            )

        mp_manager = torch_mp.Manager()
        mp_shared_space = mp_manager.Namespace()
        self._mp_shared_space = mp_shared_space

        self.imgs = self.input_list
        self.masks = self.mask_list

        ds = WSIStreamDataset(
            ioconfig=ioconfig,
            preproc=None,
            wsi_paths=self.imgs,
            mp_shared_space=mp_shared_space,
            mode="wsi"
        )
        loader = torch_data.DataLoader(
            ds,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.num_loader_workers,
            persistent_workers=self.num_loader_workers > 0,
        )
        self._loader = loader

        # iterate over list of wsi filepaths
        for wsi_idx, wsi_path in enumerate(self.imgs):
            wsi_basename = pathlib.Path(wsi_path).stem
            start = time.perf_counter()

            #* LOGGING
            dt_now = datetime.now()
            dt_string = dt_now.strftime("%d-%m-%Y_%H:%M:%S")
            wsi_logging_path = f"{self.logging_dir}/{wsi_basename}_{dt_string}_std.log"

            self.logger = logging.getLogger()
            fhandler = logging.FileHandler(filename=wsi_logging_path, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)
            self.logger.setLevel(logging.DEBUG)

            if not os.path.exists(self.output_dir + "/dat/%s.dat" % wsi_basename):
                self.logger.info(f"Processing {wsi_basename} ...")
                self.process_single_file(
                    ioconfig, ioconfig_pp, wsi_idx, wsi_basename, self.output_dir
                )
                end = time.perf_counter()
                self.logger.info("Overall Time: {0}".format(end - start))
                self.logger.info("Finish")
            else:
                self.logger.warning(f"Skip {wsi_basename}- already processed!")

            self.logger.handlers.clear() # close logger
            
        rm_n_mkdir(self.cache_path)  # clean up all cache

        # clean up memory
        self._loader = None
        return
