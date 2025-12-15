#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import random
import pandas
import torch
import torch.utils.data
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import (
    MaskingGenerator,
    MaskingGenerator3D,
    create_random_augment,
)

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
            "test_openset",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        self.dummy_output = None

        # dual-modal 开关：rgb_ir 模式
        self.dual_modal = self.cfg.DATA.MODALITY == "rgb_ir"
        if self.dual_modal:
            self._path_to_videos_ir = []
            self._video_meta_ir = {}

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test", "test_openset"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        if self.dual_modal and self.cfg.DATA.VERIFY_DUAL_MODAL_PAIRS:
            self._verify_dual_modal_pairs()

        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0
        self.cur_epoch = 0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            split_file = self.cfg.TRAIN_FILE
            split_file_ir = self.cfg.TRAIN_FILE_IR
        elif self.mode == "val":
            split_file = self.cfg.VAL_FILE
            split_file_ir = self.cfg.VAL_FILE_IR
        elif self.mode == "test":
            split_file = self.cfg.TEST_FILE
            split_file_ir = self.cfg.TEST_FILE_IR
        else:
            raise RuntimeError("Unknown split mode {}".format(self.mode))

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}".format(split_file)
        )

        # IR 注释文件逻辑：如果显式给了 IR data dir 或者 IR split 文件不同，则尝试加载
        split_file_ir = split_file_ir if split_file_ir else split_file
        ir_annotations_enabled = self.dual_modal and (
            self.cfg.DATA.PATH_TO_DATA_DIR_IR or split_file_ir != split_file
        )
        path_to_file_ir = (
            os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR_IR
                if self.cfg.DATA.PATH_TO_DATA_DIR_IR
                else self.cfg.DATA.PATH_TO_DATA_DIR,
                "{}".format(split_file_ir),
            )
            if ir_annotations_enabled
            else None
        )

        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        if path_to_file_ir:
            assert pathmgr.exists(path_to_file_ir), "{} dir not found".format(
                path_to_file_ir
            )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS

        with pathmgr.open(path_to_file, "r") as f:
            if self.use_chunk_loading:
                rows = self._get_chunk(f, self.cfg.DATA.LOADER_CHUNK_SIZE)
            else:
                rows = f.read().splitlines()

            rows_ir = None
            if path_to_file_ir:
                with pathmgr.open(path_to_file_ir, "r") as f_ir:
                    rows_ir = f_ir.read().splitlines()
                if len(rows_ir) != len(rows):
                    raise RuntimeError(
                        "RGB annotations ({} entries) and IR annotations ({} entries)"
                        " must describe the same number of clips when"
                        " DATA.PATH_TO_DATA_DIR_IR is set.".format(
                            len(rows), len(rows_ir)
                        )
                    )

            # 如果有表头，跳过第一行
            if len(rows) > 0 and "label" in rows[0]:
                rows = rows[1:]
                if rows_ir is not None and len(rows_ir) > 0:
                    rows_ir = rows_ir[1:]

            # 根据文件夹顺序自动匹配 IR（当 csv 中没写 IR 路径时）
            ir_order = None
            if self.dual_modal and self.cfg.DATA.MATCH_DUAL_MODAL_BY_ORDER:
                ir_order = utils.list_videos_by_order(self.cfg.DATA.PATH_PREFIX_IR)
                if len(ir_order) < len(rows):
                    raise RuntimeError(
                        "Expected at least {} IR clips under {} but found {}."
                        " Ensure the infrared folder is complete or disable"
                        " DATA.MATCH_DUAL_MODAL_BY_ORDER.".format(
                            len(rows), self.cfg.DATA.PATH_PREFIX_IR, len(ir_order)
                        )
                    )

            for clip_idx, path_label in enumerate(rows):
                fetch_info = [
                    item.strip()
                    for item in path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                    if item != ""
                ]

                ir_pair = None
                if rows_ir is not None:
                    ir_fetch = [
                        item.strip()
                        for item in rows_ir[clip_idx].split(
                            self.cfg.DATA.PATH_LABEL_SEPARATOR
                        )
                        if item != ""
                    ]
                    if len(ir_fetch) >= 1:
                        ir_path_candidate = ir_fetch[0]
                        ir_label_candidate = (
                            ir_fetch[-1] if len(ir_fetch) > 1 else None
                        )
                        ir_pair = (ir_path_candidate, ir_label_candidate)

                if len(fetch_info) == 2:
                    path, label = fetch_info
                    path_ir = None
                elif len(fetch_info) == 3:
                    if self.dual_modal:
                        path, path_ir, label = fetch_info
                    else:
                        path, fn, label = fetch_info
                        path_ir = None
                elif len(fetch_info) == 1:
                    path, label = fetch_info[0], 0
                    path_ir = None
                else:
                    raise RuntimeError(
                        "Failed to parse video fetch {} info {} retries.".format(
                            path_to_file, fetch_info
                        )
                    )

                # 如果 csv 有 IR 注释信息，则优先使用；并检查 label 一致性
                if self.dual_modal and path_ir is None and ir_pair is not None:
                    path_ir, ir_label = ir_pair
                    if ir_label not in [None, "", str(label)]:
                        raise RuntimeError(
                            "Mismatched labels between RGB ({}) and IR ({}) annotations"
                            " for clip {}".format(label, ir_label, clip_idx)
                        )

                # 如果还没 IR 路径，同时开启了按顺序匹配，则从 IR 目录按顺序配对
                if (
                    self.dual_modal
                    and path_ir is None
                    and self.cfg.DATA.MATCH_DUAL_MODAL_BY_ORDER
                ):
                    path_ir = ir_order[clip_idx]

                # ==== 关键修改 1：清理路径中的引号 ====
                path = str(path).strip().strip('"').strip("'")
                if path_ir is not None:
                    path_ir = str(path_ir).strip().strip('"').strip("'")

                for idx in range(self._num_clips):
                    # ==== 关键修改 2：绝对路径不再拼接 PATH_PREFIX ====
                    if os.path.isabs(path):
                        rgb_path = path
                    else:
                        rgb_path = os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    self._path_to_videos.append(rgb_path)

                    if self.dual_modal:
                        if path_ir is None:
                            raise RuntimeError(
                                "Missing infrared path while DATA.MODALITY is rgb_ir"
                            )
                        if os.path.isabs(path_ir):
                            ir_path = path_ir
                        else:
                            ir_path = os.path.join(
                                self.cfg.DATA.PATH_PREFIX_IR, path_ir
                            )
                        self._path_to_videos_ir.append(ir_path)

                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                    if self.dual_modal:
                        self._video_meta_ir[clip_idx * self._num_clips + idx] = {}

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {} skip_rows {}) from {} ".format(
                len(self._path_to_videos), self.skip_rows, path_to_file
            )
        )

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            for chunk in pandas.read_csv(
                path_to_file,
                chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                skiprows=self.skip_rows,
            ):
                break
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor or list): the frames of sampled from the video.
            label (int or list): the label(s) of the current video.
            index (int or list): original index(es).
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index
        if self.dummy_output is not None:
            return self.dummy_output
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test", "test_openset"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max-scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            if self.mode in ["train"]
            else 1
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE
                or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.DATA.TRAIN_CROP_SIZE]
                * (num_decode - len(crop_size))
            )
            assert self.mode in ["train", "val"]

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            video_container_ir = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
                if self.mode not in ["test", "test_openset"]:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue  # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test", "test_openset"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.dual_modal:
                try:
                    video_container_ir = container.get_video_container(
                        self._path_to_videos_ir[index],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load infrared video from {} with error {}".format(
                            self._path_to_videos_ir[index], e
                        )
                    )
                    if self.mode not in ["test", "test_openset"]:
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if video_container_ir is None:
                    logger.warning(
                        "Failed to meta load infrared video idx {} from {}; trial {}".format(
                            index, self._path_to_videos_ir[index], i_try
                        )
                    )
                    if self.mode not in ["test", "test_openset"] and i_try > self._num_retries // 8:
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )
            frames_ir_decoded = [None] * num_decode if self.dual_modal else None

            # RGB 帧数
            rgb_num_frames = [self.cfg.DATA.NUM_FRAMES]
            sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )
            sampling_rate = [sampling_rate]
            if len(rgb_num_frames) < num_decode:
                rgb_num_frames.extend(
                    [
                        rgb_num_frames[-1]
                        for _ in range(num_decode - len(rgb_num_frames))
                    ]
                )
                # base case where keys have same frame-rate as query
                sampling_rate.extend(
                    [
                        sampling_rate[-1]
                        for _ in range(num_decode - len(sampling_rate))
                    ]
                )
            elif len(rgb_num_frames) > num_decode:
                rgb_num_frames = rgb_num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]

            # IR 帧数（可单独配置）
            if self.dual_modal:
                ir_num_frames_value = (
                    self.cfg.DATA.NUM_FRAMES_IR
                    if self.cfg.DATA.NUM_FRAMES_IR > 0
                    else self.cfg.DATA.NUM_FRAMES
                )
                ir_num_frames = [ir_num_frames_value]
                if len(ir_num_frames) < num_decode:
                    ir_num_frames.extend(
                        [
                            ir_num_frames[-1]
                            for _ in range(num_decode - len(ir_num_frames))
                        ]
                    )
                elif len(ir_num_frames) > num_decode:
                    ir_num_frames = ir_num_frames[:num_decode]
            else:
                ir_num_frames = None

            if self.mode in ["train"]:
                assert (
                    len(min_scale)
                    == len(max_scale)
                    == len(crop_size)
                    == num_decode
                )

            target_fps = self.cfg.DATA.TARGET_FPS
            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(
                    0.0, self.cfg.DATA.TRAIN_JITTER_FPS
                )

            # Decode RGB video.
            frames, time_idx, tdiff = decoder.decode(
                video_container,
                sampling_rate,
                rgb_num_frames,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index]
                if len(self._video_meta) < 5e6
                else {},  # do not cache on huge datasets
                target_fps=target_fps,
                backend=self.cfg.DATA.DECODING_BACKEND,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=min_scale[0]
                if all(x == min_scale[0] for x in min_scale)
                else 0,  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt
                if self.mode in ["train"]
                else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )
            frames_decoded = frames
            time_idx_decoded = time_idx

            # Decode IR video.
            if self.dual_modal:
                frames_ir, time_idx_ir, _ = decoder.decode(
                    video_container_ir,
                    sampling_rate,
                    ir_num_frames,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta_ir[index]
                    if len(self._video_meta_ir) < 5e6
                    else {},
                    target_fps=target_fps,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                    max_spatial_scale=min_scale[0]
                    if all(x == min_scale[0] for x in min_scale)
                    else 0,
                    time_diff_prob=self.p_convert_dt
                    if self.mode in ["train"]
                    else 0.0,
                    temporally_rnd_clips=True,
                    min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                    max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
                )
                frames_ir_decoded = frames_ir

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_decoded is None or None in frames_decoded:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if (
                    self.mode not in ["test", "test_openset"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.dual_modal and (frames_ir_decoded is None or None in frames_ir_decoded):
                logger.warning(
                    "Failed to decode infrared video idx {} from {}; trial {}".format(
                        index, self._path_to_videos_ir[index], i_try
                    )
                )
                if (
                    self.mode not in ["test", "test_openset"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    modal_frames = OrderedDict()
                    modal_frames["rgb"] = frames_decoded[i].clone()
                    if self.dual_modal:
                        modal_frames["ir"] = frames_ir_decoded[i].clone()
                    time_idx_out[idx] = time_idx_decoded[i, :]

                    # 归一化到 [0,1]
                    for key in modal_frames:
                        modal_frames[key] = modal_frames[key].float()
                        modal_frames[key] = modal_frames[key] / 255.0

                    # 颜色抖动：RGB 和 IR 共享同一组随机参数
                    if (
                        self.mode in ["train"]
                        and self.cfg.DATA.SSL_COLOR_JITTER
                    ):
                        def _color_jitter(frames):
                            return transform.color_jitter_video_ssl(
                                frames,
                                bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                                hue=self.cfg.DATA.SSL_COLOR_HUE,
                                p_convert_gray=self.p_convert_gray,
                                moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                                gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                                gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                            )

                        modal_frames = self._apply_shared_random(
                            modal_frames, _color_jitter
                        )

                    # RandAugment：共享随机参数
                    if self.aug and self.cfg.AUG.AA_TYPE:
                        aug_transform = create_random_augment(
                            input_size=(
                                next(iter(modal_frames.values())).size(1),
                                next(iter(modal_frames.values())).size(2),
                            ),
                            auto_augment=self.cfg.AUG.AA_TYPE,
                            interpolation=self.cfg.AUG.INTERPOLATION,
                        )

                        def _auto_augment(frames):
                            # T H W C -> T C H W.
                            frames = frames.permute(0, 3, 1, 2)
                            list_img = self._frame_to_list_img(frames)
                            list_img = aug_transform(list_img)
                            frames = self._list_img_to_frames(list_img)
                            # T C H W -> T H W C
                            return frames.permute(0, 2, 3, 1)

                        modal_frames = self._apply_shared_random(
                            modal_frames, _auto_augment
                        )

                    # 颜色归一化 + 维度变换：T H W C -> C T H W
                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None
                        if (self.mode not in ["train"] or len(scl) == 0)
                        else scl
                    )
                    relative_aspect = (
                        None
                        if (self.mode not in ["train"] or len(asp) == 0)
                        else asp
                    )

                    for key in modal_frames:
                        if key != "rgb" and self.dual_modal:
                            mean = (
                                self.cfg.DATA.MEAN_IR
                                if len(self.cfg.DATA.MEAN_IR)
                                else self.cfg.DATA.MEAN
                            )
                            std = (
                                self.cfg.DATA.STD_IR
                                if len(self.cfg.DATA.STD_IR)
                                else self.cfg.DATA.STD
                            )
                        else:
                            mean = self.cfg.DATA.MEAN
                            std = self.cfg.DATA.STD

                        modal_frames[key] = utils.tensor_normalize(
                            modal_frames[key], mean, std
                        )
                        modal_frames[key] = modal_frames[key].permute(3, 0, 1, 2)

                    # 空间采样：共享随机参数
                    def _spatial(frames):
                        return utils.spatial_sampling(
                            frames,
                            spatial_idx=spatial_sample_index,
                            min_scale=min_scale[i],
                            max_scale=max_scale[i],
                            crop_size=crop_size[i],
                            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                            aspect_ratio=relative_aspect,
                            scale=relative_scales,
                            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                            if self.mode in ["train"]
                            else False,
                        )

                    modal_frames = self._apply_shared_random(
                        modal_frames, _spatial
                    )

                    # 对齐 RGB/IR clip 长度
                    modal_frames = self._match_clip_lengths(modal_frames)

                    # 随机擦除：共享随机参数
                    if self.rand_erase:
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )

                        def _erase(frames):
                            return erase_transform(
                                frames.permute(1, 0, 2, 3)
                            ).permute(1, 0, 2, 3)

                        modal_frames = self._apply_shared_random(
                            modal_frames, _erase
                        )

                    # 打包输出：单模态 / 双模态
                    if self.dual_modal:
                        rgb_frames = modal_frames["rgb"]
                        ir_frames = modal_frames["ir"]
                        f_out[idx] = utils.pack_pathway_output(
                            self.cfg, (rgb_frames, ir_frames)
                        )
                    else:
                        f_out[idx] = utils.pack_pathway_output(
                            self.cfg, next(iter(modal_frames.values()))
                        )

                    if self.cfg.AUG.GEN_MASK_LOADER:
                        mask = self._gen_mask()
                        f_out[idx] = f_out[idx] + [torch.Tensor(), mask]

            sample_index = index
            extra_info = {}
            if self.cfg.DATA.RETURN_VIDEO_PATHS:
                extra_info["rgb_path"] = self._path_to_videos[sample_index]
                if self.dual_modal:
                    extra_info["ir_path"] = self._path_to_videos_ir[sample_index]

            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)
            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode
            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (frames, label, index, time_idx, {})
            return frames, label, index, time_idx, extra_info if extra_info else {}
        else:
            logger.warning("!!!!!!!!!!!!!!!")
            logger.warning(self._path_to_videos[index])
            logger.warning(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=np.int)
            n_mask = round(
                self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask

    def _apply_shared_random(self, modal_frames, func):
        """
        Apply a stochastic transform to every modality using identical random
        parameters. The RNG state after applying the transform matches the
        single-modality case so subsequent operations remain reproducible.
        """

        if len(modal_frames) <= 1:
            for key in modal_frames:
                modal_frames[key] = func(modal_frames[key])
            return modal_frames

        keys = list(modal_frames.keys())
        torch_state_before = torch.random.get_rng_state()
        np_state_before = np.random.get_state()
        random_state_before = random.getstate()

        # 先在第一个模态上跑一遍，生成/消耗随机数
        modal_frames[keys[0]] = func(modal_frames[keys[0]])

        torch_state_after = torch.random.get_rng_state()
        np_state_after = np.random.get_state()
        random_state_after = random.getstate()

        # 其余模态恢复到同一 RNG 起点，再执行 transform
        for key in keys[1:]:
            torch.random.set_rng_state(torch_state_before)
            np.random.set_state(np_state_before)
            random.setstate(random_state_before)
            modal_frames[key] = func(modal_frames[key])

        # 最后恢复 RNG 到“单模态情况下”的状态
        torch.random.set_rng_state(torch_state_after)
        np.random.set_state(np_state_after)
        random.setstate(random_state_after)
        return modal_frames

    def _match_clip_lengths(self, modal_frames):
        """Ensure all modalities share the same temporal dimension."""
        if len(modal_frames) <= 1:
            return modal_frames

        # 统一到第一个模态的 T
        target_len = next(iter(modal_frames.values())).shape[1]
        for key, frames in modal_frames.items():
            if frames.shape[1] == target_len:
                continue
            resized = F.interpolate(
                frames.unsqueeze(0),
                size=(target_len, frames.shape[2], frames.shape[3]),
                mode="trilinear",
                align_corners=False,
            )
            modal_frames[key] = resized.squeeze(0)
        return modal_frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def _verify_dual_modal_pairs(self):
        """Validate that RGB/IR samples stay paired throughout iteration."""

        if len(self._path_to_videos) != len(self._path_to_videos_ir):
            raise RuntimeError(
                f"RGB and infrared clip counts differ ({len(self._path_to_videos)} vs "
                f"{len(self._path_to_videos_ir)}). Ensure each row in {self.mode} lists"
                " both modalities."
            )

        first_missing = None
        for idx, (rgb_path, ir_path) in enumerate(
            zip(self._path_to_videos, self._path_to_videos_ir)
        ):
            if rgb_path is None or ir_path is None:
                first_missing = (idx, rgb_path, ir_path)
                break
            if not pathmgr.exists(rgb_path):
                first_missing = (idx, rgb_path, ir_path)
                break
            if not pathmgr.exists(ir_path):
                first_missing = (idx, rgb_path, ir_path)
                break

        if first_missing is not None:
            idx, rgb_path, ir_path = first_missing
            raise RuntimeError(
                "Dual-modal sample {} is incomplete (rgb={}, ir={}). Check the"
                " CSV annotations for split {}.".format(idx, rgb_path, ir_path, self.mode)
            )

        logger.info(
            "Verified {} paired RGB/IR clips for {} split.".format(
                len(self._path_to_videos), self.mode
            )
        )
