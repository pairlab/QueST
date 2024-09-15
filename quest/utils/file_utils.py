"""
A collection of utility functions for working with files, such as reading metadata from
demonstration datasets, loading model checkpoints, or downloading dataset files.

This file is adopted from robomimic
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""
import os
import h5py
from collections import OrderedDict

import quest.utils.obs_utils as ObsUtils


def get_shape_metadata_from_dataset(dataset_path, all_obs_keys=None, verbose=False):
    """
    Retrieves shape metadata from dataset.

    Args:
        dataset_path (str): path to dataset
        all_obs_keys (list): list of all modalities used by the model. If not provided, all modalities
            present in the file are used.
        verbose (bool): if True, include print statements

    Returns:
        shape_meta (dict): shape metadata. Contains the following keys:

            :`'ac_dim'`: action space dimension
            :`'all_shapes'`: dictionary that maps observation key string to shape
            :`'all_obs_keys'`: list of all observation modalities used
            :`'use_images'`: bool, whether or not image modalities are present
            :`'use_depths'`: bool, whether or not depth modalities are present
    """

    shape_meta = {}

    # read demo file for some metadata
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    demo_id = list(f["data"].keys())[0]
    demo = f["data/{}".format(demo_id)]

    # action dimension
    shape_meta['ac_dim'] = f["data/{}/actions".format(demo_id)].shape[1]

    # observation dimensions
    all_shapes = OrderedDict()

    if all_obs_keys is None:
        # use all modalities present in the file
        all_obs_keys = [k for k in demo["obs"]]

    for k in sorted(all_obs_keys):
        initial_shape = demo["obs/{}".format(k)].shape[1:]
        if verbose:
            print("obs key {} with shape {}".format(k, initial_shape))
        # Store processed shape for each obs key
        all_shapes[k] = ObsUtils.get_processed_shape(
            obs_modality=ObsUtils.OBS_KEYS_TO_MODALITIES[k],
            input_shape=initial_shape,
        )

    f.close()

    shape_meta['all_shapes'] = all_shapes
    shape_meta['all_obs_keys'] = all_obs_keys
    shape_meta['use_images'] = ObsUtils.has_modality("rgb", all_obs_keys)
    shape_meta['use_depths'] = ObsUtils.has_modality("depth", all_obs_keys)

    return shape_meta
