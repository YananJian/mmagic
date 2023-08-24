# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Union

from mmengine import FileClient
from mmengine.dataset import BaseDataset

from mmagic.registry import DATASETS

import json


@DATASETS.register_module()
class DreamBoothDataset(BaseDataset):
    """Dataset for DreamBooth.

    Args:
        data_root (str): Path to the data root.
        concept_dir (str): Path to the concept images.
        prompt (str): Prompt of the concept.
        pipeline (list[dict | callable]): A sequence of data transforms.
    """

    def __init__(self,
                 data_root: str,
                 concept_dir: str,
                 prompt: str,
                 prompt_fname: str = '',
                 pipeline: List[Union[dict, Callable]] = []):

        data_prefix = dict(img_path=concept_dir)

        self.prompt = prompt
     
        self.prompt_fname = prompt_fname

        super().__init__(
            data_root=data_root, data_prefix=data_prefix, pipeline=pipeline)

    def load_data_list(self) -> list:
        """Load data list from concept_dir and class_dir."""
        data_list = []

        img_dir = self.data_prefix['img_path']
        file_client = FileClient.infer_client(uri=img_dir)
        img_dir = osp.abspath(img_dir)

        if self.prompt_fname != '':
            prompt_path = osp.join(osp.dirname(img_dir), self.prompt_fname)
            with open(prompt_path) as fp:
                dataname2caption = json.loads(fp.read())

        for data_name in file_client.list_dir_or_file(img_dir, list_dir=False):
            prompt = self.prompt if self.prompt_fname == '' else dataname2caption.get(data_name, '')
            if prompt == '':
                continue
            data_info = dict(
                img_path=file_client.join_path(img_dir, data_name),
                prompt=prompt)
            data_list.append(data_info)

        return data_list
