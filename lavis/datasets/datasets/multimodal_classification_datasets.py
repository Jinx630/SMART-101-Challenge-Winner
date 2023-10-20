"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
from PIL import Image
from abc import abstractmethod
from lavis.datasets.datasets.base_dataset import BaseDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class MultimodalClassificationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.class_labels = None

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        label = ann["label"]
        instance_id = ann["instance_id"]

        return {"image": image, "text_input": question, "label":label, "instance_id":instance_id}
