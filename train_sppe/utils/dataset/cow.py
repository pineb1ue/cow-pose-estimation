# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import numpy as np
import json
import cv2
from sklearn.model_selection import train_test_split
import torch.utils.data as data

from ..pose import generateSampleBox
from opt import opt


class Anno_json2list_cow(object):
    def __init__(self):
        pass

    def __call__(self, img_dir: str, anno_path: str):
        """Convert json to list

        Args:
            img_dir (str): cow instance image directory
            anno_path (str): annotation path

        Returns:
            dict: annoatation info (bounding boxes, image names, keypoints)
            list: train index
            list: val index
        """

        # load annotaion info
        with open(anno_path) as f:
            data = json.load(f)

        bndboxes, imgnames, parts = [], [], []

        for d in data:
            if not d["Label"] == "Skip":
                # get keypoints
                for label_name in [
                    "Nose",
                    "L_EarBase",
                    "R_EarBase",
                    "Withers",
                    "TailBase",
                ]:
                    keys = d["Label"][label_name]
                    v = int(keys[0]["v"])
                    # v = 0: not labeled (in which case x = y = 0)
                    if v == 0:
                        x = y = 0
                    else:
                        x = keys[0]["geometry"]["x"]
                        y = keys[0]["geometry"]["y"]
                    parts.extend([float(x), float(y)])
                imgnames.append(d["External ID"])
                # create box info
                im = cv2.imread(os.path.join("data/cow/instances", d["External ID"]))
                h, w, _ = im.shape
                bndboxes.append([0.5, 0.5, float(w) - 0.5, float(h) - 0.5])

        n_samples = len(imgnames)
        bndboxes = np.array(bndboxes).reshape(n_samples, 1, 4)
        imgnames = np.array(imgnames)
        parts = np.array(parts).reshape(n_samples, 5, 2)

        anno = {"bndbox": bndboxes, "imgname": imgnames, "part": parts}

        idx = [i for i in range(n_samples)]
        idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=0)

        return anno, idx_train, idx_val


class CowDataset(data.Dataset):
    def __init__(
        self,
        transform_anno=Anno_json2list_cow(),
        train=True,
        sigma=1,
        scale_factor=(0.2, 0.3),
        rot_factor=40,
        label_type="Gaussian",
    ):
        self.transform_anno = transform_anno  # json to list (annoatation info)
        self.img_dir = opt.imgdir
        # self.img_dir = "data/imgs/cow/instances"
        self.anno_path = opt.annopath
        # self.anno_path = "data/anno/cow_keypoints.json"
        self.is_train = train  # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_animal = 5
        self.nJoints = 5
        self.accIdxs = (1, 2, 3, 4, 5)
        self.flipRef = ((2, 3),)

        anno, idx_train, idx_val = self.transform_anno(self.img_dir, self.anno_path)

        # train
        self.imgname_train = anno["imgname"][idx_train]
        self.bndbox_train = anno["bndbox"][idx_train]
        self.pose_train = anno["part"][idx_train]

        if not os.path.isfile("data/cow/train.txt"):
            with open("data/cow/train.txt", mode="w") as f:
                for idx_tr in idx_train:
                    f.write(anno["imgname"][idx_tr] + "\n")

        # val
        self.imgname_val = anno["imgname"][idx_val]
        self.bndbox_val = anno["bndbox"][idx_val]
        self.part_val = anno["part"][idx_val]

        if not os.path.isfile("data/cow/val.txt"):
            with open("data/cow/val.txt", mode="w") as f:
                for idx_v in idx_val:
                    f.write(anno["imgname"][idx_v] + "\n")

        self.size_train = self.imgname_train.shape[0]
        self.size_val = self.imgname_val.shape[0]
        print(f"train_size: {self.size_train}, val_size: {self.size_val}")

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.pose_train[index]
            bndbox = self.bndbox_train[index]
            imgname = self.imgname_train[index]
        else:
            part = self.part_val[index]
            bndbox = self.bndbox_val[index]
            imgname = self.imgname_val[index]

        img_path = os.path.join(self.img_dir, imgname)

        metaData = generateSampleBox(
            img_path,
            bndbox,
            part,
            self.nJoints,
            "animal",
            sf,
            self,
            train=self.is_train,
        )

        inp, out, setMask = metaData

        return inp, out, setMask, "animal"
