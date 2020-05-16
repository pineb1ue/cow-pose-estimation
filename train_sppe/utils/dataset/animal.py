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


class Anno_json2list(object):
    def __init__(self, anno_path: str):
        self.anno_path = anno_path
        with open(anno_path) as f:
            self.data = json.load(f)

    def animal(self, train_txt: str, val_txt: str):
        """Convert json to list

        Args:
            train_txt (str): the name of the train

        Returns:
        """

        with open(train_txt) as f:
            _imgname_train = [s.strip() for s in f.readlines()]
        with open(val_txt) as f:
            _imgname_val = [s.strip() for s in f.readlines()]

        imgname_train, imgname_val = [], []
        bndbox_train, bndbox_val = [], []
        part_train, part_val = [], []

        # train
        for name in _imgname_train:
            for r in data[name]["regions"]:
                imgname_train.append(data[name]["filename"])
                bndbox_train.append(r["box_attributes"])
                _keypoints = r["pose_attributes"]["keypoints"]
                # only x, y (except v)
                _key = [k for i, k in enumerate(_keypoints) if (i + 1) % 3 != 0]
                part_train.append(_key)

        imgname_train = np.array(imgname_train)
        bndbox_train = np.array(bndbox_train).reshape(-1, 1, 4)
        part_train = np.array(part_train).reshape(-1, 5, 2)

        # val
        for name in _imgname_val:
            for r in data[name]["regions"]:
                imgname_val.append(data[name]["filename"])
                bndbox_val.append(r["box_attributes"])
                _keypoints = r["pose_attributes"]["keypoints"]
                # only x, y (except v)
                _key = [k for i, k in enumerate(_keypoints) if (i + 1) % 3 != 0]
                part_val.append(_key)

        imgname_val = np.array(imgname_val)
        bndbox_val = np.array(bndbox_val).reshape(-1, 1, 4)
        part_val = np.array(part_val).reshape(-1, 5, 2)

        return (
            imgname_train,
            imgname_val,
            bndbox_train,
            bndbox_val,
            part_train,
            part_val,
        )


class Anno_json2list_cow(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

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
                im = cv2.imread(os.path.join(self.img_dir, d["External ID"]))
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


class AnimalDataset(data.Dataset):
    def __init__(
        self,
        data_type="cow",  # animal or cow
        train=True,
        sigma=1,
        scale_factor=(0.2, 0.3),
        rot_factor=40,
        label_type="Gaussian",
    ):
        self.data_type = data_type  # animal or cow
        self.img_dir = opt.imgdir
        self.anno_path = opt.annopath
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

        if self.data_type == "animal":
            transform_anno = Anno_json2list_animal()
            (
                self.imgname_train,
                self.imgname_val,
                self.bndbox_train,
                self.bndbox_val,
                self.part_train,
                self.part_val,
            ) = transform_anno(self.anno_path)

        elif self.data_type == "cow":
            transform_anno = Anno_json2list_cow(img_dir=self.img_dir)
            anno, idx_train, idx_val = transform_anno(self.img_dir, self.anno_path)

            # train
            self.imgname_train = anno["imgname"][idx_train]
            self.bndbox_train = anno["bndbox"][idx_train]
            self.part_train = anno["part"][idx_train]

            if not os.path.isfile("data/cow/train.txt"):
                with open("data/imgs/cow/train.txt", mode="w") as f:
                    for idx_tr in idx_train:
                        f.write(anno["imgname"][idx_tr] + "\n")

            # val
            self.imgname_val = anno["imgname"][idx_val]
            self.bndbox_val = anno["bndbox"][idx_val]
            self.part_val = anno["part"][idx_val]

            if not os.path.isfile("data/cow/val.txt"):
                with open("data/imgs/cow/val.txt", mode="w") as f:
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
            part = self.part_train[index]
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
