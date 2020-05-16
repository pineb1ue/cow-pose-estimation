import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import torch

from opt import opt
from dataloader import DataWriter, CowDataset
from dataloader import crop_from_dets
from SPPE.src.main_fast_inference import InferenNet_fast, InferenNet
from fn import getTime
from utils import write_json, im_to_torch


class CowPoseEst:
    """Pose estimation
    """

    def __init__(
        self,
        annopath: str,
        outputpath: str,
        inputlist: list,
        fast_inference: bool,
        profile: bool,
        batch: int,
        inputResH: int,
        inputResW: int,
    ):
        self.annopath = annopath
        self.outputpath = outputpath
        self.inputlist = inputlist
        self.fast_inference = fast_inference
        self.profile = profile
        self.batchSize = batch
        self.inputResH = inputResH
        self.inputResW = inputResW

        if not os.path.exists(self.outputpath):
            os.mkdir(self.outputpath)

        print("Loading gt boxes..")
        with open(self.annopath) as f:
            self.gts = json.load(f)

        self.im_names = self.gts.keys()

    def run(self):

        pose_dataset = CowDataset()

        if opt.fast_inference:
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)

        pose_model.cuda()
        pose_model.eval()

        runtime_profile = {"dt": [], "pt": [], "pn": []}

        # Init data writer
        writer = DataWriter(opt.save_video).start()

        data_len = len(self.im_names)
        im_names_desc = tqdm(range(data_len))

        for i, im_name in zip(im_names_desc, self.im_names):
            start_time = getTime()
            with torch.no_grad():
                orig_img = cv2.imread(im_name)

                boxes = []

                for info in self.gts[im_name]["info"]:
                    box = info["bbox"]
                    boxes.append(box)

                boxes = torch.tensor(boxes)
                scores = torch.ones(boxes.size(0), 1)
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps = torch.zeros(boxes.size(0), 3, self.inputResH, self.inputResW)
                pt1 = torch.zeros(boxes.size(0), 2)
                pt2 = torch.zeros(boxes.size(0), 2)
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue

                ckpt_time, det_time = getTime(start_time)
                runtime_profile["dt"].append(det_time)

                # Pose Estimation
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % self.batchSize:
                    leftover = 1
                num_batches = datalen // self.batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[
                        j * self.batchSize : min((j + 1) * self.batchSize, datalen)
                    ].cuda()
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile["pt"].append(pose_time)
                hm = hm.cpu()
                writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name)

                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile["pn"].append(post_time)

            if self.profile:
                # TQDM
                im_names_desc.set_description(
                    "det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}".format(
                        dt=np.mean(runtime_profile["dt"]),
                        pt=np.mean(runtime_profile["pt"]),
                        pn=np.mean(runtime_profile["pn"]),
                    )
                )

        # Save json file
        final_result = writer.results()
        write_json(final_result, self.outputpath)


if __name__ == "__main__":
    opt.dataset = "animal"
    if not opt.sp:
        torch.multiprocessing.set_start_method("forkserver", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")

    pose_est = CowPoseEst(
        annopath=opt.annopath,
        outputpath=opt.outputpath,
        inputlist=opt.inputlist,
        fast_inference=opt.fast_inference,
        profile=opt.profile,
        batch=opt.posebatch,
        inputResH=opt.inputResH,
        inputResW=opt.inputResW,
    )
    pose_est.run()
