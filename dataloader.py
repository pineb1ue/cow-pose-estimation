import os
import cv2
import numpy as np
import sys
import time
from PIL import Image
import torch
import torch.utils.data as data
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from opt import opt
from pPose_nms import pose_nms
from matching import candidate_reselect as matching
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from yolo.preprocess import prep_image
from yolo.util import dynamic_write_results
from yolo.models import Darknet
from multiprocessing import Queue as pQueue
from threading import Thread

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame


class ImageLoader:
    def __init__(self, im_names, batchSize=1, format="yolo", queueSize=50):
        self.img_dir = opt.inputpath
        self.imglist = im_names
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.format = format

        self.batchSize = batchSize
        self.datalen = len(self.imglist)
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if self.format == "ssd":
            if opt.sp:
                p = Thread(target=self.getitem_ssd, args=())
            else:
                p = mp.Process(target=self.getitem_ssd, args=())
        elif self.format == "yolo":
            if opt.sp:
                p = Thread(target=self.getitem_yolo, args=())
            else:
                p = mp.Process(target=self.getitem_yolo, args=())
        else:
            raise NotImplementedError
        p.daemon = True
        p.start()
        return self

    def getitem_ssd(self):
        length = len(self.imglist)
        for index in range(length):
            im_name = self.imglist[index].rstrip("\n").rstrip("\r")
            im_name = os.path.join(self.img_dir, im_name)
            im = Image.open(im_name)
            inp = load_image(im_name)
            if im.mode == "L":
                im = im.convert("RGB")

            ow = oh = 512
            im = im.resize((ow, oh))
            im = self.transform(im)
            while self.Q.full():
                time.sleep(2)
            self.Q.put((im, inp, im_name))

    def getitem_yolo(self):
        for i in range(self.num_batches):
            img = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(
                i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)
            ):
                inp_dim = int(opt.inp_dim)
                im_name_k = self.imglist[k].rstrip("\n").rstrip("\r")
                im_name_k = os.path.join(self.img_dir, im_name_k)
                img_k, orig_img_k, im_dim_list_k = prep_image(im_name_k, inp_dim)

                img.append(img_k)
                orig_img.append(orig_img_k)
                im_name.append(im_name_k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                img = torch.cat(img)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

            while self.Q.full():
                time.sleep(2)

            self.Q.put((img, orig_img, im_name, im_dim_list))

    def getitem(self):
        return self.Q.get()

    def length(self):
        return len(self.imglist)

    def len(self):
        return self.Q.qsize()


class DetectionLoader:
    def __init__(self, dataloder, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg", (416, 256))
        # Load weights
        weights = "models/yolo/cow_80cls.pt"
        self.det_model.load_state_dict(torch.load(weights)["model"])
        self.det_inp_dim = int(opt.inp_dim)
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.det_model.to(device=device)
        self.det_model.eval()

        self.stopped = False
        self.dataloder = dataloder
        self.batchSize = batchSize
        self.datalen = self.dataloder.length()
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.num_batches):
            img, orig_img, im_name, im_dim_list = self.dataloder.getitem()
            if img is None:
                self.Q.put((None, None, None, None, None, None, None))
                return

            with torch.no_grad():
                # Human Detection
                img = img.cuda()
                prediction, _ = self.det_model(img)
                # NMS process
                dets = dynamic_write_results(
                    prediction,
                    opt.confidence,
                    opt.num_classes,
                    nms=True,
                    nms_conf=opt.nms_thesh,
                )
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_img)):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put(
                            (orig_img[k], im_name[k], None, None, None, None, None)
                        )
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(
                    -1, 1
                )

                # coordinate transfer
                dets[:, [1, 3]] -= (
                    self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)
                ) / 2
                dets[:, [2, 4]] -= (
                    self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)
                ) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(
                        dets[j, [1, 3]], 0.0, im_dim_list[j, 0]
                    )
                    dets[j, [2, 4]] = torch.clamp(
                        dets[j, [2, 4]], 0.0, im_dim_list[j, 1]
                    )
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:, 0] == k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
                if self.Q.full():
                    time.sleep(2)
                self.Q.put(
                    (
                        orig_img[k],
                        im_name[k],
                        boxes_k,
                        scores[dets[:, 0] == k],
                        inps,
                        pt1,
                        pt2,
                    )
                )

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor:
    def __init__(self, detectionLoader, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.datalen = self.detectionLoader.datalen

        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = pQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):

            with torch.no_grad():
                (
                    orig_img,
                    im_name,
                    boxes,
                    scores,
                    inps,
                    pt1,
                    pt2,
                ) = self.detectionLoader.read()
                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None, None))
                    return
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, im_name, boxes, scores, None, None))
                    continue
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DataWriter:
    def __init__(
        self,
        save_video=False,
        savepath="examples/res/1.avi",
        fourcc=cv2.VideoWriter_fourcc(*"XVID"),
        fps=25,
        frameSize=(640, 480),
        queueSize=1024,
    ):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), "Cannot open video for writing"
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + "/vis"):
                os.mkdir(opt.outputpath + "/vis")

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                if boxes is None:
                    if opt.save_img or opt.save_video or opt.vis:
                        img = orig_img
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(
                                os.path.join(opt.outputpath, "vis", im_name), img
                            )
                        if opt.save_video:
                            self.stream.write(img)
                else:
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    if opt.matching:
                        preds = getMultiPeakPrediction(
                            hm_data,
                            pt1.numpy(),
                            pt2.numpy(),
                            opt.inputResH,
                            opt.inputResW,
                            opt.outputResH,
                            opt.outputResW,
                        )
                        result = matching(boxes, scores.numpy(), preds)
                    else:
                        preds_hm, preds_img, preds_scores = getPrediction(
                            hm_data,
                            pt1,
                            pt2,
                            opt.inputResH,
                            opt.inputResW,
                            opt.outputResH,
                            opt.outputResW,
                        )
                        result = pose_nms(boxes, scores, preds_img, preds_scores)
                    result = {"imgname": im_name, "result": result}
                    self.final_result.append(result)
                    if opt.save_img or opt.save_video or opt.vis:
                        img = vis_frame(orig_img, result)
                        if opt.vis:
                            cv2.imshow("AlphaPose Demo", img)
                            cv2.waitKey(30)
                        if opt.save_img:
                            cv2.imwrite(
                                os.path.join(opt.outputpath, "vis", im_name), img
                            )
                        if opt.save_video:
                            self.stream.write(img)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()


class CowDataset(data.Dataset):
    def __init__(
        self,
        train=True,
        sigma=1,
        scale_factor=(0.2, 0.3),
        rot_factor=40,
        label_type="Gaussian",
    ):
        self.img_dir = "data/imgs/cow/instances"
        self.anno_path = "data/anno/cow_keypoints.json"
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
        self.flipRef = (2, 3)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


def crop_from_dets(img, boxes, inps, pt1, pt2):
    """
    Crop human from origin image according to Dectecion Results
    """

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor((float(box[0]), float(box[1])))
        bottomRight = torch.Tensor((float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5
        )
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5
        )

        try:
            inps[i] = cropBox(
                tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW
            )
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print("===")
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
