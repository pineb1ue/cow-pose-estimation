import numpy as np
import json
import cv2
import os.path as osp


class Params:
    """
    Params for animal evaluation api
    """

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, np.round((1.00 - 0.0) / 0.01) + 1, endpoint=True
        )
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1
        # Nose, L_EarBase, R_EarBase, Withers, TailBase
        self.kpt_oks_sigmas = np.array([0.26, 0.35, 0.35, 0.79, 1.07]) / 10.0

    def __init__(self, iouType="keypoints"):
        if iouType == "keypoints":
            self.setKpParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None


class Coweval:
    def __init__(
        self,
        dts_path: str,
        yoloes_path: str,
        gts_path: str,
        img_path: str,
        p=Params(iouType="keypoints"),
    ):
        self.p = p
        self.img_path = img_path

        with open(dts_path) as f:
            self.dts = json.load(f)
        with open(yoloes_path) as f:
            self.yoloes = json.load(f)
        with open(gts_path) as f:
            self.gts = json.load(f)

    def computeAP(self, oks, n_instances):
        AP_sum = 0.0
        for threshold in self.p.iouThrs:
            AP = len([ok for ok in oks if ok >= threshold]) / n_instances
            if threshold == 0.5:
                mAP50 = AP
            elif threshold == 0.75:
                mAP75 = AP
            AP_sum += AP
        mAP = AP_sum / len(self.p.iouThrs)
        print(f"mAP: {mAP}")
        print(f"mAP@50: {mAP50}")
        print(f"mAP@75: {mAP75}")

    def evaluate(self):
        print(f"dts: {len(self.dts)}, gts: {len(self.gts)}, yoloes: {len(self.yoloes)}")
        sigmas = self.p.kpt_oks_sigmas
        _vars = (sigmas * 2) ** 2
        k = len(sigmas)

        ok = 0.0
        oks = []
        n_instances = 0

        for i, gt in enumerate(self.gts):
            if not gt["Label"] == "Skip":
                n_instances += 1
                filename = gt["External ID"]
                # get box info
                orig_img = cv2.imread(osp.join(self.img_path, filename))
                h, w, _ = orig_img.shape
                bb = np.array([0.5, 0.5, w - 0.5, h - 0.5])
                x0 = 2 * bb[0] - bb[2]
                x1 = 2 * bb[2] - bb[0]
                y0 = 2 * bb[1] - bb[3]
                y1 = 2 * bb[3] - bb[1]
                # get keypoints info
                xg, yg, vg = [], [], []
                for key_name in [
                    "Nose",
                    "L_EarBase",
                    "R_EarBase",
                    "Withers",
                    "TailBase",
                ]:
                    xg.append(gt["Label"][key_name][0]["geometry"]["x"])
                    yg.append(gt["Label"][key_name][0]["geometry"]["y"])
                    if int(gt["Label"][key_name][0]["v"]) > 0:
                        vg.append(1)
                    else:
                        vg.append(0)
                xg = np.array(xg)
                yg = np.array(yg)
                vg = np.array(vg)
                k1 = np.count_nonzero(vg > 0)
                for dt in self.dts:
                    if dt["image_id"] == filename:
                        d = dt["keypoints"]
                        xd, yd = d[0::3], d[1::3]
                        if k1 > 0:
                            # measure the per-keypoint distance if keypoints visible
                            dx = xd - xg
                            dy = yd - yg
                        else:
                            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                            z = np.zeros((k))
                            dx = np.max((z, x0 - xd), axis=0) + np.max(
                                (z, xd - x1), axis=0
                            )
                            dy = np.max((z, y0 - yd), axis=0) + np.max(
                                (z, yd - y1), axis=0
                            )
                        area = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
                        e = (dx ** 2 + dy ** 2) / _vars / (area + np.spacing(1)) / 2
                        if k1 > 0:
                            e = e[vg > 0]
                        ok = np.sum(np.exp(-e)) / e.shape[0]
                        oks.append(ok)
                        if ok > 0.75:
                            print(f"{i}: filename: {filename}, ok: {ok}")

        print(f"n_instances: {n_instances}")

        self.computeAP(oks, n_instances)


if __name__ == "__main__":

    dts_path = "res/alphapose_cow_test.json"
    yoloes_path = "res/yolo_cow_test.json"
    gts_path = "train_sppe/data/anno/cow_20190515_0900_100.json"
    img_path = "train_sppe/data/imgs/cow/test"

    eval = Coweval(
        dts_path=dts_path, yoloes_path=yoloes_path, gts_path=gts_path, img_path=img_path
    )
    eval.evaluate()
