import os
import json
import numpy as np
import torch


def write_json(results: dict, outputpath: list, for_eval=False):
    """Output json file

    Args:
        results     (dict): keypoint info
        outputpath  (list): output path
        for_eval (bool, optional): Defaults to False.
    """

    pose_results = {}
    bbox_results = {}

    for im_res in results:

        im_name = im_res["imgname"]

        pose_result = []
        bbox_result = {}

        for cow in im_res["result"]:

            region = {}
            keypoints = []

            bb_preds = cow["bboxes"]
            kp_preds = cow["keypoints"]
            kp_scores = cow["kp_score"]
            pro_scores = cow["proposal_score"]

            for n in range(kp_scores.shape[0]):
                keypoints += [
                    float(kp_preds[n, 0]),
                    float(kp_preds[n, 1]),
                    float(kp_scores[n]),
                ]

            region["keypoints"] = keypoints
            region["score"] = float(pro_scores)

            pose_result.append(region)

        pose_results[im_name] = pose_result
        bbox_result["bbox"] = bb_preds.numpy().tolist()
        bbox_results[im_name] = bbox_result

    with open(os.path.join(outputpath, "pose-results.json"), "w") as f:
        f.write(json.dumps(pose_results))
    with open(os.path.join(outputpath, "bbox-results.json"), "w") as f:
        f.write(json.dumps(bbox_results))


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray
