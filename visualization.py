import argparse
import cv2
import json
import os


class PoseVis:
    def __init__(self, pose_path: str, bbox_path: str):
        self.pose_path = pose_path
        self.bbox_path = bbox_path

        with open(self.pose_path) as f:
            self.pose_data = json.load(f)
        with open(self.bbox_path) as f:
            self.bbox_data = json.load(f)

        self.threshold = 0.2
        self.num_keypoints = 5
        self.points_color = [
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (128, 0, 0),
            (0, 255, 255),
        ]
        self.joints = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
        self.joints_color = [
            (128, 0, 128),
            (128, 128, 0),
            (0, 128, 128),
            (64, 0, 128),
            (0, 255, 128),
            (64, 128, 128),
        ]

    def __call__(self):

        for img_path, annos in self.pose_data.items():

            img_name = img_path.split("/")[-1]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # draw keypoints
            for anno in annos:
                key = anno["keypoints"]
                x, y, scores = key[::3], key[1::3], key[2::3]

                is_high_score = [0 for i in range(self.num_keypoints)]
                for i, (x_, y_) in enumerate(zip(x, y)):
                    points = (int(x_), int(y_))
                    if scores[i] > self.threshold:
                        is_high_score[i] += 1
                        img = cv2.circle(
                            img,
                            points,
                            3,
                            self.points_color[i],
                            thickness=5,
                            lineType=cv2.LINE_AA,
                        )

                for i, joint in enumerate(self.joints):
                    p, q = joint
                    if is_high_score[p] and is_high_score[q]:
                        img = cv2.line(
                            img,
                            (int(x[p]), int(y[p])),
                            (int(x[q]), int(y[q])),
                            self.joints_color[i],
                            thickness=5,
                            lineType=cv2.LINE_AA,
                        )

            # draw bbox
            img = self.draw_bbox(img_path=img_path, img=img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # save images
            cv2.imwrite(os.path.join("res/imgs", img_name), img)

    def draw_bbox(self, img_path: str, img: list):

        bboxes = self.bbox_data[img_path]["bbox"]

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(bb) for bb in bbox]
            img = cv2.rectangle(
                img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=3,
            )

        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results Visualization")
    parser.add_argument(
        "--pose_path", default="default", type=str, help="input pose json path"
    )
    parser.add_argument(
        "--bbox_path", default="default", type=str, help="input bbox json path"
    )
    args = parser.parse_args()

    pose_visualization = PoseVis(pose_path=args.pose_path, bbox_path=args.bbox_path)
    pose_visualization()
