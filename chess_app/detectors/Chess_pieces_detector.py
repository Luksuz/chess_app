import cv2
from ultralytics import YOLO
import torch
from dictionaries.dictionaries import cls_to_name

class ChessPiecesDetector:
    def __init__(self, model_path="./model_weights/chess_pieces.pt", conf_threshold=0.4):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        self.conf_threshold = conf_threshold
        self.bb_desc = []

    def predict(self, img):
        results = self.model(img)
        bbs = results.xyxy[0]
        new_bb_desc = []
        for bb in bbs:
            x1, y1, x2, y2 = map(int, bb[:4])
            conf = bb[4]
            cls = int(bb[5])
            cls_name = cls_to_name[cls]
            if conf > self.conf_threshold:
                new_bb_desc.append((x1, y1, x2, y2, cls_name))
        self.bb_desc = new_bb_desc

    def get_pieces_bb(self, img):
        self.predict(img)
        return self.bb_desc
