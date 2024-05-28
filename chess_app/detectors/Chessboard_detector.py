from ultralytics import YOLO
import torch
import numpy as np

class Chessboard_detector:
    def __init__(self, weights_path='./model_weights/chessboard_segment_v2.pt'):
        self.model = YOLO(weights_path)
        
    def _get_prediction(self, img):
        results = self.model.predict(source=img.copy(), save=False, save_txt=False, stream=True)
        return results
    
    def calculate_mask(self, img):
        results = self._get_prediction(img)
        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]
            chessboard_indices = torch.where(clss == 0)[0]

            if len(chessboard_indices) > 0:
                mask = masks[chessboard_indices]
                combined_mask = torch.any(mask, dim=0).int() * 255
                final_mask = combined_mask.cpu().numpy().astype(np.uint8)
                return final_mask
    
    def get_mask(self, img):
        mask = self.calculate_mask(img)
        return mask
                
    

        
