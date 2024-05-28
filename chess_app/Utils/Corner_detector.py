import cv2
import numpy as np
from helper_functions.extend_lines import extend_lines

class Corner_detector:
    def __init__(self):
        pass

    def detect_lines(self, mask):
        canny = cv2.Canny(mask, 100, 200)
        lines = cv2.HoughLinesP(canny, 1, np.pi / 300, threshold=40, minLineLength=100, maxLineGap=100)
        return lines

    def filter_lines_direction(self, mask):
        lines = self.detect_lines(mask)
        horizontal_lines_canvas = np.zeros_like(mask)
        vertical_lines_canvas = np.zeros_like(mask)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x1, y1, x2, y2 = extend_lines(x1, y1, x2, y2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 45:
                    cv2.line(horizontal_lines_canvas, (x1, y1), (x2, y2), 255, 10)
                elif abs(angle) > 45:
                    cv2.line(vertical_lines_canvas, (x1, y1), (x2, y2), 255, 10)

        return horizontal_lines_canvas, vertical_lines_canvas

    def get_bitwise_corners(self, mask):
        horizontal_lines_canvas, vertical_lines_canvas = self.filter_lines_direction(mask)

        _, binary_horizontal = cv2.threshold(horizontal_lines_canvas, 0, 255, cv2.THRESH_BINARY)
        _, binary_vertical = cv2.threshold(vertical_lines_canvas, 0, 255, cv2.THRESH_BINARY)

        corners = cv2.bitwise_and(binary_vertical, binary_horizontal)
        return corners

    def get_sorted_corner_pts(self, mask):
        corners = self.get_bitwise_corners(mask)
        contours, _ = cv2.findContours(corners, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corners = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                corners.append((cx, cy))
        
        return sorted(corners, key=lambda c: c[1])
