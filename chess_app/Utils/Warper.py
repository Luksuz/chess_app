import cv2
import numpy as np

class Warper:
    def __init__(self, img_size=500, n_of_pts=9):
        self.img_size = img_size
        self.n_of_pts = n_of_pts
        self.step_size = img_size / (n_of_pts - 1)
        self.M = None
        self.M_inv = None
    
    def warp_img(self, src, dst, resized_img):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
        warped_img = cv2.warpPerspective(resized_img, self.M, (self.img_size, self.img_size))
        return warped_img

    def generate_chessboard_points(self):
        chessboard_outer_corner_pts = np.linspace(0, self.img_size, self.n_of_pts)
        chessboard_outer_corner_pts = [int(pt) for pt in chessboard_outer_corner_pts]

        zeros = np.zeros((self.n_of_pts), dtype=np.int32)
        end_pts = np.zeros((self.n_of_pts), dtype=np.int32) + self.img_size

        horizontal_upper_lines = list(zip(chessboard_outer_corner_pts, zeros))
        horizontal_lower_lines = list(zip(chessboard_outer_corner_pts, end_pts))
        vertical_left_lines = list(zip(zeros, chessboard_outer_corner_pts))
        vertical_right_lines = list(zip(end_pts, chessboard_outer_corner_pts))

        chessboard_grid_pts = []
        for i in range(8):  # ROWS
            for j in range(8):  # COLS
                x1 = j * self.step_size  # Top-left corner x-coordinate
                y1 = i * self.step_size  # Top-left corner y-coordinate
                x2 = (j + 1) * self.step_size  # Top-right corner x-coordinate
                y2 = (i + 1) * self.step_size  # Bottom-right corner y-coordinate
                chessboard_grid_pts.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        chessboard_grid_pts = np.array(chessboard_grid_pts, dtype=np.float32)

        points_in_warped_img = np.array(
            horizontal_upper_lines + horizontal_lower_lines + vertical_left_lines + vertical_right_lines,
            dtype=np.float32
        ).reshape(-1, 1, 2)
        
        return chessboard_grid_pts

    def unwarp_grid_pts(self):
        chessboard_grid_pts = self.generate_chessboard_points()
        chessboard_grid_pts_in_warped = chessboard_grid_pts.reshape(-1, 4, 2)
        chessboard_grid_pts_in_original = cv2.perspectiveTransform(chessboard_grid_pts_in_warped, self.M_inv)
        chessboard_grid_pts_in_original = chessboard_grid_pts_in_original.reshape(-1, 4, 2)
        return chessboard_grid_pts_in_original
