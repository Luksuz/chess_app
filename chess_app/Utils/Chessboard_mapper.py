import cv2
from dictionaries.dictionaries import fields_dict
import numpy as np

Y_OFFSET = lambda y1, y2: (y2-y1)/10

class Chessboard_mapper:
    def __init__(self):
        pass
    
    def map_pieces_to_chessboard(self, pieces_coords, chessboard_grid_pts_in_original):
        chessboard = np.full((8, 8), "")
        for bb in pieces_coords:
            x1, y1, x2, y2, cls = bb
            bottom_mid = (x1 + x2) // 2
            for i, grid_pts in enumerate(chessboard_grid_pts_in_original):
                (x_grid1, y_grid1), (x_grid2, y_grid2), (x_grid3, y_grid3), (x_grid4, y_grid4) = grid_pts                
                if (x_grid1 < bottom_mid < x_grid2 or x_grid4 < bottom_mid < x_grid3) and (y_grid1 < y2 - Y_OFFSET(y1, y2) < y_grid4):
                    row = i // 8
                    col = i % 8
                    chessboard[row, col] = cls
                    break
        print(chessboard)
        return chessboard
        
    def get_fen_from_chessboard(self, chessboard):   
        count = 0
        new_chessboard = []
        for row in chessboard:
            count = 0
            curr_row = []
            
            for i in range(8):
                if row[i] == "":
                    count += 1
                else:
                    if count > 0:
                        curr_row.append(str(count))
                        count = 0
                    curr_row.append(row[i])
            
            if count > 0:
                curr_row.append(str(count))
            
            new_chessboard.append("".join(curr_row))
            
        fen = "/".join(new_chessboard)
        
        return fen
    
    def get_move_pts(self, chessboard_grid_pts_in_original, move):
        from_pt, to_pt = fields_dict[move.from_square], fields_dict[move.to_square]
        pt1, pt2 = [int(x) for x in chessboard_grid_pts_in_original[from_pt][0]], [int(x) for x in chessboard_grid_pts_in_original[from_pt][2]]
        pt3, pt4 = [int(x) for x in chessboard_grid_pts_in_original[to_pt][0]], [int(x) for x in chessboard_grid_pts_in_original[to_pt][2]]

        center1 = ((pt1[0]+pt2[0]) // 2, (pt1[1]+pt2[1]) // 2)
        center2 = ((pt3[0]+pt4[0]) // 2, (pt3[1]+pt4[1]) // 2)
        
        return center1, center2