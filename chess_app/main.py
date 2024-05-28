import cv2
from ultralytics import YOLO
import numpy as np
import torch
from dictionaries.dictionaries import fields_dict
from detectors.Chess_pieces_detector import ChessPiecesDetector
from detectors.Chessboard_detector import Chessboard_detector
from Utils.Corner_detector import Corner_detector
from Utils.Warper import Warper
from Utils.Chessboard_mapper import Chessboard_mapper
from Utils.Stockfish import Stockfish




def process_frame(img, chessboard):
    mask = chessboard_detector.get_mask(img)
    resized_img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    corners = corner_detector.get_sorted_corner_pts(mask)
    corners_img = corner_detector.get_bitwise_corners(mask)
    pieces_coords = chess_pieces_detector.get_pieces_bb(resized_img)


    #######
    if len(corners) < 4:
        cv2.imshow("Mask found", mask)
        cv2.imshow("corners", corners_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        quit()
    ########
    upper_left, upper_right = sorted(corners[:2], key=lambda c: c[0])
    lower_left, lower_right = sorted(corners[2:], key=lambda c: c[0])

    src = np.array([
        upper_left,
        upper_right,
        lower_right,
        lower_left
        ],
        dtype = np.float32)

    dst = np.array([
        [0, 0],
        [500, 0],
        [500, 500],
        [0, 500]],
        dtype = np.float32)

    warper.warp_img(src, dst, resized_img)
    chessboard_grid_pts_in_original = warper.unwarp_grid_pts()
    
    chessboard = chessboard_mapper.map_pieces_to_chessboard(pieces_coords, chessboard_grid_pts_in_original)
    fen = chessboard_mapper.get_fen_from_chessboard(chessboard)
    
    ####
    fen = "".join(["k" if i == 4 else char for i, char in enumerate(fen)])
    fen += " b"
    ####    
    """gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    p0 = []"""
    for bb in pieces_coords:
        pt1, pt2 = bb[:2], bb[2:4]
        cv2.rectangle(resized_img, pt1, pt2, (0, 255, 0), 2)
        """ (x1, y1), (x2, y2) = pt1, pt2
        roi = gray[y1:y2, x1:x2]
        feature = cv2.goodFeaturesToTrack(roi, 10, 0.3, 1, None, None)
        for f in feature:
            f[0][0] = f[0][0] + x1
            f[0][1] = f[0][1] + y1
            x, y = int(f[0][0]), int(f[0][1])

            cv2.circle(resized_img, (x, y), 5, (0, 255, 0), -1)
            p0.append(f)"""
    for pt1, pt2, _, _ in chessboard_grid_pts_in_original:
        x1, y1 = pt1
        x2, y2 = pt2
        center = (int((x1+x2)//2), int((y1+y2)//2))
    #rnbkkbnr/bnbbqbbn/8/8/8/8/BBR2R1Q/QNKKKKNQ w KQkq - 0 1
    #print(fen)
    move = stockfish.get_move(fen)
    
    from_pt, to_pt = fields_dict[move.from_square], fields_dict[move.to_square]
    pt1, pt2 = [int(x) for x in chessboard_grid_pts_in_original[from_pt][0]], [int(x) for x in chessboard_grid_pts_in_original[from_pt][2]]
    pt3, pt4 = [int(x) for x in chessboard_grid_pts_in_original[to_pt][0]], [int(x) for x in chessboard_grid_pts_in_original[to_pt][2]]

    center1 = ((pt1[0]+pt2[0]) // 2, (pt1[1]+pt2[1]) // 2)
    center2 = ((pt3[0]+pt4[0]) // 2, (pt3[1]+pt4[1]) // 2)

    cv2.line(resized_img, center1, center2, (255, 0, 0), 4)
    #############
    if move is None:
        cv2.imshow("Chessboard", resized_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        quit()
    ##############
    #center1, center2 = chessboard_mapper.get_move_pts(chessboard_grid_pts_in_original, move)
    return resized_img


chess_pieces_detector = ChessPiecesDetector()
chessboard_detector = Chessboard_detector()
corner_detector = Corner_detector()
warper = Warper()
chessboard_mapper = Chessboard_mapper()
stockfish = Stockfish()

def main(cap):
    counter = 0
    chessboard = np.full((8, 8), "")
    
    while True:
        ret, frame = cap.read()
        counter += 1
        if counter < 300:
            continue
        if not ret:
            break
        
        if counter % 100 == 0:
            processed_frame = process_frame(frame, chessboard)
            cv2.imshow("frame", processed_frame)   
            cv2.waitKey(0)
        else:
            cv2.imshow("frame", frame)   
                    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    cap = cv2.VideoCapture('/Users/luksuz/Desktop/chess_app/images/chess_video.mp4')
    main(cap)

