import cv2
from ultralytics import YOLO
import numpy as np
import torch
import logging
from dictionaries.dictionaries import fields_dict
from detectors.Chess_pieces_detector import ChessPiecesDetector
from detectors.Chessboard_detector import Chessboard_detector
from Utils.Corner_detector import Corner_detector
from Utils.Warper import Warper
from Utils.Chessboard_mapper import Chessboard_mapper
from Utils.Stockfish import Stockfish

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("chess_app.log"),
        logging.StreamHandler()
    ]
)

def process_frame(img, chessboard):
    try:
        mask = chessboard_detector.get_mask(img)
    except Exception as e:
        logging.error(f"Error getting mask: {e}")
        return img  # Return original image or a placeholder

    try:
        resized_img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return img

    try:
        corners = corner_detector.get_sorted_corner_pts(mask)
    except Exception as e:
        logging.error(f"Error getting sorted corners: {e}")
        return img

    try:
        corners_img = corner_detector.get_bitwise_corners(mask)
    except Exception as e:
        logging.error(f"Error getting bitwise corners: {e}")
        corners_img = mask  # Fallback to mask image

    try:
        pieces_coords = chess_pieces_detector.get_pieces_bb(resized_img)
    except Exception as e:
        logging.error(f"Error detecting chess pieces: {e}")
        pieces_coords = []

    # Check for sufficient corners
    if len(corners) < 4:
        logging.warning("Less than 4 corners detected. Displaying mask and corners image.")
        try:
            cv2.imshow("Mask found", mask)
            cv2.imshow("corners", corners_img)
            cv2.waitKey(1)  # Changed from 0 to 1 to prevent blocking
            cv2.destroyWindow("Mask found")
            cv2.destroyWindow("corners")
        except Exception as e:
            logging.error(f"Error displaying images: {e}")
        return img  # Skip processing this frame

    try:
        upper_left, upper_right = sorted(corners[:2], key=lambda c: c[0])
        lower_left, lower_right = sorted(corners[2:], key=lambda c: c[0])
    except Exception as e:
        logging.error(f"Error sorting corners: {e}")
        return img

    try:
        src = np.array([
            upper_left,
            upper_right,
            lower_right,
            lower_left
            ],
            dtype=np.float32)
    
        dst = np.array([
            [0, 0],
            [500, 0],
            [500, 500],
            [0, 500]],
            dtype=np.float32)
    except Exception as e:
        logging.error(f"Error creating src and dst arrays: {e}")
        return img

    try:
        warper.warp_img(src, dst, resized_img)
        chessboard_grid_pts_in_original = warper.unwarp_grid_pts()
    except Exception as e:
        logging.error(f"Error warping image: {e}")
        return img

    try:
        chessboard = chessboard_mapper.map_pieces_to_chessboard(pieces_coords, chessboard_grid_pts_in_original)
        fen = chessboard_mapper.get_fen_from_chessboard(chessboard)
    except Exception as e:
        logging.error(f"Error mapping pieces to chessboard: {e}")
        fen = ""

    try:
        fen = "".join(["k" if i == 4 else char for i, char in enumerate(fen)])
        fen += " b"
    except Exception as e:
        logging.error(f"Error modifying FEN string: {e}")
        fen = ""

    # Draw bounding boxes around detected pieces
    try:
        for bb in pieces_coords:
            pt1, pt2 = bb[:2], bb[2:4]
            cv2.rectangle(resized_img, pt1, pt2, (0, 255, 0), 2)
    except Exception as e:
        logging.error(f"Error drawing bounding boxes: {e}")

    # Optionally, draw corners or other features
    try:
        for pt1, pt2, _, _ in chessboard_grid_pts_in_original:
            x1, y1 = pt1
            x2, y2 = pt2
            center = (int((x1+x2)//2), int((y1+y2)//2))
            cv2.circle(resized_img, center, 5, (0, 0, 255), -1)
    except Exception as e:
        logging.error(f"Error drawing grid points: {e}")

    # Get move from Stockfish
    try:
        if fen:
            move = stockfish.get_move(fen)
        else:
            move = None
    except Exception as e:
        logging.error(f"Error getting move from Stockfish: {e}")
        move = None

    if move is None:
        logging.warning("No move returned from Stockfish.")
        try:
            cv2.imshow("Chessboard", resized_img)
            cv2.waitKey(1)  # Changed from 0 to 1
            cv2.destroyWindow("Chessboard")
        except Exception as e:
            logging.error(f"Error displaying Chessboard image: {e}")
        return resized_img  # Continue without drawing move

    try:
        from_pt, to_pt = fields_dict.get(move.from_square), fields_dict.get(move.to_square)
        if from_pt is None or to_pt is None:
            raise ValueError("Invalid move squares.")
    except Exception as e:
        logging.error(f"Error retrieving move squares: {e}")
        return resized_img

    try:
        pt1, pt2 = [int(x) for x in chessboard_grid_pts_in_original[from_pt][0]], [int(x) for x in chessboard_grid_pts_in_original[from_pt][2]]
        pt3, pt4 = [int(x) for x in chessboard_grid_pts_in_original[to_pt][0]], [int(x) for x in chessboard_grid_pts_in_original[to_pt][2]]
    
        center1 = ((pt1[0]+pt2[0]) // 2, (pt1[1]+pt2[1]) // 2)
        center2 = ((pt3[0]+pt4[0]) // 2, (pt3[1]+pt4[1]) // 2)
    
        cv2.line(resized_img, center1, center2, (255, 0, 0), 4)
    except Exception as e:
        logging.error(f"Error drawing move line: {e}")

    # Display the processed chessboard
    try:
        cv2.imshow("Chessboard", resized_img)
        cv2.waitKey(1)  # Changed from 0 to 1
    except Exception as e:
        logging.error(f"Error displaying Chessboard image: {e}")

    return resized_img


# Initialize detectors and utilities with error handling
try:
    chess_pieces_detector = ChessPiecesDetector()
    chessboard_detector = Chessboard_detector()
    corner_detector = Corner_detector()
    warper = Warper()
    chessboard_mapper = Chessboard_mapper()
    stockfish = Stockfish()
except Exception as e:
    logging.critical(f"Error initializing detectors or utilities: {e}")
    exit(1)  # Exit if initialization fails


def main(cap):
    counter = 0
    chessboard = np.full((8, 8), "", dtype=object)  # Specify dtype for clarity

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from capture. Exiting loop.")
                break

            counter += 1
            if counter < 300:
                continue

            if counter % 100 == 0:
                try:
                    processed_frame = process_frame(frame, chessboard)
                    cv2.imshow("Processed Frame", processed_frame)
                except Exception as e:
                    logging.error(f"Error processing frame {counter}: {e}")
            else:
                try:
                    cv2.imshow("Frame", frame)
                except Exception as e:
                    logging.error(f"Error displaying frame {counter}: {e}")

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logging.info("Exit key pressed. Exiting loop.")
                break

        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            continue  # Continue processing the next frame

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Released video capture and destroyed all windows.")


if __name__ == "__main__":
    video_path = "/Users/lukamindek/Desktop/i/chess_app/images/chess_video.mp4"
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        logging.info(f"Successfully opened video file: {video_path}")
    except Exception as e:
        logging.critical(f"Error opening video file: {e}")
        exit(1)

    main(cap)
