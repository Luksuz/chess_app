# ♟️ Chess AI App ♟️

## Overview

Welcome to the Chess AI App! This project leverages advanced AI techniques to segment chessboards, detect chess pieces, and predict the next best move using the Stockfish neural engine. The app utilizes two YOLO models (v8 for chessboard segmentation and v5 for chess piece detection) and OpenCV for chessboard corner detection and grid creation. This project was trained and tested on Kaggle's GPU P100s.

## Features

- **♜ Chessboard Segmentation**: Uses YOLO v8 to segment the chessboard from the image.
- **♞ Chess Piece Detection**: Uses YOLO v5 to detect and classify chess pieces on the segmented chessboard.
- **♟️ Grid Mapping**: Utilizes OpenCV to detect chessboard corners and create a grid of 64 tiles for mapping detected pieces.
- **♛ Move Prediction**: Uses the Stockfish neural engine to predict the next best move in the given chess situation.

## Installation

To install the app, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/chess_ai_app.git
cd chess_app
pip install -r requirements.txt
python main.py
```

Contributions to enhance or fix issues in the Chess AI App are welcome. If you have suggestions, improvements, or bug fixes, please feel free to:

	1.	Fork the repository
	2.	Create your feature branch (git checkout -b feature/AmazingFeature)
	3.	Commit your changes (git commit -am 'Add some amazing feature')
	4.	Push to the branch (git push origin feature/AmazingFeature)
	5.	Open a pull request
