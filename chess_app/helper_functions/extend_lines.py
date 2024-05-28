import numpy as np

def extend_lines(x1, y1, x2, y2, distance=1000):
    diff = np.arctan2(y1 - y2, x1 - x2)
    p3_x = int(x1 + distance * np.cos(diff))
    p3_y = int(y1 + distance * np.sin(diff))
    p4_x = int(x1 - distance * np.cos(diff))
    p4_y = int(y1 - distance * np.sin(diff))
    return p3_x, p3_y, p4_x, p4_y