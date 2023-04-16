import sys, os, cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # Chessboard configuration
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard_create(18, 9, 0.02, 0.015, aruco_dict)

# # check if the board is correct
# image = board.draw((1280, 720))

# plt.figure()
# plt.imshow(image, cmap='gray')
# plt.title('18x9 ChAruco pattern')
# plt.show()


parameters =  cv2.aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
all_corners = []
all_ids = []
input_files = 'data/intrinsic/*.jpeg'

for i in sorted(glob(input_files)):
    frame = cv2.imread(i)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if len(corners) > 0:
        ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        print(f'{i}  found {ret} corners')
        if ret > 0:
            all_corners.append(c_corners)
            all_ids.append(c_ids)

    imsize = (gray.shape[1], gray.shape[0])

# show sample image
# plt.figure()
# plt.imshow(frame)
# plt.show()

ret, K, d, rvec, tvec = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, imsize, None, None,
                                                         flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_RATIONAL_MODEL)

print("Reprojection error = ", ret)
print("Intrinsic parameter K = ", K)
print("Distortion parameters d = (k1, k2, p1, p2, k3, k4, k5, k6) = ", d)