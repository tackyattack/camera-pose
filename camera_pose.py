import sys
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_cam(ax, R_c2w, t_c2w, name=None, scale=50.0):
    ex = R_c2w[:, 0] * scale + t_c2w
    ey = R_c2w[:, 1] * scale + t_c2w
    ez = R_c2w[:, 2] * scale + t_c2w
    p = np.array([ex, t_c2w, ey, t_c2w, ez, t_c2w])
    ax.plot(p[:, 0], p[:, 1], p[:, 2])
    ax.text(p[0, 0], p[0, 1], p[0, 2], 'x')
    ax.text(p[2, 0], p[2, 1], p[2, 2], 'y')
    ax.text(p[4, 0], p[4, 1], p[4, 2], 'z')
    if name is not None:
        ax.text(p[1, 0], p[1, 1], p[1, 2], name)


def visualize(R_list, t_list, v, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, (R, t) in enumerate(zip(R_list, t_list)):
        plot_cam(ax, R, t, name=f'C{i}')

    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)
    ax.set_title(title)
    plt.show()


class ImagePoints:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path, 1)
        self.points = []

    def get_points(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img)
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return np.array(self.points, dtype=np.float64)

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', self.img)
            self.points.append([x, y])


def calibRt(X, x, K, distCoeffs):
    X = np.copy(X)
    x = np.copy(x)
    ret, rvec, tvec = cv2.solvePnP(X, x, K, distCoeffs)
    rmat = cv2.Rodrigues(rvec)[0]
    return rmat, tvec, ret


# iPhone 11, 26mm
K = np.array([[3.04394013e+03, 0.00000000e+00, 1.97865411e+03],
              [0.00000000e+00, 3.04394013e+03, 1.48784274e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

d = np.array([[-2.21760091e+01, 1.77875346e+02,  1.66830131e-03, -2.34664887e-03,
               3.07765888e+01,  -2.22197052e+01, 1.78730770e+02, 2.21446963e+01]])


print("Intrinsic parameter K = ", K)
print("Distortion parameters d = (k1, k2, p1, p2, k3, k4, k5, k6) = ", d)

# Point board in cm
X = np.array([[-4.0, 3.0,  0.],
              [5.5,  3.0,  0.],
              [5.0, -2.5, 0.],
              [-4.5, -2.5,  0.]])*np.array([10.0])

# x = np.array([[1275, 1110],
#               [2876, 1108],
#               [2883, 2084],
#               [1019, 2015]], dtype=np.float64)

image_points = ImagePoints(
    'data/intrinsic/A4566FC7-8A32-41B3-AA87-061A4E11C9A4.jpeg')
x = image_points.get_points()
print(x)
print(X)
R, t, ret = calibRt(X, x, K, d)
visualize([R], [t[:, 0]], X, "cam position")
