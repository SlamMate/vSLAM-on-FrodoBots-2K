import cv2
import numpy as np

# 你的相机内参矩阵
camera_matrix = np.array([[407.86023253, 0., 533.30109486],
                          [0., 407.86605705, 278.69939958],
                          [0., 0., 1.]])

# 你的畸变系数
dist_coeffs = np.array([-0.2171785, 0.05372816, 0.00185307, -0.0021051, -0.00599918])

# 读取图像
image = cv2.imread('screenshot_1723289045.png')

# 去畸变
undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

# 保存或显示结果
cv2.imwrite('undistorted_image.jpg', undistorted_image)
cv2.imshow('undistorted', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

