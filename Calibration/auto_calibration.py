import os
import requests
import time
import base64
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API token and bot name from environment variables
SDK_API_TOKEN = os.getenv('SDK_API_TOKEN')
BOT_NAME = os.getenv('BOT_NAME')

# Base URL for the SDK
BASE_URL = 'http://localhost:8000'

# Headers for the requests
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {SDK_API_TOKEN}'
}

def send_control_command(linear, angular):
    """
    Send control command to the robot.
    """
    url = f'{BASE_URL}/control'
    payload = {
        'command': {'linear': linear, 'angular': angular}
    }
    response = requests.post(url, json=payload, headers=HEADERS)
    return response.json()

def capture_screenshot():
    """
    Capture screenshot from the robot's camera.
    """
    url = f'{BASE_URL}/screenshot'
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    
    # Decode the base64 image
    image_data = base64.b64decode(data['frame'])
    timestamp = data['timestamp']
    
    return image_data, timestamp

def save_image(image_data, timestamp):
    """
    Save the image to the local filesystem.
    """
    filename = f'screenshot_{timestamp}.jpg'
    with open(filename, 'wb') as f:
        f.write(image_data)
    print(f'Saved {filename}')
    return filename

def capture_images(movement_patterns):
    image_files = []
    
    for linear, angular in movement_patterns:
        # Send control command
        send_control_command(linear, angular)
        # Wait for the robot to move
        time.sleep(2)
        # Capture screenshot
        image_data, timestamp = capture_screenshot()
        # Save screenshot
        filename = save_image(image_data, timestamp)
        image_files.append(filename)
        # Stop the robot
        send_control_command(0, 0)
        # Wait a bit before the next command
        time.sleep(2)
    
    return image_files

def calibrate_camera(image_files):
    # Termination criteria for corner sub-pix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

def main():
    # Define movement patterns for calibration (linear, angular)
    movement_patterns = [
        (0.5, 0),   # Move forward
        (-0.5, 0),  # Move backward
        (0, 0.5),   # Rotate clockwise
        (0, -0.5)   # Rotate counterclockwise
    ]

    desired_accuracy = 0.01  # Desired calibration accuracy
    achieved_accuracy = float('inf')  # Initialize achieved accuracy

    while achieved_accuracy > desired_accuracy:
        # Capture images with current movement patterns
        image_files = capture_images(movement_patterns)
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(image_files)
        # Calculate the calibration error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        achieved_accuracy = total_error / len(objpoints)
        print(f'Calibration error: {achieved_accuracy}')
    
    print("Camera calibration achieved desired accuracy")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)

if __name__ == '__main__':
    main()

