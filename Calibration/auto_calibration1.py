import os
import requests
import time
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
import random

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

def capture_screenshot(retries=3, delay=2):
    """
    Capture screenshot from the robot's camera with retries.
    """
    url = f'{BASE_URL}/screenshot'
    for attempt in range(retries):
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            if 'front_video_frame' in data and 'timestamp' in data:
                # Decode the base64 image
                image_data = base64.b64decode(data['front_video_frame'])
                timestamp = data['timestamp']
                return image_data, timestamp
            else:
                print("Invalid response format, 'front_video_frame' or 'timestamp' not found")
        else:
            print(f"Failed to get screenshot, status code: {response.status_code}")
            print("Response content:", response.content)
        
        if attempt < retries - 1:
            time.sleep(delay)
    
    return None, None

def save_image(image_data, timestamp):
    """
    Save the image to the local filesystem.
    """
    if image_data is None or timestamp is None:
        print("Invalid image data or timestamp, skipping save")
        return None

    # Check the file format based on the first few bytes
    if image_data[:2] == b'\xff\xd8':
        file_extension = 'jpg'
    elif image_data[:4] == b'\x89PNG':
        file_extension = 'png'
    else:
        print("Unknown file format, skipping save")
        return None

    filename = f'screenshot_{timestamp}.{file_extension}'
    with open(filename, 'wb') as f:
        f.write(image_data)
    print(f'Saved {filename}')
    return filename

def capture_images(movement_patterns, chessboard_size):
    image_files = []
    last_linear, last_angular = 0, 0
    
    for linear, angular in movement_patterns:
        # Send control command
        send_control_command(linear, angular)
        # Wait for the robot to move
        time.sleep(2)
        # Capture screenshot
        image_data, timestamp = capture_screenshot()
        # Save screenshot
        filename = save_image(image_data, timestamp)
        
        if filename:
            # Load the saved image to check for chessboard corners
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                # If corners are found, add the image file and update the last movement
                image_files.append(filename)
                last_linear, last_angular = linear, angular
            else:
                # If corners are not found, move in the opposite direction
                print(f"Chessboard corners not found in image {filename}. Reversing last movement.")
                send_control_command(-last_linear, last_angular)
                time.sleep(2)  # Wait for the robot to move
                send_control_command(0, 0)  # Stop the robot
            
        # Stop the robot
        send_control_command(0, 0)
        # Wait a bit before the next command
        time.sleep(2)
    
    return image_files

def calibrate_camera(image_files, chessboard_size, frame_size):
    # Termination criteria for corner sub-pix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...,(chessboard_size[0]-1,chessboard_size[1]-1,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            print(f"Chessboard corners not found in image {fname}")

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("No chessboard corners were found in any of the images.")
        return None, None, None, None, None, None, None

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints

def main():
    # Instructions for placing the chessboard
    print("Ensure the chessboard is placed on a flat surface in a well-lit area.")
    print("Capture images of the chessboard from different angles and distances.")
    print("Ensure the entire chessboard is visible in each image.")

    # Define movement patterns for calibration (linear, angular)
    movement_patterns = [
        (1, 0),   # Move forward
        (-1, 0),  # Move backward
        (0, 1),   # Rotate clockwise
        (0, -1),  # Rotate counterclockwise
        (1, 1), # Move forward-right
        (-1, -1), # Move backward-right
    ]

    # Repeat the entire pattern 20 times with a random adjustment for each group
    repeated_patterns = [
        (random.uniform(0, 1) * x, random.uniform(0, 1) * y)
        for _ in range(20)
        for x, y in movement_patterns
    ]

    # Load one image to get the frame size
    frame_size = (1024, 576)

    # Chessboard size
    chessboard_size = (8, 6)

    desired_accuracy = 0.1  # Desired calibration accuracy
    achieved_accuracy = float('inf')  # Initialize achieved accuracy

    while achieved_accuracy > desired_accuracy:
        # Capture images with current movement patterns
        image_files = capture_images(repeated_patterns, chessboard_size)
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = calibrate_camera(image_files, chessboard_size, frame_size)
        if ret is None:
            print("Calibration failed. Retrying...")
            continue
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

