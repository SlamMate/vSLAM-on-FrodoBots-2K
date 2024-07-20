import cv2
import numpy as np
import requests
import base64
from time import sleep
from ORB_SLAM3 import System, ORB_SLAM3
import threading
import json
from flask import Flask, request

app = Flask(__name__)

# Function to load images (not used in real-time mode)
def LoadImages(strFile, vstrImageFilenames, vTimestamps):
    with open(strFile, 'r') as f:
        # Skip the first three lines
        for _ in range(3):
            next(f)

        for line in f:
            if line.strip():
                parts = line.strip().split()
                vTimestamps.append(float(parts[0]))
                vstrImageFilenames.append(parts[1])

# Function to fetch images from the Earth Rovers SDK
def fetch_image():
    response = requests.get('http://localhost:8000/screenshot')
    data = response.json()
    image_data = base64.b64decode(data['frame'])
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    timestamp = data['timestamp']
    return image, timestamp

# Function to process images using ORB-SLAM3
def process_images():
    while True:
        image, timestamp = fetch_image()
        if image_scale != 1.0:
            width = int(image.shape[1] * image_scale)
            height = int(image.shape[0] * image_scale)
            image = cv2.resize(image, (width, height))
        with lock:
            SLAM.TrackMonocular(image, timestamp)
        sleep(0.1)  # Adjust sleep duration as needed

@app.route('/control', methods=['POST'])
def control():
    command = request.json['command']
    linear = command['linear']
    angular = command['angular']
    # Implement control logic based on linear and angular values
    # Send control commands to the robot via Earth Rovers SDK
    control_robot(linear, angular)
    return json.dumps({'message': 'Command sent successfully'}), 200

@app.route('/data', methods=['GET'])
def data():
    # Retrieve the latest data from the robot
    response = requests.get('http://localhost:8000/data')
    return response.json()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    with lock:
        SLAM.Shutdown()
    return json.dumps({'message': 'SLAM system shut down successfully'}), 200

def control_robot(linear, angular):
    # Implement the control logic for the robot
    payload = {
        'command': {
            'linear': linear,
            'angular': angular
        }
    }
    headers = {'Content-Type': 'application/json'}
    requests.post('http://localhost:8000/control', headers=headers, json=payload)

if __name__ == '__main__':
    path_to_vocabulary = "path/to/ORBvoc.txt"
    path_to_settings = "path/to/settings.yaml"
    path_to_sequence = "path/to/sequence"  # Replace with the appropriate path

    # Initialize ORB-SLAM3 system
    SLAM = ORB_SLAM3.System(path_to_vocabulary, path_to_settings, ORB_SLAM3.System.MONOCULAR, True)
    image_scale = SLAM.GetImageScale()
    lock = threading.Lock()

    # Start the image processing thread
    threading.Thread(target=process_images, daemon=True).start()

    # Start Flask server
    app.run(host='0.0.0.0', port=8000)

