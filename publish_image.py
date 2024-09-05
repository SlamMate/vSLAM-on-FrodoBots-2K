import rospy
import requests
import base64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

# Initialize ROS node
rospy.init_node('earth_rovers_camera_publisher')

# Create a publisher for the /camera/image_raw topic
image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

# Initialize CvBridge
bridge = CvBridge()

# SDK endpoint to get the camera images
url = "http://localhost:8000/screenshot?view_types=front,rear"

def get_camera_image(view_type):
    try:
        # Make the request to the SDK
        response = requests.get(url)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Decode the base64 image based on the view type
        img_data = base64.b64decode(data[f"{view_type}_video_frame"])

        # Convert the image data to a numpy array
        np_arr = np.frombuffer(img_data, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return img_np

    except requests.exceptions.RequestException as e:
        rospy.logerr(f"Error getting {view_type} image from SDK: {e}")
        return None

def save_image(img_np, view_type):
    # Save the image with the current timestamp
    timestamp = rospy.get_time()
    filename = f"/home/zhangqi/Downloads/Dataset/rear_image/{view_type}_image_{int(timestamp)}.png"
    cv2.imwrite(filename, img_np)
    # rospy.loginfo(f"Saved {view_type} image to {filename}")

def publish_image():
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        front_img_np = get_camera_image("front")
        rear_img_np = get_camera_image("rear")

        if front_img_np is not None:
            # Convert the numpy array (OpenCV format) to a ROS Image message
            ros_image = bridge.cv2_to_imgmsg(front_img_np, "bgr8")
            # Publish the image to the ROS topic
            image_pub.publish(ros_image)

        if rear_img_np is not None:
            # Save the rear image locally
            save_image(rear_img_np, "rear")

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass

