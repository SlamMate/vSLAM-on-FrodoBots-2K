import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

# Initialize ROS node
rospy.init_node('image_saver')

# Initialize CvBridge
bridge = CvBridge()

# Directory to save images
save_dir = "/home/zhangqi/Downloads/Dataset/image"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def image_callback(msg):
    try:
        rospy.loginfo(f"Image received with encoding: {msg.encoding}")
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Log image data for debugging
        rospy.loginfo(f"Image data: {cv_image}")

        # Generate a unique filename
        timestamp = rospy.Time.now().to_nsec()
        filename = os.path.join(save_dir, f"image_{timestamp}.jpg")

        # Save the image
        cv2.imwrite(filename, cv_image)
        rospy.loginfo(f"Saved image to {filename}")

    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")

# Subscribe to the /camera/image_raw topic
rospy.Subscriber('/camera/image_raw', Image, image_callback)

# Keep the script running
rospy.spin()

