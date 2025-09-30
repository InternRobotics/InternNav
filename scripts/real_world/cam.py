# requirements: sudo apt-get install ros-noetic-cv-bridge python3-opencv
import threading

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraSubscriber:
    def __init__(self, topic="/camera/image_raw", queue_size=1):
        self.bridge = CvBridge()
        self.latest_bgr = None
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber(topic, Image, self._cb, queue_size=queue_size)

    def _cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_bgr = frame
        except Exception as e:
            rospy.logerr(f"[CameraSubscriber] decode failed: {e}")

    def get_latest(self):
        with self.lock:
            return None if self.latest_bgr is None else self.latest_bgr.copy()
