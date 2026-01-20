import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageSubscriber:
    def __init__(self, topic="/usb_cam/image_raw"):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.new_frame_available = False
        
        # 카메라 토픽 구독
        self.sub = rospy.Subscriber(topic, Image, self._callback)
        rospy.loginfo(f"Subscribing to {topic}...")

    def _callback(self, msg):
        try:
            # ROS Image -> OpenCV BGR
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.new_frame_available = True
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def read(self):
        """
        기존 VideoReader.read()와 유사한 인터페이스 유지
        (성공 여부, 프레임) 반환
        """
        if self.latest_frame is not None:
            # 읽어간 후에는 일단 False로 처리 (선택 사항)
            ret = True
            frame = self.latest_frame.copy()
            self.new_frame_available = False
            return ret, frame
        return False, None

    def release(self):
        # ROS에서는 딱히 release가 필요 없으나 구조 유지를 위해 정의
        self.sub.unregister()
        rospy.loginfo("Camera topic unsubscribed.")