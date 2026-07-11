import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from geometry_msgs.msg import TransformStamped
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO  # Import YOLO-v8

class HeadPosePublisher(Node):
    def __init__(self):
        super().__init__('head_pose_publisher')
        
        self.detector = YOLO('../../yolov8n.pt')
        self.device = 'cuda:0'
        self.detector.to(self.device)
        
        self.bridge = CvBridge()
        self.depth_intrinsics = None
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.intrinsics_callback,
            10
        )
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        self.color_image = None
        self.depth_image = None
        
        # Rate control
        self.timer = self.create_timer(1.0 / 15.0, self.run_loop)

    def intrinsics_callback(self, data):
        self.depth_intrinsics = rs.intrinsics()
        self.depth_intrinsics.width = data.width
        self.depth_intrinsics.height = data.height
        self.depth_intrinsics.ppx = data.k[2]
        self.depth_intrinsics.ppy = data.k[5]
        self.depth_intrinsics.fx = data.k[0]
        self.depth_intrinsics.fy = data.k[4]
        self.depth_intrinsics.model = rs.distortion.brown_conrady
        self.depth_intrinsics.coeffs = data.d
        # self.get_logger().info("Camera intrinsics updated.")

    def color_callback(self, data):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting color image: {e}")

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def get_bb_yolo(self, frame: np.ndarray):
        results = self.detector(frame, stream=True, conf=0.5, verbose=False)
        if results is not None:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    (x_min, y_min, x_max, y_max) = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                    return (x_min, y_min, x_max, y_max)
        return None

    def run_loop(self):
        if self.color_image is None or self.depth_image is None or self.depth_intrinsics is None:
            return
        bbox = self.get_bb_yolo(self.color_image)
        if bbox is None:
            return

        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min, y_min = max(0, x_min), max(0, y_min)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        depth = self.depth_image[center_y, center_x]
        
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [center_x, center_y], depth)
        translation = (depth_point[0] / 1000, depth_point[1] / 1000, depth_point[2] / 1000)
        
        self.publish_head(translation)

    def publish_head(self, translation):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "camera_color_optical_frame"
        t.child_frame_id = "exp/head"
        t.transform.translation.x = float(translation[0]) 
        t.transform.translation.y = float(translation[1]) 
        t.transform.translation.z = float(translation[2]) 
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)
        # self.get_logger().info(f"Published head transform: {t}")

def main(args=None):
    rclpy.init(args=args)
    node = HeadPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
