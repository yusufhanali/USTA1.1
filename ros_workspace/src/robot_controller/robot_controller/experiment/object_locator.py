import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
import threading
import traceback

import pyrealsense2 as rs
import numpy as np
import cv2

from geometry_msgs.msg import PointStamped

from robot_control.new_controller import spin_thread

class ObjectLocatorNode(Node):
    def init_rs(self):
        self.rs_pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        config.enable_device("313522301714")

        self.rs_profile = self.rs_pipeline.start(config)

        align_to = rs.stream.color
        self.rs_align = rs.align(
            align_to
        )
        
    def init_cv2(self):
        self.clicked_u, self.clicked_v = -1, -1        
        self.last_clicked_u, self.last_clicked_v = -1, -1
        
        cv2.namedWindow("interactive_rs_feed")
        cv2.setMouseCallback("interactive_rs_feed", self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_u, self.clicked_v = x, y
    
    def __init__(self, base_frame="overhead_hri_camera_link"):
        super().__init__("object_locator_node")
        
        self.init_rs()
        self.init_cv2()
        
        self.base_frame = base_frame
        
        self.point_publisher = self.create_publisher(PointStamped, "/clicked_point_coordinates", 10)
        
        self.get_logger().info("ObjectLocatorNode initialized.")
       
    def publish_point(self, x, y, z):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = self.base_frame
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z

        self.point_publisher.publish(point_msg)
        
    def run(self):        
        try:
            for i in range(10):
                self.rs_pipeline.wait_for_frames()

            while True:
                frames = self.rs_pipeline.wait_for_frames()

                if not frames:
                    self.get_logger().warning("Frames not available.")
                    continue

                if self.clicked_u != -1 and self.clicked_v != -1:
                    u, v = self.clicked_u, self.clicked_v
                    
                    aligned_frames = self.rs_align.process(frames)                    
                    aligned_depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                
                    if not aligned_depth_frame or not color_frame:
                        self.get_logger().warning("Frame not available.")
                        continue
                
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    depth_value = aligned_depth_frame.get_distance(u, v)
                    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_value)
                    
                    self.last_clicked_u, self.last_clicked_v = self.clicked_u, self.clicked_v
                    self.clicked_u, self.clicked_v = -1, -1
                    
                    if depth_value == 0:
                        self.get_logger().warning(f"Depth value at pixel ({u}, {v}) is zero. Cannot compute 3D coordinates.")
                        continue
                    
                    self.publish_point(depth_point[2], -depth_point[0], -depth_point[1])
                    self.get_logger().info(f"Clicked Pixel\n(u, v): ({self.clicked_u}, {self.clicked_v})\n3D Coordinates: ({depth_point[0]:.3f}, {depth_point[1]:.3f}, {depth_point[2]:.3f})")
                else:
                    color_frame = frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())

                if self.last_clicked_u != -1 and self.last_clicked_v != -1:
                    cv2.circle(color_image, (self.last_clicked_u, self.last_clicked_v), 5, (0, 0, 180), -1)
                    window_text = f"X:{depth_point[0]:.2f} Y:{depth_point[1]:.2f} Z:{depth_point[2]:.2f}m"
                    cv2.putText(
                        color_image,
                        window_text,
                        (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 180),
                        2,
                    )

                cv2.imshow("interactive_rs_feed", color_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()
            
    def shutdown(self):
        try:
            self.rs_pipeline.stop()
            cv2.destroyAllWindows()
        except:
            pass
            

def main(args=None):

    try:
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)    
        object_locator = ObjectLocatorNode()
        
        locator_spin_thread = threading.Thread(target=spin_thread, args=(object_locator,))
        locator_spin_thread.start()
        
        object_locator.run()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
    except Exception as e:    
        print("Error in main: ", traceback.format_exc())   
        pass
    
    object_locator.shutdown()
    
    rclpy.try_shutdown()
    locator_spin_thread.join()
    
    print("\n take care of yourself \n")